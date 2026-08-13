"""Project frozen HPM context onto players without changing its predictions.

This module is deliberately an *attribution* audit.  It regresses the frozen
context offset used by HPM onto the same signed player matrix as RAPM and keeps
the unexplained portion as matchup-specific residual synergy.  It never uses
the projection to refit the HPM outcome model or to claim a new frozen forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.models.baselines import (
    RidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_TARGET_SEASON = "2025-26"
SOURCE_MODEL = (
    "forward_centered_value_conditioned_aging_"
    "bounded_hierarchical_portable_matchup_contextual_rapm"
)
MODEL = "context_reattributed_rapm_audit"
RUN_PREFIX = "context-reattributed-rapm-audit"
MINIMUM_LINEUP_POSSESSIONS = 25.0


@dataclass(frozen=True)
class ContextProjection:
    """Ridge projection of an HPM context vector onto signed player features."""

    player_ids: tuple[int, ...]
    coefficients: np.ndarray
    intercept: float
    prediction: np.ndarray
    residual: np.ndarray
    selected_lambda: float


def build_context_reattributed_rapm_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    target_season: str = DEFAULT_TARGET_SEASON,
    minimum_lineup_possessions: float = MINIMUM_LINEUP_POSSESSIONS,
) -> Path:
    """Build an HPM context-reallocation audit for one realized target season.

    The frozen prior-season HPM context state is evaluated on the realized
    target-season stints, then projected onto target-season player identities.
    This makes CR-RAPM a retrospective interpretation of an already frozen HPM
    forecast, rather than a preseason model evaluation.
    """

    if minimum_lineup_possessions <= 0:
        raise ValueError("minimum_lineup_possessions must be positive")
    root = Path(artifacts_dir)
    source = Path(source_run_dir) if source_run_dir is not None else _latest_run(
        root / SOURCE_MODEL / target_season
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != SOURCE_MODEL:
        raise ValueError("CR-RAPM audit requires the published Value-Conditioned Aging HPM")

    source_season = _previous_season(target_season)
    context_models = joblib.load(source / "season_context_models.joblib")
    context_model = context_models.get(source_season)
    if not isinstance(context_model, MatchupContextualModel):
        raise ValueError(f"Source artifact has no {source_season} portable context state")
    profiles = pd.read_parquet(source / "target_player_profiles.parquet")
    hpm_coefficients = pd.read_parquet(source / "historical_player_coefficients.parquet")
    target_coefficients = hpm_coefficients.loc[
        hpm_coefficients["season"].eq(target_season),
        ["player_id", "rapm", "prior_rapm", "rapm_adjustment_from_prior", "selected_lambda"],
    ].copy()
    if target_coefficients.empty:
        raise ValueError(f"Source artifact has no HPM coefficients for {target_season}")
    selected_lambdas = target_coefficients["selected_lambda"].dropna().unique()
    if len(selected_lambdas) != 1:
        raise ValueError("HPM target season must expose one selected player regularization")
    projection_lambda = float(selected_lambdas[0])

    stints = read_rapm_stints(target_season, analytical_dir=analytical_dir)
    context_target = context_model.predict_lineups(
        stints["home_player_ids"].tolist(),
        stints["away_player_ids"].tolist(),
        profiles,
    )
    projection = fit_context_projection(stints, context_target, projection_lambda)
    players = _player_ledger(
        stints,
        projection,
        target_coefficients,
        profiles,
        target_season=target_season,
        source_season=source_season,
    )
    stints_ledger = _stint_ledger(stints, context_target, projection)
    lineups = _lineup_residuals(
        stints_ledger,
        profiles,
        minimum_possessions=minimum_lineup_possessions,
    )
    metrics = _audit_metrics(context_target, projection.prediction, projection.residual, stints)
    _assert_ledger_identity(context_target, projection.prediction, projection.residual)
    return _write_run(
        source=source,
        target_season=target_season,
        source_season=source_season,
        projection_lambda=projection_lambda,
        players=players,
        stints=stints_ledger,
        lineups=lineups,
        metrics=metrics,
        artifacts_dir=root,
    )


def fit_context_projection(
    stints: pd.DataFrame,
    context_target: np.ndarray,
    regularization: float,
) -> ContextProjection:
    """Fit the CR-RAPM player projection using HPM's own lambda convention."""

    values = np.asarray(context_target, dtype=float)
    if len(stints) != len(values):
        raise ValueError("Context target rows must match stints")
    if not np.isfinite(values).all():
        raise ValueError("Context target must be finite")
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        vocabulary_mapping(player_ids),
        multiple=True,
    )
    weights = stints["possessions"].to_numpy(dtype=float)
    fitted = RidgeLineupModel(regularization).fit(matrix, values, weights)
    prediction = fitted.predict(matrix)
    residual = values - prediction
    return ContextProjection(
        player_ids=player_ids,
        coefficients=fitted.coef_,
        intercept=fitted.intercept_,
        prediction=prediction,
        residual=residual,
        selected_lambda=float(regularization),
    )


def _player_ledger(
    stints: pd.DataFrame,
    projection: ContextProjection,
    hpm_coefficients: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    target_season: str,
    source_season: str,
) -> pd.DataFrame:
    exposure = _player_exposure(stints)
    projection_frame = pd.DataFrame(
        {
            "player_id": projection.player_ids,
            "context_reattribution": projection.coefficients,
        }
    )
    names = profiles.loc[:, ["player_id", "player_name"]].drop_duplicates("player_id")
    output = (
        projection_frame.merge(hpm_coefficients, on="player_id", how="left", validate="one_to_one")
        .merge(exposure, on="player_id", how="left", validate="one_to_one")
        .merge(names, on="player_id", how="left", validate="one_to_one")
    )
    if output["rapm"].isna().any() or output["player_name"].isna().any():
        raise ValueError("All projected players must have HPM ratings and profiles")
    output.insert(0, "season", target_season)
    output.insert(1, "source_context_season", source_season)
    output["cr_rapm"] = output["rapm"] + output["context_reattribution"]
    output["projection_lambda"] = projection.selected_lambda
    output = output.rename(columns={"rapm": "hpm_rapm", "possessions": "on_court_possessions"})
    return output.sort_values(
        ["cr_rapm", "on_court_possessions", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _player_exposure(stints: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[int, float]] = []
    for lineups in (stints["home_player_ids"], stints["away_player_ids"]):
        for lineup, possessions in zip(lineups, stints["possessions"], strict=True):
            rows.extend((int(player_id), float(possessions)) for player_id in lineup)
    return (
        pd.DataFrame(rows, columns=["player_id", "possessions"])
        .groupby("player_id", as_index=False, sort=True)["possessions"]
        .sum()
    )


def _stint_ledger(
    stints: pd.DataFrame,
    context_target: np.ndarray,
    projection: ContextProjection,
) -> pd.DataFrame:
    columns = [
        "game_id",
        "game_date",
        "game_time_utc",
        "stint_index",
        "possessions",
        "home_team_tricode",
        "away_team_tricode",
        "home_player_ids",
        "away_player_ids",
    ]
    output = stints.loc[:, columns].copy()
    output["hpm_context"] = np.asarray(context_target, dtype=float)
    output["reattributed_player_context"] = projection.prediction
    output["residual_synergy"] = projection.residual
    output["context_projection_intercept"] = projection.intercept
    return output


def _lineup_residuals(
    stints: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    minimum_possessions: float,
) -> pd.DataFrame:
    names = dict(zip(profiles["player_id"].astype(int), profiles["player_name"], strict=True))
    frame = stints.copy()
    frame["home_lineup_key"] = frame["home_player_ids"].map(_lineup_key)
    frame["away_lineup_key"] = frame["away_player_ids"].map(_lineup_key)
    for value_column in ("hpm_context", "reattributed_player_context", "residual_synergy"):
        frame[f"weighted_{value_column}"] = frame[value_column] * frame["possessions"]
    grouped = frame.groupby(["home_lineup_key", "away_lineup_key"], as_index=False, sort=False).agg(
        possessions=("possessions", "sum"),
        games=("game_id", "nunique"),
        weighted_hpm_context=("weighted_hpm_context", "sum"),
        weighted_reattributed_player_context=("weighted_reattributed_player_context", "sum"),
        weighted_residual_synergy=("weighted_residual_synergy", "sum"),
    )
    for value_column in ("hpm_context", "reattributed_player_context", "residual_synergy"):
        grouped[value_column] = grouped[f"weighted_{value_column}"] / grouped["possessions"]
    grouped["home_lineup"] = grouped["home_lineup_key"].map(
        lambda value: _lineup_names(value, names)
    )
    grouped["away_lineup"] = grouped["away_lineup_key"].map(
        lambda value: _lineup_names(value, names)
    )
    output = grouped.loc[grouped["possessions"].ge(minimum_possessions)].copy()
    return output.sort_values(
        ["residual_synergy", "possessions"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)


def _lineup_key(lineup: object) -> str:
    return ",".join(str(int(player_id)) for player_id in sorted(lineup))  # type: ignore[arg-type]


def _lineup_names(key: str, names: dict[int, object]) -> str:
    return ", ".join(str(names[int(player_id)]) for player_id in key.split(","))


def _audit_metrics(
    target: np.ndarray,
    projection: np.ndarray,
    residual: np.ndarray,
    stints: pd.DataFrame,
) -> pd.DataFrame:
    weights = stints["possessions"].to_numpy(dtype=float)
    mean = float(np.average(target, weights=weights))
    total_ss = float(np.sum(weights * np.square(target - mean)))
    residual_ss = float(np.sum(weights * np.square(residual)))
    correlation = float(np.corrcoef(target, projection)[0, 1])
    return pd.DataFrame(
        [
            {
                "weighted_context_mean": mean,
                "weighted_context_rmse": float(
                    np.sqrt(np.average(np.square(target), weights=weights))
                ),
                "weighted_projection_rmse": float(
                    np.sqrt(np.average(np.square(residual), weights=weights))
                ),
                "weighted_projection_mae": float(np.average(np.abs(residual), weights=weights)),
                "weighted_r_squared": 1.0 - residual_ss / total_ss if total_ss else float("nan"),
                "unweighted_pearson_correlation": correlation,
                "residual_p05": float(np.quantile(residual, 0.05)),
                "residual_p50": float(np.quantile(residual, 0.50)),
                "residual_p95": float(np.quantile(residual, 0.95)),
                "stint_count": len(stints),
                "game_count": int(stints["game_id"].nunique()),
                "possessions": float(weights.sum()),
            }
        ]
    )


def _assert_ledger_identity(
    target: np.ndarray,
    projection: np.ndarray,
    residual: np.ndarray,
) -> None:
    if not np.allclose(target, projection + residual, atol=1e-10, rtol=1e-10):
        raise ValueError("Context reattribution ledger does not reconcile")


def _write_run(
    *,
    source: Path,
    target_season: str,
    source_season: str,
    projection_lambda: float,
    players: pd.DataFrame,
    stints: pd.DataFrame,
    lineups: pd.DataFrame,
    metrics: pd.DataFrame,
    artifacts_dir: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        players.to_parquet(temporary / "player_context_reattribution.parquet", index=False)
        stints.to_parquet(temporary / "stint_context_ledger.parquet", index=False)
        lineups.to_parquet(temporary / "lineup_residual_synergy.parquet", index=False)
        metrics.to_parquet(temporary / "audit_metrics.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL,
            "target_season": target_season,
            "source_season": source_season,
            "source_run_dir": str(source),
            "projection_lambda": projection_lambda,
            "created_at": now.isoformat(),
            "method": (
                "weighted ridge projection of frozen HPM context onto the target-season "
                "signed player matrix, preserving a residual-synergy ledger"
            ),
            "interpretation_contract": (
                "retrospective attribution only: realized target-season lineups project "
                "a prior-season frozen context state and do not form a new preseason forecast"
            ),
            "ledger_identity": "C = (X gamma + intercept_gamma) + residual_synergy",
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build the CR-RAPM attribution audit artifact."""

    parser = argparse.ArgumentParser(description="Build a context-reattributed RAPM audit")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument(
        "--minimum-lineup-possessions",
        type=float,
        default=MINIMUM_LINEUP_POSSESSIONS,
    )
    args = parser.parse_args()
    run = build_context_reattributed_rapm_audit(
        target_season=args.through_season,
        minimum_lineup_possessions=args.minimum_lineup_possessions,
    )
    print(f"Context-reattributed RAPM audit: run={run}")


if __name__ == "__main__":
    main()
