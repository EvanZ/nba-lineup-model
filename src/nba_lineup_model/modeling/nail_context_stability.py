"""Audit frozen versus completed NAIL non-additive lineup effects.

Frozen target-season scores use the prior completed context state. Completed
scores use the target's post-season context state. Both use the same strictly
lagged player profiles; observed stints are only an exposure-allocation device.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    lineup_side_context_features,
)
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty import (
    MODEL_NAME,
)
from nba_lineup_model.modeling.linear_hpm_x3_compilation_audit import (
    linear_raw_context_coefficients,
)
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_ARTIFACT_SEASON = "2025-26"
OUTPUT_ROOT = Path("artifacts/models/analysis/nail_context_stability")
MINIMUM_LINEUP_POSSESSIONS = 100.0
MINIMUM_PLAYER_POSSESSIONS = 250.0


@dataclass(frozen=True)
class NailContextStabilityRun:
    """Persisted frozen-versus-completed NALE stability audit."""

    run_dir: Path
    source_run_dir: Path
    seasons: tuple[str, ...]


def build_nail_context_stability_audit(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    output_root: Path | str = OUTPUT_ROOT,
    minimum_lineup_possessions: float = MINIMUM_LINEUP_POSSESSIONS,
    minimum_player_possessions: float = MINIMUM_PLAYER_POSSESSIONS,
) -> NailContextStabilityRun:
    """Compare frozen and completed NALE on observed target-season stints."""

    targets = tuple(validate_season(season) for season in seasons)
    if not targets:
        raise ValueError("At least one target season is required")
    if minimum_lineup_possessions <= 0 or minimum_player_possessions <= 0:
        raise ValueError("Minimum possession thresholds must be positive")
    artifacts_path = Path(artifacts_dir)
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(artifacts_path / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("NALE stability audit requires the published NAIL-RAPM v1.0 artifact")
    models = joblib.load(source / "season_context_models.joblib")
    panel = pd.read_parquet(player_season_panel_path)
    profile_cache_path = (
        Path("artifacts/web/historical_profiles") / MODEL_NAME / f"{source.name}.parquet"
    )
    historical_profiles = (
        pd.read_parquet(profile_cache_path) if profile_cache_path.is_file() else pd.DataFrame()
    )
    exposure_cohort: pd.DataFrame | None = None

    coefficient_rows: list[pd.DataFrame] = []
    stint_rows: list[pd.DataFrame] = []
    lineup_rows: list[pd.DataFrame] = []
    player_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for target in targets:
        source_season = _previous_season(target)
        frozen_model = _nail_model(models, source_season)
        completed_model = _nail_model(models, target)
        print(f"[{target}] scoring frozen {source_season} and completed NALE", flush=True)
        stints = read_rapm_stints(target, analytical_dir=analytical_dir)
        participant_ids = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
        profiles = _target_profiles(
            historical_profiles,
            panel,
            target=target,
            participant_ids=participant_ids,
            analytical_dir=analytical_dir,
            curated_dir=curated_dir,
            exposure_cohort=exposure_cohort,
        )
        print(f"[{target}] loaded {len(profiles):,} lagged player profiles", flush=True)
        coefficients = _coefficient_ledger(
            frozen_model, completed_model, target=target, source=source_season
        )
        print(f"[{target}] scoring {len(stints):,} stints", flush=True)
        stints = _stint_ledger(
            stints,
            profiles,
            frozen_model=frozen_model,
            completed_model=completed_model,
            target=target,
            source=source_season,
        )
        print(f"[{target}] aggregating observed lineups", flush=True)
        lineups = _lineup_ledger(stints, minimum_possessions=minimum_lineup_possessions)
        print(f"[{target}] aggregating player exposure", flush=True)
        players = _player_ledger(stints, profiles, minimum_possessions=minimum_player_possessions)
        coefficient_rows.append(coefficients)
        stint_rows.append(stints)
        lineup_rows.append(lineups)
        player_rows.append(players)
        summary_rows.append(_summary(coefficients, stints, lineups, players))

    return _write_run(
        source=source,
        seasons=targets,
        coefficients=pd.concat(coefficient_rows, ignore_index=True),
        stints=pd.concat(stint_rows, ignore_index=True),
        lineups=pd.concat(lineup_rows, ignore_index=True),
        players=pd.concat(player_rows, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        output_root=Path(output_root),
        minimum_lineup_possessions=minimum_lineup_possessions,
        minimum_player_possessions=minimum_player_possessions,
    )


def _nail_model(models: dict[str, MatchupContextualModel], season: str) -> MatchupContextualModel:
    model = models.get(season)
    if not isinstance(model, MatchupContextualModel):
        raise ValueError(f"NAIL-RAPM artifact has no context state for {season}")
    if model.feature_set != CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
        raise ValueError("NALE stability audit requires the canonical NAIL feature contract")
    if tuple(model.pipeline.named_steps) != ("scale", "ridge"):
        raise ValueError("NALE stability audit requires a linear Ridge context model")
    return model


def _target_profiles(
    historical_profiles: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    target: str,
    participant_ids: set[int],
    analytical_dir: Path | str,
    curated_dir: Path | str,
    exposure_cohort: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load published lagged profiles, rebuilding only for local cache misses."""

    cached = historical_profiles.loc[
        historical_profiles.get("season", pd.Series(dtype=str)).astype(str).eq(target)
    ].copy()
    if not cached.empty:
        cached = cached.drop(columns="season")
        cached_ids = set(cached["player_id"].astype(int))
        if cached_ids == participant_ids and not cached["player_id"].duplicated().any():
            return cached
    if exposure_cohort is None:
        exposure_cohort = prepare_player_exposure_cohort(
            panel.loc[panel["season"].astype(str).le(target)],
            through_season=target,
            analytical_dir=analytical_dir,
        )
    return build_contextual_player_profiles(
        panel,
        target_season=target,
        target_player_ids=participant_ids,
        analytical_dir=str(analytical_dir),
        curated_dir=str(curated_dir),
        exposure_cohort=exposure_cohort,
    )


def _nonadditive_coefficients(model: MatchupContextualModel) -> pd.Series:
    return linear_raw_context_coefficients(model).drop(
        labels=list(LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES)
    )


def _edge(home: pd.DataFrame, away: pd.DataFrame, model: MatchupContextualModel) -> np.ndarray:
    coefficients = _nonadditive_coefficients(model)
    relative = home.loc[:, coefficients.index].to_numpy(float) - away.loc[
        :, coefficients.index
    ].to_numpy(float)
    return relative @ coefficients.to_numpy(float)


def _coefficient_ledger(
    frozen_model: MatchupContextualModel,
    completed_model: MatchupContextualModel,
    *,
    target: str,
    source: str,
) -> pd.DataFrame:
    frozen = _nonadditive_coefficients(frozen_model).rename("frozen_raw_coefficient")
    completed = _nonadditive_coefficients(completed_model).rename("completed_raw_coefficient")
    output = pd.concat([frozen, completed], axis=1).rename_axis("feature").reset_index()
    output.insert(0, "source_context_season", source)
    output.insert(0, "season", target)
    output["coefficient_revision"] = (
        output["completed_raw_coefficient"] - output["frozen_raw_coefficient"]
    )
    output["sign_agrees"] = np.sign(output["completed_raw_coefficient"]) == np.sign(
        output["frozen_raw_coefficient"]
    )
    return output


def _stint_ledger(
    stints: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    frozen_model: MatchupContextualModel,
    completed_model: MatchupContextualModel,
    target: str,
    source: str,
) -> pd.DataFrame:
    home = lineup_side_context_features(
        stints["home_player_ids"].tolist(), profiles, feature_set=frozen_model.feature_set
    )
    away = lineup_side_context_features(
        stints["away_player_ids"].tolist(), profiles, feature_set=frozen_model.feature_set
    )
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
    output.insert(0, "source_context_season", source)
    output.insert(0, "season", target)
    output["frozen_nale"] = _edge(home, away, frozen_model)
    output["completed_nale"] = _edge(home, away, completed_model)
    output["nale_revision"] = output["completed_nale"] - output["frozen_nale"]
    return output


def _lineup_ledger(stints: pd.DataFrame, *, minimum_possessions: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for side, team_column, lineup_column, sign in (
        ("home", "home_team_tricode", "home_player_ids", 1.0),
        ("away", "away_team_tricode", "away_player_ids", -1.0),
    ):
        frame = stints.loc[:, ["season", "source_context_season", "game_id", "possessions"]].copy()
        frame["side"] = side
        frame["team"] = stints[team_column].astype("string")
        frame["lineup_key"] = stints[lineup_column].map(_lineup_key)
        frame["player_ids"] = stints[lineup_column]
        for column in ("frozen_nale", "completed_nale", "nale_revision"):
            frame[f"weighted_{column}"] = (
                sign * stints[column].to_numpy(float) * frame["possessions"]
            )
        frames.append(frame)
    grouped = (
        pd.concat(frames, ignore_index=True)
        .groupby(
            ["season", "source_context_season", "team", "lineup_key"], as_index=False, sort=True
        )
        .agg(
            possessions=("possessions", "sum"),
            games=("game_id", "nunique"),
            player_ids=("player_ids", "first"),
            weighted_frozen_nale=("weighted_frozen_nale", "sum"),
            weighted_completed_nale=("weighted_completed_nale", "sum"),
            weighted_nale_revision=("weighted_nale_revision", "sum"),
        )
    )
    grouped = grouped.loc[grouped["possessions"].ge(minimum_possessions)].copy()
    for column in ("frozen_nale", "completed_nale", "nale_revision"):
        grouped[column] = grouped.pop(f"weighted_{column}") / grouped["possessions"]
    return grouped.sort_values(["season", "possessions"], ascending=[True, False], kind="stable")


def _player_ledger(
    stints: pd.DataFrame, profiles: pd.DataFrame, *, minimum_possessions: float
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_columns = [
        "season",
        "source_context_season",
        "possessions",
        "frozen_nale",
        "completed_nale",
    ]
    for lineup_column, sign in (("home_player_ids", 1.0), ("away_player_ids", -1.0)):
        frame = stints.loc[:, base_columns].copy()
        frame["player_id"] = stints[lineup_column]
        frame = frame.explode("player_id", ignore_index=True)
        frame["player_id"] = frame["player_id"].astype("int64")
        frame["on_court_possessions"] = frame.pop("possessions").astype(float)
        frame["weighted_frozen_nale"] = (
            sign * frame.pop("frozen_nale").astype(float) * frame["on_court_possessions"]
        )
        frame["weighted_completed_nale"] = (
            sign * frame.pop("completed_nale").astype(float) * frame["on_court_possessions"]
        )
        frames.append(frame)
    output = (
        pd.concat(frames, ignore_index=True)
        .groupby(["season", "source_context_season", "player_id"], as_index=False, sort=True)
        .sum()
    )
    output = output.loc[output["on_court_possessions"].ge(minimum_possessions)].copy()
    output["frozen_nale"] = output.pop("weighted_frozen_nale") / output["on_court_possessions"]
    output["completed_nale"] = (
        output.pop("weighted_completed_nale") / output["on_court_possessions"]
    )
    output["nale_revision"] = output["completed_nale"] - output["frozen_nale"]
    names = profiles.loc[:, ["player_id", "player_name"]].drop_duplicates("player_id")
    return output.merge(names, on="player_id", how="left", validate="many_to_one").sort_values(
        ["season", "on_court_possessions"], ascending=[True, False], kind="stable"
    )


def _summary(
    coefficients: pd.DataFrame, stints: pd.DataFrame, lineups: pd.DataFrame, players: pd.DataFrame
) -> dict[str, object]:
    return {
        "season": str(stints["season"].iloc[0]),
        "source_context_season": str(stints["source_context_season"].iloc[0]),
        "coefficient_count": len(coefficients),
        "coefficient_pearson": _pearson(
            coefficients["frozen_raw_coefficient"], coefficients["completed_raw_coefficient"]
        ),
        "coefficient_spearman": _spearman(
            coefficients["frozen_raw_coefficient"], coefficients["completed_raw_coefficient"]
        ),
        "coefficient_sign_agreement": float(coefficients["sign_agrees"].mean()),
        "stint_count": len(stints),
        "stint_possessions": float(stints["possessions"].sum()),
        "stint_weighted_pearson": _weighted_pearson(
            stints["frozen_nale"], stints["completed_nale"], stints["possessions"]
        ),
        "stint_weighted_spearman": _weighted_spearman(
            stints["frozen_nale"], stints["completed_nale"], stints["possessions"]
        ),
        "lineup_count": len(lineups),
        "lineup_weighted_pearson": _weighted_pearson(
            lineups["frozen_nale"], lineups["completed_nale"], lineups["possessions"]
        ),
        "lineup_weighted_spearman": _weighted_spearman(
            lineups["frozen_nale"], lineups["completed_nale"], lineups["possessions"]
        ),
        "player_count": len(players),
        "player_weighted_pearson": _weighted_pearson(
            players["frozen_nale"], players["completed_nale"], players["on_court_possessions"]
        ),
        "player_weighted_spearman": _weighted_spearman(
            players["frozen_nale"], players["completed_nale"], players["on_court_possessions"]
        ),
    }


def _pearson(left: pd.Series, right: pd.Series) -> float:
    return (
        float(left.corr(right, method="pearson"))
        if left.nunique() > 1 and right.nunique() > 1
        else float("nan")
    )


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return (
        float(left.corr(right, method="spearman"))
        if left.nunique() > 1 and right.nunique() > 1
        else float("nan")
    )


def _weighted_pearson(left: pd.Series, right: pd.Series, weights: pd.Series) -> float:
    x, y, w = left.to_numpy(float), right.to_numpy(float), weights.to_numpy(float)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or not (w > 0).all():
        return float("nan")
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    denominator = np.sqrt(np.average(x**2, weights=w) * np.average(y**2, weights=w))
    return float(np.average(x * y, weights=w) / denominator) if denominator else float("nan")


def _weighted_spearman(left: pd.Series, right: pd.Series, weights: pd.Series) -> float:
    return _weighted_pearson(left.rank(), right.rank(), weights)


def _lineup_key(lineup: object) -> str:
    return ":".join(str(player_id) for player_id in sorted(int(player_id) for player_id in lineup))


def _write_run(
    *,
    source: Path,
    seasons: tuple[str, ...],
    coefficients: pd.DataFrame,
    stints: pd.DataFrame,
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    summary: pd.DataFrame,
    output_root: Path,
    minimum_lineup_possessions: float,
    minimum_player_possessions: float,
) -> NailContextStabilityRun:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"nail-context-stability-{seasons[0]}-to-{seasons[-1]}-{timestamp}-{uuid4().hex[:8]}"
    run_dir = output_root / f"{seasons[0]}_to_{seasons[-1]}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    coefficients.to_parquet(run_dir / "coefficient_stability.parquet", index=False)
    stints.to_parquet(run_dir / "stint_nale_stability.parquet", index=False)
    lineups.to_parquet(run_dir / "lineup_nale_stability.parquet", index=False)
    players.to_parquet(run_dir / "player_nale_stability.parquet", index=False)
    summary.to_parquet(run_dir / "summary.parquet", index=False)
    metadata = {
        "model": MODEL_NAME,
        "source_run_dir": str(source),
        "seasons": list(seasons),
        "frozen_context_rule": "target t uses completed context state t-1",
        "completed_context_rule": "target t uses completed context state t",
        "profile_information_rule": "both scores use target profiles formed before t",
        "allocation_rule": "observed target stints are used only for exposure aggregation",
        "minimum_lineup_possessions": minimum_lineup_possessions,
        "minimum_player_possessions": minimum_player_possessions,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (run_dir.parent / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailContextStabilityRun(run_dir=run_dir, source_run_dir=source, seasons=seasons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen versus completed NAIL context")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument(
        "--minimum-lineup-possessions", type=float, default=MINIMUM_LINEUP_POSSESSIONS
    )
    parser.add_argument(
        "--minimum-player-possessions", type=float, default=MINIMUM_PLAYER_POSSESSIONS
    )
    args = parser.parse_args()
    run = build_nail_context_stability_audit(
        seasons=tuple(args.seasons),
        minimum_lineup_possessions=args.minimum_lineup_possessions,
        minimum_player_possessions=args.minimum_player_possessions,
    )
    print(f"NAIL context stability audit: run={run.run_dir}")


if __name__ == "__main__":
    main()
