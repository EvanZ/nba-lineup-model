"""Compile additive Linear-HPM x3 context into player adjustments and audit identity."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    _playoff_partition_exists,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _read_playoff_possessions
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "forward_hpm_x3_linear_ridge_without_uncertainty"
MODEL_ROOT = Path("artifacts/models") / MODEL_NAME / "2025-26"
OUTPUT_ROOT = Path("artifacts/models/analysis/linear_hpm_x3_compilation_audit")

# These feature-map coordinates are exact sums of per-player values. Every
# remaining x3 coordinate is a non-additive five-player unit shape.
_BASKETBALL_ADDITIVE_FEATURE_TO_PLAYER_PROFILE = {
    "three_pa_per_100": "three_pa_per_100",
    "three_pm_per_100": "three_pm_per_100",
    "assists_per_100": "assists_per_100",
    "turnovers_per_100": "turnovers_per_100",
    "usage_per_100": "usage_per_100",
    "steals_per_100": "steals_per_100",
    "blocks_per_100": "blocks_per_100",
    "offensive_rebound_claim_total": "offensive_rebound_pct",
}
_LEGACY_CALIBRATION_FEATURE_TO_PLAYER_PROFILE = {
    "imputed_count": "profile_imputed",
    "replacement_weight": "profile_replacement_weight",
}

# Public canonical mapping. The historical 16-feature contract is selected
# explicitly through _additive_feature_map_for_contract below.
ADDITIVE_FEATURE_TO_PLAYER_PROFILE = _BASKETBALL_ADDITIVE_FEATURE_TO_PLAYER_PROFILE
ADDITIVE_FEATURE_COLUMNS = tuple(ADDITIVE_FEATURE_TO_PLAYER_PROFILE)
CHUNK_SIZE = 20_000


def run_linear_hpm_x3_compilation_audit(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    model_root: Path | str = MODEL_ROOT,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    output_root: Path | str = OUTPUT_ROOT,
) -> Path:
    """Audit whether linear additive context exactly compiles into player values."""

    target_seasons = tuple(validate_season(season) for season in seasons)
    run_dir = _latest_run(Path(model_root))
    models: dict[str, MatchupContextualModel] = joblib.load(
        run_dir / "season_context_models.joblib"
    )
    priors = pd.read_parquet(run_dir / "season_player_priors.parquet")
    panel = pd.read_parquet(player_season_panel_path)
    summaries: list[dict[str, object]] = []
    adjustments: list[pd.DataFrame] = []
    for target in target_seasons:
        source = _previous_season(target)
        print(f"[{target}] loading frozen linear context from {source}", flush=True)
        model = models.get(source)
        if model is None:
            raise ValueError(f"Linear HPM x3 artifact lacks context state for {source}")
        if model.feature_set not in {
            CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
            CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        }:
            raise ValueError("Compilation audit requires the HPM x3 feature contract")
        cohorts = {"regular_season": read_neural_possessions(target, analytical_dir=analytical_dir)}
        if _playoff_partition_exists(target, Path(curated_dir)):
            cohorts["playoffs"] = _read_playoff_possessions(target, curated_dir)[0]
        player_ids = _participant_ids(cohorts.values())
        profiles = build_contextual_player_profiles(
            panel,
            target_season=target,
            target_player_ids=player_ids,
            analytical_dir=str(analytical_dir),
            curated_dir=str(curated_dir),
        )
        lineup_features = _cached_lineup_features(
            cohorts.values(),
            profiles=profiles,
            feature_set=model.feature_set,
        )
        print(
            f"[{target}] cached {len(lineup_features):,} observed lineups across "
            f"{len(cohorts)} cohort(s)",
            flush=True,
        )
        raw_coefficients = linear_raw_context_coefficients(model)
        additive_feature_map = _additive_feature_map_for_contract(model.feature_set)
        target_priors = priors.loc[
            priors["season"].eq(target), ["player_id", "prior_rapm"]
        ]
        if target_priors.empty:
            raise ValueError(f"Linear HPM x3 artifact lacks player priors for {target}")
        adjustments.append(
            _compiled_player_adjustments(
                profiles,
                raw_coefficients=raw_coefficients,
                additive_feature_map=additive_feature_map,
                target_priors=target_priors,
                target_season=target,
                source_season=source,
            )
        )
        for cohort, possessions in cohorts.items():
            print(f"[{target} {cohort}] auditing {len(possessions):,} possessions", flush=True)
            summaries.append(
                _audit_possessions(
                    possessions,
                    model=model,
                    profiles=profiles,
                    lineup_features=lineup_features,
                    raw_coefficients=raw_coefficients,
                    additive_feature_map=additive_feature_map,
                    target_priors=target_priors,
                    target_season=target,
                    source_season=source,
                    cohort=cohort,
                )
            )
    print("writing compilation identity artifact", flush=True)
    return _write_report(
        pd.DataFrame(summaries),
        pd.concat(adjustments, ignore_index=True),
        source_run_dir=run_dir,
        feature_set=_single_feature_set(models, target_seasons),
        output_root=Path(output_root),
    )


def linear_raw_context_coefficients(model: MatchupContextualModel) -> pd.Series:
    """Return unstandardized coefficients for the antisymmetric linear context."""

    if tuple(model.pipeline.named_steps) != ("scale", "ridge"):
        raise ValueError("Compilation requires the stored linear scale-plus-ridge pipeline")
    scale = model.pipeline.named_steps["scale"]
    ridge = model.pipeline.named_steps["ridge"]
    columns = side_context_feature_columns(model.feature_set)
    values = np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float)
    if len(values) != len(columns):
        raise ValueError("Linear context coefficient count does not match its feature contract")
    return pd.Series(values, index=columns, name="raw_context_coefficient")


def _audit_possessions(
    possessions: pd.DataFrame,
    *,
    model: MatchupContextualModel,
    profiles: pd.DataFrame,
    lineup_features: dict[tuple[int, ...], np.ndarray],
    raw_coefficients: pd.Series,
    additive_feature_map: dict[str, str],
    target_priors: pd.DataFrame,
    target_season: str,
    source_season: str,
    cohort: str,
) -> dict[str, object]:
    additive_columns = tuple(additive_feature_map)
    player_adjustments = _player_adjustment_map(
        profiles, raw_coefficients, additive_feature_map
    )
    nonadditive_columns = tuple(
        column for column in raw_coefficients.index if column not in additive_columns
    )
    additive_indices = _feature_indices(raw_coefficients, additive_columns)
    nonadditive_indices = _feature_indices(raw_coefficients, nonadditive_columns)
    additive_coefficients = raw_coefficients.loc[list(additive_columns)].to_numpy(float)
    nonadditive_coefficients = raw_coefficients.loc[list(nonadditive_columns)].to_numpy(float)
    maximum_total_error = 0.0
    maximum_additive_error = 0.0
    squared_additive = 0.0
    squared_nonadditive = 0.0
    maximum_full_prediction_error = 0.0
    base_prior_values = dict(
        zip(
            target_priors["player_id"].astype(int),
            target_priors["prior_rapm"].astype(float),
            strict=True,
        )
    )
    observation_count = 0
    for start in range(0, len(possessions), CHUNK_SIZE):
        frame = possessions.iloc[start : start + CHUNK_SIZE]
        home, away = _home_away_lineups(frame)
        home_features = _cached_feature_frame(home, lineup_features, model.feature_set)
        away_features = _cached_feature_frame(away, lineup_features, model.feature_set)
        relative = home_features.to_numpy(float) - away_features.to_numpy(float)
        direct_total = model.predict_side_pairs(home_features, away_features)
        additive_direct = relative[:, additive_indices] @ additive_coefficients
        nonadditive_direct = relative[:, nonadditive_indices] @ nonadditive_coefficients
        additive_compiled = _lineup_adjustment_difference(home, away, player_adjustments)
        total_compiled = additive_compiled + nonadditive_direct
        offense, defense = _offense_defense_lineups(frame)
        sign = frame["home_offense_sign"].to_numpy(dtype=float)
        original_player_edge = _lineup_adjustment_difference(offense, defense, base_prior_values)
        compiled_prior_values = {
            player_id: base_prior_values.get(player_id, 0.0) + adjustment
            for player_id, adjustment in player_adjustments.items()
        }
        compiled_player_edge = _lineup_adjustment_difference(
            offense, defense, compiled_prior_values
        )
        maximum_full_prediction_error = max(
            maximum_full_prediction_error,
            float(
                np.max(
                    np.abs(
                        original_player_edge
                        + sign * direct_total
                        - compiled_player_edge
                        - sign * nonadditive_direct
                    )
                )
            ),
        )
        maximum_total_error = max(
            maximum_total_error, float(np.max(np.abs(direct_total - total_compiled)))
        )
        maximum_additive_error = max(
            maximum_additive_error, float(np.max(np.abs(additive_direct - additive_compiled)))
        )
        squared_additive += float(np.dot(additive_direct, additive_direct))
        squared_nonadditive += float(np.dot(nonadditive_direct, nonadditive_direct))
        observation_count += len(frame)
    return {
        "target_season": target_season,
        "source_season": source_season,
        "cohort": cohort,
        "possession_count": observation_count,
        "additive_feature_count": len(additive_columns),
        "nonadditive_feature_count": len(nonadditive_columns),
        "max_abs_additive_compilation_error": maximum_additive_error,
        "max_abs_total_reconstruction_error": maximum_total_error,
        "max_abs_full_prediction_component_error": maximum_full_prediction_error,
        "rms_additive_context": float(np.sqrt(squared_additive / observation_count)),
        "rms_nonadditive_context": float(np.sqrt(squared_nonadditive / observation_count)),
    }


def _compiled_player_adjustments(
    profiles: pd.DataFrame,
    *,
    raw_coefficients: pd.Series,
    additive_feature_map: dict[str, str],
    target_priors: pd.DataFrame,
    target_season: str,
    source_season: str,
) -> pd.DataFrame:
    output = _player_adjustment_frame(profiles, raw_coefficients, additive_feature_map)
    base = target_priors.rename(columns={"prior_rapm": "base_player_prior_rapm"})
    output = output.merge(base, on="player_id", how="left", validate="one_to_one")
    output["base_player_prior_rapm"] = output["base_player_prior_rapm"].fillna(0.0)
    output["compiled_player_prior_rapm"] = (
        output["base_player_prior_rapm"] + output["compiled_additive_context_adjustment"]
    )
    output.insert(0, "source_season", source_season)
    output.insert(0, "target_season", target_season)
    return output


def _player_adjustment_map(
    profiles: pd.DataFrame,
    raw_coefficients: pd.Series,
    additive_feature_map: dict[str, str],
) -> dict[int, float]:
    adjustments = _player_adjustment_frame(profiles, raw_coefficients, additive_feature_map)
    return dict(
        zip(
            adjustments["player_id"].astype(int),
            adjustments["compiled_additive_context_adjustment"],
            strict=True,
        )
    )


def _player_adjustment_frame(
    profiles: pd.DataFrame,
    raw_coefficients: pd.Series,
    additive_feature_map: dict[str, str] = ADDITIVE_FEATURE_TO_PLAYER_PROFILE,
) -> pd.DataFrame:
    required = {"player_id", *additive_feature_map.values()}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Player profiles lack additive compilation inputs: {sorted(missing)}")
    values = np.zeros(len(profiles), dtype=float)
    for feature, profile_column in additive_feature_map.items():
        values += raw_coefficients[feature] * profiles[profile_column].to_numpy(float)
    return pd.DataFrame(
        {
            "player_id": profiles["player_id"].astype(int),
            "compiled_additive_context_adjustment": values,
        }
    )


def _participant_ids(cohorts: Iterable[pd.DataFrame]) -> set[int]:
    players: set[int] = set()
    for frame in cohorts:
        for lineup in frame["offense_player_ids"]:
            players.update(int(player_id) for player_id in lineup)
        for lineup in frame["defense_player_ids"]:
            players.update(int(player_id) for player_id in lineup)
    return players


def _home_away_lineups(
    possessions: pd.DataFrame,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    home = [
        tuple(
            int(player_id)
            for player_id in (offense if home_offense_sign > 0 else defense)
        )
        for offense, defense, home_offense_sign in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense_sign"],
            strict=True,
        )
    ]
    away = [
        tuple(
            int(player_id)
            for player_id in (defense if home_offense_sign > 0 else offense)
        )
        for offense, defense, home_offense_sign in zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense_sign"],
            strict=True,
        )
    ]
    return home, away


def _offense_defense_lineups(
    possessions: pd.DataFrame,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    offense = [
        tuple(int(player_id) for player_id in lineup)
        for lineup in possessions["offense_player_ids"]
    ]
    defense = [
        tuple(int(player_id) for player_id in lineup)
        for lineup in possessions["defense_player_ids"]
    ]
    return offense, defense


def _cached_lineup_features(
    cohorts: Iterable[pd.DataFrame],
    *,
    profiles: pd.DataFrame,
    feature_set: str,
) -> dict[tuple[int, ...], np.ndarray]:
    """Materialize each observed five-player unit once for a target season."""

    unique: dict[tuple[int, ...], None] = {}
    for possessions in cohorts:
        home, away = _home_away_lineups(possessions)
        for lineup in [*home, *away]:
            unique[lineup] = None
    lineups = list(unique)
    features = lineup_side_context_features(lineups, profiles, feature_set=feature_set)
    return dict(zip(lineups, features.to_numpy(dtype=float), strict=True))


def _cached_feature_frame(
    lineups: list[tuple[int, ...]],
    cached: dict[tuple[int, ...], np.ndarray],
    feature_set: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        np.vstack([cached[lineup] for lineup in lineups]),
        columns=side_context_feature_columns(feature_set),
    )


def _lineup_adjustment_difference(
    home: list[tuple[int, ...]], away: list[tuple[int, ...]], adjustments: dict[int, float]
) -> np.ndarray:
    return np.fromiter(
        (
            sum(adjustments.get(int(player_id), 0.0) for player_id in home_lineup)
            - sum(adjustments.get(int(player_id), 0.0) for player_id in away_lineup)
            for home_lineup, away_lineup in zip(home, away, strict=True)
        ),
        dtype=float,
        count=len(home),
    )


def _feature_indices(coefficients: pd.Series, columns: tuple[str, ...]) -> list[int]:
    positions = {column: position for position, column in enumerate(coefficients.index)}
    return [positions[column] for column in columns]


def _write_report(
    summary: pd.DataFrame,
    adjustments: pd.DataFrame,
    *,
    source_run_dir: Path,
    feature_set: str,
    output_root: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"linear-hpm-x3-compilation-audit-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{run_id}.tmp"
    run_dir = output_root / run_id
    temporary.mkdir()
    try:
        summary.to_parquet(temporary / "identity_summary.parquet", index=False)
        adjustments.to_parquet(temporary / "compiled_player_adjustments.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "created_at": now.isoformat(),
            "source_run_dir": str(source_run_dir),
            "feature_set": feature_set,
            "additive_feature_columns": list(ADDITIVE_FEATURE_COLUMNS),
            "nonadditive_feature_definition": "all remaining x3 side-feature columns",
            "identity": (
                "linear total context equals compiled additive player edge plus "
                "nonadditive context"
            ),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(run_dir)
        (output_root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return run_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _single_feature_set(
    models: dict[str, MatchupContextualModel], target_seasons: tuple[str, ...]
) -> str:
    """Ensure the audited frozen states share one immutable feature contract."""

    feature_sets = {
        models[_previous_season(target)].feature_set
        for target in target_seasons
        if _previous_season(target) in models
    }
    if len(feature_sets) != 1:
        raise ValueError(
            "Compilation audit spans inconsistent feature sets: "
            + str(sorted(feature_sets))
        )
    return next(iter(feature_sets))


def _additive_feature_map_for_contract(feature_set: str) -> dict[str, str]:
    """Return player-compilable coordinates for one immutable x3 contract."""

    if feature_set == CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
        return dict(_BASKETBALL_ADDITIVE_FEATURE_TO_PLAYER_PROFILE)
    if feature_set == CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT:
        return {
            **_BASKETBALL_ADDITIVE_FEATURE_TO_PLAYER_PROFILE,
            **_LEGACY_CALIBRATION_FEATURE_TO_PLAYER_PROFILE,
        }
    raise ValueError(f"Unsupported HPM x3 compilation contract: {feature_set}")


def _latest_run(root: Path) -> Path:
    latest = root / "latest.json"
    if not latest.is_file():
        raise ValueError(f"Model root has no latest pointer: {root}")
    run_dir = root / str(json.loads(latest.read_text())["run_id"])
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"Model root latest pointer is invalid: {root}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit additive compilation of linear HPM x3")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run_dir = run_linear_hpm_x3_compilation_audit(seasons=tuple(args.seasons))
    print(f"Linear HPM x3 compilation audit: run={run_dir}")


if __name__ == "__main__":
    main()
