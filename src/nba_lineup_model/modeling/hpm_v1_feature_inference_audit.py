"""Fast fixed-model family perturbation audit for Original HPM v1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    V1_KNOCKOUT_EXCLUSIONS,
    lineup_side_context_features,
)
from nba_lineup_model.modeling.contextual_prior import (
    _contextual_stint_predictions,
    _score_possessions,
)
from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.frozen_model_tournament import _paired_metrics
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    BacktestModel,
    _latest_recursive_run,
    _load_state,
    _target_contextual_profiles,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import read_rapm_stints

SEASONS = ("2023-24", "2024-25", "2025-26")
MODEL = BacktestModel(
    "forward_centered_value_conditioned_aging_bounded_hierarchical_portable_matchup_contextual_rapm",
    "Original HPM v1",
)
ARTIFACTS_DIR = Path("artifacts/models")
PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
ANALYTICAL_DIR = Path("data/analytical")
REBOUNDING_COMPONENTS: dict[str, frozenset[str]] = {
    "offensive_rebounds_per_100": frozenset({"offensive_rebounds_per_100"}),
    "defensive_rebounds_per_100": frozenset({"defensive_rebounds_per_100"}),
    "sqrt_offensive_rebounds": frozenset({"sqrt_offensive_rebounds"}),
    "sqrt_defensive_rebounds": frozenset({"sqrt_defensive_rebounds"}),
    "rebounding_usage_interaction": frozenset({"rebounding_usage_interaction"}),
}
DEFENSIVE_EVENT_COMPONENTS: dict[str, frozenset[str]] = {
    "steals_per_100": frozenset({"steals_per_100"}),
    "blocks_per_100": frozenset({"blocks_per_100"}),
}
CREATION_COMPONENTS: dict[str, frozenset[str]] = {
    "assists_per_100": frozenset({"assists_per_100"}),
    "top_two_assists": frozenset({"top_two_assists"}),
    "shooter_passing_interaction": frozenset({"shooter_passing_interaction"}),
}


def run_hpm_v1_feature_inference_audit(
    *,
    feature_groups: Mapping[str, frozenset[str]] | None = None,
    audit_name: str = "feature-family",
    draws: int = 10_000,
) -> Path:
    """Neutralize fixed context groups at inference and compare with saved HPM v1."""

    groups = feature_groups or {
        feature_set.removeprefix("v1_without_"): columns
        for feature_set, columns in V1_KNOCKOUT_EXCLUSIONS.items()
    }
    root = ARTIFACTS_DIR / MODEL.model / "2025-26"
    state = _load_state(_latest_recursive_run(root), candidate=MODEL, target_seasons=SEASONS)
    panel = pd.read_parquet(PANEL_PATH)
    cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(SEASONS[-1])],
        through_season=SEASONS[-1],
        analytical_dir=ANALYTICAL_DIR,
    )
    baseline_games: list[pd.DataFrame] = []
    baseline_possessions: list[pd.DataFrame] = []
    candidates: dict[str, dict[str, list[pd.DataFrame]]] = {
        name: {"games": [], "possessions": []} for name in groups
    }
    for target in SEASONS:
        print(f"Replaying {target} baseline and feature neutralizations...", flush=True)
        profiles = _target_contextual_profiles(
            target,
            panel=panel,
            exposure_cohort=cohort,
            analytical_dir=ANALYTICAL_DIR,
        )
        source = _previous_season(target)
        model = state.context_models[source]
        priors = state.priors.loc[
            state.priors["season"].eq(target), ["player_id", "prior_rapm"]
        ].rename(columns={"prior_rapm": "prior_rapm_mean"})
        coefficients = state.coefficients.loc[
            state.coefficients["season"].eq(source), ["player_id", "rapm"]
        ]
        intercept = _recover_home_intercept(read_rapm_stints(source), coefficients)
        stints = read_rapm_stints(target)
        possessions = read_neural_possessions(target)
        predictors = _cached_predictors(model, profiles, groups)
        baseline_game, _ = _contextual_stint_predictions(
            stints,
            profiles=profiles,
            context_predictor=predictors["baseline"],
            priors=priors,
            source_home_intercept=intercept,
        )
        baseline_games.append(baseline_game.assign(season=target, model=MODEL.model))
        baseline_possessions.append(
            _score_possessions(
                possessions,
                cohort="regular_season",
                profiles=profiles,
                context_predictor=predictors["baseline"],
                priors=priors,
                source_mean=0.0,
                source_home_intercept=intercept,
            ).assign(model=MODEL.model)
        )
        for name in groups:
            predictor = predictors[name]
            games, _ = _contextual_stint_predictions(
                stints,
                profiles=profiles,
                context_predictor=predictor,
                priors=priors,
                source_home_intercept=intercept,
            )
            candidates[name]["games"].append(games.assign(season=target, model=name))
            candidates[name]["possessions"].append(
                _score_possessions(
                    possessions,
                    cohort="regular_season",
                    profiles=profiles,
                    context_predictor=predictor,
                    priors=priors,
                    source_mean=0.0,
                    source_home_intercept=intercept,
                ).assign(model=name)
            )
    baseline = {
        "games": pd.concat(baseline_games, ignore_index=True),
        "possessions": pd.concat(baseline_possessions, ignore_index=True),
    }
    comparisons = []
    for index, (name, frames) in enumerate(candidates.items(), start=1):
        print(f"Bootstrapping {name}...", flush=True)
        candidate = {key: pd.concat(value, ignore_index=True) for key, value in frames.items()}
        metrics = _paired_metrics(baseline, candidate, draws=draws, seed=20_260_814 + index)
        metrics.insert(0, "feature_group", name)
        comparisons.append(metrics)
    metrics = pd.concat(comparisons, ignore_index=True)
    run_dir = (
        ARTIFACTS_DIR
        / "analysis"
        / "hpm_v1_feature_inference_audit"
        / (
            f"hpm-v1-{audit_name}-inference-audit-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        )
    )
    run_dir.mkdir(parents=True)
    metrics.to_parquet(run_dir / "paired_metrics.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL.model,
                "seasons": SEASONS,
                "draws": draws,
                "method": "fixed_model_reference_neutralization",
                "audit_name": audit_name,
                "feature_groups": {name: sorted(columns) for name, columns in groups.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return run_dir


def _cached_predictors(model, profiles: pd.DataFrame, feature_groups: Mapping[str, frozenset[str]]):
    """Return baseline and family predictors that reuse unit and pair features."""

    reference = model.reference_features
    reference_values = {
        column: float(np.average(reference[column], weights=model.reference_weights))
        for column in reference
    }
    unit_cache: dict[tuple[int, ...], pd.Series] = {}

    def resolve_units(lineups: list[list[int]] | list[tuple[int, ...]]) -> pd.DataFrame:
        keys = [tuple(int(player_id) for player_id in lineup) for lineup in lineups]
        missing = list(dict.fromkeys(key for key in keys if key not in unit_cache))
        if missing:
            computed = lineup_side_context_features(
                missing, profiles, feature_set=model.feature_set
            )
            unit_cache.update(dict(zip(missing, computed.iloc, strict=True)))
        return pd.DataFrame([unit_cache[key] for key in keys], columns=reference.columns)

    def make_predictor(columns: frozenset[str]):
        pair_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}

        def predict(home_lineups, away_lineups, _profiles):
            pairs = [
                (
                    tuple(int(player_id) for player_id in home),
                    tuple(int(player_id) for player_id in away),
                )
                for home, away in zip(home_lineups, away_lineups, strict=True)
            ]
            missing = list(dict.fromkeys(pair for pair in pairs if pair not in pair_cache))
            if missing:
                home = resolve_units([pair[0] for pair in missing])
                away = resolve_units([pair[1] for pair in missing])
                for column in columns:
                    home[column] = reference_values[column]
                    away[column] = reference_values[column]
                corrections = model.predict_side_pairs(home, away)
                pair_cache.update(dict(zip(missing, corrections, strict=True)))
            return np.asarray([pair_cache[pair] for pair in pairs], dtype=float)

        return predict

    return {
        "baseline": make_predictor(frozenset()),
        **{name: make_predictor(columns) for name, columns in feature_groups.items()},
    }


def main() -> None:
    """Run the fixed-model HPM v1 family-reliance audit."""

    print(f"HPM v1 feature inference audit: run={run_hpm_v1_feature_inference_audit()}")


def rebounding_components_main() -> None:
    """Run the fixed-model audit for each original HPM v1 rebounding feature."""

    run = run_hpm_v1_feature_inference_audit(
        feature_groups=REBOUNDING_COMPONENTS,
        audit_name="rebounding-components",
    )
    print(f"HPM v1 rebounding-component inference audit: run={run}")


def defensive_event_components_main() -> None:
    """Run the fixed-model audit for each original HPM v1 defensive-event feature."""

    run = run_hpm_v1_feature_inference_audit(
        feature_groups=DEFENSIVE_EVENT_COMPONENTS,
        audit_name="defensive-event-components",
    )
    print(f"HPM v1 defensive-event component inference audit: run={run}")


def creation_components_main() -> None:
    """Run the fixed-model audit for each original HPM v1 creation signal."""

    run = run_hpm_v1_feature_inference_audit(
        feature_groups=CREATION_COMPONENTS,
        audit_name="creation-components",
    )
    print(f"HPM v1 creation component inference audit: run={run}")


if __name__ == "__main__":
    main()
