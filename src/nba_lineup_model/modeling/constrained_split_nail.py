"""Descriptive O/D allocation for a locked production NAIL-RAPM release.

Constrained Split NAIL never refits scalar player value or the total context
function.  It holds those completed production quantities fixed and learns only
offense-minus-defense coordinates against two offensive scoring observations per
stint.  The published O/D values therefore reconcile exactly to Total NAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import _Tee
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.split_nail import (
    SPLIT_NAIL_ADDITIVE_FEATURES,
    SplitNailDesign,
    _raw_coefficient_pairs,
    build_split_nail_design,
    build_split_nail_design_from_side_features,
    fit_fixed_total_split_nail,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "constrained_split_nail"
RUN_PREFIX = "constrained-split-nail"
DISPLAY_SEASON = "2025-26"
FIRST_SEASON = "1996-97"
DEVELOPMENT_TARGET_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
SPECIALIZATION_PRECISION_GRID = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
PRODUCTION_MODEL_ARTIFACT = "forward_nail_rapm_v1212_residualized_lambda"
PRODUCTION_RUN_ID = (
    "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25"
)
ADDITIVE_PROFILE_COLUMNS = {
    "three_pa_per_100": "three_pa_per_100",
    "three_pm_per_100": "three_pm_per_100",
    "assists_per_100": "assists_per_100",
    "turnovers_per_100": "turnovers_per_100",
    "usage_pct": "usage_pct",
    "steals_per_100": "steals_per_100",
    "blocks_per_100": "blocks_per_100",
    "offensive_rebound_claim_total": "offensive_rebound_pct",
}


@dataclass(frozen=True)
class ProductionState:
    """Completed scalar NAIL state required for an exact side allocation."""

    player_completed: pd.DataFrame
    player_priors: pd.DataFrame
    context_models: dict[str, object]
    schedule_metadata: pd.DataFrame
    context_metadata: pd.DataFrame


@dataclass(frozen=True)
class ConstrainedSplitState:
    """One completed season's learned O/D difference coordinates."""

    season: str
    player_difference: dict[int, float]
    feature_difference: dict[str, float]
    back_to_back_difference: float
    home_court_difference: float
    intercept: float
    selected_lambda: float
    r_player: float
    r_context: float


@dataclass(frozen=True)
class ConstrainedSplitNailRun:
    run_dir: Path
    run_id: str
    selected_r_player: float
    selected_r_context: float


def train_constrained_split_nail(
    *,
    through_season: str = DISPLAY_SEASON,
    production_run_dir: Path | str | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    game_catalog_path: Path | str = Path("data/catalog/games.parquet"),
    design_cache_path: Path | str | None = None,
) -> ConstrainedSplitNailRun:
    """Select context allocation precision, then materialize a locked O/D split."""

    target = validate_season(through_season)
    source_dir = Path(production_run_dir or _default_production_run_dir())
    production = _load_production_state(source_dir)
    panel = pd.read_parquet(player_season_panel_path)
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    seasons = _seasons_through(target)
    development_seasons = tuple(
        season for season in DEVELOPMENT_TARGET_SEASONS if season in seasons
    )
    if not development_seasons:
        raise ValueError("Constrained Split NAIL requires at least one development season")
    development_endpoint = development_seasons[-1]
    cache_path = Path(design_cache_path or _default_design_cache_path(target))
    if cache_path.is_file():
        print(f"Loading immutable constrained-side designs from {cache_path}", flush=True)
        designs = joblib.load(cache_path)
        if tuple(designs) != seasons:
            raise ValueError("Constrained Split NAIL design cache does not match target seasons")
    else:
        print("Preparing immutable constrained-side designs", flush=True)
        designs = {}
        for index, season in enumerate(seasons, start=1):
            print(f"  Preparing {season} ({index}/{len(seasons)})", flush=True)
            designs[season] = _season_design(
                season,
                panel=panel,
                schedule_features=schedule_features,
                analytical_dir=analytical_dir,
                curated_dir=curated_dir,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(designs, cache_path)
        print(f"Cached immutable constrained-side designs at {cache_path}", flush=True)

    selection_rows: list[dict[str, object]] = []
    for r_player in SPECIALIZATION_PRECISION_GRID:
        for r_context in SPECIALIZATION_PRECISION_GRID:
            print(
                "Selecting constrained O/D precision "
                f"r_player={r_player:g}; r_context={r_context:g}",
                flush=True,
            )
            states = _fit_completed_states(
                seasons=_seasons_through(_previous_season(development_endpoint)),
                designs=designs,
                r_player=r_player,
                r_context=r_context,
                production=production,
            )
            for season in development_seasons:
                source = _previous_season(season)
                if source not in states:
                    raise ValueError(f"Constrained Split NAIL missing development source {source}")
                metrics = _score_forward_side_predictions(
                    season,
                    states[source],
                    design=designs[season],
                    production=production,
                )
                selection_rows.append(
                    {
                        "r_player": r_player,
                        "r_context": r_context,
                        "season": season,
                        **metrics,
                    }
                )

    selection = pd.DataFrame(selection_rows)
    aggregate = (
        selection.groupby(["r_player", "r_context"], as_index=False)
        .agg(
            squared_error=("squared_error", "sum"),
            absolute_error=("absolute_error", "sum"),
            offensive_possessions=("offensive_possessions", "sum"),
        )
        .sort_values(["r_player", "r_context"], kind="stable")
    )
    aggregate["side_scoring_rmse"] = np.sqrt(
        aggregate["squared_error"] / aggregate["offensive_possessions"]
    )
    aggregate["side_scoring_mae"] = (
        aggregate["absolute_error"] / aggregate["offensive_possessions"]
    )
    selected = aggregate.sort_values(
        ["side_scoring_rmse", "r_player", "r_context"], kind="stable"
    ).iloc[0]
    selected_r_player = float(selected["r_player"])
    selected_r_context = float(selected["r_context"])
    boundary = {SPECIALIZATION_PRECISION_GRID[0], SPECIALIZATION_PRECISION_GRID[-1]}
    if selected_r_player in boundary or selected_r_context in boundary:
        raise ValueError(
            "Constrained Split NAIL precision selection reached the grid boundary: "
            f"r_player={selected_r_player:g}, r_context={selected_r_context:g}. "
            "Expand the grid before publishing an O/D allocation."
        )

    states = _fit_completed_states(
        seasons=seasons,
        designs=designs,
        r_player=selected_r_player,
        r_context=selected_r_context,
        production=production,
    )
    ratings, feature_allocations, control_allocations = _materialize_display_ratings(
        target,
        states[target],
        design=designs[target],
        source_dir=source_dir,
        production=production,
        panel=panel,
    )
    return _write_run(
        target=target,
        source_dir=source_dir,
        states=states,
        selection=selection,
        aggregate=aggregate,
        ratings=ratings,
        feature_allocations=feature_allocations,
        control_allocations=control_allocations,
        selected_r_player=selected_r_player,
        selected_r_context=selected_r_context,
        artifacts_dir=Path(artifacts_dir),
    )


def _fit_completed_states(
    *,
    seasons: tuple[str, ...],
    designs: dict[str, SplitNailDesign],
    r_player: float,
    r_context: float,
    production: ProductionState,
) -> dict[str, ConstrainedSplitState]:
    states: dict[str, ConstrainedSplitState] = {}
    for index, season in enumerate(seasons, start=1):
        print(
            "Fitting constrained O/D allocation "
            f"{season} ({index}/{len(seasons)}; r_player={r_player:g}; "
            f"r_context={r_context:g})",
            flush=True,
        )
        design = designs[season]
        mean, selected_lambda = _completed_mean_coordinates(design, season, production)
        prior = _specialization_prior(design, states.get(_previous_season(season)))
        fitted = fit_fixed_total_split_nail(
            design,
            mean,
            specialization_prior=prior,
            regularization=selected_lambda,
            player_specialization_relative_precision=r_player,
            context_specialization_relative_precision=r_context,
        )
        states[season] = _state_from_fit(
            season,
            design,
            fitted.coef_,
            fitted.intercept_,
            selected_lambda,
            r_player,
            r_context,
        )
    return states


def _score_forward_side_predictions(
    season: str,
    state: ConstrainedSplitState,
    *,
    design: SplitNailDesign,
    production: ProductionState,
) -> dict[str, float]:
    mean = _forecast_mean_coordinates(design, season, production)
    specialization = _state_specialization_for_design(design, state)
    raw = _raw_coefficients(design, mean, specialization)
    prediction = np.asarray(design.features @ raw, dtype=float).reshape(-1) + state.intercept
    residual = design.target - prediction
    weights = design.weights
    return {
        "squared_error": float(np.dot(weights, np.square(residual))),
        "absolute_error": float(np.dot(weights, np.abs(residual))),
        "offensive_possessions": float(weights.sum()),
    }


def _season_design(
    season: str,
    *,
    panel: pd.DataFrame,
    schedule_features: pd.DataFrame,
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> SplitNailDesign:
    stints = _attach_back_to_back_flags(
        read_rapm_stints(season, analytical_dir=analytical_dir), schedule_features
    )
    participants = {
        int(player)
        for column in ("home_player_ids", "away_player_ids")
        for lineup in stints[column]
        for player in lineup
    }
    if season == FIRST_SEASON:
        columns = (*SPLIT_NAIL_ADDITIVE_FEATURES, "top_two_assists", "usage_concentration")
        zero = pd.DataFrame(0.0, index=stints.index, columns=columns)
        return build_split_nail_design_from_side_features(stints, zero, zero)
    profiles = build_contextual_player_profiles(
        panel,
        target_season=season,
        target_player_ids=participants,
        analytical_dir=str(analytical_dir),
        curated_dir=str(curated_dir),
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )
    return build_split_nail_design(stints, profiles)


def _completed_mean_coordinates(
    design: SplitNailDesign, season: str, production: ProductionState
) -> tuple[np.ndarray, float]:
    players = production.player_completed.loc[
        production.player_completed["season"].astype(str).eq(season)
    ]
    if players.empty:
        raise ValueError(f"Production NAIL has no completed player state for {season}")
    values = dict(zip(players["player_id"].astype(int), players["rapm"], strict=True))
    selected_lambda = float(players["selected_lambda"].iloc[0])
    return _mean_coordinates(
        design,
        player_values=values,
        context_season=season,
        schedule_season=season,
        production=production,
    ), selected_lambda


def _forecast_mean_coordinates(
    design: SplitNailDesign, season: str, production: ProductionState
) -> np.ndarray:
    priors = production.player_priors.loc[
        production.player_priors["season"].astype(str).eq(season)
    ]
    values = dict(zip(priors["player_id"].astype(int), priors["prior_rapm"], strict=True))
    source = _previous_season(season)
    return _mean_coordinates(
        design,
        player_values=values,
        context_season=source,
        schedule_season=source,
        production=production,
    )


def _mean_coordinates(
    design: SplitNailDesign,
    *,
    player_values: dict[int, float],
    context_season: str,
    schedule_season: str,
    production: ProductionState,
) -> np.ndarray:
    pair_count = design.coefficient_count // 2
    output = np.zeros(pair_count, dtype=float)
    output[: design.player_count] = [player_values.get(player, 0.0) for player in design.player_ids]
    feature_start = design.player_count
    coefficients = _context_raw_coefficients(production.context_models.get(context_season))
    for offset, feature in enumerate((*design.additive_features, *design.nonadditive_features)):
        output[feature_start + offset] = coefficients.get(feature, 0.0)
    cursor = feature_start + len((*design.additive_features, *design.nonadditive_features))
    if design.includes_back_to_back:
        output[cursor] = _schedule_raw_coefficient(schedule_season, production)
        cursor += 1
    output[cursor] = _home_court_total(context_season, production)
    return output


def _context_raw_coefficients(model: object | None) -> dict[str, float]:
    if model is None:
        return {}
    pipeline = getattr(model, "pipeline", None)
    if pipeline is None or not {"scale", "ridge"}.issubset(pipeline.named_steps):
        raise ValueError("Production context model must expose linear scale and ridge steps")
    scale = pipeline.named_steps["scale"]
    ridge = pipeline.named_steps["ridge"]
    columns = list(scale.feature_names_in_)
    raw = np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float)
    return {
        str(column).removeprefix("home_minus_away_"): float(value)
        for column, value in zip(columns, raw, strict=True)
    }


def _schedule_raw_coefficient(season: str, production: ProductionState) -> float:
    rows = production.schedule_metadata.loc[
        production.schedule_metadata["season"].astype(str).eq(season)
    ]
    return float(rows["schedule_control_raw_weight"].iloc[0]) if not rows.empty else 0.0


def _home_court_total(season: str, production: ProductionState) -> float:
    rows = production.context_metadata.loc[
        production.context_metadata["season"].astype(str).eq(season)
    ]
    return float(rows["context_home_intercept"].iloc[0]) if not rows.empty else 0.0


def _specialization_prior(
    design: SplitNailDesign, previous: ConstrainedSplitState | None
) -> np.ndarray:
    output = np.zeros(design.coefficient_count // 2, dtype=float)
    if previous is None:
        return output
    output[: design.player_count] = [
        previous.player_difference.get(player, 0.0) for player in design.player_ids
    ]
    return output


def _state_specialization_for_design(
    design: SplitNailDesign, state: ConstrainedSplitState
) -> np.ndarray:
    output = _specialization_prior(design, state)
    feature_start = design.player_count
    for offset, feature in enumerate((*design.additive_features, *design.nonadditive_features)):
        output[feature_start + offset] = state.feature_difference.get(feature, 0.0)
    cursor = feature_start + len((*design.additive_features, *design.nonadditive_features))
    if design.includes_back_to_back:
        output[cursor] = state.back_to_back_difference
        cursor += 1
    output[cursor] = state.home_court_difference
    return output


def _raw_coefficients(
    design: SplitNailDesign, mean: np.ndarray, specialization: np.ndarray
) -> np.ndarray:
    pair_count = design.coefficient_count // 2
    if mean.shape != (pair_count,) or specialization.shape != (pair_count,):
        raise ValueError("Constrained Split coordinates do not match design pairs")
    raw = np.zeros(design.coefficient_count, dtype=float)
    for index, (offense, defense) in enumerate(_raw_coefficient_pairs(design)):
        raw[offense] = 0.5 * (mean[index] + specialization[index])
        raw[defense] = 0.5 * (mean[index] - specialization[index])
    return raw


def _state_from_fit(
    season: str,
    design: SplitNailDesign,
    raw: np.ndarray,
    intercept: float,
    selected_lambda: float,
    r_player: float,
    r_context: float,
) -> ConstrainedSplitState:
    pairs = _raw_coefficient_pairs(design)
    differences = np.asarray([raw[offense] - raw[defense] for offense, defense in pairs])
    players = dict(zip(design.player_ids, differences[: design.player_count], strict=True))
    feature_start = design.player_count
    features = (*design.additive_features, *design.nonadditive_features)
    feature_difference = dict(
        zip(features, differences[feature_start : feature_start + len(features)], strict=True)
    )
    cursor = feature_start + len(features)
    back_to_back = float(differences[cursor]) if design.includes_back_to_back else 0.0
    cursor += int(design.includes_back_to_back)
    return ConstrainedSplitState(
        season=season,
        player_difference={int(key): float(value) for key, value in players.items()},
        feature_difference={str(key): float(value) for key, value in feature_difference.items()},
        back_to_back_difference=back_to_back,
        home_court_difference=float(differences[cursor]),
        intercept=float(intercept),
        selected_lambda=float(selected_lambda),
        r_player=float(r_player),
        r_context=float(r_context),
    )


def _materialize_display_ratings(
    season: str,
    state: ConstrainedSplitState,
    *,
    design: SplitNailDesign,
    source_dir: Path,
    production: ProductionState,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    published_path = (
        Path("artifacts/web/lineup_rankings")
        / PRODUCTION_MODEL_ARTIFACT
        / source_dir.name
        / "player_ratings.parquet"
    )
    published = pd.read_parquet(published_path)
    ratings = published.loc[published["season"].astype(str).eq(season)].copy()
    completed = production.player_completed.loc[
        production.player_completed["season"].astype(str).eq(season), ["player_id", "rapm"]
    ].rename(columns={"rapm": "base_nail"})
    ratings = ratings.merge(completed, on="player_id", how="left", validate="one_to_one")
    ratings["base_nail"] = ratings["base_nail"].fillna(
        ratings["rapm"] - ratings["additive_profile_adjustment"].fillna(0.0)
    )
    ratings["additive_profile"] = ratings["rapm"] - ratings["base_nail"]
    ratings["base_difference"] = ratings["player_id"].map(state.player_difference).fillna(0.0)
    ratings["profile_difference"] = _display_profile_differences(
        season,
        ratings,
        state,
        panel=panel,
    )
    ratings["offense_base"] = 0.5 * (ratings["base_nail"] + ratings["base_difference"])
    ratings["defense_base"] = 0.5 * (ratings["base_nail"] - ratings["base_difference"])
    ratings["offense_additive_profile"] = 0.5 * (
        ratings["additive_profile"] + ratings["profile_difference"]
    )
    ratings["defense_additive_profile"] = 0.5 * (
        ratings["additive_profile"] - ratings["profile_difference"]
    )
    ratings["offense_rating"] = ratings["offense_base"] + ratings["offense_additive_profile"]
    ratings["defense_rating"] = ratings["defense_base"] + ratings["defense_additive_profile"]
    ratings["total_nail"] = ratings["rapm"]
    ratings["od_sum_error"] = (
        ratings["offense_rating"] + ratings["defense_rating"] - ratings["total_nail"]
    )

    feature_allocations = _materialize_feature_allocations(
        season,
        state,
        design=design,
        production=production,
    )
    mean, _ = _completed_mean_coordinates(design, season, production)
    specialization = _state_specialization_for_design(design, state)
    feature_start = design.player_count
    features = (*design.additive_features, *design.nonadditive_features)
    cursor = feature_start + len(features)
    control_rows = []
    if design.includes_back_to_back:
        total = float(mean[cursor])
        difference = float(specialization[cursor])
        control_rows.append(_control_row(season, "back_to_back", total, difference))
        cursor += 1
    control_rows.append(
        _control_row(
            season,
            "home_court",
            float(mean[cursor]),
            float(specialization[cursor]),
        )
    )
    return ratings, feature_allocations, pd.DataFrame(control_rows)


def _materialize_feature_allocations(
    season: str,
    state: ConstrainedSplitState,
    *,
    design: SplitNailDesign,
    production: ProductionState,
) -> pd.DataFrame:
    """Return the exact side allocation for every fixed scalar feature coefficient."""

    mean, _ = _completed_mean_coordinates(design, season, production)
    specialization = _state_specialization_for_design(design, state)
    feature_start = design.player_count
    rows = []
    for offset, feature in enumerate((*design.additive_features, *design.nonadditive_features)):
        total = float(mean[feature_start + offset])
        difference = float(specialization[feature_start + offset])
        rows.append(
            {
                "season": season,
                "feature": feature,
                "feature_layer": (
                    "additive_profile"
                    if feature in design.additive_features
                    else "nonadditive_lineup"
                ),
                "total_coefficient": total,
                "offense_coefficient": 0.5 * (total + difference),
                "defense_coefficient": 0.5 * (total - difference),
                "od_sum_error": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _display_profile_differences(
    season: str,
    ratings: pd.DataFrame,
    state: ConstrainedSplitState,
    *,
    panel: pd.DataFrame,
) -> pd.Series:
    """Allocate the locked additive profile into O/D around its display center.

    Published NAIL ratings center additive profiles at a possession-weighted
    reference player.  This mirrors that convention for the learned O-minus-D
    feature coefficients, while retaining the already published scalar profile
    total.  Both layers therefore receive an interpretable split without
    changing total NAIL.
    """

    features = tuple(
        feature for feature in ADDITIVE_PROFILE_COLUMNS if feature in state.feature_difference
    )
    if not features:
        return pd.Series(0.0, index=ratings.index, dtype=float)
    profiles = build_contextual_player_profiles(
        panel,
        target_season=season,
        target_player_ids=ratings["player_id"].astype(int).tolist(),
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    ).set_index("player_id")
    raw = pd.Series(0.0, index=profiles.index, dtype=float)
    for feature in features:
        raw += profiles[ADDITIVE_PROFILE_COLUMNS[feature]].astype(float) * float(
            state.feature_difference[feature]
        )
    values = ratings["player_id"].astype(int).map(raw).astype(float)
    observed = ratings["additive_profile"].notna()
    exposure_column = (
        "on_court_possessions" if "on_court_possessions" in ratings else "possessions"
    )
    weights = (
        ratings[exposure_column].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    )
    if observed.any() and float(weights[observed.to_numpy()].sum()) > 0.0:
        center = float(np.average(values.loc[observed], weights=weights[observed.to_numpy()]))
    elif observed.any():
        center = float(values.loc[observed].mean())
    else:
        center = 0.0
    return (values - center).where(observed, 0.0)


def _control_row(season: str, control: str, total: float, difference: float) -> dict[str, object]:
    return {
        "season": season,
        "control": control,
        "total_coefficient": total,
        "offense_coefficient": 0.5 * (total + difference),
        "defense_coefficient": 0.5 * (total - difference),
        "od_sum_error": 0.0,
    }


def _load_production_state(source_dir: Path) -> ProductionState:
    required = (
        "historical_player_coefficients.parquet",
        "season_player_priors.parquet",
        "season_context_models.joblib",
        "season_schedule_control_metadata.parquet",
        "season_context_metadata.parquet",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Production NAIL artifact missing: " + ", ".join(missing))
    return ProductionState(
        player_completed=pd.read_parquet(source_dir / "historical_player_coefficients.parquet"),
        player_priors=pd.read_parquet(source_dir / "season_player_priors.parquet"),
        context_models=joblib.load(source_dir / "season_context_models.joblib"),
        schedule_metadata=pd.read_parquet(source_dir / "season_schedule_control_metadata.parquet"),
        context_metadata=pd.read_parquet(source_dir / "season_context_metadata.parquet"),
    )


def _attach_back_to_back_flags(rows: pd.DataFrame, schedule_features: pd.DataFrame) -> pd.DataFrame:
    lookup = schedule_features.loc[:, ["game_id", "home_back_to_back", "away_back_to_back"]].copy()
    lookup["game_id"] = lookup["game_id"].astype(str)
    output = rows.copy()
    original = output["game_id"].copy()
    output["game_id"] = output["game_id"].astype(str)
    output = output.drop(columns=["home_back_to_back", "away_back_to_back"], errors="ignore").merge(
        lookup, on="game_id", how="left", validate="many_to_one"
    )
    if output[["home_back_to_back", "away_back_to_back"]].isna().any().any():
        raise ValueError("Constrained Split NAIL schedule flags are incomplete")
    output["game_id"] = original.to_numpy(copy=False)
    return output


def _seasons_through(target: str) -> tuple[str, ...]:
    return tuple(
        f"{year}-{str(year + 1)[-2:]}" for year in range(int(FIRST_SEASON[:4]), int(target[:4]) + 1)
    )


def _default_production_run_dir() -> Path:
    return Path("artifacts/models") / PRODUCTION_MODEL_ARTIFACT / DISPLAY_SEASON / PRODUCTION_RUN_ID


def _default_design_cache_path(target: str) -> Path:
    return Path("artifacts/cache/constrained_split_nail") / target / "season_designs.joblib"


def _write_run(
    *,
    target: str,
    source_dir: Path,
    states: dict[str, ConstrainedSplitState],
    selection: pd.DataFrame,
    aggregate: pd.DataFrame,
    ratings: pd.DataFrame,
    feature_allocations: pd.DataFrame,
    control_allocations: pd.DataFrame,
    selected_r_player: float,
    selected_r_context: float,
    artifacts_dir: Path,
) -> ConstrainedSplitNailRun:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{RUN_PREFIX}-{target}-{timestamp}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME / target
    staging = root / f".{run_id}.tmp"
    run_dir = root / run_id
    staging.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": MODEL_NAME,
        "through_season": target,
        "source_production_run": str(source_dir),
        "contract": "locked_production_totals_side_scoring_allocation",
        "specialization_precision_grid": list(SPECIALIZATION_PRECISION_GRID),
        "selected_r_player": selected_r_player,
        "selected_r_context": selected_r_context,
        "development_target_seasons": sorted(selection["season"].unique().tolist()),
        "selection_metric": "out_of_sample_possession_weighted_offensive_scoring_rmse",
        "net_margin_contract": "exact_production_nail_parity_by_construction",
        "code_fingerprint": modeling_code_fingerprint(),
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    selection.to_parquet(staging / "side_scoring_selection_by_season.parquet", index=False)
    aggregate.to_parquet(staging / "side_scoring_selection_summary.parquet", index=False)
    ratings.to_parquet(staging / "player_ratings.parquet", index=False)
    feature_allocations.to_parquet(staging / "feature_allocations.parquet", index=False)
    control_allocations.to_parquet(staging / "control_allocations.parquet", index=False)
    joblib.dump(states, staging / "season_states.joblib")
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "files": sorted(path.name for path in staging.iterdir()),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if run_dir.exists():
        raise FileExistsError(run_dir)
    staging.replace(run_dir)
    latest = root / "latest.json"
    latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return ConstrainedSplitNailRun(
        run_dir=run_dir,
        run_id=run_id,
        selected_r_player=selected_r_player,
        selected_r_context=selected_r_context,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train descriptive Constrained Split NAIL")
    parser.add_argument("--through-season", default=DISPLAY_SEASON)
    parser.add_argument("--production-run-dir")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    kwargs = {
        "through_season": args.through_season,
        **({"production_run_dir": args.production_run_dir} if args.production_run_dir else {}),
    }
    if args.log_path:
        path = Path(args.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            stdout = sys.stdout
            sys.stdout = _Tee(stdout, handle)  # type: ignore[assignment]
            try:
                run = train_constrained_split_nail(**kwargs)
                print(
                    "Constrained Split NAIL: "
                    f"run={run.run_dir}; r_player={run.selected_r_player:g}; "
                    f"r_context={run.selected_r_context:g}"
                )
            finally:
                sys.stdout = stdout
        return
    run = train_constrained_split_nail(**kwargs)
    print(
        "Constrained Split NAIL: "
        f"run={run.run_dir}; r_player={run.selected_r_player:g}; "
        f"r_context={run.selected_r_context:g}"
    )


if __name__ == "__main__":
    main()
