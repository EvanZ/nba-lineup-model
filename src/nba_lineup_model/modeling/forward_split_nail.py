"""Recursive forward Split NAIL-RAPM training with O/D schedule controls.

Split NAIL keeps the scalar NAIL v1.2.1.2 prior as the total player state,
learns offense-minus-defense specialization recursively, and fits known
before-tipoff home/away back-to-back controls in every completed season.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    DEFAULT_TARGET_SEASON,
    _previous_season,
)
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import _Tee
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _returning_priors,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.prior_rapm import (
    DEFAULT_LAMBDA_GRID,
    ForwardLaggedRapmSeason,
)
from nba_lineup_model.modeling.replacement_level import (
    player_exposure_shares,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.progress import format_progress_bar
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.split_nail import (
    SPLIT_NAIL_ADDITIVE_FEATURES,
    SPLIT_NAIL_NONADDITIVE_FEATURES,
    SplitNailDesign,
    SplitNailSeasonFit,
    build_split_nail_design,
    build_split_nail_design_from_side_features,
    fit_split_nail_season,
    split_nail_prior_vector,
    _constrained_precision,
    _paired_mean_specialization_transform,
    _paired_raw_to_parameters,
    _raw_coefficient_pairs,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.modeling.train import chronological_game_splits
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.models.baselines import PriorPrecisionRidgeLineupModel
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "forward_split_nail_rapm"
RUN_PREFIX = "forward-split-nail-rapm"
FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_FEATURE_RELATIVE_PRECISION = 1.5
DEFAULT_GAME_CATALOG_PATH = Path("data/catalog/games.parquet")
FIRST_SEASON = "1996-97"

_ADDITIVE_PROFILE_COLUMNS = {
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
class ForwardSplitNailRun:
    """Immutable Split NAIL training artifact."""

    run_dir: Path
    run_id: str


def train_forward_split_nail(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    feature_relative_precision: float = DEFAULT_FEATURE_RELATIVE_PRECISION,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
) -> ForwardSplitNailRun:
    """Fit a standalone constrained O/D state and frozen evaluation.

    No learned production NAIL artifact is read. Combined player priors are
    rebuilt from preceding Split NAIL ``R`` states under the production aging,
    gap-returner, and exposure-gated cold-start contracts.
    """

    target = validate_season(through_season)
    if feature_relative_precision <= 0:
        raise ValueError("Feature relative precision must be positive")
    panel = pd.read_parquet(player_season_panel_path)
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    seasons = _seasons_through(target)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
        analytical_dir=analytical_dir,
    )
    season_fits: dict[str, SplitNailSeasonFit] = {}
    completed_results: list[ForwardLaggedRapmSeason] = []
    exposure_history: list[pd.DataFrame] = []
    replacement_tokens: list[dict[str, object]] = []
    priors_by_season: dict[str, dict[int, float]] = {}
    rating_rows: list[pd.DataFrame] = []
    feature_rows: list[pd.DataFrame] = []
    schedule_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    profiles_by_season: dict[str, pd.DataFrame] = {}
    side_differences: dict[int, float] = {}

    for index, season in enumerate(seasons, start=1):
        print(f"Fitting Split NAIL state for {season} ({index}/{len(seasons)})", flush=True)
        stints = _attach_back_to_back_flags(
            read_rapm_stints(season, analytical_dir=analytical_dir),
            schedule_features,
        )
        participants = _stint_participants(stints)
        priors, _ = build_centered_value_conditioned_aging_gap_returner_priors(
            season=season,
            panel=panel,
            completed_results=completed_results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        scalar_priors = dict(zip(priors["player_id"].astype(int), priors["lagged_rapm_prior"], strict=True))
        priors_by_season[season] = scalar_priors
        profiles = _profiles_for_season(
            season,
            participants,
            panel=panel,
            analytical_dir=analytical_dir,
            curated_dir=curated_dir,
        )
        design = (
            build_split_nail_design(stints, profiles)
            if profiles is not None
            else _zero_profile_design(stints)
        )
        regularization = _select_standalone_lambda(
            design,
            stints=stints,
            scalar_priors=scalar_priors,
            previous_specialization=side_differences,
        )
        fit = fit_split_nail_season(
            design,
            scalar_priors,
            carried_side_differences=side_differences,
            regularization=regularization,
            feature_relative_precision=feature_relative_precision,
        )
        season_fits[season] = fit
        completed = _as_combined_result(season, fit, priors, regularization)
        completed_results.append(completed)
        exposure = player_exposure_shares(stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(_fit_replacement_token(season, stints, exposure, completed, panel))
        season_ratings = _season_player_ratings(
            fit,
            profiles,
            panel=panel,
            season=season,
        )
        rating_rows.append(season_ratings)
        feature_rows.append(fit.feature_coefficients.assign(season=season))
        if not fit.schedule_coefficients.empty:
            schedule_rows.append(fit.schedule_coefficients.assign(season=season))
        metadata_rows.append(
            {
                "season": season,
                "source_scalar_model": "none_standalone_split_nail",
                "combined_prior_player_count": len(scalar_priors),
                "player_count": design.player_count,
                "scoring_row_count": len(design.target),
                "selected_player_lambda": regularization,
                "feature_relative_precision": feature_relative_precision,
                "feature_regularization": regularization * feature_relative_precision,
                "schedule_relative_precision": feature_relative_precision,
                "schedule_regularization": regularization * feature_relative_precision,
                "includes_back_to_back": design.includes_back_to_back,
                "back_to_back_scale": design.back_to_back_scale,
                "profile_timing": "prior" if profiles is not None else "unavailable_initial_season",
                "league_scoring_intercept": fit.model.intercept_,
                "home_offense_coefficient": float(
                    fit.model.coef_[design.home_court_column(side="offense")]
                ),
                "home_defense_coefficient": float(
                    fit.model.coef_[design.home_court_column(side="defense")]
                ),
            }
        )
        side_differences = dict(
            zip(
                fit.player_coefficients["player_id"].astype(int),
                (
                    fit.player_coefficients["offense_base_rating"]
                    - fit.player_coefficients["defense_base_rating"]
                ),
                strict=True,
            )
        )
        if profiles is not None:
            profiles_by_season[season] = profiles
        print(
            format_progress_bar(index, len(seasons), label=f"Completed Split NAIL {season}"),
            flush=True,
        )

    frozen = _evaluate_frozen_seasons(
        season_fits,
        scalar_priors=priors_by_season,
        panel=panel,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        seasons=tuple(season for season in FROZEN_SEASONS if season in seasons),
        schedule_features=schedule_features,
    )
    return _write_run(
        target=target,
        source_run_dir=None,
        season_fits=season_fits,
        ratings=pd.concat(rating_rows, ignore_index=True),
        feature_coefficients=pd.concat(feature_rows, ignore_index=True),
        schedule_coefficients=(
            pd.concat(schedule_rows, ignore_index=True) if schedule_rows else pd.DataFrame()
        ),
        priors=pd.concat(
            [
                pd.DataFrame(
                    {
                        "season": season,
                        "player_id": list(values),
                        "prior_rapm": list(values.values()),
                    }
                )
                for season, values in priors_by_season.items()
            ],
            ignore_index=True,
        ),
        metadata=pd.DataFrame(metadata_rows),
        profiles=profiles_by_season,
        frozen=frozen,
        feature_relative_precision=feature_relative_precision,
        artifacts_dir=Path(artifacts_dir),
    )


def _load_scalar_source_state(
    source_run_dir: Path,
) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    prior_path = source_run_dir / "season_player_priors.parquet"
    coefficients_path = source_run_dir / "historical_player_coefficients.parquet"
    if not prior_path.is_file() or not coefficients_path.is_file():
        raise FileNotFoundError("Split NAIL scalar source artifact is incomplete")
    priors = pd.read_parquet(prior_path)
    coefficients = pd.read_parquet(coefficients_path)
    prior_by_season = {
        str(season): dict(zip(group["player_id"].astype(int), group["prior_rapm"], strict=True))
        for season, group in priors.groupby("season", sort=False)
    }
    lambda_schedule = {
        str(season): float(group["selected_lambda"].iloc[0])
        for season, group in coefficients.groupby("season", sort=False)
    }
    return prior_by_season, lambda_schedule


def _select_standalone_lambda(
    design: SplitNailDesign,
    *,
    stints: pd.DataFrame,
    scalar_priors: dict[int, float],
    previous_specialization: dict[int, float],
) -> float:
    """Select the player-state lambda on Split NAIL's own chronological folds."""

    raw_prior = split_nail_prior_vector(design, scalar_priors, previous_specialization)
    transform = _paired_mean_specialization_transform(
        design.coefficient_count, _raw_coefficient_pairs(design)
    )
    features = design.features @ transform
    prior = _paired_raw_to_parameters(raw_prior, _raw_coefficient_pairs(design))
    precision = _constrained_precision(
        design,
        feature_relative_precision=DEFAULT_FEATURE_RELATIVE_PRECISION,
        schedule_relative_precision=DEFAULT_FEATURE_RELATIVE_PRECISION,
    )
    split_plan = chronological_game_splits(stints, config=ChronologicalSplitConfig())
    scores: list[tuple[float, float, float]] = []
    for regularization in (value for value in DEFAULT_LAMBDA_GRID if value > 0.0):
        squared_error = 0.0
        exposure = 0.0
        for fold in split_plan.folds:
            train = np.isin(design.game_ids, fold.train_game_ids)
            validation = np.isin(design.game_ids, fold.validation_game_ids)
            if not train.any() or not validation.any():
                continue
            model = PriorPrecisionRidgeLineupModel(float(regularization)).fit(
                features[train], design.target[train], design.weights[train], prior, precision
            )
            residual = design.target[validation] - model.predict(features[validation])
            squared_error += float(np.dot(design.weights[validation], np.square(residual)))
            exposure += float(design.weights[validation].sum())
        if exposure:
            scores.append((squared_error / exposure, float(regularization), exposure))
    if not scores:
        return float(DEFAULT_LAMBDA_GRID[0])
    return min(scores, key=lambda row: (row[0], row[1]))[1]


def _as_combined_result(
    season: str,
    fit: SplitNailSeasonFit,
    priors: pd.DataFrame,
    selected_lambda: float,
) -> ForwardLaggedRapmSeason:
    """Adapt the fitted combined R state to the established prior-builder API."""

    estimates = fit.player_coefficients.loc[:, ["player_id", "net_base_rating"]].rename(
        columns={"net_base_rating": "rapm"}
    )
    estimates["season"] = season
    estimates["selected_lambda"] = selected_lambda
    prior_frame = priors.rename(columns={"lagged_rapm_prior": "prior_rapm"}).copy()
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=selected_lambda,
        cv_results=pd.DataFrame(),
        player_estimates=estimates,
        player_priors=prior_frame,
    )


def _seasons_through(target: str) -> tuple[str, ...]:
    first_year = int(FIRST_SEASON[:4])
    final_year = int(target[:4])
    if final_year < first_year:
        raise ValueError(f"Split NAIL requires a season no earlier than {FIRST_SEASON}")
    return tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(first_year, final_year + 1))


def _stint_participants(stints: pd.DataFrame) -> set[int]:
    return {
        int(player_id)
        for column in ("home_player_ids", "away_player_ids")
        for lineup in stints[column]
        for player_id in lineup
    }


def _profiles_for_season(
    season: str,
    player_ids: set[int],
    *,
    panel: pd.DataFrame,
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> pd.DataFrame | None:
    if season == FIRST_SEASON:
        return None
    return build_contextual_player_profiles(
        panel,
        target_season=season,
        target_player_ids=player_ids,
        analytical_dir=str(analytical_dir),
        curated_dir=str(curated_dir),
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )


def _zero_profile_design(stints: pd.DataFrame) -> SplitNailDesign:
    columns = (*SPLIT_NAIL_ADDITIVE_FEATURES, *SPLIT_NAIL_NONADDITIVE_FEATURES)
    features = pd.DataFrame(0.0, index=stints.index, columns=columns)
    return build_split_nail_design_from_side_features(stints, features, features)


def _attach_back_to_back_flags(
    rows: pd.DataFrame,
    schedule_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach side-specific known-before-tipoff schedule flags by game id."""

    required = {"game_id", "home_back_to_back", "away_back_to_back"}
    missing = required - set(schedule_features)
    if missing:
        raise ValueError(f"Schedule feature frame lacks: {sorted(missing)}")
    lookup = schedule_features.loc[
        :, ["game_id", "home_back_to_back", "away_back_to_back"]
    ].copy()
    lookup["game_id"] = lookup["game_id"].astype(str)
    if lookup["game_id"].duplicated().any():
        raise ValueError("Schedule features must be unique by game_id")
    output = rows.copy()
    original_game_ids = output["game_id"].copy()
    output["game_id"] = original_game_ids.astype(str)
    output = output.drop(
        columns=["home_back_to_back", "away_back_to_back"], errors="ignore"
    ).merge(lookup, on="game_id", how="left", validate="many_to_one")
    if output[["home_back_to_back", "away_back_to_back"]].isna().any().any():
        missing_ids = sorted(
            output.loc[
                output[["home_back_to_back", "away_back_to_back"]].isna().any(axis=1),
                "game_id",
            ].unique()
        )
        raise ValueError("Schedule flags missing game ids: " + ", ".join(missing_ids[:10]))
    output["game_id"] = original_game_ids.to_numpy(copy=False)
    return output


def _season_player_ratings(
    fit: SplitNailSeasonFit,
    profiles: pd.DataFrame | None,
    *,
    panel: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    output = fit.player_coefficients.copy()
    output["season"] = season
    output["offense_additive_profile_raw"] = 0.0
    output["defense_additive_profile_raw"] = 0.0
    if profiles is not None:
        feature_lookup = fit.feature_coefficients.set_index("feature")
        profile_index = profiles.set_index("player_id")
        for feature, profile_column in _ADDITIVE_PROFILE_COLUMNS.items():
            values = output["player_id"].map(profile_index[profile_column]).fillna(0.0)
            output["offense_additive_profile_raw"] += (
                values * float(feature_lookup.loc[feature, "offense_raw_coefficient"])
            )
            output["defense_additive_profile_raw"] += (
                values * float(feature_lookup.loc[feature, "defense_raw_coefficient"])
            )
    references = _additive_profile_reference_per_player(fit)
    output["offense_additive_profile_reference"] = references["offense"]
    output["defense_additive_profile_reference"] = references["defense"]
    output["offense_additive_profile"] = (
        output["offense_additive_profile_raw"] - references["offense"]
    )
    output["defense_additive_profile"] = (
        output["defense_additive_profile_raw"] - references["defense"]
    )
    output["offense_rating"] = (
        output["offense_base_rating"] + output["offense_additive_profile"]
    )
    output["defense_rating"] = (
        output["defense_base_rating"] + output["defense_additive_profile"]
    )
    output["split_nail_rating"] = output["offense_rating"] + output["defense_rating"]
    names = panel.loc[
        panel["season"].eq(season),
        [column for column in ("player_id", "player_name", "age") if column in panel],
    ].drop_duplicates("player_id")
    output = output.merge(names, on="player_id", how="left", validate="one_to_one")
    return (
        output.sort_values("split_nail_rating", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def _additive_profile_reference_per_player(fit: SplitNailSeasonFit) -> dict[str, float]:
    """Return the exposure-weighted average-player additive profile on each side.

    The scoring design uses lineup sums, so an uncentered beta-times-profile is not
    a portable player contribution: the common five-player baseline cancels
    between offense and defense and is absorbed by the scoring intercept.
    Materialized player ratings instead subtract the completed-season,
    possession-weighted average lineup profile divided by five player slots.
    """

    additive = fit.design.additive_features
    feature_count = len((*fit.design.additive_features, *fit.design.nonadditive_features))
    start = 2 * fit.design.player_count
    scales = np.asarray(fit.design.feature_scales, dtype=float)
    offense_values = fit.design.features[:, start : start + len(additive)].toarray()
    defense_start = start + feature_count
    defense_values = -fit.design.features[
        :, defense_start : defense_start + len(additive)
    ].toarray()
    offense_values *= scales[: len(additive)]
    defense_values *= scales[: len(additive)]
    coefficients = fit.feature_coefficients.set_index("feature")
    weights = np.asarray(fit.design.weights, dtype=float)
    offense = np.average(
        offense_values
        @ coefficients.loc[list(additive), "offense_raw_coefficient"].to_numpy(dtype=float),
        weights=weights,
    ) / 5.0
    defense = np.average(
        defense_values
        @ coefficients.loc[list(additive), "defense_raw_coefficient"].to_numpy(dtype=float),
        weights=weights,
    ) / 5.0
    return {"offense": float(offense), "defense": float(defense)}


def _evaluate_frozen_seasons(
    season_fits: dict[str, SplitNailSeasonFit],
    *,
    scalar_priors: dict[str, dict[int, float]],
    panel: pd.DataFrame,
    analytical_dir: Path | str,
    curated_dir: Path | str,
    seasons: tuple[str, ...],
    schedule_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    game_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for season in seasons:
        source_season = _previous_season(season)
        source_fit = season_fits.get(source_season)
        if source_fit is None:
            continue
        stints = _attach_back_to_back_flags(
            read_rapm_stints(season, analytical_dir=analytical_dir),
            schedule_features,
        )
        profiles = _profiles_for_season(
            season,
            _stint_participants(stints),
            panel=panel,
            analytical_dir=analytical_dir,
            curated_dir=curated_dir,
        )
        design = (
            build_split_nail_design(stints, profiles)
            if profiles is not None
            else _zero_profile_design(stints)
        )
        source_differences = dict(
            zip(
                source_fit.player_coefficients["player_id"].astype(int),
                (
                    source_fit.player_coefficients["offense_base_rating"]
                    - source_fit.player_coefficients["defense_base_rating"]
                ),
                strict=True,
            )
        )
        prediction = _frozen_scoring_prediction(
            design,
            source_fit,
            scalar_priors.get(season, {}),
            source_differences,
        )
        rows = pd.DataFrame(
            {
                "season": season,
                "source_season": source_season,
                "game_id": design.game_ids,
                "home_offense": design.home_offense,
                "possessions": design.weights,
                "actual_ppp": design.target,
                "predicted_ppp": prediction,
            }
        )
        rows["squared_error"] = np.square(rows["actual_ppp"] - rows["predicted_ppp"])
        rows["absolute_error"] = np.abs(rows["actual_ppp"] - rows["predicted_ppp"])
        prediction_rows.append(rows)
        games = _game_predictions(rows)
        game_rows.append(games)
        metric_rows.append(_frozen_metric_row(season, rows, games))
    return {
        "scoring_predictions": (
            pd.concat(prediction_rows, ignore_index=True)
            if prediction_rows
            else pd.DataFrame()
        ),
        "game_predictions": (
            pd.concat(game_rows, ignore_index=True) if game_rows else pd.DataFrame()
        ),
        "metrics": pd.DataFrame(metric_rows),
    }


def _frozen_scoring_prediction(
    target_design: SplitNailDesign,
    source_fit: SplitNailSeasonFit,
    scalar_priors: dict[int, float],
    source_differences: dict[int, float],
) -> np.ndarray:
    coefficients = split_nail_prior_vector(
        target_design,
        scalar_priors,
        source_differences,
    )
    source_features = source_fit.feature_coefficients.set_index("feature")
    for feature in (*target_design.additive_features, *target_design.nonadditive_features):
        scale = target_design.feature_scale(feature)
        coefficients[target_design.feature_column(feature, side="offense")] = (
            float(source_features.loc[feature, "offense_raw_coefficient"]) * scale
        )
        coefficients[target_design.feature_column(feature, side="defense")] = (
            float(source_features.loc[feature, "defense_raw_coefficient"]) * scale
        )
    if target_design.includes_back_to_back:
        source_schedule = source_fit.schedule_coefficients.set_index("schedule_control")
        if "back_to_back" not in source_schedule.index:
            raise ValueError("Source Split NAIL state lacks back-to-back coefficients")
        if target_design.back_to_back_scale is None:
            raise ValueError("Target Split NAIL state lacks a back-to-back scale")
        schedule_scale = float(target_design.back_to_back_scale)
        coefficients[target_design.back_to_back_column(side="offense")] = (
            float(source_schedule.loc["back_to_back", "offense_raw_coefficient"])
            * schedule_scale
        )
        coefficients[target_design.back_to_back_column(side="defense")] = (
            float(source_schedule.loc["back_to_back", "defense_raw_coefficient"])
            * schedule_scale
        )
    coefficients[target_design.home_court_column(side="offense")] = source_fit.model.coef_[
        source_fit.design.home_court_column(side="offense")
    ]
    coefficients[target_design.home_court_column(side="defense")] = source_fit.model.coef_[
        source_fit.design.home_court_column(side="defense")
    ]
    return (
        np.asarray(target_design.features @ coefficients, dtype=float).reshape(-1)
        + source_fit.model.intercept_
    )


def _game_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    points = rows.copy()
    points["predicted_points"] = points["predicted_ppp"] * points["possessions"] / 100.0
    points["actual_points"] = points["actual_ppp"] * points["possessions"] / 100.0
    grouped = points.groupby(
        ["season", "source_season", "game_id", "home_offense"], sort=False
    ).agg(
        predicted_points=("predicted_points", "sum"),
        actual_points=("actual_points", "sum"),
        possessions=("possessions", "sum"),
    )
    wide = grouped.unstack("home_offense")
    home = wide.xs(True, axis=1, level="home_offense")
    away = wide.xs(False, axis=1, level="home_offense")
    output = home.loc[:, ["predicted_points", "actual_points", "possessions"]].rename(
        columns=lambda name: f"home_{name}"
    )
    output = output.join(
        away.loc[:, ["predicted_points", "actual_points", "possessions"]].rename(
            columns=lambda name: f"away_{name}"
        )
    )
    output["predicted_margin"] = output["home_predicted_points"] - output["away_predicted_points"]
    output["actual_margin"] = output["home_actual_points"] - output["away_actual_points"]
    output["margin_error"] = output["predicted_margin"] - output["actual_margin"]
    output["correct_winner"] = (
        np.sign(output["predicted_margin"]) == np.sign(output["actual_margin"])
    )
    return output.reset_index()


def _frozen_metric_row(
    season: str,
    scoring: pd.DataFrame,
    games: pd.DataFrame,
) -> dict[str, object]:
    weights = scoring["possessions"].to_numpy(dtype=float)
    return {
        "season": season,
        "scoring_row_count": len(scoring),
        "game_count": len(games),
        "scoring_ppp_rmse": float(np.sqrt(np.average(scoring["squared_error"], weights=weights))),
        "scoring_ppp_mae": float(np.average(scoring["absolute_error"], weights=weights)),
        "game_margin_rmse": float(np.sqrt(np.mean(np.square(games["margin_error"])))),
        "game_winner_accuracy": float(games["correct_winner"].mean()),
    }


def _write_run(
    *,
    target: str,
    source_run_dir: Path | None,
    season_fits: dict[str, SplitNailSeasonFit],
    ratings: pd.DataFrame,
    feature_coefficients: pd.DataFrame,
    schedule_coefficients: pd.DataFrame,
    priors: pd.DataFrame,
    metadata: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    frozen: dict[str, pd.DataFrame],
    feature_relative_precision: float,
    artifacts_dir: Path,
) -> ForwardSplitNailRun:
    now = datetime.now(UTC)
    run_id = f"{RUN_PREFIX}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "player_season_ratings.parquet": ratings,
            "season_feature_coefficients.parquet": feature_coefficients,
            "season_schedule_coefficients.parquet": schedule_coefficients,
            "season_player_priors.parquet": priors,
            "season_model_metadata.parquet": metadata,
            "frozen_scoring_predictions.parquet": frozen["scoring_predictions"],
            "frozen_game_predictions.parquet": frozen["game_predictions"],
            "frozen_metrics.parquet": frozen["metrics"],
            "target_player_profiles.parquet": profiles.get(target, pd.DataFrame()),
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        joblib.dump(season_fits, temporary / "season_split_nail_models.joblib")
        metadata_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "target_season": target,
            "created_at": now.isoformat(),
            "source_scalar_model": "none_standalone_split_nail",
            "source_scalar_run_dir": None,
            "feature_relative_precision": feature_relative_precision,
            "state_contract": (
                "standalone forward combined prior plus prior Split NAIL O-minus-D "
                "base specialization"
            ),
            "feature_contract": (
                "all additive and retained non-additive lineup features have offense and "
                "defense coefficients; no semantic side-zero constraints"
            ),
            "schedule_contract": (
                "home-offense and home-defense terms plus side-specific back-to-back controls "
                "are fit in every completed source season and carried only into its next-season forecast"
            ),
            "code_version": modeling_code_fingerprint(
                (Path(__file__), Path(__file__).with_name("split_nail.py"))
            ),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata_payload, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata_payload, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ForwardSplitNailRun(run_dir=output, run_id=run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train recursive Split NAIL-RAPM")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument(
        "--feature-relative-precision",
        type=float,
        default=DEFAULT_FEATURE_RELATIVE_PRECISION,
    )
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_forward_split_nail(
                    through_season=args.through_season,
                    feature_relative_precision=args.feature_relative_precision,
                )
                print(f"Split NAIL-RAPM: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_forward_split_nail(
        through_season=args.through_season,
        feature_relative_precision=args.feature_relative_precision,
    )
    print(f"Split NAIL-RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
