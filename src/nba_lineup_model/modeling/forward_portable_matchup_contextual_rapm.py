"""Recursive RAPM with portable unit context plus matchup residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.context_reattributed_rapm import (
    ContextProjection,
    fit_context_projection,
)
from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_V1,
    CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
    CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
    CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO,
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    contextual_feature_columns,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.contextual_prior import _evaluate_target, _lineup_effects
from nba_lineup_model.modeling.contextual_profiles import build_contextual_player_profiles
from nba_lineup_model.modeling.forward_aging_player_prior import center_player_priors
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _fit_replacement_token,
    _latest_run,
    _returning_priors,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.matchup_contextual import (
    MatchupContextualModel,
    fit_matchup_contextual_model,
    model_metadata,
)
from nba_lineup_model.modeling.prior_rapm import (
    HISTORICAL_SEASONS,
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
    _available_processed_playoff_game_ids,
    fit_forward_lagged_rapm_season,
)
from nba_lineup_model.modeling.rebound_opportunity import (
    ReboundOpportunityModel,
    fit_rebound_opportunity_model,
)
from nba_lineup_model.modeling.replacement_level import (
    player_exposure_shares,
    prepare_player_exposure_cohort,
)
from nba_lineup_model.modeling.stints import (
    build_rapm_stints_from_legacy_processed_games,
    modeling_code_fingerprint,
    read_rapm_stints,
)
from nba_lineup_model.modeling.usage_allocation import (
    UsageAllocationModel,
    fit_usage_allocation_model,
)
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "forward_portable_matchup_contextual_rapm"
RUN_PREFIX = "forward-portable-matchup-contextual-rapm"

_COMPILED_ADDITIVE_PROFILE_COLUMNS = {
    "three_pa_per_100": "three_pa_per_100",
    "three_pm_per_100": "three_pm_per_100",
    "assists_per_100": "assists_per_100",
    "turnovers_per_100": "turnovers_per_100",
    "usage_per_100": "usage_per_100",
    "steals_per_100": "steals_per_100",
    "blocks_per_100": "blocks_per_100",
    "offensive_rebound_claim_total": "offensive_rebound_pct",
}


@dataclass(frozen=True)
class ForwardPortableMatchupContextualRapmRun:
    """One immutable recursive portable-plus-matchup context artifact."""

    run_dir: Path
    run_id: str


def train_forward_portable_matchup_contextual_rapm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    context_curvature_alpha: float = 0.0,
    context_temporal_alpha: float = 0.0,
    context_reattribution_weight: float = 0.0,
    compiled_additive_prior: bool = False,
    include_historical_playoffs: bool = False,
    use_context: bool = True,
    evaluate_target: bool = True,
    model_name: str = MODEL_NAME,
    run_prefix: str = RUN_PREFIX,
    player_prior_builder: Callable[..., tuple[pd.DataFrame, dict[str, object]]] | None = None,
    player_prior_description: str = "forward exposure-gated RAPM plus portable-matchup context",
    context_fit: Callable[..., MatchupContextualModel] = fit_matchup_contextual_model,
    context_metadata: Callable[[MatchupContextualModel], dict[str, object]] = model_metadata,
    context_feature_set: str = CONTEXT_FEATURE_SET_V1,
    profile_builder: Callable[..., pd.DataFrame] | None = None,
    profile_contract_metadata: dict[str, object] | None = None,
    profile_transformer: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Roll additive RAPM and reference-identified contextual state forward.

    When enabled, completed historical playoffs are appended to the regular
    season that precedes the next state update. The target-season playoffs are
    never used here: they remain part of the frozen evaluation contract.
    """

    target = validate_season(through_season)
    if use_context and (
        context_alpha <= 0 or context_curvature_alpha < 0 or context_temporal_alpha < 0
    ):
        raise ValueError("Contextual penalties must be non-negative, with alpha positive")
    if not 0.0 <= context_reattribution_weight <= 1.0:
        raise ValueError("context_reattribution_weight must be between zero and one")
    if context_reattribution_weight and not use_context:
        raise ValueError("Context reattribution requires context_enabled")
    if compiled_additive_prior:
        if not use_context:
            raise ValueError("Compiled additive prior transfer requires context_enabled")
        if context_feature_set != CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
            raise ValueError(
                "Compiled additive prior transfer requires the canonical HPM x3 feature set"
            )
        if context_reattribution_weight:
            raise ValueError(
                "Compiled additive prior transfer cannot be combined with context reattribution"
            )
    panel = pd.read_parquet(player_season_panel_path)
    artifact_root = Path(artifacts_dir)
    reference_root = _latest_reference_run(
        artifact_root / "forward_exposure_gated_rapm",
        target,
    )
    reference_coefficients = pd.read_parquet(
        reference_root / "historical_player_coefficients.parquet"
    )
    lambda_schedule = reference_coefficients.groupby("season", as_index=True)[
        "selected_lambda"
    ].agg(lambda values: float(values.iloc[0]))
    seasons = tuple(season for season in HISTORICAL_SEASONS if season <= target)
    if target not in seasons:
        seasons = (*seasons, target)
    source = _previous_season(target)
    print(
        f"Preparing exposure cohort through {target} for {len(seasons):,} seasonal fits",
        flush=True,
    )
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target)],
        through_season=target,
        analytical_dir=analytical_dir,
    )
    results: list[ForwardLaggedRapmSeason] = []
    priors_by_season: list[pd.DataFrame] = []
    exposure_history: list[pd.DataFrame] = []
    replacement_tokens: list[dict[str, object]] = []
    prior_metadata: list[dict[str, object]] = []
    aging_models: dict[str, object] = {}
    aging_curve_grids: list[pd.DataFrame] = []
    box_score_residual_models: dict[str, object] = {}
    box_score_residual_selections: list[pd.DataFrame] = []
    contextual_models: dict[str, MatchupContextualModel] = {}
    contextual_metadata: list[dict[str, object]] = []
    rebound_calibration_metadata: list[dict[str, object]] = []
    usage_allocation_metadata: list[dict[str, object]] = []
    context_reattributions: dict[str, ContextProjection] = {}
    context_reattribution_metadata: list[dict[str, object]] = []
    compiled_additive_prior_coefficients: list[pd.DataFrame] = []
    training_metadata: list[dict[str, object]] = []
    target_priors: pd.DataFrame | None = None
    target_profiles: pd.DataFrame | None = None
    prior_builder = player_prior_builder or _exposure_gated_player_priors
    resolved_profile_builder = profile_builder or build_contextual_player_profiles

    for season in seasons:
        print(f"Fitting portable-plus-matchup context state for {season}", flush=True)
        raw_stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        regular_stint_count = len(raw_stints)
        playoff_game_count = 0
        excluded_playoff_game_count = 0
        if include_historical_playoffs and season != target:
            playoff_ids = _available_processed_playoff_game_ids(season)
            if playoff_ids:
                (
                    playoff_stints,
                    excluded_playoff_ids,
                ) = build_rapm_stints_from_legacy_processed_games(playoff_ids)
                raw_stints = (
                    pd.concat([raw_stints, playoff_stints], ignore_index=True)
                    .sort_values(["game_time_utc", "game_id", "stint_index"], kind="stable")
                    .reset_index(drop=True)
                )
                playoff_game_count = len(playoff_ids) - len(excluded_playoff_ids)
                excluded_playoff_game_count = len(excluded_playoff_ids)
            print(
                f"  Added {playoff_game_count:,} completed playoff games to {season}",
                flush=True,
            )
        training_metadata.append(
            {
                "season": season,
                "training_regular_stint_count": regular_stint_count,
                "training_playoff_game_count": playoff_game_count,
                "training_excluded_playoff_game_count": excluded_playoff_game_count,
                "training_stint_count": len(raw_stints),
                "historical_playoffs_included": include_historical_playoffs and season != target,
            }
        )
        participants = set().union(*raw_stints["home_player_ids"], *raw_stints["away_player_ids"])
        previous_model = contextual_models.get(_previous_season(season))
        previous_reattribution = context_reattributions.get(_previous_season(season))
        priors, prior_row = prior_builder(
            season=season,
            panel=panel,
            completed_results=results,
            exposure_history=exposure_history,
            replacement_tokens=replacement_tokens,
        )
        profile_ids = participants
        if compiled_additive_prior:
            profile_ids = profile_ids | set(priors["player_id"].astype(int))
        profiles = (
            resolved_profile_builder(
                panel,
                target_season=season,
                target_player_ids=profile_ids,
                analytical_dir=str(analytical_dir),
                curated_dir=str(curated_dir),
                exposure_cohort=exposure_cohort,
            )
            if use_context and season != seasons[0]
            else None
        )
        if profiles is not None and profile_transformer is not None:
            profiles = profile_transformer(profiles, season)
        compiled_prior_metadata: dict[str, object] = {}
        if compiled_additive_prior and previous_model is not None and profiles is not None:
            priors, compiled_prior_metadata = _add_compiled_additive_context_to_priors(
                priors,
                previous_model,
                profiles,
                previous_exposure=exposure_history[-1] if exposure_history else None,
            )
            compiled_additive_prior_coefficients.append(
                _compiled_additive_coefficient_frame(
                    previous_model,
                    target_season=season,
                    source_season=_previous_season(season),
                )
            )
        if use_context and previous_model is not None and profiles is not None:
            offset = (
                _compiled_additive_residual_context_offset(
                    raw_stints, previous_model, profiles
                )
                if compiled_additive_prior
                else _context_offset(
                    raw_stints,
                    previous_model,
                    profiles,
                    reattribution=previous_reattribution,
                    reattribution_weight=context_reattribution_weight,
                )
                if context_reattribution_weight
                else _context_offset(raw_stints, previous_model, profiles)
            )
        else:
            offset = np.zeros(len(raw_stints), dtype=float)
        adjusted_stints = raw_stints.copy()
        adjusted_stints["target_home_net_rating"] = (
            raw_stints["target_home_net_rating"].to_numpy(dtype=float) - offset
        )
        priors, reattribution_prior_metadata = _add_context_reattribution_to_priors(
            priors,
            previous_reattribution,
            reattribution_weight=context_reattribution_weight,
        )
        prior_row = dict(prior_row)
        prior_row.update(reattribution_prior_metadata)
        prior_row.update(compiled_prior_metadata)
        aging_model = prior_row.pop("_aging_model", None)
        aging_curve_grid = prior_row.pop("_aging_curve_grid", None)
        box_score_residual_model = prior_row.pop("_box_score_residual_model", None)
        box_score_residual_selection = prior_row.pop("box_score_residual_selection", None)
        if aging_model is not None:
            aging_models[season] = aging_model
        if isinstance(aging_curve_grid, pd.DataFrame):
            aging_curve_grids.append(aging_curve_grid)
        if box_score_residual_model is not None:
            box_score_residual_models[season] = box_score_residual_model
        if isinstance(box_score_residual_selection, list):
            box_score_residual_selections.append(
                pd.DataFrame(box_score_residual_selection).assign(season=season)
            )
        prior_metadata.append(prior_row)
        prior_rows = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm"}).copy()
        prior_rows["season"] = season
        prior_rows["context_offset_source_season"] = (
            _previous_season(season) if previous_model is not None else pd.NA
        )
        priors_by_season.append(prior_rows)
        season_lambda = float(lambda_schedule.loc[season])
        if not np.isfinite(season_lambda) or season_lambda < 0:
            raise ValueError(f"Published RAPM lambda is invalid for {season}")
        fitted = fit_forward_lagged_rapm_season(
            season,
            adjusted_stints,
            priors,
            lambda_grid=(season_lambda,),
        )
        results.append(fitted)
        exposure = player_exposure_shares(raw_stints)
        exposure_history.append(
            panel.loc[panel["season"].eq(season)].merge(
                exposure, on="player_id", how="inner", validate="one_to_one"
            )
        )
        replacement_tokens.append(
            _fit_replacement_token(season, adjusted_stints, exposure, fitted, panel)
        )
        if use_context and profiles is not None:
            rebound_model = (
                fit_rebound_opportunity_model(season, profiles, curated_dir=curated_dir)
                if context_feature_set
                in {
                    CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
                    CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
                    CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO,
                }
                else None
            )
            usage_model = (
                fit_usage_allocation_model(season, profiles, curated_dir=curated_dir)
                if context_feature_set
                in {
                    CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
                    CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO,
                }
                else None
            )
            if rebound_model is not None:
                print(
                    f"  Calibrated rebound realization on "
                    f"{rebound_model.training_opportunity_count:,} opportunities",
                    flush=True,
                )
                rebound_calibration_metadata.append(
                    {
                        "season": season,
                        "calibration_season": rebound_model.training_season,
                        "training_opportunity_count": rebound_model.training_opportunity_count,
                        "reference_grid_size": len(rebound_model.reference_defensive_claims),
                    }
                )
            if usage_model is not None:
                print(
                    "  Calibrated usage allocation on "
                    f"{usage_model.training_event_count:,} actions",
                    flush=True,
                )
                usage_allocation_metadata.append(
                    {
                        "season": season,
                        "calibration_season": usage_model.training_season,
                        "training_event_count": usage_model.training_event_count,
                        "temperature": usage_model.temperature,
                        "claim_budget": usage_model.claim_budget,
                    }
                )
            model, row = _fit_matchup_contextual_season(
                raw_stints,
                fitted,
                profiles,
                alpha=context_alpha,
                curvature_alpha=context_curvature_alpha,
                temporal_alpha=context_temporal_alpha if previous_model is not None else 0.0,
                previous_model=previous_model,
                context_fit=context_fit,
                context_metadata=context_metadata,
                context_feature_set=context_feature_set,
                rebound_model=rebound_model,
                usage_model=usage_model,
            )
            contextual_models[season] = model
            contextual_metadata.append(row)
            if context_reattribution_weight:
                full_context = _context_offset(raw_stints, model, profiles)
                projection = fit_context_projection(raw_stints, full_context, season_lambda)
                context_reattributions[season] = projection
                context_reattribution_metadata.append(
                    _context_reattribution_metadata(
                        season=season,
                        stints=raw_stints,
                        full_context=full_context,
                        projection=projection,
                        reattribution_weight=context_reattribution_weight,
                    )
                )
        if season == target:
            target_priors = priors.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm_mean"})
            target_profiles = profiles if profiles is not None else pd.DataFrame()

    if target_priors is None or target_profiles is None:
        raise ValueError("Forward RAPM did not create the target state")
    forecast_model = contextual_models.get(source) if use_context else None
    if use_context and forecast_model is None:
        raise ValueError("Portable-matchup contextual RAPM has no source context state")
    historical_coefficients = pd.concat(
        [result.player_estimates for result in results], ignore_index=True
    )
    state_priors = pd.concat(priors_by_season, ignore_index=True)
    evaluation = (
        _evaluate_target(
            target,
            model=forecast_model,  # type: ignore[arg-type]
            profiles=target_profiles,
            priors=state_priors,
            coefficients=historical_coefficients,
            analytical_dir=Path(analytical_dir),
            curated_dir=Path(curated_dir),
            evaluation_model=model_name,
            context_predictor=(
                _context_predictor_with_reattribution(
                    forecast_model,
                    context_reattributions.get(source),
                    context_reattribution_weight,
                )
                if forecast_model is not None and context_reattribution_weight
                else forecast_model.predict_lineups
                if forecast_model is not None and not compiled_additive_prior
                else _context_predictor_with_compiled_additive_prior(forecast_model)
                if forecast_model is not None and compiled_additive_prior
                else _zero_context_predictor
            ),
        )
        if evaluate_target
        else _empty_target_evaluation(target=target, source=source)
    )
    evaluation["source_state"] = {
        **evaluation["source_state"],  # type: ignore[arg-type]
        "player_prior_method": player_prior_description,
        "context_enabled": use_context,
        "context_contract": (
            "C_shape_(t-1)(home, away) = C_full_(t-1)(home, away) - "
            "beta_(t-1)'(z(home) - z(away))"
            if compiled_additive_prior
            else "C_(t-1)(home, away) = h(home) - h(away) + q(home, away)"
            if not context_reattribution_weight
            else "C_residual_(t-1) = C_(t-1) - rho X gamma_(t-1)"
            if use_context
            else "C_(t-1)(home, away) = 0"
        ),
        "reference_context_source_season": source if use_context else None,
        "target_evaluation_performed": evaluate_target,
        "context_feature_set": context_feature_set if use_context else None,
        "context_feature_columns": list(contextual_feature_columns(context_feature_set))
        if use_context
        else [],
        "profile_padding_contract": profile_contract_metadata,
    }
    return _write_run(
        target=target,
        results=results,
        priors=state_priors,
        contextual_models=contextual_models,
        contextual_metadata=pd.DataFrame(contextual_metadata),
        rebound_calibration_metadata=pd.DataFrame(rebound_calibration_metadata),
        usage_allocation_metadata=pd.DataFrame(usage_allocation_metadata),
        prior_metadata=pd.DataFrame(prior_metadata),
        target_priors=target_priors,
        target_profiles=target_profiles,
        forecast_reference=(
            forecast_model.reference_features.assign(
                reference_weight=forecast_model.reference_weights
            )
            if forecast_model is not None
            else pd.DataFrame()
        ),
        evaluation=evaluation,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        context_enabled=use_context,
        context_reattribution_weight=context_reattribution_weight,
        context_reattributions=context_reattributions,
        context_reattribution_metadata=pd.DataFrame(context_reattribution_metadata),
        compiled_additive_prior=compiled_additive_prior,
        compiled_additive_prior_coefficients=(
            pd.concat(compiled_additive_prior_coefficients, ignore_index=True)
            if compiled_additive_prior_coefficients
            else pd.DataFrame()
        ),
        model_name=model_name,
        run_prefix=run_prefix,
        exposure_history=exposure_history,
        aging_models=aging_models,
        aging_curve_grid=(
            pd.concat(aging_curve_grids, ignore_index=True) if aging_curve_grids else pd.DataFrame()
        ),
        box_score_residual_models=box_score_residual_models,
        box_score_residual_selection=(
            pd.concat(box_score_residual_selections, ignore_index=True)
            if box_score_residual_selections
            else pd.DataFrame()
        ),
        training_metadata=pd.DataFrame(training_metadata),
        profile_contract_metadata=profile_contract_metadata,
        artifacts_dir=artifact_root,
    )


def _context_offset(
    stints: pd.DataFrame,
    model: MatchupContextualModel,
    profiles: pd.DataFrame,
    *,
    reattribution: ContextProjection | None = None,
    reattribution_weight: float = 0.0,
) -> np.ndarray:
    full_context = model.predict_lineups(
        stints["home_player_ids"].tolist(),
        stints["away_player_ids"].tolist(),
        profiles,
    )
    if reattribution is None or not reattribution_weight:
        return full_context
    projected = _project_reattributed_player_context(stints, reattribution)
    return full_context - reattribution_weight * projected


def _linear_raw_context_coefficients(model: MatchupContextualModel) -> pd.Series:
    """Return original-unit coefficients from the canonical linear x3 model."""

    if model.feature_set != CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
        raise ValueError("Compiled additive transfer requires canonical HPM x3 features")
    if tuple(model.pipeline.named_steps) != ("scale", "ridge"):
        raise ValueError("Compiled additive transfer requires a linear scale-plus-ridge model")
    scale = model.pipeline.named_steps["scale"]
    ridge = model.pipeline.named_steps["ridge"]
    values = np.asarray(ridge.coef_, dtype=float) / np.asarray(scale.scale_, dtype=float)
    columns = side_context_feature_columns(model.feature_set)
    if len(values) != len(columns):
        raise ValueError("Linear HPM x3 coefficient count does not match its feature contract")
    return pd.Series(values, index=columns, name="raw_context_coefficient")


def _compiled_additive_context(
    home: pd.DataFrame,
    away: pd.DataFrame,
    model: MatchupContextualModel,
) -> np.ndarray:
    """Evaluate only the exactly player-compilable part of linear x3 context."""

    coefficients = _linear_raw_context_coefficients(model)
    columns = list(LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES)
    relative = home.loc[:, columns].to_numpy(dtype=float) - away.loc[:, columns].to_numpy(
        dtype=float
    )
    return relative @ coefficients.loc[columns].to_numpy(dtype=float)


def _compiled_additive_residual_context_offset(
    stints: pd.DataFrame,
    model: MatchupContextualModel,
    profiles: pd.DataFrame,
) -> np.ndarray:
    """Return carried context after removing its transferred additive player part."""

    home = lineup_side_context_features(
        stints["home_player_ids"].tolist(), profiles, feature_set=model.feature_set
    )
    away = lineup_side_context_features(
        stints["away_player_ids"].tolist(), profiles, feature_set=model.feature_set
    )
    full = model.predict_side_pairs(home, away)
    return full - _compiled_additive_context(home, away, model)


def _context_predictor_with_compiled_additive_prior(
    model: MatchupContextualModel,
) -> Callable[[object, object, object], np.ndarray]:
    """Expose only non-additive shape context once beta has entered the prior."""

    def predict(home_lineups: object, away_lineups: object, profiles: object) -> np.ndarray:
        stints = pd.DataFrame({"home_player_ids": home_lineups, "away_player_ids": away_lineups})
        return _compiled_additive_residual_context_offset(
            stints, model, profiles  # type: ignore[arg-type]
        )

    return predict


def _add_compiled_additive_context_to_priors(
    priors: pd.DataFrame,
    model: MatchupContextualModel,
    profiles: pd.DataFrame,
    *,
    previous_exposure: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add prior-season beta times current lagged player profiles to the prior."""

    coefficients = _linear_raw_context_coefficients(model)
    columns = list(LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES)
    profile_columns = [_COMPILED_ADDITIVE_PROFILE_COLUMNS[column] for column in columns]
    required = {"player_id", *profile_columns}
    missing = required - set(profiles)
    if missing:
        raise ValueError("Compiled additive prior profiles lack: " + ", ".join(sorted(missing)))
    values = profiles.loc[:, ["player_id", *profile_columns]].copy()
    values["compiled_additive_prior_adjustment"] = values.loc[:, profile_columns].to_numpy(
        dtype=float
    ) @ coefficients.loc[columns].to_numpy(dtype=float)
    output = priors.merge(
        values.loc[:, ["player_id", "compiled_additive_prior_adjustment"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    available = output["compiled_additive_prior_adjustment"].notna()
    output[PRIOR_MEAN_COLUMN] += output["compiled_additive_prior_adjustment"].fillna(0.0)
    centered, center_metadata = center_player_priors(
        output.drop(columns="compiled_additive_prior_adjustment"),
        previous_exposure=previous_exposure,
    )
    return centered, {
        "compiled_additive_prior_enabled": True,
        "compiled_additive_prior_source_feature_set": model.feature_set,
        "compiled_additive_prior_feature_count": len(columns),
        "compiled_additive_prior_available_player_count": int(available.sum()),
        "compiled_additive_prior_missing_player_count": int((~available).sum()),
        **{f"compiled_additive_{key}": value for key, value in center_metadata.items()},
    }


def _compiled_additive_coefficient_frame(
    model: MatchupContextualModel,
    *,
    target_season: str,
    source_season: str,
) -> pd.DataFrame:
    """Materialize the beta state carried into one following-season player prior."""

    coefficients = _linear_raw_context_coefficients(model)
    columns = list(LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES)
    return pd.DataFrame(
        {
            "target_season": target_season,
            "source_season": source_season,
            "feature": columns,
            "raw_context_coefficient": coefficients.loc[columns].to_numpy(dtype=float),
        }
    )


def _context_predictor_with_reattribution(
    model: MatchupContextualModel,
    reattribution: ContextProjection | None,
    reattribution_weight: float,
) -> Callable[[object, object, object], np.ndarray]:
    """Return residual context after its transferred player component is removed."""

    def predict(home_lineups: object, away_lineups: object, profiles: object) -> np.ndarray:
        full_context = model.predict_lineups(home_lineups, away_lineups, profiles)  # type: ignore[arg-type]
        if reattribution is None or not reattribution_weight:
            return full_context
        stints = pd.DataFrame({"home_player_ids": home_lineups, "away_player_ids": away_lineups})
        return full_context - reattribution_weight * _project_reattributed_player_context(
            stints, reattribution
        )

    return predict


def _project_reattributed_player_context(
    stints: pd.DataFrame,
    reattribution: ContextProjection,
) -> np.ndarray:
    """Evaluate one fitted player projection on lineups with no unseen-player credit."""

    values = dict(zip(reattribution.player_ids, reattribution.coefficients, strict=True))
    home = np.fromiter(
        (
            sum(values.get(int(player_id), 0.0) for player_id in lineup)
            for lineup in stints["home_player_ids"]
        ),
        dtype=float,
        count=len(stints),
    )
    away = np.fromiter(
        (
            sum(values.get(int(player_id), 0.0) for player_id in lineup)
            for lineup in stints["away_player_ids"]
        ),
        dtype=float,
        count=len(stints),
    )
    # The projection intercept is not portable player value. It remains in the
    # context residual so transferring gamma cannot shift every matchup.
    return home - away


def _add_context_reattribution_to_priors(
    priors: pd.DataFrame,
    reattribution: ContextProjection | None,
    *,
    reattribution_weight: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Promote a fixed fraction of completed context into returning-player priors."""

    output = priors.copy()
    metadata: dict[str, object] = {
        "context_reattribution_enabled": bool(reattribution_weight),
        "context_reattribution_weight": reattribution_weight,
        "context_reattribution_source_available": reattribution is not None,
        "context_reattribution_returning_player_count": 0,
    }
    if reattribution is None or not reattribution_weight:
        return output, metadata
    values = pd.DataFrame(
        {
            "player_id": reattribution.player_ids,
            "context_reattribution": reattribution.coefficients,
        }
    )
    output = output.merge(values, on="player_id", how="left", validate="one_to_one")
    available = output["context_reattribution"].notna()
    output.loc[available, PRIOR_MEAN_COLUMN] += (
        reattribution_weight * output.loc[available, "context_reattribution"]
    )
    metadata["context_reattribution_returning_player_count"] = int(available.sum())
    metadata["context_reattribution_intercept_not_transferred"] = reattribution.intercept
    return output.drop(columns="context_reattribution"), metadata


def _context_reattribution_metadata(
    *,
    season: str,
    stints: pd.DataFrame,
    full_context: np.ndarray,
    projection: ContextProjection,
    reattribution_weight: float,
) -> dict[str, object]:
    weights = stints["possessions"].to_numpy(dtype=float)
    mean = float(np.average(full_context, weights=weights))
    total = float(np.sum(weights * np.square(full_context - mean)))
    residual = projection.residual
    residual_sum = float(np.sum(weights * np.square(residual)))
    return {
        "season": season,
        "context_reattribution_weight": reattribution_weight,
        "context_reattribution_lambda": projection.selected_lambda,
        "context_reattribution_intercept": projection.intercept,
        "context_reattribution_player_count": len(projection.player_ids),
        "context_reattribution_weighted_r_squared": (
            1.0 - residual_sum / total if total else float("nan")
        ),
        "context_reattribution_residual_rmse": float(
            np.sqrt(np.average(np.square(residual), weights=weights))
        ),
    }


def _zero_context_predictor(
    home_lineups: object,
    away_lineups: object,
    profiles: object,
) -> np.ndarray:
    """Provide the player-only control with a compatible zero context state."""

    del away_lineups, profiles
    return np.zeros(len(home_lineups), dtype=float)  # type: ignore[arg-type]


def _empty_target_evaluation(*, target: str, source: str) -> dict[str, object]:
    """Persist recursive state without requiring target-season scoring data."""

    return {
        "source_state": {
            "target_season": target,
            "source_season": source,
            "target_evaluation_performed": False,
        },
        "cohort_metrics": pd.DataFrame(),
        "possession_predictions": pd.DataFrame(),
        "game_predictions": pd.DataFrame(),
        "regular_game_predictions": pd.DataFrame(),
        "team_net_rating_predictions": pd.DataFrame(),
        "team_net_rating_metrics": pd.DataFrame(),
        "team_win_predictions": pd.DataFrame(),
        "team_win_metrics": pd.DataFrame(),
    }


def _latest_reference_run(root: Path, target: str) -> Path:
    """Use the nearest complete exposure-gated run that reaches ``target``."""

    exact = root / target
    try:
        return _latest_run(exact)
    except FileNotFoundError:
        pass
    if not root.is_dir():
        raise ValueError(f"No exposure-gated reference artifact reaches {target}: {root}")
    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name >= target and (path / "latest.json").is_file()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise ValueError(f"No exposure-gated reference artifact reaches {target}: {root}")
    return _latest_run(candidates[0])


def _exposure_gated_player_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Preserve the established lagged-returner and cold-start prior state."""

    cold, cold_metadata = _cold_start_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,
        exposure_history=exposure_history,
        replacement_tokens=replacement_tokens,
    )
    return _combine_priors(_returning_priors(completed_results), cold), {
        "season": season,
        "player_prior_method": "lagged_rapm_plus_exposure_gated_cold_start",
        "cold_start": cold_metadata,
    }


def _fit_matchup_contextual_season(
    stints: pd.DataFrame,
    fitted: ForwardLaggedRapmSeason,
    profiles: pd.DataFrame,
    *,
    alpha: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    context_fit: Callable[..., MatchupContextualModel] = fit_matchup_contextual_model,
    context_metadata: Callable[[MatchupContextualModel], dict[str, object]] = model_metadata,
    context_feature_set: str = CONTEXT_FEATURE_SET_V1,
    rebound_model: ReboundOpportunityModel | None = None,
    usage_model: UsageAllocationModel | None = None,
) -> tuple[MatchupContextualModel, dict[str, object]]:
    coefficients = fitted.player_estimates.loc[:, ["player_id", "rapm"]]
    values = dict(zip(coefficients["player_id"].astype(int), coefficients["rapm"], strict=True))
    effects, unknown = _lineup_effects(stints, values)
    intercept = _recover_home_intercept(stints, coefficients)
    home = lineup_side_context_features(
        stints["home_player_ids"].tolist(),
        profiles,
        feature_set=context_feature_set,
        rebound_model=rebound_model,
        usage_model=usage_model,
    )
    away = lineup_side_context_features(
        stints["away_player_ids"].tolist(),
        profiles,
        feature_set=context_feature_set,
        rebound_model=rebound_model,
        usage_model=usage_model,
    )
    target = stints["target_home_net_rating"].to_numpy(dtype=float) - effects - intercept
    print(f"  Fitting antisymmetric context model on {len(stints):,} stints", flush=True)
    model = context_fit(
        home,
        away,
        target,
        stints["possessions"].to_numpy(dtype=float),
        alpha=alpha,
        curvature_alpha=curvature_alpha,
        temporal_alpha=temporal_alpha,
        previous_model=previous_model,
        feature_set=context_feature_set,
    )
    if rebound_model is not None or usage_model is not None:
        model = replace(model, rebound_model=rebound_model, usage_model=usage_model)
    print(
        f"  Stored {len(model.reference_features):,} reference units for {fitted.season}",
        flush=True,
    )
    return model, {
        "season": fitted.season,
        "context_alpha": alpha,
        "context_curvature_alpha": curvature_alpha,
        "context_temporal_alpha": temporal_alpha,
        "context_temporal_source_season": (
            _previous_season(fitted.season) if previous_model is not None else pd.NA
        ),
        "context_training_stint_count": len(stints),
        "context_unknown_player_exposures": int(unknown.sum()),
        "context_home_intercept": intercept,
        "context_feature_set": context_feature_set,
        "context_feature_columns": list(contextual_feature_columns(context_feature_set)),
        "rebound_calibration_source_season": rebound_model.training_season
        if rebound_model is not None
        else pd.NA,
        "rebound_calibration_opportunity_count": rebound_model.training_opportunity_count
        if rebound_model is not None
        else pd.NA,
        "usage_allocation_source_season": usage_model.training_season
        if usage_model is not None
        else pd.NA,
        "usage_allocation_event_count": usage_model.training_event_count
        if usage_model is not None
        else pd.NA,
        **context_metadata(model),
    }


def _write_run(
    *,
    target: str,
    results: list[ForwardLaggedRapmSeason],
    priors: pd.DataFrame,
    contextual_models: dict[str, MatchupContextualModel],
    contextual_metadata: pd.DataFrame,
    rebound_calibration_metadata: pd.DataFrame,
    usage_allocation_metadata: pd.DataFrame,
    prior_metadata: pd.DataFrame,
    target_priors: pd.DataFrame,
    target_profiles: pd.DataFrame,
    forecast_reference: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame | dict[str, object]],
    context_alpha: float,
    context_curvature_alpha: float,
    context_temporal_alpha: float,
    context_enabled: bool,
    context_reattribution_weight: float,
    context_reattributions: dict[str, ContextProjection],
    context_reattribution_metadata: pd.DataFrame,
    compiled_additive_prior: bool,
    compiled_additive_prior_coefficients: pd.DataFrame,
    model_name: str,
    run_prefix: str,
    exposure_history: list[pd.DataFrame],
    aging_models: dict[str, object],
    aging_curve_grid: pd.DataFrame,
    box_score_residual_models: dict[str, object],
    box_score_residual_selection: pd.DataFrame,
    training_metadata: pd.DataFrame,
    profile_contract_metadata: dict[str, object] | None,
    artifacts_dir: Path,
) -> ForwardPortableMatchupContextualRapmRun:
    now = datetime.now(UTC)
    run_id = f"{run_prefix}-{target}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / model_name / target
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        historical_coefficients = pd.concat(
            [result.player_estimates for result in results], ignore_index=True
        )
        player_season_ratings = _player_season_ratings(
            historical_coefficients,
            exposure_history=exposure_history,
        )
        season_model_metadata = _season_model_metadata(
            target=target,
            model_name=model_name,
            run_id=run_id,
            results=results,
            prior_metadata=prior_metadata,
            contextual_metadata=contextual_metadata,
            context_alpha=context_alpha,
            context_curvature_alpha=context_curvature_alpha,
            context_temporal_alpha=context_temporal_alpha,
            training_metadata=training_metadata,
        )
        tables: dict[str, pd.DataFrame] = {
            "historical_player_coefficients.parquet": historical_coefficients,
            "player_season_ratings.parquet": player_season_ratings,
            "season_model_metadata.parquet": season_model_metadata,
            "season_player_priors.parquet": priors,
            "season_context_metadata.parquet": contextual_metadata,
            "season_rebound_calibration_metadata.parquet": rebound_calibration_metadata,
            "season_usage_allocation_metadata.parquet": usage_allocation_metadata,
            "season_context_reattribution_metadata.parquet": context_reattribution_metadata,
            "season_compiled_additive_prior_coefficients.parquet": (
                compiled_additive_prior_coefficients
            ),
            "season_player_prior_metadata.parquet": prior_metadata,
            "frozen_2025_26_player_priors.parquet": target_priors,
            "target_player_profiles.parquet": target_profiles,
            "forecast_reference_units.parquet": forecast_reference,
            "cohort_metrics.parquet": evaluation["cohort_metrics"],  # type: ignore[dict-item]
            "possession_predictions.parquet": evaluation["possession_predictions"],  # type: ignore[dict-item]
            "game_predictions.parquet": evaluation["game_predictions"],  # type: ignore[dict-item]
            "regular_game_predictions.parquet": evaluation["regular_game_predictions"],  # type: ignore[dict-item]
            "team_net_rating_predictions.parquet": evaluation["team_net_rating_predictions"],  # type: ignore[dict-item]
            "team_net_rating_metrics.parquet": evaluation["team_net_rating_metrics"],  # type: ignore[dict-item]
            "team_win_predictions.parquet": evaluation["team_win_predictions"],  # type: ignore[dict-item]
            "team_win_metrics.parquet": evaluation["team_win_metrics"],  # type: ignore[dict-item]
        }
        if not aging_curve_grid.empty:
            tables["aging_curve_grid.parquet"] = aging_curve_grid.assign(
                run_id=run_id,
                model=model_name,
            )
        if not box_score_residual_selection.empty:
            tables["season_box_score_residual_selection.parquet"] = box_score_residual_selection
        if context_reattributions:
            reattribution_rows = [
                pd.DataFrame(
                    {
                        "season": season,
                        "player_id": projection.player_ids,
                        "context_reattribution": projection.coefficients,
                        "context_reattribution_intercept": projection.intercept,
                        "context_reattribution_lambda": projection.selected_lambda,
                    }
                )
                for season, projection in context_reattributions.items()
            ]
            tables["season_context_reattributions.parquet"] = pd.concat(
                reattribution_rows, ignore_index=True
            )
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        joblib.dump(contextual_models, temporary / "season_context_models.joblib")
        if aging_models:
            joblib.dump(aging_models, temporary / "season_aging_models.joblib")
        if box_score_residual_models:
            joblib.dump(
                box_score_residual_models,
                temporary / "season_box_score_residual_models.joblib",
            )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": model_name,
            "target_season": target,
            "context_alpha": context_alpha,
            "context_curvature_alpha": context_curvature_alpha,
            "context_temporal_alpha": context_temporal_alpha,
            "context_enabled": context_enabled,
            "context_reattribution_weight": context_reattribution_weight,
            "compiled_additive_prior": compiled_additive_prior,
            "profile_padding_contract": profile_contract_metadata,
            "contextual_offset_contract": (
                "The prior-season linear additive profile term is added to player priors; "
                "only the remaining lineup-shape context is carried forward"
                if compiled_additive_prior
                else
                "C_(t-1)(home, away) less its transferred player projection is subtracted "
                "before season t RAPM"
                if context_enabled
                else "C_(t-1)(home, away) is identically zero in the controlled ablation"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint(
                (Path(__file__), Path(__file__).with_name("matchup_contextual.py"))
            ),
            "source_state": evaluation["source_state"],
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.replace(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return ForwardPortableMatchupContextualRapmRun(output, run_id)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _player_season_ratings(
    coefficients: pd.DataFrame,
    *,
    exposure_history: list[pd.DataFrame],
) -> pd.DataFrame:
    """Normalize annual player estimates with exposure and deterministic ranks."""

    exposure_columns = (
        "season",
        "player_id",
        "player_name",
        "age",
        "nba_experience_years",
        "is_rookie",
        "rapm_seconds",
        "rapm_exposure_eligible",
        "on_court_possessions",
        "team_opportunity_possessions",
        "exposure_share",
        "team_count",
    )
    available = [
        column
        for column in exposure_columns
        if any(column in frame.columns for frame in exposure_history)
    ]
    exposure = pd.concat(exposure_history, ignore_index=True).loc[:, available]
    exposure = exposure.drop_duplicates(["season", "player_id"], keep="last")
    frame = coefficients.merge(
        exposure,
        on=["season", "player_id"],
        how="left",
        suffixes=("", "_panel"),
        validate="one_to_one",
    )
    if "player_name_panel" in frame:
        frame["player_name"] = frame["player_name"].fillna(frame.pop("player_name_panel"))
    frame = frame.sort_values(
        ["season", "rapm", "player_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["rank_all_players"] = frame.groupby("season", sort=False).cumcount() + 1
    frame["rank_exposure_eligible"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    eligible = frame["rapm_exposure_eligible"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    frame.loc[eligible, "rank_exposure_eligible"] = (
        frame.loc[eligible].groupby("season", sort=False).cumcount() + 1
    ).astype("Int64")
    group_sizes = frame.groupby("season")["player_id"].transform("size")
    denominator = (group_sizes - 1).clip(lower=1)
    frame["percentile_all_players"] = 1.0 - (frame["rank_all_players"] - 1) / denominator
    return frame


def _season_model_metadata(
    *,
    target: str,
    model_name: str,
    run_id: str,
    results: list[ForwardLaggedRapmSeason],
    prior_metadata: pd.DataFrame,
    contextual_metadata: pd.DataFrame,
    context_alpha: float,
    context_curvature_alpha: float,
    context_temporal_alpha: float,
    training_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Write one inspectable row for every recursive seasonal fit."""

    lambdas = pd.DataFrame(
        {
            "season": [result.season for result in results],
            "selected_lambda": [result.selected_lambda for result in results],
            "estimated_player_count": [len(result.player_estimates) for result in results],
        }
    )
    prior = _serialize_nested_metadata(prior_metadata)
    context = _serialize_nested_metadata(contextual_metadata)
    output = lambdas.merge(prior, on="season", how="left", validate="one_to_one")
    if not context.empty:
        output = output.merge(context, on="season", how="left", validate="one_to_one")
    output = output.merge(
        training_metadata,
        on="season",
        how="left",
        validate="one_to_one",
    )
    output.insert(0, "model", model_name)
    output.insert(1, "run_id", run_id)
    output["source_season"] = output["season"].map(_previous_season)
    output["information_cutoff"] = output["source_season"].map(lambda season: f"end_of_{season}")
    output["is_target_completed_refit"] = output["season"].eq(target)
    output["frozen_forecast_target_season"] = target
    output["is_frozen_forecast_source_season"] = output["season"].eq(_previous_season(target))
    output["configured_context_alpha"] = context_alpha
    output["configured_context_curvature_alpha"] = context_curvature_alpha
    output["configured_context_temporal_alpha"] = context_temporal_alpha
    return output.sort_values("season", kind="stable").reset_index(drop=True)


def _serialize_nested_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Make list/dict metadata Parquet-friendly without losing its structure."""

    if frame.empty:
        return frame.copy()
    output = frame.copy()
    for column in output:
        if output[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            output[column] = output[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            )
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Train the published portable-plus-matchup contextual RAPM exemplar."""

    parser = argparse.ArgumentParser(
        description="Train forward RAPM with portable lineup context and matchup residuals"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_forward_portable_matchup_contextual_rapm(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Forward portable-matchup contextual RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
