"""Strictly forward age-adjusted returning-player priors for recursive RAPM."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.aging import (
    AGING_FEATURE_COLUMNS,
    DEFAULT_AGING_REGULARIZATION_GRID,
    fit_aging_pipeline,
    prepare_aging_prior_features,
    prepare_aging_transitions,
    run_aging_experiment,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import (
    _cold_start_priors,
    _combine_priors,
    _returning_priors,
)
from nba_lineup_model.modeling.prior_rapm import (
    PRIOR_MEAN_COLUMN,
    ForwardLaggedRapmSeason,
)

_BIO_COLUMNS = (
    "player_id",
    "player_name",
    "age",
    "nba_experience_years",
    "is_rookie",
    "draft_year",
    "draft_number",
    "height_inches",
    "weight_pounds",
    "is_undrafted",
    "rapm_seconds",
    "rapm_exposure_eligible",
)


def build_aging_exposure_gated_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a forward aging prior for returners plus the existing rookie branch.

    Every aging fit uses transition labels from completed recursive RAPM seasons
    only.  The target-season row supplies known biographical inputs and the
    immediately prior completed RAPM/exposure state, never target outcomes.
    """

    cold, cold_metadata = _cold_start_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,
        exposure_history=exposure_history,
        replacement_tokens=replacement_tokens,
    )
    returning = _returning_priors(completed_results)
    metadata: dict[str, object] = {
        "season": season,
        "player_prior_method": "lagged_rapm_fallback",
        "aging_enabled": False,
        "cold_start": cold_metadata,
    }
    if returning.empty:
        return _combine_priors(returning, cold), metadata

    try:
        transitions = _aging_transition_history(panel, completed_results, exposure_history)
        experiment = run_aging_experiment(
            transitions,
            regularization_grid=DEFAULT_AGING_REGULARIZATION_GRID,
        )
        training = prepare_aging_transitions(transitions)
        model = fit_aging_pipeline(
            training,
            regularization=experiment.selected_regularization,
            age_spline_knots=5,
            age_spline_degree=2,
        )
        target = _target_returning_features(
            panel,
            season=season,
            returning=returning,
            latest_exposure=exposure_history[-1],
        )
        target = prepare_aging_prior_features(target)
        predicted = model.predict(target.loc[:, AGING_FEATURE_COLUMNS])
        age_returning = target.loc[:, ["player_id"]].copy()
        age_returning[PRIOR_MEAN_COLUMN] = np.asarray(predicted, dtype=float)
        metadata.update(
            {
                "player_prior_method": "forward_aging_returner_plus_exposure_gated_cold_start",
                "aging_enabled": True,
                "aging_selected_regularization": experiment.selected_regularization,
                "aging_training_transition_count": int(len(training)),
                "aging_training_target_seasons": list(experiment.training_target_seasons),
                "aging_selection_holdout_target_season": experiment.holdout_target_season,
                "aging_returning_player_count": int(len(age_returning)),
            }
        )
        return _combine_priors(age_returning, cold), metadata
    except ValueError as error:
        metadata["aging_reason"] = str(error)
        return _combine_priors(returning, cold), metadata


def _aging_transition_history(
    panel: pd.DataFrame,
    results: Sequence[ForwardLaggedRapmSeason],
    exposure_history: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Build labeled consecutive-season transitions from completed model state."""

    if len(results) != len(exposure_history):
        raise ValueError("Completed RAPM results and exposure history must align")
    exposures = {
        result.season: _exposure_frame(exposure)
        for result, exposure in zip(results, exposure_history, strict=True)
    }
    rows: list[pd.DataFrame] = []
    for prior, target in zip(results, results[1:], strict=False):
        if int(target.season[:4]) != int(prior.season[:4]) + 1:
            continue
        target_bios = _season_bios(panel, target.season)
        target_estimates = target.player_estimates.loc[:, ["player_id", "rapm"]].rename(
            columns={"rapm": "target_rapm"}
        )
        prior_estimates = prior.player_estimates.loc[:, ["player_id", "rapm"]].rename(
            columns={"rapm": "prior_rapm"}
        )
        transition = (
            target_bios.merge(target_estimates, on="player_id", how="inner", validate="one_to_one")
            .merge(exposures[target.season], on="player_id", how="inner", validate="one_to_one")
            .merge(prior_estimates, on="player_id", how="inner", validate="one_to_one")
            .merge(
                exposures[prior.season].rename(
                    columns={"on_court_possessions": "prior_rapm_possessions"}
                ),
                on="player_id",
                how="inner",
                validate="one_to_one",
            )
        )
        transition = transition.rename(
            columns={
                "age": "target_age",
                "nba_experience_years": "target_nba_experience_years",
                "on_court_possessions": "target_rapm_possessions",
                "rapm_seconds": "target_rapm_seconds",
                "rapm_exposure_eligible": "target_rapm_exposure_eligible",
            }
        )
        transition.insert(0, "target_season", target.season)
        transition.insert(1, "prior_season", prior.season)
        transition["has_prior_season"] = True
        rows.append(transition)
    if not rows:
        raise ValueError("Aging prior requires at least one completed transition")
    return pd.concat(rows, ignore_index=True).sort_values(
        ["target_season", "player_id"], kind="stable"
    ).reset_index(drop=True)


def _target_returning_features(
    panel: pd.DataFrame,
    *,
    season: str,
    returning: pd.DataFrame,
    latest_exposure: pd.DataFrame,
) -> pd.DataFrame:
    previous_exposure = _exposure_frame(latest_exposure).rename(
        columns={"on_court_possessions": "prior_rapm_possessions"}
    )
    output = (
        _season_bios(panel, season)
        .merge(
            returning.rename(columns={PRIOR_MEAN_COLUMN: "prior_rapm"}),
            on="player_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(previous_exposure, on="player_id", how="inner", validate="one_to_one")
    ).rename(
        columns={
            "age": "target_age",
            "nba_experience_years": "target_nba_experience_years",
        }
    )
    output["target_season"] = season
    output["has_prior_season"] = True
    return output


def _season_bios(panel: pd.DataFrame, season: str) -> pd.DataFrame:
    missing = set(_BIO_COLUMNS) - set(panel)
    if missing:
        raise ValueError(f"Player-season panel missing aging bios: {sorted(missing)}")
    output = panel.loc[panel["season"].eq(season), _BIO_COLUMNS].copy()
    if output.empty:
        raise ValueError(f"Player-season panel has no aging bios for {season}")
    if output["player_id"].duplicated().any():
        raise ValueError("Player-season aging bios must be unique by season and player")
    return output


def _exposure_frame(exposure: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "on_court_possessions"}
    missing = required - set(exposure)
    if missing:
        raise ValueError(f"Player exposure history missing columns: {sorted(missing)}")
    output = exposure.loc[:, ["player_id", "on_court_possessions"]].copy()
    output["player_id"] = output["player_id"].astype(int)
    output["on_court_possessions"] = pd.to_numeric(
        output["on_court_possessions"], errors="raise"
    )
    if output["player_id"].duplicated().any() or not output["on_court_possessions"].gt(0).all():
        raise ValueError("Player exposure history is invalid")
    return output
