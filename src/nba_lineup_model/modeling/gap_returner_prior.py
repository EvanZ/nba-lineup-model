"""Forward-safe priors for established players returning after an absence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.aging import (
    VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
    prepare_aging_prior_features,
)
from nba_lineup_model.modeling.forward_aging_player_prior import (
    _season_bios,
    build_aging_exposure_gated_priors,
    center_player_priors,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _combine_priors
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


GAP_RETURNER_METHOD = "last_observed_annual_aging_bridge"


def build_centered_value_conditioned_aging_gap_returner_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Extend the v1.1 prior with projected states for gap returners.

    The existing v1.1 branch remains unchanged for immediate returners and
    rookies. A player who last appeared before the immediately prior season is
    advanced one age-model transition per missing season, then once more into
    the target season. No unobserved season receives a RAPM update.
    """

    raw_priors, metadata = build_aging_exposure_gated_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,
        exposure_history=exposure_history,
        replacement_tokens=replacement_tokens,
        feature_columns=VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
        model_name="forward_value_conditioned_aging_gap_returner_ridge",
    )
    aging_model = metadata.get("_aging_model")
    gap_priors, projected_states = _gap_returner_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,
        exposure_history=exposure_history,
        existing_prior_ids=set(raw_priors["player_id"].astype(int)),
        aging_model=aging_model,
    )
    combined = _combine_priors(raw_priors, gap_priors)
    centered, centering_metadata = center_player_priors(
        combined,
        previous_exposure=exposure_history[-1] if exposure_history else None,
    )
    metadata.update(
        {
            **centering_metadata,
            "player_prior_method": (
                "value_conditioned_aging_returner_plus_gap_returner_plus_exposure_gated_cold_start"
            ),
            "gap_returner_method": GAP_RETURNER_METHOD,
            "gap_returner_count": int(len(gap_priors)),
            "gap_returner_projected_state_count": int(len(projected_states)),
            "_gap_returner_states": projected_states,
        }
    )
    return centered, metadata


def _gap_returner_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: Sequence[ForwardLaggedRapmSeason],
    exposure_history: Sequence[pd.DataFrame],
    existing_prior_ids: set[int],
    aging_model: object | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create only non-immediate returner priors and their projected states."""

    empty_priors = pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN])
    state_columns = (
        "target_season",
        "player_id",
        "last_observed_season",
        "projected_season",
        "gap_seasons",
        "projection_step",
        "is_return_season",
        "prior_rapm",
        "projected_prior_rapm",
        "prior_rapm_possessions",
    )
    empty_states = pd.DataFrame(columns=state_columns)
    if aging_model is None or not completed_results:
        return empty_priors, empty_states
    if len(completed_results) != len(exposure_history):
        raise ValueError("Gap returner state requires aligned result and exposure histories")

    target_bios = _season_bios(panel, season).set_index("player_id", verify_integrity=True)
    target_year = int(season[:4])
    observed = _last_observed_states(completed_results, exposure_history)
    rows: list[dict[str, object]] = []
    projected: list[dict[str, object]] = []

    for player_id, bio in target_bios.iterrows():
        player_id = int(player_id)
        if player_id in existing_prior_ids or player_id not in observed:
            continue
        last = observed[player_id]
        last_year = int(str(last["season"])[:4])
        gap_seasons = target_year - last_year - 1
        if gap_seasons < 1:
            continue
        if pd.isna(bio["age"]) or pd.isna(bio["nba_experience_years"]):
            continue

        current_rating = float(last["rapm"])
        prior_possessions = float(last["on_court_possessions"])
        for step_year in range(last_year + 1, target_year + 1):
            step_season = _season_label(step_year)
            elapsed = target_year - step_year
            step = step_year - last_year
            step_features = pd.DataFrame(
                {
                    "target_season": [step_season],
                    "player_id": [player_id],
                    "target_age": [float(bio["age"]) - elapsed],
                    "target_nba_experience_years": [
                        float(bio["nba_experience_years"]) - elapsed
                    ],
                    "is_rookie": [False],
                    "has_prior_season": [True],
                    "prior_rapm": [current_rating],
                    "prior_rapm_possessions": [prior_possessions],
                    "draft_year": [bio["draft_year"]],
                    "draft_number": [bio["draft_number"]],
                    "height_inches": [bio["height_inches"]],
                    "weight_pounds": [bio["weight_pounds"]],
                    "is_undrafted": [bio["is_undrafted"]],
                }
            )
            transformed = prepare_aging_prior_features(step_features)
            next_rating = float(
                aging_model.predict(
                    transformed.loc[:, VALUE_CONDITIONED_AGING_FEATURE_COLUMNS]
                )[0]
            )
            projected.append(
                {
                    "target_season": season,
                    "player_id": player_id,
                    "last_observed_season": last["season"],
                    "projected_season": step_season,
                    "gap_seasons": gap_seasons,
                    "projection_step": step,
                    "is_return_season": step_year == target_year,
                    "prior_rapm": current_rating,
                    "projected_prior_rapm": next_rating,
                    "prior_rapm_possessions": prior_possessions,
                }
            )
            current_rating = next_rating
        rows.append({"player_id": player_id, PRIOR_MEAN_COLUMN: current_rating})

    priors = pd.DataFrame(rows, columns=["player_id", PRIOR_MEAN_COLUMN])
    if not priors.empty:
        priors["player_id"] = priors["player_id"].astype(int)
    states = pd.DataFrame(projected, columns=state_columns)
    return priors, states


def _last_observed_states(
    results: Sequence[ForwardLaggedRapmSeason],
    exposure_history: Sequence[pd.DataFrame],
) -> dict[int, dict[str, object]]:
    """Return the latest completed RAPM/exposure state for each player."""

    states: dict[int, dict[str, object]] = {}
    for result, exposure in zip(results, exposure_history, strict=True):
        rating = result.player_estimates.loc[:, ["player_id", "rapm"]].copy()
        prior_exposure = exposure.loc[:, ["player_id", "on_court_possessions"]].copy()
        merged = rating.merge(prior_exposure, on="player_id", how="inner", validate="one_to_one")
        for row in merged.itertuples(index=False):
            states[int(row.player_id)] = {
                "season": result.season,
                "rapm": float(row.rapm),
                "on_court_possessions": float(row.on_court_possessions),
            }
    return states


def _season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"
