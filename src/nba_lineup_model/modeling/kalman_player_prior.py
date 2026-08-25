"""Forward-only Kalman filtering for established-player RAPM priors.

The recursive RAPM fit supplies one noisy annual observation per player.  This
module filters that observation against the prior that was actually used in
that season's RAPM fit, then passes the posterior mean through the existing
aging transition for the following season.  No target-season outcome enters a
target-season prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from nba_lineup_model.modeling.aging import (
    VALUE_CONDITIONED_AGING_FEATURE_COLUMNS,
    prepare_aging_prior_features,
)
from nba_lineup_model.modeling.forward_aging_player_prior import (
    _season_bios,
    center_player_priors,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    _season_label,
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


@dataclass(frozen=True)
class PlayerKalmanConfig:
    """Fixed, interpretable uncertainty assumptions for annual RAPM states."""

    initial_variance: float = 9.0
    process_variance_per_season: float = 1.0
    observation_variance_possession_scale: float = 4000.0

    def validate(self) -> None:
        if self.initial_variance <= 0:
            raise ValueError("Initial player-state variance must be positive")
        if self.process_variance_per_season < 0:
            raise ValueError("Player-state process variance must be non-negative")
        if self.observation_variance_possession_scale <= 0:
            raise ValueError("Observation variance possession scale must be positive")


DEFAULT_PLAYER_KALMAN_CONFIG = PlayerKalmanConfig()


def build_centered_value_conditioned_aging_gap_returner_kalman_priors(
    *,
    season: str,
    panel: pd.DataFrame,
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    replacement_tokens: list[dict[str, object]],
    config: PlayerKalmanConfig = DEFAULT_PLAYER_KALMAN_CONFIG,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the existing v1.2 prior after filtering observed player states.

    The existing aging/cold-start builder remains the contract for cold starts,
    model selection, and center point.  Only established-player input ratings
    are replaced by their forward Kalman posterior before the aging transition.
    """

    config.validate()
    base_priors, metadata = build_centered_value_conditioned_aging_gap_returner_priors(
        season=season,
        panel=panel,
        completed_results=completed_results,
        exposure_history=exposure_history,
        replacement_tokens=replacement_tokens,
    )
    states = filter_completed_player_states(
        completed_results,
        exposure_history,
        config=config,
    )
    aging_model = metadata.get("_aging_model")
    if states.empty or aging_model is None or not completed_results:
        metadata.update(_kalman_metadata(config, states, 0, 0))
        metadata["_kalman_player_states"] = states
        return base_priors, metadata

    center_offset = float(metadata.get("player_prior_center_offset", 0.0))
    uncentered = base_priors.copy()
    uncentered[PRIOR_MEAN_COLUMN] = (
        uncentered[PRIOR_MEAN_COLUMN].to_numpy(dtype=float) + center_offset
    )
    latest = _latest_states(states)
    source_ids = set(
        completed_results[-1].player_estimates["player_id"].astype(int).tolist()
    )

    immediate = latest.loc[
        latest["player_id"].isin(source_ids),
        ["player_id", "posterior_mean", "posterior_variance", "on_court_possessions"],
    ]
    direct_updates = _transition_immediate_returners(
        season=season,
        panel=panel,
        states=immediate,
        aging_model=aging_model,
    )
    gap_updates = _transition_gap_returners(
        season=season,
        panel=panel,
        states=latest.loc[~latest["player_id"].isin(source_ids)].copy(),
        aging_model=aging_model,
    )
    updates = pd.concat([direct_updates, gap_updates], ignore_index=True)
    if not updates.empty:
        known = set(uncentered["player_id"].astype(int))
        updates = updates.loc[updates["player_id"].isin(known)].copy()
        replacement = updates.set_index("player_id")["kalman_prior_rapm"]
        uncentered[PRIOR_MEAN_COLUMN] = uncentered["player_id"].map(replacement).fillna(
            uncentered[PRIOR_MEAN_COLUMN]
        )
    centered, centering_metadata = center_player_priors(
        uncentered,
        previous_exposure=exposure_history[-1] if exposure_history else None,
    )
    metadata.update(centering_metadata)
    metadata.update(_kalman_metadata(config, states, len(direct_updates), len(gap_updates)))
    metadata["_kalman_player_states"] = states.assign(target_season=season)
    return centered, metadata


def filter_completed_player_states(
    completed_results: list[ForwardLaggedRapmSeason],
    exposure_history: list[pd.DataFrame],
    *,
    config: PlayerKalmanConfig = DEFAULT_PLAYER_KALMAN_CONFIG,
) -> pd.DataFrame:
    """Return each completed annual posterior using only contemporaneous inputs."""

    config.validate()
    if len(completed_results) != len(exposure_history):
        raise ValueError("Completed results and exposure history must align")
    states: dict[int, dict[str, float | str]] = {}
    rows: list[dict[str, float | int | str]] = []
    for result, exposure in zip(completed_results, exposure_history, strict=True):
        observed = result.player_estimates.loc[
            :, ["player_id", "rapm", "prior_rapm"]
        ].merge(
            exposure.loc[:, ["player_id", "on_court_possessions"]],
            on="player_id",
            how="inner",
            validate="one_to_one",
        )
        year = int(result.season[:4])
        for row in observed.itertuples(index=False):
            player_id = int(row.player_id)
            previous = states.get(player_id)
            if previous is None:
                prior_variance = config.initial_variance
                gap_seasons = 1
            else:
                gap_seasons = max(year - int(str(previous["season"])[:4]), 1)
                prior_variance = float(previous["posterior_variance"]) + (
                    gap_seasons * config.process_variance_per_season
                )
            possessions = max(float(row.on_court_possessions), 1.0)
            observation_variance = config.observation_variance_possession_scale / possessions
            kalman_gain = prior_variance / (prior_variance + observation_variance)
            prior_mean = float(row.prior_rapm)
            observation = float(row.rapm)
            posterior_mean = prior_mean + kalman_gain * (observation - prior_mean)
            posterior_variance = (1.0 - kalman_gain) * prior_variance
            states[player_id] = {
                "season": result.season,
                "posterior_variance": posterior_variance,
            }
            rows.append(
                {
                    "season": result.season,
                    "player_id": player_id,
                    "prior_mean": prior_mean,
                    "prior_variance": prior_variance,
                    "observation_rapm": observation,
                    "observation_variance": observation_variance,
                    "kalman_gain": kalman_gain,
                    "posterior_mean": posterior_mean,
                    "posterior_variance": posterior_variance,
                    "on_court_possessions": possessions,
                    "gap_seasons": gap_seasons,
                }
            )
    return pd.DataFrame(rows)


def _latest_states(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return states.copy()
    return (
        states.sort_values(["player_id", "season"], kind="stable")
        .groupby("player_id", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _transition_immediate_returners(
    *,
    season: str,
    panel: pd.DataFrame,
    states: pd.DataFrame,
    aging_model: Any,
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame(columns=["player_id", "kalman_prior_rapm"])
    bios = _season_bios(panel, season)
    target = bios.merge(states, on="player_id", how="inner", validate="one_to_one").rename(
        columns={
            "age": "target_age",
            "nba_experience_years": "target_nba_experience_years",
            "posterior_mean": "prior_rapm",
            "on_court_possessions": "prior_rapm_possessions",
        }
    )
    target["target_season"] = season
    target["has_prior_season"] = True
    transformed = prepare_aging_prior_features(target)
    output = target.loc[:, ["player_id"]].copy()
    output["kalman_prior_rapm"] = aging_model.predict(
        transformed.loc[:, VALUE_CONDITIONED_AGING_FEATURE_COLUMNS]
    )
    return output


def _transition_gap_returners(
    *,
    season: str,
    panel: pd.DataFrame,
    states: pd.DataFrame,
    aging_model: Any,
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame(columns=["player_id", "kalman_prior_rapm"])
    bios = _season_bios(panel, season).set_index("player_id", verify_integrity=True)
    target_year = int(season[:4])
    rows: list[dict[str, float | int]] = []
    for state in states.itertuples(index=False):
        player_id = int(state.player_id)
        if player_id not in bios.index:
            continue
        last_year = int(str(state.season)[:4])
        if target_year - last_year <= 1:
            continue
        bio = bios.loc[player_id]
        if pd.isna(bio["age"]) or pd.isna(bio["nba_experience_years"]):
            continue
        rating = float(state.posterior_mean)
        for step_year in range(last_year + 1, target_year + 1):
            elapsed = target_year - step_year
            features = pd.DataFrame(
                {
                    "target_season": [_season_label(step_year)],
                    "player_id": [player_id],
                    "target_age": [float(bio["age"]) - elapsed],
                    "target_nba_experience_years": [
                        float(bio["nba_experience_years"]) - elapsed
                    ],
                    "is_rookie": [False],
                    "has_prior_season": [True],
                    "prior_rapm": [rating],
                    "prior_rapm_possessions": [float(state.on_court_possessions)],
                    "draft_year": [bio["draft_year"]],
                    "draft_number": [bio["draft_number"]],
                    "height_inches": [bio["height_inches"]],
                    "weight_pounds": [bio["weight_pounds"]],
                    "is_undrafted": [bio["is_undrafted"]],
                }
            )
            transformed = prepare_aging_prior_features(features)
            rating = float(
                aging_model.predict(
                    transformed.loc[:, VALUE_CONDITIONED_AGING_FEATURE_COLUMNS]
                )[0]
            )
        rows.append({"player_id": player_id, "kalman_prior_rapm": rating})
    return pd.DataFrame(rows, columns=["player_id", "kalman_prior_rapm"])


def _kalman_metadata(
    config: PlayerKalmanConfig,
    states: pd.DataFrame,
    direct_count: int,
    gap_count: int,
) -> dict[str, object]:
    return {
        "player_prior_method": (
            "forward_value_conditioned_aging_gap_returner_plus_kalman_player_state_"
            "plus_exposure_gated_cold_start"
        ),
        "kalman_player_state_enabled": True,
        "kalman_initial_variance": config.initial_variance,
        "kalman_process_variance_per_season": config.process_variance_per_season,
        "kalman_observation_variance_possession_scale": (
            config.observation_variance_possession_scale
        ),
        "kalman_observation_count": int(len(states)),
        "kalman_direct_returner_count": int(direct_count),
        "kalman_gap_returner_count": int(gap_count),
        "kalman_contract": (
            "Annual player RAPM is a noisy observation; the posterior mean is advanced "
            "through the forward value-conditioned aging transition."
        ),
    }
