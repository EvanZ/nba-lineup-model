"""Lineup-composition features for the contextual residual prior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import PROFILE_COLUMNS, PROFILE_RATE_COLUMNS

if TYPE_CHECKING:
    from nba_lineup_model.modeling.rebound_opportunity import ReboundOpportunityModel
    from nba_lineup_model.modeling.usage_allocation import UsageAllocationModel

SHOOTING_COLUMN = "three_pm_per_100"
PASSING_COLUMN = "assists_per_100"
USAGE_COLUMN = "usage_per_100"
OFFENSIVE_REBOUND_COLUMN = "offensive_rebounds_per_100"
DEFENSIVE_REBOUND_COLUMN = "defensive_rebounds_per_100"
CREDIBLE_SHOOTER_THREE_PM_PER_100 = 2.0
SHOOTER_CAP_THREE_PM_PER_100 = 2.0

CONTEXT_FEATURE_SET_V1 = "v1"
CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING = "v2_depth_aware_shooting"
CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY = "v2_1_empirical_rebound_capacity"
CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION = "v2_2_usage_allocation"


def available_context_feature_sets() -> tuple[str, ...]:
    """Return supported, versioned contextual feature contracts."""

    return (
        CONTEXT_FEATURE_SET_V1,
        CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING,
        CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
        CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
    )


def lineup_context_features(
    home_lineups: Sequence[Sequence[int]],
    away_lineups: Sequence[Sequence[int]],
    profiles: pd.DataFrame,
    *,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
    rebound_model: ReboundOpportunityModel | None = None,
    usage_model: UsageAllocationModel | None = None,
) -> pd.DataFrame:
    """Encode fixed five-player home and away lineup compositions.

    Values are home-minus-away to align with a home-net-rating residual. The
    profile map is deliberately complete; missing player profiles are a data
    contract error rather than an implicit zero-value player.
    """

    if len(home_lineups) != len(away_lineups):
        raise ValueError("Contextual home and away lineup sequences must align")
    required = {"player_id", *_required_profile_columns(feature_set), "profile_imputed", "profile_replacement_weight"}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Contextual player profiles missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Contextual player profiles contain duplicate player IDs")
    home = lineup_side_context_features(
        home_lineups, profiles, feature_set=feature_set, rebound_model=rebound_model, usage_model=usage_model
    )
    away = lineup_side_context_features(
        away_lineups, profiles, feature_set=feature_set, rebound_model=rebound_model, usage_model=usage_model
    )
    return pd.DataFrame(
        {
            f"home_minus_away_{column}": home[column].to_numpy(dtype=float)
            - away[column].to_numpy(dtype=float)
            for column in side_context_feature_columns(feature_set)
        },
        columns=contextual_feature_columns(feature_set),
    )


def contextual_feature_columns(
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> tuple[str, ...]:
    """Return the fixed feature order used by the contextual residual model."""

    return tuple(
        f"home_minus_away_{column}" for column in side_context_feature_columns(feature_set)
    )


def lineup_side_context_features(
    lineups: Sequence[Sequence[int]],
    profiles: pd.DataFrame,
    *,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
    rebound_model: ReboundOpportunityModel | None = None,
    usage_model: UsageAllocationModel | None = None,
) -> pd.DataFrame:
    """Encode five-player units under one versioned contextual feature contract."""

    required = {"player_id", *_required_profile_columns(feature_set), "profile_imputed", "profile_replacement_weight"}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Contextual player profiles missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Contextual player profiles contain duplicate player IDs")
    if feature_set in {CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY, CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION} and rebound_model is None:
        raise ValueError("Empirical rebound-capacity features require a rebound opportunity model")
    if feature_set == CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION and usage_model is None:
        raise ValueError("Usage-allocation features require a usage allocation model")
    values = profiles.set_index("player_id")
    lineup_values = [_lineup_values(lineup, values) for lineup in lineups]
    rebound_rates = _rebound_rates(lineup_values, rebound_model)
    usage_features = _usage_features(lineup_values, usage_model)
    rows = [
        _side_feature_row(lineup, feature_set, rebound_rate=rate, usage_feature=usage)
        for lineup, rate, usage in zip(lineup_values, rebound_rates, usage_features, strict=True)
    ]
    return pd.DataFrame(rows, columns=side_context_feature_columns(feature_set))


def side_context_feature_columns(
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> tuple[str, ...]:
    """Return the per-lineup form of every published contextual feature."""

    if feature_set == CONTEXT_FEATURE_SET_V1:
        return (
            *PROFILE_RATE_COLUMNS,
            "bottom_two_three_pm",
            "credible_shooter_count",
            "top_two_assists",
            "usage_concentration",
            "sqrt_offensive_rebounds",
            "sqrt_defensive_rebounds",
            "imputed_count",
            "replacement_weight",
            "shooting_usage_interaction",
            "shooter_passing_interaction",
            "rebounding_usage_interaction",
        )
    if feature_set == CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING:
        return (
            *(column for column in PROFILE_RATE_COLUMNS if column not in {
                "three_pa_per_100",
                "three_pm_per_100",
            }),
            "bottom_three_three_pm",
            "credible_shooter_count",
            "capped_three_pm",
            "shooting_concentration",
            "top_two_assists",
            "usage_concentration",
            "sqrt_offensive_rebounds",
            "sqrt_defensive_rebounds",
            "imputed_count",
            "replacement_weight",
            "depth_usage_interaction",
            "shooter_passing_interaction",
            "rebounding_usage_interaction",
        )
    if feature_set == CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY:
        return (
            *(column for column in PROFILE_RATE_COLUMNS if column not in {
                "three_pa_per_100",
                "three_pm_per_100",
                OFFENSIVE_REBOUND_COLUMN,
                DEFENSIVE_REBOUND_COLUMN,
            }),
            "bottom_three_three_pm",
            "credible_shooter_count",
            "capped_three_pm",
            "shooting_concentration",
            "top_two_assists",
            "usage_concentration",
            "expected_offensive_rebound_pct",
            "expected_defensive_rebound_pct",
            "imputed_count",
            "replacement_weight",
            "depth_usage_interaction",
            "shooter_passing_interaction",
        )
    if feature_set == CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION:
        return (
            *(column for column in PROFILE_RATE_COLUMNS if column not in {
                "three_pa_per_100", "three_pm_per_100", OFFENSIVE_REBOUND_COLUMN,
                DEFENSIVE_REBOUND_COLUMN, "usage_per_100", "turnovers_per_100",
            }),
            "bottom_three_three_pm", "credible_shooter_count", "capped_three_pm",
            "shooting_concentration", "top_two_assists",
            "expected_offensive_rebound_pct", "expected_defensive_rebound_pct",
            "imputed_count", "replacement_weight", "shooter_passing_interaction",
            "excess_usage_demand", "allocation_entropy", "role_reallocation_js",
            "allocation_weighted_turnover_burden",
        )
    raise ValueError(f"Unknown contextual feature set: {feature_set}")


def _lineup_values(lineup: Sequence[int], values: pd.DataFrame) -> pd.DataFrame:
    player_ids = [int(player_id) for player_id in lineup]
    if len(player_ids) != 5 or len(set(player_ids)) != 5:
        raise ValueError("Contextual lineup features require five unique players")
    missing = sorted(set(player_ids) - set(values.index.astype(int)))
    if missing:
        raise ValueError(f"Contextual player profiles missing lineup players: {missing}")
    return values.loc[player_ids]


def _required_profile_columns(feature_set: str) -> tuple[str, ...]:
    if feature_set in {CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY, CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION}:
        return PROFILE_COLUMNS
    return PROFILE_RATE_COLUMNS


def _side_feature_row(
    lineup: pd.DataFrame,
    feature_set: str,
    *,
    rebound_rate: tuple[float, float] | None = None,
    usage_feature: dict[str, float] | None = None,
) -> dict[str, float]:
    if feature_set == CONTEXT_FEATURE_SET_V1:
        result = {column: float(lineup[column].sum()) for column in PROFILE_RATE_COLUMNS}
        result.update(_summary_v1(lineup))
    elif feature_set == CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING:
        result = {
            column: float(lineup[column].sum())
            for column in PROFILE_RATE_COLUMNS
            if column not in {"three_pa_per_100", "three_pm_per_100"}
        }
        result.update(_summary_v2(lineup))
    elif feature_set == CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY:
        result = {
            column: float(lineup[column].sum())
            for column in PROFILE_RATE_COLUMNS
            if column not in {
                "three_pa_per_100",
                "three_pm_per_100",
                OFFENSIVE_REBOUND_COLUMN,
                DEFENSIVE_REBOUND_COLUMN,
            }
        }
        if rebound_rate is None:
            raise ValueError("Empirical rebound capacity is required for v2.1 features")
        result.update(_summary_v21(lineup, rebound_rate=rebound_rate))
    elif feature_set == CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION:
        result = {
            column: float(lineup[column].sum())
            for column in PROFILE_RATE_COLUMNS
            if column not in {"three_pa_per_100", "three_pm_per_100", OFFENSIVE_REBOUND_COLUMN, DEFENSIVE_REBOUND_COLUMN, "usage_per_100", "turnovers_per_100"}
        }
        if rebound_rate is None or usage_feature is None:
            raise ValueError("Rebound and usage allocation features are required for v2.2")
        result.update(_summary_v22(lineup, rebound_rate=rebound_rate, usage_feature=usage_feature))
    else:
        raise ValueError(f"Unknown contextual feature set: {feature_set}")
    return result


def _summary_v1(lineup: pd.DataFrame) -> dict[str, float]:
    shooting = np.sort(lineup[SHOOTING_COLUMN].to_numpy(dtype=float))
    assists = np.sort(lineup[PASSING_COLUMN].to_numpy(dtype=float))
    usage = lineup[USAGE_COLUMN].to_numpy(dtype=float)
    total_usage = float(usage.sum())
    usage_concentration = float(np.sort(usage)[-2:].sum() / total_usage) if total_usage else 0.0
    bottom_two = float(shooting[:2].sum())
    credible_shooters = float((shooting >= CREDIBLE_SHOOTER_THREE_PM_PER_100).sum())
    top_two_assists = float(assists[-2:].sum())
    sqrt_offensive_rebounds = float(np.sqrt(lineup[OFFENSIVE_REBOUND_COLUMN].sum()))
    sqrt_defensive_rebounds = float(np.sqrt(lineup[DEFENSIVE_REBOUND_COLUMN].sum()))
    return {
        "bottom_two_three_pm": bottom_two,
        "credible_shooter_count": credible_shooters,
        "top_two_assists": top_two_assists,
        "usage_concentration": usage_concentration,
        "sqrt_offensive_rebounds": sqrt_offensive_rebounds,
        "sqrt_defensive_rebounds": sqrt_defensive_rebounds,
        "imputed_count": float(lineup["profile_imputed"].sum()),
        "replacement_weight": float(lineup["profile_replacement_weight"].sum()),
        "shooting_usage_interaction": bottom_two * usage_concentration,
        "shooter_passing_interaction": credible_shooters * top_two_assists,
        "rebounding_usage_interaction": sqrt_offensive_rebounds * usage_concentration,
    }


def _summary_v2(lineup: pd.DataFrame) -> dict[str, float]:
    """Summarize shooting as shared lineup capacity, not raw star volume."""

    shooting = np.sort(lineup[SHOOTING_COLUMN].to_numpy(dtype=float))
    assists = np.sort(lineup[PASSING_COLUMN].to_numpy(dtype=float))
    usage = lineup[USAGE_COLUMN].to_numpy(dtype=float)
    total_usage = float(usage.sum())
    usage_concentration = float(np.sort(usage)[-2:].sum() / total_usage) if total_usage else 0.0
    total_shooting = float(shooting.sum())
    bottom_three = float(shooting[:3].sum())
    credible_shooters = float((shooting >= CREDIBLE_SHOOTER_THREE_PM_PER_100).sum())
    capped_shooting = float(np.minimum(shooting, SHOOTER_CAP_THREE_PM_PER_100).sum())
    shooting_concentration = (
        float(np.square(shooting / total_shooting).sum()) if total_shooting > 0.0 else 0.0
    )
    top_two_assists = float(assists[-2:].sum())
    sqrt_offensive_rebounds = float(np.sqrt(lineup[OFFENSIVE_REBOUND_COLUMN].sum()))
    sqrt_defensive_rebounds = float(np.sqrt(lineup[DEFENSIVE_REBOUND_COLUMN].sum()))
    return {
        "bottom_three_three_pm": bottom_three,
        "credible_shooter_count": credible_shooters,
        "capped_three_pm": capped_shooting,
        "shooting_concentration": shooting_concentration,
        "top_two_assists": top_two_assists,
        "usage_concentration": usage_concentration,
        "sqrt_offensive_rebounds": sqrt_offensive_rebounds,
        "sqrt_defensive_rebounds": sqrt_defensive_rebounds,
        "imputed_count": float(lineup["profile_imputed"].sum()),
        "replacement_weight": float(lineup["profile_replacement_weight"].sum()),
        "depth_usage_interaction": bottom_three * usage_concentration,
        "shooter_passing_interaction": credible_shooters * top_two_assists,
        "rebounding_usage_interaction": sqrt_offensive_rebounds * usage_concentration,
    }


def _summary_v21(
    lineup: pd.DataFrame,
    *,
    rebound_rate: tuple[float, float],
) -> dict[str, float]:
    """Retain v2 shooting with empirically calibrated rebound realization."""

    shooting = np.sort(lineup[SHOOTING_COLUMN].to_numpy(dtype=float))
    assists = np.sort(lineup[PASSING_COLUMN].to_numpy(dtype=float))
    usage = lineup[USAGE_COLUMN].to_numpy(dtype=float)
    total_usage = float(usage.sum())
    usage_concentration = float(np.sort(usage)[-2:].sum() / total_usage) if total_usage else 0.0
    total_shooting = float(shooting.sum())
    bottom_three = float(shooting[:3].sum())
    credible_shooters = float((shooting >= CREDIBLE_SHOOTER_THREE_PM_PER_100).sum())
    capped_shooting = float(np.minimum(shooting, SHOOTER_CAP_THREE_PM_PER_100).sum())
    shooting_concentration = (
        float(np.square(shooting / total_shooting).sum()) if total_shooting > 0.0 else 0.0
    )
    offensive_rate, defensive_rate = rebound_rate
    return {
        "bottom_three_three_pm": bottom_three,
        "credible_shooter_count": credible_shooters,
        "capped_three_pm": capped_shooting,
        "shooting_concentration": shooting_concentration,
        "top_two_assists": float(assists[-2:].sum()),
        "usage_concentration": usage_concentration,
        "expected_offensive_rebound_pct": float(offensive_rate),
        "expected_defensive_rebound_pct": float(defensive_rate),
        "imputed_count": float(lineup["profile_imputed"].sum()),
        "replacement_weight": float(lineup["profile_replacement_weight"].sum()),
        "depth_usage_interaction": bottom_three * usage_concentration,
        "shooter_passing_interaction": credible_shooters * float(assists[-2:].sum()),
    }


def _summary_v22(
    lineup: pd.DataFrame,
    *,
    rebound_rate: tuple[float, float],
    usage_feature: dict[str, float],
) -> dict[str, float]:
    result = _summary_v21(lineup, rebound_rate=rebound_rate)
    result.pop("usage_concentration")
    result.pop("depth_usage_interaction")
    result.update(usage_feature)
    return result


def _rebound_rates(
    lineups: Sequence[pd.DataFrame],
    rebound_model: ReboundOpportunityModel | None,
) -> list[tuple[float, float] | None]:
    """Return portable expected ORB%/DRB% for v2.1 without touching raw sums."""

    if rebound_model is None:
        return [None] * len(lineups)
    offensive_claims = np.array(
        [lineup["offensive_rebound_pct"].sum() for lineup in lineups], dtype=float
    )
    defensive_claims = np.array(
        [lineup["defensive_rebound_pct"].sum() for lineup in lineups], dtype=float
    )
    offensive, defensive = rebound_model.predict_unit_rebound_rates(
        offensive_claims, defensive_claims
    )
    return list(zip(offensive.tolist(), defensive.tolist(), strict=True))


def _usage_features(
    lineups: Sequence[pd.DataFrame],
    usage_model: UsageAllocationModel | None,
) -> list[dict[str, float] | None]:
    if usage_model is None:
        return [None] * len(lineups)
    claims = np.array([lineup[USAGE_COLUMN].to_numpy(dtype=float) for lineup in lineups])
    turnovers = np.array([lineup["turnovers_per_100"].to_numpy(dtype=float) for lineup in lineups])
    frame = usage_model.lineup_features(claims, turnovers)
    return frame.to_dict(orient="records")
