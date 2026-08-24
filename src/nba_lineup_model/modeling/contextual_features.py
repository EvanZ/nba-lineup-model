"""Lineup-composition features for the contextual residual prior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS

if TYPE_CHECKING:
    from nba_lineup_model.modeling.rebound_opportunity import ReboundOpportunityModel
    from nba_lineup_model.modeling.usage_allocation import UsageAllocationModel

SHOOTING_COLUMN = "three_pm_per_100"
PASSING_COLUMN = "assists_per_100"
USAGE_COLUMN = "usage_per_100"
STANDARD_USAGE_COLUMN = "usage_pct"
OFFENSIVE_REBOUND_COLUMN = "offensive_rebounds_per_100"
DEFENSIVE_REBOUND_COLUMN = "defensive_rebounds_per_100"
CREDIBLE_SHOOTER_THREE_PM_PER_100 = 2.0
SHOOTER_CAP_THREE_PM_PER_100 = 2.0

CONTEXT_FEATURE_SET_V1 = "v1"
CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING = "v2_depth_aware_shooting"
CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY = "v2_1_empirical_rebound_capacity"
CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION = "v2_2_usage_allocation"
CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO = "v2_3_shot_portfolio"
CONTEXT_FEATURE_SET_X1_ORB_CLAIM_TOTAL = "x1_orb_claim_total"
CONTEXT_FEATURE_SET_X2_ORB_PER_100_TOTAL = "x2_orb_per_100_total"
CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT = "x3_v1_orb_claim_replacement"
CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY = "x3_without_uncertainty"
CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE = "nail_v12_1_pruned_nonadditive"
CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING = "nail_critical_spacing"
CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE = "nail_critical_spacing_quintile"
CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUARTILE_STANDARD_USAGE = (
    "nail_critical_spacing_quartile_standard_usage"
)
CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE = "nail_v12_1_1_standard_usage"
CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE = "nail_v12_2_defensive_rebound_profile"
CONTEXT_FEATURE_SET_NAIL_V123_FREE_THROW_PROFILE = "nail_v12_3_free_throw_profile"
CONTEXT_FEATURE_SET_NAIL_V124_FREE_THROW_REPLACEMENT = (
    "nail_v12_4_free_throw_replacement"
)
CONTEXT_FEATURE_SET_NAIL_V13 = "nail_v13_additive_profile"
CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE = "nail_v13_1_pruned_additive_profile"
CONTEXT_FEATURE_SET_NAIL_ADDITIVE_ONLY = "nail_additive_only"
CONTEXT_FEATURE_SET_X4_ORB_CLAIM_BLOCKS_ONLY = "x4_orb_claim_blocks_only"
CONTEXT_FEATURE_SET_X5_ORB_CLAIM_INTERACTION_CREATION = "x5_orb_claim_interaction_creation"
CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE = "linear_nonadditive_context"
CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE = "x3_nonadditive_shape_context"
CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE = "linear_x3_additive_quadratic_side"
CONTEXT_FEATURE_SET_V1_WITHOUT_SHOOTING = "v1_without_shooting"
CONTEXT_FEATURE_SET_V1_WITHOUT_CREATION = "v1_without_creation"
CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING = "v1_without_rebounding"
CONTEXT_FEATURE_SET_V1_WITHOUT_DEFENSIVE_EVENTS = "v1_without_defensive_events"
CONTEXT_FEATURE_SET_V1_WITHOUT_UNCERTAINTY = "v1_without_uncertainty"

V1_KNOCKOUT_EXCLUSIONS = {
    CONTEXT_FEATURE_SET_V1_WITHOUT_SHOOTING: frozenset(
        {
            "three_pa_per_100",
            "three_pm_per_100",
            "bottom_two_three_pm",
            "credible_shooter_count",
            "shooting_usage_interaction",
            "shooter_passing_interaction",
        }
    ),
    CONTEXT_FEATURE_SET_V1_WITHOUT_CREATION: frozenset({"assists_per_100", "top_two_assists"}),
    CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING: frozenset(
        {
            "offensive_rebounds_per_100",
            "defensive_rebounds_per_100",
            "sqrt_offensive_rebounds",
            "sqrt_defensive_rebounds",
            "rebounding_usage_interaction",
        }
    ),
    CONTEXT_FEATURE_SET_V1_WITHOUT_DEFENSIVE_EVENTS: frozenset(
        {"steals_per_100", "blocks_per_100"}
    ),
    CONTEXT_FEATURE_SET_V1_WITHOUT_UNCERTAINTY: frozenset({"imputed_count", "replacement_weight"}),
}

# The basketball-profile coordinates in the canonical compiled-linear x3
# contract. They are exact sums of player-profile values and can therefore be
# compiled into player ratings after fitting without changing a prediction.
LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES = (
    "three_pa_per_100",
    "three_pm_per_100",
    "assists_per_100",
    "turnovers_per_100",
    "usage_per_100",
    "steals_per_100",
    "blocks_per_100",
    "offensive_rebound_claim_total",
)
LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES = tuple(
    STANDARD_USAGE_COLUMN if feature == USAGE_COLUMN else feature
    for feature in LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES
)
LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES = (
    *LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    "defensive_rebound_pct",
    "free_throw_attempts_per_100",
    "unassisted_rim_makes_per_100",
    "unassisted_three_makes_per_100",
)
LINEAR_NAIL_V122_BASKETBALL_ADDITIVE_FEATURES = (
    *LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    "defensive_rebound_pct",
)
LINEAR_NAIL_V123_BASKETBALL_ADDITIVE_FEATURES = (
    *LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    "free_throw_attempts_per_100",
)
LINEAR_NAIL_V124_BASKETBALL_ADDITIVE_FEATURES = tuple(
    feature
    for feature in LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES
    if feature != "usage_per_100"
) + ("free_throw_attempts_per_100",)
LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES = tuple(
    feature
    for feature in LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES
    if feature not in {"three_pa_per_100", "usage_per_100"}
)

# The original x3 and quadratic feature contracts additionally used two
# profile-quality calibration coordinates. Keep the immutable legacy contract
# available so stored artifacts remain scoreable, but do not use it for new
# canonical training.
LINEAR_X3_LEGACY_ADDITIVE_FEATURES = (
    *LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES[:-1],
    "imputed_count",
    "replacement_weight",
    "offensive_rebound_claim_total",
)

# Backward-compatible name for the existing quadratic-side feature contract.
LINEAR_X3_ADDITIVE_FEATURES = LINEAR_X3_LEGACY_ADDITIVE_FEATURES


def available_context_feature_sets() -> tuple[str, ...]:
    """Return supported, versioned contextual feature contracts."""

    return (
        CONTEXT_FEATURE_SET_V1,
        CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING,
        CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
        CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
        CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO,
        CONTEXT_FEATURE_SET_X1_ORB_CLAIM_TOTAL,
        CONTEXT_FEATURE_SET_X2_ORB_PER_100_TOTAL,
        CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
        CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
        CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING,
        CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE,
        CONTEXT_FEATURE_SET_NAIL_V123_FREE_THROW_PROFILE,
        CONTEXT_FEATURE_SET_NAIL_V124_FREE_THROW_REPLACEMENT,
        CONTEXT_FEATURE_SET_NAIL_V13,
        CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
        CONTEXT_FEATURE_SET_NAIL_ADDITIVE_ONLY,
        CONTEXT_FEATURE_SET_X4_ORB_CLAIM_BLOCKS_ONLY,
        CONTEXT_FEATURE_SET_X5_ORB_CLAIM_INTERACTION_CREATION,
        CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE,
        CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE,
        CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE,
        *V1_KNOCKOUT_EXCLUSIONS,
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
    required = {
        "player_id",
        *_required_profile_columns(feature_set),
        "profile_imputed",
        "profile_replacement_weight",
    }
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Contextual player profiles missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Contextual player profiles contain duplicate player IDs")
    home = lineup_side_context_features(
        home_lineups,
        profiles,
        feature_set=feature_set,
        rebound_model=rebound_model,
        usage_model=usage_model,
    )
    away = lineup_side_context_features(
        away_lineups,
        profiles,
        feature_set=feature_set,
        rebound_model=rebound_model,
        usage_model=usage_model,
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

    required = {
        "player_id",
        *_required_profile_columns(feature_set),
        "profile_imputed",
        "profile_replacement_weight",
    }
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Contextual player profiles missing columns: {sorted(missing)}")
    if profiles["player_id"].duplicated().any():
        raise ValueError("Contextual player profiles contain duplicate player IDs")
    if (
        feature_set
        in {
            CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
            CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
            CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO,
        }
        and rebound_model is None
    ):
        raise ValueError("Empirical rebound-capacity features require a rebound opportunity model")
    if (
        feature_set
        in {CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION, CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO}
        and usage_model is None
    ):
        raise ValueError("Usage-allocation features require a usage allocation model")
    values = profiles.set_index("player_id")
    lineup_values = [_lineup_values(lineup, values) for lineup in lineups]
    critical_spacing_quantile = _critical_spacing_quantile(feature_set)
    critical_spacing_threshold = (
        _critical_spacing_threshold(profiles, quantile=critical_spacing_quantile)
        if critical_spacing_quantile is not None
        else None
    )
    rebound_rates = _rebound_rates(lineup_values, rebound_model)
    usage_features = _usage_features(lineup_values, usage_model)
    rows = [
        _side_feature_row(
            lineup,
            feature_set,
            rebound_rate=rate,
            usage_feature=usage,
            critical_spacing_threshold=critical_spacing_threshold,
        )
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
    if feature_set in V1_KNOCKOUT_EXCLUSIONS:
        return tuple(
            column
            for column in side_context_feature_columns(CONTEXT_FEATURE_SET_V1)
            if column not in V1_KNOCKOUT_EXCLUSIONS[feature_set]
        )
    if feature_set == CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING:
        return (
            *(
                column
                for column in PROFILE_RATE_COLUMNS
                if column
                not in {
                    "three_pa_per_100",
                    "three_pm_per_100",
                }
            ),
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
            *(
                column
                for column in PROFILE_RATE_COLUMNS
                if column
                not in {
                    "three_pa_per_100",
                    "three_pm_per_100",
                    OFFENSIVE_REBOUND_COLUMN,
                    DEFENSIVE_REBOUND_COLUMN,
                }
            ),
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
            *(
                column
                for column in PROFILE_RATE_COLUMNS
                if column
                not in {
                    "three_pa_per_100",
                    "three_pm_per_100",
                    OFFENSIVE_REBOUND_COLUMN,
                    DEFENSIVE_REBOUND_COLUMN,
                    "usage_per_100",
                    "turnovers_per_100",
                }
            ),
            "bottom_three_three_pm",
            "credible_shooter_count",
            "capped_three_pm",
            "shooting_concentration",
            "top_two_assists",
            "expected_offensive_rebound_pct",
            "expected_defensive_rebound_pct",
            "imputed_count",
            "replacement_weight",
            "shooter_passing_interaction",
            "excess_usage_demand",
            "allocation_entropy",
            "role_reallocation_js",
            "allocation_weighted_turnover_burden",
        )
    if feature_set == CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO:
        return (
            *(
                column
                for column in side_context_feature_columns(CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION)
                if column
                not in {
                    "bottom_three_three_pm",
                    "credible_shooter_count",
                    "capped_three_pm",
                    "shooting_concentration",
                    "shooter_passing_interaction",
                }
            ),
            "bottom_three_three_pm",
            "credible_shooter_count",
            "capped_three_pm",
            "shooting_concentration",
            "shooter_passing_interaction",
            "rim_pressure",
            "spacing_capacity",
            "rim_spacing_interaction",
        )
    if feature_set == CONTEXT_FEATURE_SET_X1_ORB_CLAIM_TOTAL:
        return ("offensive_rebound_claim_total",)
    if feature_set == CONTEXT_FEATURE_SET_X2_ORB_PER_100_TOTAL:
        return (OFFENSIVE_REBOUND_COLUMN,)
    if feature_set == CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT:
        excluded = V1_KNOCKOUT_EXCLUSIONS[CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING]
        return (
            *(
                column
                for column in side_context_feature_columns(CONTEXT_FEATURE_SET_V1)
                if column not in excluded
            ),
            "offensive_rebound_claim_total",
        )
    if feature_set == CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
        return tuple(
            column
            for column in side_context_feature_columns(
                CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT
            )
            if column not in {"imputed_count", "replacement_weight"}
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE:
        return (
            *LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
            "top_two_assists",
            "usage_concentration",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING:
        return (
            *side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE),
            "critical_spacing",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE:
        return (
            *side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE),
            "critical_spacing",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUARTILE_STANDARD_USAGE:
        return (
            *side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE),
            "critical_spacing",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE:
        return (
            *LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
            "top_two_assists",
            "usage_concentration",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE:
        return (
            *LINEAR_NAIL_V122_BASKETBALL_ADDITIVE_FEATURES,
            "top_two_assists",
            "usage_concentration",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V123_FREE_THROW_PROFILE:
        return (
            *LINEAR_NAIL_V123_BASKETBALL_ADDITIVE_FEATURES,
            "top_two_assists",
            "usage_concentration",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V124_FREE_THROW_REPLACEMENT:
        return (
            *LINEAR_NAIL_V124_BASKETBALL_ADDITIVE_FEATURES,
            "top_two_assists",
            "usage_concentration",
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_ADDITIVE_ONLY:
        return LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V13:
        return (
            *LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES,
            *side_context_feature_columns(CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE),
        )
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE:
        return (
            *LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES,
            *side_context_feature_columns(CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE),
        )
    if feature_set == CONTEXT_FEATURE_SET_X4_ORB_CLAIM_BLOCKS_ONLY:
        excluded = (
            V1_KNOCKOUT_EXCLUSIONS[CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING]
            | frozenset({"steals_per_100"})
        )
        return (
            *(
                column
                for column in side_context_feature_columns(CONTEXT_FEATURE_SET_V1)
                if column not in excluded
            ),
            "offensive_rebound_claim_total",
        )
    if feature_set == CONTEXT_FEATURE_SET_X5_ORB_CLAIM_INTERACTION_CREATION:
        excluded = (
            V1_KNOCKOUT_EXCLUSIONS[CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING]
            | frozenset({"assists_per_100", "top_two_assists"})
        )
        return (
            *(
                column
                for column in side_context_feature_columns(CONTEXT_FEATURE_SET_V1)
                if column not in excluded
            ),
            "offensive_rebound_claim_total",
        )
    if feature_set == CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE:
        return (
            "bottom_two_three_pm",
            "credible_shooter_count",
            "usage_concentration",
            "shooting_usage_interaction",
            "shooter_passing_interaction",
        )
    if feature_set == CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE:
        return (
            "bottom_two_three_pm",
            "credible_shooter_count",
            "top_two_assists",
            "usage_concentration",
            "shooting_usage_interaction",
            "shooter_passing_interaction",
        )
    if feature_set == CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE:
        return (
            *LINEAR_X3_ADDITIVE_FEATURES,
            *(f"{column}_squared" for column in LINEAR_X3_ADDITIVE_FEATURES),
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


def _critical_spacing_quantile(feature_set: str) -> float | None:
    """Return the forward-safe low-shooting quantile for a spacing candidate."""

    if feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING:
        return 1.0 / 3.0
    if feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE:
        return 1.0 / 5.0
    if feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUARTILE_STANDARD_USAGE:
        return 1.0 / 4.0
    return None


def _critical_spacing_threshold(profiles: pd.DataFrame, *, quantile: float) -> float:
    """Return a season-specific lower-quantile shrunk 3PM/100 cutoff."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("Critical-spacing quantile must be strictly between zero and one")

    shooting = pd.to_numeric(profiles[SHOOTING_COLUMN], errors="raise").to_numpy(dtype=float)
    if shooting.size == 0 or not np.isfinite(shooting).all():
        raise ValueError("Critical-spacing profiles require finite 3PM/100 values")
    return float(np.quantile(shooting, quantile))


def _required_profile_columns(feature_set: str) -> tuple[str, ...]:
    if feature_set in {
        CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
        CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
        CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO,
    }:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct", "defensive_rebound_pct")
    if feature_set == CONTEXT_FEATURE_SET_X1_ORB_CLAIM_TOTAL:
        return ("offensive_rebound_pct",)
    if feature_set == CONTEXT_FEATURE_SET_X2_ORB_PER_100_TOTAL:
        return (OFFENSIVE_REBOUND_COLUMN,)
    if feature_set in {
        CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
        CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
        CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING,
        CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE,
        CONTEXT_FEATURE_SET_NAIL_ADDITIVE_ONLY,
    }:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct")
    if feature_set in {
        CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUARTILE_STANDARD_USAGE,
    }:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct", STANDARD_USAGE_COLUMN)
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct", "defensive_rebound_pct")
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V123_FREE_THROW_PROFILE:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct")
    if feature_set == CONTEXT_FEATURE_SET_NAIL_V124_FREE_THROW_REPLACEMENT:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct")
    if feature_set == CONTEXT_FEATURE_SET_X4_ORB_CLAIM_BLOCKS_ONLY:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct")
    if feature_set == CONTEXT_FEATURE_SET_X5_ORB_CLAIM_INTERACTION_CREATION:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct")
    if feature_set in {
        CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE,
        CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE,
    }:
        return PROFILE_RATE_COLUMNS
    if feature_set == CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE:
        return (*PROFILE_RATE_COLUMNS, "offensive_rebound_pct")
    return PROFILE_RATE_COLUMNS


def _side_feature_row(
    lineup: pd.DataFrame,
    feature_set: str,
    *,
    rebound_rate: tuple[float, float] | None = None,
    usage_feature: dict[str, float] | None = None,
    critical_spacing_threshold: float | None = None,
) -> dict[str, float]:
    if feature_set == CONTEXT_FEATURE_SET_V1 or feature_set in V1_KNOCKOUT_EXCLUSIONS:
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
            if column
            not in {
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
            if column
            not in {
                "three_pa_per_100",
                "three_pm_per_100",
                OFFENSIVE_REBOUND_COLUMN,
                DEFENSIVE_REBOUND_COLUMN,
                "usage_per_100",
                "turnovers_per_100",
            }
        }
        if rebound_rate is None or usage_feature is None:
            raise ValueError("Rebound and usage allocation features are required for v2.2")
        result.update(_summary_v22(lineup, rebound_rate=rebound_rate, usage_feature=usage_feature))
    elif feature_set == CONTEXT_FEATURE_SET_V23_SHOT_PORTFOLIO:
        if rebound_rate is None or usage_feature is None:
            raise ValueError("Rebound and usage allocation features are required for v2.3")
        # Preserve every base aggregate retained by v2.2, then append the
        # portfolio signals.  The shot features supplement rather than replace
        # defensive events, passing, or other established profile aggregates.
        result = {
            column: float(lineup[column].sum())
            for column in PROFILE_RATE_COLUMNS
            if column
            not in {
                "three_pa_per_100",
                "three_pm_per_100",
                OFFENSIVE_REBOUND_COLUMN,
                DEFENSIVE_REBOUND_COLUMN,
                "usage_per_100",
                "turnovers_per_100",
            }
        }
        result.update(_summary_v22(lineup, rebound_rate=rebound_rate, usage_feature=usage_feature))
        rim = float(lineup["rim_pressure"].sum())
        spacing = float(lineup["spacing_capacity"].sum())
        result.update(
            rim_pressure=rim,
            spacing_capacity=spacing,
            rim_spacing_interaction=rim * spacing,
        )
    elif feature_set == CONTEXT_FEATURE_SET_X1_ORB_CLAIM_TOTAL:
        result = {"offensive_rebound_claim_total": float(lineup["offensive_rebound_pct"].sum())}
    elif feature_set == CONTEXT_FEATURE_SET_X2_ORB_PER_100_TOTAL:
        result = {OFFENSIVE_REBOUND_COLUMN: float(lineup[OFFENSIVE_REBOUND_COLUMN].sum())}
    elif feature_set == CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT:
        excluded = V1_KNOCKOUT_EXCLUSIONS[CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING]
        result = {column: float(lineup[column].sum()) for column in PROFILE_RATE_COLUMNS}
        result.update(_summary_v1(lineup))
        result = {column: value for column, value in result.items() if column not in excluded}
        result["offensive_rebound_claim_total"] = float(lineup["offensive_rebound_pct"].sum())
    elif feature_set == CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY:
        result = _side_feature_row(lineup, CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT)
        result = {
            column: value
            for column, value in result.items()
            if column not in {"imputed_count", "replacement_weight"}
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY)
        result = {
            column: base[column]
            for column in side_context_feature_columns(
                CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE
            )
        }
    elif feature_set in {
        CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING,
        CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE,
    }:
        if critical_spacing_threshold is None:
            raise ValueError("Critical-spacing features require a season profile threshold")
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE)
        result = {
            **base,
            "critical_spacing": float(
                (lineup[SHOOTING_COLUMN] < critical_spacing_threshold).sum() >= 2
            ),
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUARTILE_STANDARD_USAGE:
        if critical_spacing_threshold is None:
            raise ValueError("Critical-spacing features require a season profile threshold")
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE)
        result = {
            **base,
            "critical_spacing": float(
                (lineup[SHOOTING_COLUMN] < critical_spacing_threshold).sum() >= 2
            ),
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE:
        standard_usage_lineup = lineup.assign(**{USAGE_COLUMN: lineup[STANDARD_USAGE_COLUMN]})
        base = _side_feature_row(
            standard_usage_lineup,
            CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
        )
        result = {
            (STANDARD_USAGE_COLUMN if column == USAGE_COLUMN else column): value
            for column, value in base.items()
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE)
        base["defensive_rebound_pct"] = float(lineup["defensive_rebound_pct"].sum())
        result = {
            column: base[column]
            for column in side_context_feature_columns(
                CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE
            )
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V123_FREE_THROW_PROFILE:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE)
        base["free_throw_attempts_per_100"] = float(
            lineup["free_throw_attempts_per_100"].sum()
        )
        result = {
            column: base[column]
            for column in side_context_feature_columns(
                CONTEXT_FEATURE_SET_NAIL_V123_FREE_THROW_PROFILE
            )
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V124_FREE_THROW_REPLACEMENT:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE)
        base.pop("usage_per_100")
        base["free_throw_attempts_per_100"] = float(
            lineup["free_throw_attempts_per_100"].sum()
        )
        result = {
            column: base[column]
            for column in side_context_feature_columns(
                CONTEXT_FEATURE_SET_NAIL_V124_FREE_THROW_REPLACEMENT
            )
        }
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_ADDITIVE_ONLY:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY)
        result = {column: base[column] for column in LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES}
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V13:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY)
        result = {column: base[column] for column in LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES}
        result.update(
            {
                column: float(lineup[column].sum())
                for column in LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES[8:]
            }
        )
        result.update(_side_feature_row(lineup, CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE))
    elif feature_set == CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_NAIL_V13)
        result = {
            column: base[column]
            for column in side_context_feature_columns(
                CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE
            )
        }
    elif feature_set == CONTEXT_FEATURE_SET_X4_ORB_CLAIM_BLOCKS_ONLY:
        excluded = (
            V1_KNOCKOUT_EXCLUSIONS[CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING]
            | frozenset({"steals_per_100"})
        )
        result = {column: float(lineup[column].sum()) for column in PROFILE_RATE_COLUMNS}
        result.update(_summary_v1(lineup))
        result = {column: value for column, value in result.items() if column not in excluded}
        result["offensive_rebound_claim_total"] = float(lineup["offensive_rebound_pct"].sum())
    elif feature_set == CONTEXT_FEATURE_SET_X5_ORB_CLAIM_INTERACTION_CREATION:
        excluded = (
            V1_KNOCKOUT_EXCLUSIONS[CONTEXT_FEATURE_SET_V1_WITHOUT_REBOUNDING]
            | frozenset({"assists_per_100", "top_two_assists"})
        )
        result = {column: float(lineup[column].sum()) for column in PROFILE_RATE_COLUMNS}
        result.update(_summary_v1(lineup))
        result = {column: value for column, value in result.items() if column not in excluded}
        result["offensive_rebound_claim_total"] = float(lineup["offensive_rebound_pct"].sum())
    elif feature_set == CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE:
        result = _summary_v1(lineup)
        result = {
            column: result[column]
            for column in side_context_feature_columns(CONTEXT_FEATURE_SET_LINEAR_NONADDITIVE)
        }
    elif feature_set == CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE:
        result = _summary_v1(lineup)
        result = {
            column: result[column]
            for column in side_context_feature_columns(CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE)
        }
    elif feature_set == CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE:
        base = _side_feature_row(lineup, CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT)
        result = {column: base[column] for column in LINEAR_X3_ADDITIVE_FEATURES}
        result.update({f"{column}_squared": value**2 for column, value in result.items()})
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
