from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
    CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
    CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING,
    lineup_context_features,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS
from nba_lineup_model.modeling.rebound_opportunity import ReboundOpportunityModel
from nba_lineup_model.modeling.usage_allocation import UsageAllocationModel


def test_contextual_features_capture_lineup_shooting_and_uncertainty() -> None:
    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **_rates(player_id),
                "profile_imputed": int(player_id == 5),
                "profile_replacement_weight": 0.8 if player_id == 5 else 0.0,
            }
            for player_id in range(1, 11)
        ]
    )

    features = lineup_context_features([[1, 2, 3, 4, 5]], [[6, 7, 8, 9, 10]], profiles)

    assert features.loc[0, "home_minus_away_three_pm_per_100"] < 0.0
    assert features.loc[0, "home_minus_away_imputed_count"] == 1.0
    assert features.loc[0, "home_minus_away_replacement_weight"] == 0.8


def test_relative_context_features_equal_the_difference_of_side_features() -> None:
    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **_rates(player_id),
                "profile_imputed": 0,
                "profile_replacement_weight": 0.0,
            }
            for player_id in range(1, 11)
        ]
    )
    home = [[1, 2, 3, 4, 5]]
    away = [[6, 7, 8, 9, 10]]

    relative = lineup_context_features(home, away, profiles)
    home_side = lineup_side_context_features(home, profiles)
    away_side = lineup_side_context_features(away, profiles)

    for column in side_context_feature_columns():
        assert relative.loc[0, f"home_minus_away_{column}"] == (
            home_side.loc[0, column] - away_side.loc[0, column]
        )


def test_depth_aware_shooting_does_not_treat_one_star_as_five_shooters() -> None:
    """Identical raw totals separate when their lineup shooting depth differs."""

    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **_rates(player_id),
                "profile_imputed": 0,
                "profile_replacement_weight": 0.0,
            }
            for player_id in range(1, 11)
        ]
    )
    profiles.loc[profiles["player_id"].between(1, 5), "three_pm_per_100"] = [5, 0, 0, 0, 0]
    profiles.loc[profiles["player_id"].between(6, 10), "three_pm_per_100"] = [2, 2, 1, 0, 0]

    v1 = lineup_side_context_features([[1, 2, 3, 4, 5]], profiles)
    v2 = lineup_side_context_features(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        profiles,
        feature_set=CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING,
    )

    assert v1.loc[0, "three_pm_per_100"] == 5.0
    assert v2.loc[0, "capped_three_pm"] == 2.0
    assert v2.loc[1, "capped_three_pm"] == 5.0
    assert v2.loc[0, "bottom_three_three_pm"] == 0.0
    assert v2.loc[1, "bottom_three_three_pm"] == 1.0
    assert v2.loc[0, "shooting_concentration"] > v2.loc[1, "shooting_concentration"]


def test_rebound_capacity_uses_empirical_reference_field() -> None:
    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **_rates(player_id),
                "offensive_rebound_pct": 8.0,
                "defensive_rebound_pct": 25.0,
                "profile_imputed": 0,
                "profile_replacement_weight": 0.0,
            }
            for player_id in range(1, 6)
        ]
    )

    features = lineup_side_context_features(
        [[1, 2, 3, 4, 5]],
        profiles,
        feature_set=CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
        rebound_model=_rebound_model(),
    )

    assert features.loc[0, "expected_offensive_rebound_pct"] > 0.0
    assert features.loc[0, "expected_defensive_rebound_pct"] > 0.0
    assert features.loc[0, "expected_defensive_rebound_pct"] < 100.0


def test_rebound_capacity_requires_fitted_calibration() -> None:
    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **_rates(player_id),
                "offensive_rebound_pct": 5.0,
                "defensive_rebound_pct": 15.0,
                "profile_imputed": 0,
                "profile_replacement_weight": 0.0,
            }
            for player_id in range(1, 6)
        ]
    )

    try:
        lineup_side_context_features(
            [[1, 2, 3, 4, 5]],
            profiles,
            feature_set=CONTEXT_FEATURE_SET_V21_REBOUND_CAPACITY,
        )
    except ValueError as exc:
        assert "rebound opportunity model" in str(exc)
    else:
        raise AssertionError("v2.1 must reject an unfitted rebound calibration")


def test_usage_allocation_replaces_raw_usage_features_without_cutoffs() -> None:
    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **_rates(player_id),
                "offensive_rebound_pct": 5.0,
                "defensive_rebound_pct": 15.0,
                "profile_imputed": 0,
                "profile_replacement_weight": 0.0,
            }
            for player_id in range(1, 6)
        ]
    )
    features = lineup_side_context_features(
        [[1, 2, 3, 4, 5]],
        profiles,
        feature_set=CONTEXT_FEATURE_SET_V22_USAGE_ALLOCATION,
        rebound_model=_rebound_model(),
        usage_model=UsageAllocationModel(
            temperature=1.0,
            claim_scale=1.0,
            claim_budget=50.0,
            training_season="2024-25",
            training_event_count=1_000,
        ),
    )

    assert "usage_per_100" not in features
    assert "turnovers_per_100" not in features
    assert "usage_concentration" not in features
    assert "depth_usage_interaction" not in features
    assert features.loc[0, "excess_usage_demand"] > 0.0
    assert 0.0 < features.loc[0, "allocation_entropy"] <= 1.0
    assert features.loc[0, "role_reallocation_js"] >= 0.0


def _rates(player_id: int) -> dict[str, float]:
    return {
        column: float(player_id if column != "usage_per_100" else player_id + 10)
        for column in PROFILE_RATE_COLUMNS
    }


def _rebound_model() -> ReboundOpportunityModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    features = pd.DataFrame(
        {
            "defensive_claims": [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "offensive_claims": [15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
        }
    )
    pipeline = Pipeline(
        [
            ("spline", SplineTransformer(n_knots=3, degree=2, extrapolation="linear")),
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, max_iter=500)),
        ]
    )
    pipeline.fit(features, [0, 0, 0, 1, 1, 1])
    return ReboundOpportunityModel(
        pipeline=pipeline,
        reference_defensive_claims=np.array([50.0, 60.0, 70.0]),
        reference_offensive_claims=np.array([20.0, 30.0, 40.0]),
        training_season="2024-25",
        training_opportunity_count=len(features),
    )
