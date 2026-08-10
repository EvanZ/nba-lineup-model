from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    lineup_context_features,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS


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


def _rates(player_id: int) -> dict[str, float]:
    return {
        column: float(player_id if column != "usage_per_100" else player_id + 10)
        for column in PROFILE_RATE_COLUMNS
    }
