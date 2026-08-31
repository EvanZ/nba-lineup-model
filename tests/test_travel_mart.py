from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.travel_mart import (
    build_short_rest_travel_game_features,
    build_team_game_travel_features,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["001", "002", "003"],
            "season": ["2024-25"] * 3,
            "season_type": ["regular"] * 3,
            "game_status": ["final"] * 3,
            "game_time_utc": [
                "2024-10-20T00:00:00Z",
                "2024-10-21T18:00:00Z",
                "2024-10-24T00:00:00Z",
            ],
            "home_team_id": [1, 2, 1],
            "home_team_tricode": ["NYK", "LAL", "NYK"],
            "away_team_id": [2, 1, 2],
            "away_team_tricode": ["LAL", "NYK", "LAL"],
        }
    )


def test_travel_mart_uses_prior_game_venue_and_exact_tipoff_window() -> None:
    travel = build_team_game_travel_features(_catalog()).set_index(["game_id", "team_id"])

    assert pd.isna(travel.loc[("001", 1), "travel_miles"])
    assert travel.loc[("002", 1), "has_prior_competitive_game"]
    assert travel.loc[("002", 1), "previous_venue_tricode"] == "NYK"
    assert travel.loc[("002", 1), "hours_since_previous_tipoff"] == 42.0
    assert travel.loc[("002", 1), "travel_within_48_miles"] > 2_000
    assert travel.loc[("003", 1), "hours_since_previous_tipoff"] == 54.0
    assert travel.loc[("003", 1), "travel_within_48_miles"] == 0.0


def test_travel_mart_preserves_zero_distance_within_the_window() -> None:
    catalog = pd.DataFrame(
        {
            "game_id": ["001", "002"],
            "season": ["2024-25"] * 2,
            "season_type": ["regular"] * 2,
            "game_status": ["final"] * 2,
            "game_time_utc": ["2024-10-20T00:00:00Z", "2024-10-21T18:00:00Z"],
            "home_team_id": [2, 2],
            "home_team_tricode": ["LAL", "LAL"],
            "away_team_id": [1, 3],
            "away_team_tricode": ["NYK", "CHI"],
        }
    )
    travel = build_team_game_travel_features(catalog).set_index(["game_id", "team_id"])

    assert travel.loc[("002", 2), "travel_miles"] == 0.0
    assert travel.loc[("002", 2), "travel_within_48_miles"] == 0.0


def test_travel_game_feature_is_signed_and_preserves_opening_missingness() -> None:
    travel = build_team_game_travel_features(_catalog())
    features = build_short_rest_travel_game_features(travel).set_index("game_id")

    assert not features.loc["001", "has_complete_short_rest_travel"]
    assert pd.isna(features.loc["001", "home_minus_away_short_rest_travel_thousand_miles"])
    assert features.loc["002", "has_complete_short_rest_travel"]
    assert features.loc["002", "home_minus_away_short_rest_travel_thousand_miles"] == 0.0
