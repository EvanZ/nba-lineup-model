from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.season.official_schedule_serving import (
    _attach_game_prediction_attribution,
    _official_retrospective,
)


def test_historical_game_uses_one_frozen_rating_map_for_both_ledgers() -> None:
    ratings = {1: 1.0, 2: -1.0, 3: 2.0, 4: 0.0}
    pregame = _attach_game_prediction_attribution(
        {
            "game_id": "game",
            "away_team": "AWY",
            "home_team": "HME",
            "projection": {"model": "test"},
            "projected_minutes": {
                "away": [
                    {"player_id": 1, "player_name": "Away One", "projected_minutes": 120.0},
                    {"player_id": 2, "player_name": "Away Two", "projected_minutes": 120.0},
                ],
                "home": [
                    {"player_id": 3, "player_name": "Home One", "projected_minutes": 120.0},
                    {"player_id": 4, "player_name": "Home Two", "projected_minutes": 120.0},
                ],
            },
        },
        ratings=ratings,
        home_court=1.5,
        back_to_back=-0.75,
        home_back_to_back=True,
        away_back_to_back=False,
        win_probability_scale=0.2,
        provenance={"rating_as_of": "2026-10-31"},
    )
    actual_minutes = pd.DataFrame(
        [
            {
                "game_id": "game",
                "team": "AWY",
                "player_id": 1,
                "player_name": "Away One",
                "minutes": 48.0,
            },
            {
                "game_id": "game",
                "team": "AWY",
                "player_id": 2,
                "player_name": "Away Two",
                "minutes": 192.0,
            },
            {
                "game_id": "game",
                "team": "HME",
                "player_id": 3,
                "player_name": "Home One",
                "minutes": 192.0,
            },
            {
                "game_id": "game",
                "team": "HME",
                "player_id": 4,
                "player_name": "Home Two",
                "minutes": 48.0,
            },
        ]
    )

    retrospective = _official_retrospective(
        final_game={"game_id": "game", "away_team": "AWY", "home_team": "HME"},
        pregame=pregame,
        actual_minutes=actual_minutes,
        ratings=ratings,
    )

    pregame_ratings = {
        row["player_id"]: row["frozen_nail"]
        for side in ("away", "home")
        for row in pregame["projection"]["ledger"][side]["players"]
    }
    actual_ratings = {
        row["player_id"]: row["frozen_nail"]
        for side in ("away", "home")
        for row in retrospective["ledger"][side]["players"]
    }
    assert actual_ratings == pregame_ratings == ratings
    assert retrospective["home_margin"] == pytest.approx(
        retrospective["ledger"]["home_margin"]
    )
