from __future__ import annotations

import copy
import pandas as pd
import pytest

from nba_lineup_model.season.official_schedule_serving import (
    _object_sha256,
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


def test_schedule_release_identity_changes_when_upcoming_pregame_changes() -> None:
    pregame = {
        "game_id": "0022600156",
        "projection": {
            "serving_bundle_sha256": "a" * 64,
            "home_margin": 2.5,
            "ledger": {"home_margin": 2.5},
        },
        "projected_minutes": {"away": [], "home": []},
    }
    changed = copy.deepcopy(pregame)
    changed["projection"]["home_margin"] = 3.0
    changed["projection"]["ledger"]["home_margin"] = 3.0

    first_identity = {
        "upcoming_game_ids": [pregame["game_id"]],
        "pregame_sha256_by_game": {pregame["game_id"]: _object_sha256(pregame)},
    }
    second_identity = {
        "upcoming_game_ids": [changed["game_id"]],
        "pregame_sha256_by_game": {changed["game_id"]: _object_sha256(changed)},
    }

    assert _object_sha256(first_identity) != _object_sha256(second_identity)
