from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.season.daily_data_replay import (
    _actual_minutes,
    _player_identity_lookup,
)


def test_actual_minutes_blocks_an_unresolved_dnp_player() -> None:
    summary = {
        "game_id": "0022600001",
        "game_date": "2026-10-20",
        "home_team": "DET",
        "away_team": "BOS",
        "pregame": {"start_time": "2026-10-20T23:30:00Z"},
    }
    espn = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "BOS"},
                    "statistics": [
                        {
                            "labels": ["MIN", "PTS"],
                            "athletes": [
                                {
                                    "athlete": {
                                        "id": "999",
                                        "displayName": "Unmapped Player",
                                    },
                                    "stats": [],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }

    with pytest.raises(ValueError, match="Could not map ESPN athlete 'Unmapped Player'"):
        _actual_minutes(summary=summary, espn=espn, name_lookup={})


def test_current_player_identity_wins_over_retired_namesake() -> None:
    current = pd.DataFrame(
        {"player_id": [1630314], "player_name": ["Brandon Williams"]}
    )
    historical = pd.DataFrame(
        {
            "player_id": [1585, 1630314],
            "player_name": ["Brandon Williams", "Brandon Williams"],
        }
    )

    lookup = _player_identity_lookup(current, historical)

    assert lookup["brandon williams"] == 1630314
