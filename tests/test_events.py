from __future__ import annotations

import json
from pathlib import Path

import pytest

from nba_lineup_model.events.normalize import (
    canonical_events,
    events_frame,
    format_game_clock,
    parse_nba_clock,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_canonical_events_are_ordered_and_compute_score_deltas():
    events = canonical_events(load_fixture("playbyplay_minimal.json"))

    assert [event.event_type for event in events] == ["period", "3pt"]
    assert events[0].seconds_remaining_period == 720
    assert events[1].source_clock == "PT11M46.00S"
    assert events[1].clock == "11:46.00"
    assert events[1].elapsed_game_seconds == 14
    assert events[1].score_home_delta == 3
    assert events[1].related_player_ids == (201952, 1627759)


def test_overtime_elapsed_time_uses_five_minute_periods():
    payload = {
        "game": {
            "gameId": "0020000002",
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 10000,
                    "period": 5,
                    "periodType": "OVERTIME",
                    "clock": "PT05M00.00S",
                    "actionType": "period",
                    "subType": "start",
                    "scoreHome": "100",
                    "scoreAway": "100",
                },
                {
                    "actionNumber": 2,
                    "orderNumber": 20000,
                    "period": 5,
                    "periodType": "OVERTIME",
                    "clock": "PT04M59.00S",
                    "actionType": "2pt",
                    "subType": "Jump Shot",
                    "scoreHome": "102",
                    "scoreAway": "100",
                },
            ],
        }
    }

    events = canonical_events(payload)

    assert events[0].elapsed_game_seconds == 48 * 60
    assert events[1].elapsed_game_seconds == 48 * 60 + 1
    assert parse_nba_clock("PT05M00.00S") == 300
    assert format_game_clock(299.2) == "04:59.20"


def test_event_identifier_columns_use_nullable_integer_dtype():
    frame = events_frame(load_fixture("playbyplay_minimal.json"))

    assert str(frame["team_id"].dtype) == "Int64"
    assert str(frame["player_id"].dtype) == "Int64"
    assert str(frame["source_possession_team_id"].dtype) == "Int64"
    assert frame.loc[1, "team_id"] == 1610612738


def test_canonical_events_reject_identifier_type_changes():
    payload = load_fixture("playbyplay_minimal.json")
    payload["game"]["actions"][1]["teamId"] = "1610612738"

    with pytest.raises(ValueError, match="positive integer identifier"):
        canonical_events(payload)
