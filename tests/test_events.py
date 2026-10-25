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


def test_nonmonotonic_source_clock_clamps_derived_elapsed_time():
    payload = {
        "game": {
            "gameId": "0020000003",
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 10000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT10M00.00S",
                    "actionType": "2pt",
                    "subType": "Jump Shot",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 2,
                    "orderNumber": 20000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT10M05.00S",
                    "actionType": "2pt",
                    "subType": "Jump Shot",
                    "scoreHome": "2",
                    "scoreAway": "2",
                },
            ],
        }
    }

    events = canonical_events(payload)

    assert events[0].elapsed_game_seconds == 120
    assert events[1].source_clock == "PT10M05.00S"
    assert events[1].elapsed_game_seconds == 120
    assert events[1].validation_flags == ("nonmonotonic_source_clock",)


def test_late_cdn_correction_is_repaired_into_semantic_game_order():
    payload = {
        "game": {
            "gameId": "0020000004",
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 10000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT12M00.00S",
                    "actionType": "period",
                    "subType": "start",
                    "scoreHome": "0",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 20,
                    "orderNumber": 20000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT00M00.00S",
                    "actionType": "period",
                    "subType": "end",
                    "scoreHome": "25",
                    "scoreAway": "22",
                },
                {
                    "actionNumber": 21,
                    "orderNumber": 30000,
                    "period": 2,
                    "periodType": "REGULAR",
                    "clock": "PT12M00.00S",
                    "actionType": "period",
                    "subType": "start",
                    "scoreHome": "25",
                    "scoreAway": "22",
                },
                {
                    "actionNumber": 19,
                    "orderNumber": 90000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT02M55.00S",
                    "actionType": "freethrow",
                    "subType": "1 of 1",
                    "scoreHome": "25",
                    "scoreAway": "22",
                },
            ],
        }
    }

    events = canonical_events(payload)

    assert [event.event_id for event in events] == [
        "0020000004:10000",
        "0020000004:90000",
        "0020000004:20000",
        "0020000004:30000",
    ]
    assert [event.source_order_number for event in events] == [1, 2, 3, 4]
    assert all("source_order_repaired" in event.validation_flags for event in events)


def test_positive_clock_correction_after_period_end_is_repaired():
    payload = {
        "game": {
            "gameId": "0020000005",
            "actions": [
                {
                    "actionNumber": 2,
                    "orderNumber": 10000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT00M00.00S",
                    "actionType": "period",
                    "subType": "end",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 1,
                    "orderNumber": 90000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT00M00.01S",
                    "actionType": "2pt",
                    "subType": "Layup",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
            ],
        }
    }

    events = canonical_events(payload)

    assert [event.event_type for event in events] == ["2pt", "period"]


def test_repaired_stream_preserves_raw_order_at_the_same_clock():
    payload = {
        "game": {
            "gameId": "0020000006",
            "actions": [
                {
                    "actionNumber": 20,
                    "orderNumber": 10000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT10M00.00S",
                    "actionType": "2pt",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 10,
                    "orderNumber": 20000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT10M00.00S",
                    "actionType": "substitution",
                    "subType": "in",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 30,
                    "orderNumber": 30000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT00M00.00S",
                    "actionType": "period",
                    "subType": "end",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 5,
                    "orderNumber": 90000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT05M00.00S",
                    "actionType": "timeout",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
            ],
        }
    }

    events = canonical_events(payload)

    assert [event.event_id for event in events[:2]] == [
        "0020000006:10000",
        "0020000006:20000",
    ]
    assert not any("negative_score_delta" in event.validation_flags for event in events)


def test_repaired_stream_clamps_stale_score_snapshots_without_double_counting():
    payload = {
        "game": {
            "gameId": "0020000007",
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 10000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT10M00.00S",
                    "actionType": "2pt",
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 2,
                    "orderNumber": 20000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT00M00.00S",
                    "actionType": "period",
                    "subType": "end",
                    "scoreHome": "4",
                    "scoreAway": "0",
                },
                {
                    "actionNumber": 3,
                    "orderNumber": 90000,
                    "period": 1,
                    "periodType": "REGULAR",
                    "clock": "PT05M00.00S",
                    "actionType": "substitution",
                    "subType": "in",
                    "scoreHome": "1",
                    "scoreAway": "0",
                },
            ],
        }
    }

    events = canonical_events(payload)

    assert [event.score_home for event in events] == [2, 2, 4]
    assert [event.score_home_delta for event in events] == [2, 0, 2]
    assert "negative_score_delta" in events[1].validation_flags


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
