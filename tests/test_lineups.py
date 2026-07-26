from __future__ import annotations

import json
from pathlib import Path

import pytest

from nba_lineup_model.events.normalize import canonical_events
from nba_lineup_model.lineups.reconstruct import (
    LineupReconstructionError,
    LineupState,
    normalize_lineup,
    reconstruct_lineups,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_normalize_lineup_sorts_ids():
    assert normalize_lineup([5, 3, 1, 4, 2]) == (1, 2, 3, 4, 5)


def test_lineup_state_validate_accepts_five_on_five():
    state = LineupState(
        home_player_ids=(1, 2, 3, 4, 5),
        away_player_ids=(6, 7, 8, 9, 10),
    )

    state.validate()


def test_reconstruction_batches_substitutions_and_splits_periods():
    events = canonical_events(load_fixture("playbyplay_lineup_scenario.json"))
    boxscore = load_fixture("boxscore_lineup_scenario.json")

    result = reconstruct_lineups(events, boxscore)

    assignments = {
        assignment.source_order_number: assignment for assignment in result.event_lineups
    }
    assert assignments[20000].lineup_before.home_player_ids == (101, 102, 103, 104, 105)
    assert assignments[30000].lineup_before.home_player_ids == (101, 102, 103, 104, 105)
    assert assignments[30000].lineup_after.home_player_ids == (101, 102, 103, 104, 106)
    assert assignments[70000].lineup_before.home_player_ids == (101, 102, 103, 104, 106)
    assert assignments[110000].lineup_before.home_player_ids == (101, 102, 103, 104, 105)

    assert [stint.duration_seconds for stint in result.stints] == [120, 600, 720]
    assert [stint.points_home for stint in result.stints] == [1, 1, 0]
    assert [stint.points_away for stint in result.stints] == [0, 0, 2]
    assert result.issues == []


def test_reconstruction_rejects_subbing_out_player_not_on_court():
    events = canonical_events(load_fixture("playbyplay_lineup_scenario.json"))
    boxscore = load_fixture("boxscore_lineup_scenario.json")
    events[2] = events[2].model_copy(update={"player_id": 999})

    with pytest.raises(LineupReconstructionError, match="not on court"):
        reconstruct_lineups(events, boxscore)


def test_reconstruction_handles_overtime_period():
    payload = {
        "game": {
            "gameId": "0020000001",
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
                    "clock": "PT00M00.00S",
                    "actionType": "period",
                    "subType": "end",
                    "scoreHome": "110",
                    "scoreAway": "108",
                },
            ],
        }
    }

    result = reconstruct_lineups(
        canonical_events(payload),
        load_fixture("boxscore_lineup_scenario.json"),
        validate_boxscore_minutes=False,
    )

    assert len(result.stints) == 1
    assert result.stints[0].period == 5
    assert result.stints[0].period_type == "OVERTIME"
    assert result.stints[0].duration_seconds == 300
