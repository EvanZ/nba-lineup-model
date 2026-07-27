from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from nba_lineup_model.events import canonical_events
from nba_lineup_model.lineups import reconstruct_lineups
from nba_lineup_model.possessions import (
    PossessionStartReason,
    PossessionTerminalReason,
    build_possession_segments,
    reconstruct_possessions,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def action(
    number: int,
    *,
    period: int = 1,
    clock: str,
    event_type: str,
    subtype: str | None = None,
    descriptor: str | None = None,
    team_id: int | None = None,
    possession: int = 0,
    score_home: int = 0,
    score_away: int = 0,
    shot_result: str | None = None,
    is_field_goal: bool = False,
) -> dict:
    record = {
        "actionNumber": number,
        "orderNumber": number * 10000,
        "period": period,
        "periodType": "OVERTIME" if period > 4 else "REGULAR",
        "clock": clock,
        "actionType": event_type,
        "possession": possession,
        "scoreHome": str(score_home),
        "scoreAway": str(score_away),
    }
    if subtype is not None:
        record["subType"] = subtype
    if descriptor is not None:
        record["descriptor"] = descriptor
    if team_id is not None:
        record["teamId"] = team_id
        record["personId"] = team_id + 1
    if shot_result is not None:
        record["shotResult"] = shot_result
    if is_field_goal:
        record["isFieldGoal"] = 1
    return record


def event_payload(*actions: dict, game_id: str = "0020000009") -> dict:
    return {"game": {"gameId": game_id, "actions": list(actions)}}


def test_free_throw_trip_remains_one_possession_across_substitutions():
    events = canonical_events(load_fixture("playbyplay_lineup_scenario.json"))
    lineups = reconstruct_lineups(
        events,
        load_fixture("boxscore_lineup_scenario.json"),
    )

    possessions = reconstruct_possessions(
        events,
        home_team_id=100,
        away_team_id=200,
    )
    segmentation = build_possession_segments(events, possessions, lineups)

    assert len(possessions.possessions) == 2
    first, second = possessions.possessions
    assert first.offense_team_id == 100
    assert first.offense_points == 2
    assert first.terminal_reason == PossessionTerminalReason.MADE_FINAL_FREE_THROW
    assert second.offense_team_id == 200
    assert second.offense_points == 2
    assert second.terminal_reason == PossessionTerminalReason.MADE_FIELD_GOAL

    first_segments = [
        segment
        for segment in segmentation.segments
        if segment.possession_index == first.possession_index
    ]
    assert len(first_segments) == 2
    assert [segment.points_home for segment in first_segments] == [1, 1]
    assert first_segments[0].home_player_ids == (101, 102, 103, 104, 105)
    assert first_segments[1].home_player_ids == (101, 102, 103, 104, 106)
    assert possessions.issues == []
    assert segmentation.issues == []


def test_legacy_terminal_labels_do_not_require_is_field_goal():
    payload = load_fixture("playbyplay_lineup_scenario.json")
    actions = payload["game"]["actions"]
    actions[1]["subType"] = "1of2"
    actions[6]["subType"] = "2of2"
    actions[11].pop("isFieldGoal")
    events = canonical_events(payload)

    result = reconstruct_possessions(
        events,
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 2
    assert result.possessions[0].terminal_reason == (
        PossessionTerminalReason.MADE_FINAL_FREE_THROW
    )
    assert result.possessions[1].terminal_reason == PossessionTerminalReason.MADE_FIELD_GOAL
    assert result.issues == []


def test_boundary_scoring_before_delayed_period_start_is_preserved():
    payload = deepcopy(load_fixture("playbyplay_lineup_scenario.json"))
    payload["game"]["actions"].extend(
        [
            {
                "actionNumber": 14,
                "orderNumber": 105000,
                "period": 2,
                "periodType": "REGULAR",
                "clock": "PT12M00.00S",
                "actionType": "foul",
                "subType": "technical",
                "teamId": 100,
                "teamTricode": "HOM",
                "personId": 0,
                "possession": 0,
                "scoreHome": "2",
                "scoreAway": "0",
                "description": "TEAM foul technical",
            },
            {
                "actionNumber": 15,
                "orderNumber": 106000,
                "period": 2,
                "periodType": "REGULAR",
                "clock": "PT12M00.00S",
                "actionType": "freethrow",
                "subType": "1 of 1",
                "descriptor": "technical",
                "teamId": 200,
                "teamTricode": "AWY",
                "personId": 201,
                "possession": 200,
                "scoreHome": "2",
                "scoreAway": "1",
                "shotResult": "Made",
            },
        ]
    )
    for action in payload["game"]["actions"]:
        if action["orderNumber"] >= 110000:
            action["scoreAway"] = str(int(action["scoreAway"]) + 1)

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 2
    assert result.possessions[-1].points_away == 3
    assert [issue.code for issue in result.issues] == [
        "delayed_period_start_marker"
    ]


def test_segments_include_assigned_event_after_nominal_possession_end():
    payload = deepcopy(load_fixture("playbyplay_lineup_scenario.json"))
    payload["game"]["actions"].append(
        {
            "actionNumber": 14,
            "orderNumber": 75000,
            "period": 1,
            "periodType": "REGULAR",
            "clock": "PT10M00.00S",
            "actionType": "freethrow",
            "subType": "1 of 1",
            "descriptor": "technical",
            "teamId": 200,
            "teamTricode": "AWY",
            "personId": 201,
            "possession": 100,
            "scoreHome": "2",
            "scoreAway": "1",
            "shotResult": "Made",
        }
    )
    for source_action in payload["game"]["actions"]:
        if source_action["orderNumber"] >= 80000:
            source_action["scoreAway"] = str(int(source_action["scoreAway"]) + 1)

    events = canonical_events(payload)
    lineups = reconstruct_lineups(
        events,
        load_fixture("boxscore_lineup_scenario.json"),
        validate_boxscore_minutes=False,
    )
    possessions = reconstruct_possessions(
        events,
        home_team_id=100,
        away_team_id=200,
    )
    segmentation = build_possession_segments(events, possessions, lineups)
    first_possession = possessions.possessions[0]
    first_segments = [
        segment
        for segment in segmentation.segments
        if segment.possession_index == first_possession.possession_index
    ]

    assert first_possession.end_order_number == 70000
    assert first_possession.points_away == 1
    assert first_segments[-1].end_order_number == 75000
    assert sum(segment.points_away for segment in first_segments) == 1
    assert segmentation.issues == []


def test_turnover_and_defensive_rebound_are_explicit_terminals():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M50.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            shot_result="Missed",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT11M48.00S",
            event_type="rebound",
            subtype="offensive",
            team_id=100,
            possession=100,
        ),
        action(
            4,
            clock="PT11M40.00S",
            event_type="turnover",
            subtype="bad pass",
            team_id=100,
            possession=100,
        ),
        action(
            5,
            clock="PT11M40.00S",
            event_type="steal",
            team_id=200,
            possession=200,
        ),
        action(
            6,
            clock="PT11M30.00S",
            event_type="2pt",
            team_id=200,
            possession=200,
            shot_result="Missed",
            is_field_goal=True,
        ),
        action(
            7,
            clock="PT11M28.00S",
            event_type="rebound",
            subtype="defensive",
            team_id=100,
            possession=100,
        ),
        action(
            8,
            clock="PT11M20.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            9,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert [possession.offense_team_id for possession in result.possessions] == [
        100,
        200,
        100,
    ]
    assert [possession.terminal_reason for possession in result.possessions] == [
        PossessionTerminalReason.TURNOVER,
        PossessionTerminalReason.DEFENSIVE_REBOUND,
        PossessionTerminalReason.MADE_FIELD_GOAL,
    ]
    assert result.possessions[1].start_reason == PossessionStartReason.TURNOVER
    assert result.possessions[2].start_reason == PossessionStartReason.DEFENSIVE_REBOUND
    assert result.issues == []


def test_offensive_foul_companion_events_do_not_create_phantom_possession():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT10M00.00S",
            event_type="foul",
            subtype="offensive",
            team_id=200,
            possession=100,
        ),
        action(
            3,
            clock="PT10M00.00S",
            event_type="turnover",
            subtype="offensive foul",
            team_id=200,
            possession=200,
        ),
        action(
            4,
            clock="PT10M00.00S",
            event_type="timeout",
            team_id=100,
            possession=100,
        ),
        action(
            5,
            clock="PT09M45.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            6,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 2
    assert [possession.offense_team_id for possession in result.possessions] == [200, 100]
    assert result.possessions[0].terminal_reason == PossessionTerminalReason.TURNOVER
    assert result.possessions[0].source_possession_mismatch_count == 1
    assert result.issues == []


def test_held_ball_turnover_companions_form_one_terminal():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M50.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            shot_result="Missed",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT11M45.00S",
            event_type="jumpball",
            subtype="recovered",
            descriptor="heldball",
            team_id=200,
            possession=100,
        ),
        action(
            4,
            clock="PT11M45.00S",
            event_type="turnover",
            subtype="lost ball",
            team_id=100,
            possession=100,
        ),
        action(
            5,
            clock="PT11M45.00S",
            event_type="steal",
            team_id=200,
            possession=200,
        ),
        action(
            6,
            clock="PT11M35.00S",
            event_type="2pt",
            team_id=200,
            possession=200,
            score_away=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            7,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_away=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 2
    assert [possession.offense_team_id for possession in result.possessions] == [100, 200]
    assert result.possessions[0].terminal_reason == PossessionTerminalReason.TURNOVER
    assert result.possessions[1].terminal_reason == PossessionTerminalReason.MADE_FIELD_GOAL
    assert result.issues == []


def test_held_ball_waits_for_later_turnover_companion():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M50.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            shot_result="Missed",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT11M45.00S",
            event_type="jumpball",
            subtype="recovered",
            descriptor="heldball",
            team_id=200,
            possession=100,
        ),
        action(
            4,
            clock="PT11M42.00S",
            event_type="turnover",
            subtype="lost ball",
            team_id=100,
            possession=100,
        ),
        action(
            5,
            clock="PT11M30.00S",
            event_type="2pt",
            team_id=200,
            possession=200,
            score_away=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            6,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_away=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert [possession.offense_team_id for possession in result.possessions] == [
        100,
        200,
    ]
    assert result.possessions[0].terminal_reason == PossessionTerminalReason.TURNOVER
    assert result.issues == []


def test_post_score_held_ball_preserves_intervening_possession():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT10M50.00S",
            event_type="jumpball",
            subtype="recovered",
            descriptor="heldball",
            team_id=100,
            possession=200,
            score_home=2,
        ),
        action(
            4,
            clock="PT10M40.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=4,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            5,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=4,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert [possession.offense_team_id for possession in result.possessions] == [
        100,
        200,
        100,
    ]
    assert result.possessions[1].terminal_reason == PossessionTerminalReason.HELD_BALL
    assert result.issues == []


def test_out_of_bounds_jump_ball_preserves_intervening_possession():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="turnover",
            team_id=100,
            possession=100,
        ),
        action(
            3,
            clock="PT10M50.00S",
            event_type="jumpball",
            subtype="recovered",
            descriptor="outofbounds",
            team_id=100,
            possession=200,
        ),
        action(
            4,
            clock="PT10M40.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            5,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert [possession.offense_team_id for possession in result.possessions] == [
        100,
        200,
        100,
    ]
    assert result.possessions[1].terminal_reason == PossessionTerminalReason.HELD_BALL
    assert result.issues == []


def test_possession_retaining_free_throw_does_not_end_possession():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="freethrow",
            subtype="1 of 2",
            descriptor="clear-path",
            team_id=100,
            possession=100,
            score_home=1,
            shot_result="Made",
        ),
        action(
            3,
            clock="PT11M00.00S",
            event_type="freethrow",
            subtype="2 of 2",
            descriptor="clear-path",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
        ),
        action(
            4,
            clock="PT10M45.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=4,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            5,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=4,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 1
    assert result.possessions[0].points_home == 4
    assert result.possessions[0].terminal_reason == (
        PossessionTerminalReason.MADE_FIELD_GOAL
    )
    assert result.issues == []


def test_foul_descriptor_marks_undecorated_free_throws_as_possession_retaining():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="foul",
            subtype="personal",
            descriptor="flagrant-type-1",
            team_id=200,
            possession=100,
        ),
        action(
            3,
            clock="PT11M00.00S",
            event_type="freethrow",
            subtype="1 of 2",
            team_id=100,
            possession=100,
            score_home=1,
            shot_result="Made",
        ),
        action(
            4,
            clock="PT11M00.00S",
            event_type="freethrow",
            subtype="2 of 2",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
        ),
        action(
            5,
            clock="PT10M45.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=4,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            6,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=4,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 1
    assert result.possessions[0].points_home == 4
    assert result.possessions[0].terminal_reason == (
        PossessionTerminalReason.MADE_FIELD_GOAL
    )
    assert result.issues == []


def test_small_clock_drift_does_not_split_and_one_sequence():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT10M00.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT09M59.90S",
            event_type="foul",
            subtype="personal",
            descriptor="shooting",
            team_id=200,
            possession=100,
            score_home=2,
        ),
        action(
            4,
            clock="PT09M59.90S",
            event_type="freethrow",
            subtype="1 of 1",
            team_id=100,
            possession=100,
            score_home=3,
            shot_result="Made",
        ),
        action(
            5,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=3,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 1
    assert result.possessions[0].points_home == 3
    assert result.possessions[0].terminal_reason == (
        PossessionTerminalReason.MADE_FINAL_FREE_THROW
    )
    assert result.issues == []


def test_same_clock_shooting_foul_overrides_defensive_rebound():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            shot_result="Missed",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT10M58.00S",
            event_type="rebound",
            subtype="defensive",
            team_id=200,
            possession=200,
        ),
        action(
            4,
            clock="PT10M58.00S",
            event_type="foul",
            subtype="personal",
            descriptor="shooting",
            team_id=200,
            possession=100,
        ),
        action(
            5,
            clock="PT10M58.00S",
            event_type="freethrow",
            subtype="1 of 2",
            team_id=100,
            possession=100,
            score_home=1,
            shot_result="Made",
        ),
        action(
            6,
            clock="PT10M58.00S",
            event_type="freethrow",
            subtype="2 of 2",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
        ),
        action(
            7,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 1
    assert result.possessions[0].offense_team_id == 100
    assert result.possessions[0].terminal_reason == (
        PossessionTerminalReason.MADE_FINAL_FREE_THROW
    )
    assert [issue.code for issue in result.issues] == [
        "rebound_overridden_by_same_clock_foul"
    ]


def test_opponent_technical_free_throw_is_preserved_as_possession_exception():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="2pt",
            team_id=200,
            possession=200,
            score_away=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT11M00.00S",
            event_type="timeout",
            team_id=100,
            possession=100,
            score_away=2,
        ),
        action(
            4,
            clock="PT11M00.00S",
            event_type="foul",
            subtype="technical",
            team_id=100,
            possession=100,
            score_away=2,
        ),
        action(
            5,
            clock="PT11M00.00S",
            event_type="freethrow",
            subtype="1 of 1",
            descriptor="technical",
            team_id=200,
            possession=100,
            score_away=3,
            shot_result="Made",
        ),
        action(
            6,
            clock="PT10M50.00S",
            event_type="turnover",
            subtype="bad pass",
            team_id=100,
            possession=100,
            score_away=3,
        ),
        action(
            7,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_away=3,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 2
    assert result.possessions[1].offense_team_id == 100
    assert result.possessions[1].points_away == 1
    assert result.possessions[1].validation_flags == (
        "opponent_technical_free_throw",
    )
    assert result.issues == []


def test_dead_ball_technical_free_throw_uses_next_possession_team():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT11M00.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT10M50.00S",
            event_type="foul",
            subtype="technical",
            team_id=200,
            possession=200,
            score_home=2,
        ),
        action(
            4,
            clock="PT10M50.00S",
            event_type="freethrow",
            subtype="1 of 1",
            descriptor="technical",
            team_id=100,
            possession=200,
            score_home=3,
            shot_result="Made",
        ),
        action(
            5,
            clock="PT10M40.00S",
            event_type="2pt",
            team_id=200,
            possession=200,
            score_home=3,
            score_away=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            6,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=3,
            score_away=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert [possession.offense_team_id for possession in result.possessions] == [
        100,
        200,
    ]
    assert result.possessions[1].points_home == 1
    assert result.possessions[1].points_away == 2
    assert result.possessions[1].validation_flags == (
        "opponent_technical_free_throw",
    )
    assert result.issues == []


def test_post_score_loose_ball_free_throw_continues_scoring_possession():
    payload = event_payload(
        action(1, clock="PT12M00.00S", event_type="period", subtype="start"),
        action(
            2,
            clock="PT10M00.00S",
            event_type="3pt",
            team_id=100,
            possession=100,
            score_home=3,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            3,
            clock="PT10M00.00S",
            event_type="foul",
            subtype="personal",
            descriptor="loose ball",
            team_id=200,
            possession=200,
            score_home=3,
        ),
        action(
            4,
            clock="PT10M00.00S",
            event_type="freethrow",
            subtype="1 of 1",
            team_id=100,
            possession=100,
            score_home=3,
            shot_result="Missed",
        ),
        action(
            5,
            clock="PT09M57.00S",
            event_type="rebound",
            subtype="defensive",
            team_id=200,
            possession=200,
            score_home=3,
        ),
        action(
            6,
            clock="PT09M45.00S",
            event_type="2pt",
            team_id=200,
            possession=200,
            score_home=3,
            score_away=2,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            7,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=3,
            score_away=2,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 2
    assert result.possessions[0].offense_team_id == 100
    assert result.possessions[0].points_home == 3
    assert result.possessions[0].terminal_reason == (
        PossessionTerminalReason.DEFENSIVE_REBOUND
    )
    assert result.possessions[1].offense_team_id == 200
    assert result.issues == []


def test_overtime_uses_overtime_clock_and_score_baseline():
    payload = event_payload(
        action(
            1,
            period=5,
            clock="PT05M00.00S",
            event_type="period",
            subtype="start",
            score_home=100,
            score_away=100,
        ),
        action(
            2,
            period=5,
            clock="PT04M59.00S",
            event_type="2pt",
            team_id=100,
            possession=100,
            score_home=102,
            score_away=100,
            shot_result="Made",
            is_field_goal=True,
        ),
        action(
            3,
            period=5,
            clock="PT00M00.00S",
            event_type="period",
            subtype="end",
            score_home=102,
            score_away=100,
        ),
    )

    result = reconstruct_possessions(
        canonical_events(payload),
        home_team_id=100,
        away_team_id=200,
    )

    assert len(result.possessions) == 1
    assert result.possessions[0].period == 5
    assert result.possessions[0].start_elapsed_game_seconds == 48 * 60 + 1
    assert result.possessions[0].offense_points == 2
    assert result.issues == []
