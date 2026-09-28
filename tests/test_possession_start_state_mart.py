from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.possession_start_state_mart import (
    MART_COLUMNS,
    build_possession_start_state_frame,
)


def _events(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "game_id": "001",
        "event_id": None,
        "event_type": "2pt",
        "event_subtype": None,
        "descriptor": None,
        "team_id": None,
        "player_id": None,
        "source_order_number": None,
        "elapsed_game_seconds": 0.0,
        "shot_result": None,
        "description": None,
    }
    normalized = []
    for row in rows:
        event = {**defaults, **row}
        event["event_id"] = event["event_id"] or f"001:{event['event_index']}"
        event["source_order_number"] = event["source_order_number"] or int(event["event_index"]) * 10
        normalized.append(event)
    return pd.DataFrame(normalized)


def _possessions(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "season": "2025-26",
        "season_type": "regular",
        "game_id": "001",
        "possession_id": None,
        "period_possession_index": 0,
        "period": 1,
        "offense_team_id": 1,
        "defense_team_id": 2,
        "start_event_id": None,
        "end_event_id": None,
    }
    normalized = []
    for row in rows:
        possession = {**defaults, **row}
        possession["possession_id"] = possession["possession_id"] or f"001:{possession['possession_index']:04d}"
        possession["start_event_id"] = possession["start_event_id"] or f"001:{possession['start_event_index']}"
        possession["end_event_id"] = possession["end_event_id"] or f"001:{possession['end_event_index']}"
        normalized.append(possession)
    return pd.DataFrame(normalized)


def test_start_state_mart_preserves_granular_terminal_taxonomy() -> None:
    events = _events(
        [
            {"event_index": 0, "event_type": "period"},
            {"event_index": 1, "event_type": "2pt", "team_id": 1, "player_id": 11, "shot_result": "Made"},
            {"event_index": 2, "event_type": "2pt", "team_id": 2, "shot_result": "Missed"},
            {"event_index": 3, "event_type": "3pt", "team_id": 2, "player_id": 22, "shot_result": "Made"},
            {"event_index": 4, "event_type": "2pt", "team_id": 1, "shot_result": "Missed"},
            {"event_index": 5, "event_type": "freethrow", "event_subtype": "1 of 1", "team_id": 1, "player_id": 12, "shot_result": "Made"},
            {"event_index": 6, "event_type": "2pt", "team_id": 2, "shot_result": "Missed"},
            {"event_index": 7, "event_type": "turnover", "event_subtype": "bad pass", "team_id": 2, "player_id": 23},
            {"event_index": 8, "event_type": "steal", "team_id": 1, "player_id": 13},
            {"event_index": 9, "event_type": "turnover", "event_subtype": "out-of-bounds", "team_id": 1, "player_id": 14},
            {"event_index": 10, "event_type": "3pt", "team_id": 2, "shot_result": "Missed"},
            {"event_index": 11, "event_type": "rebound", "event_subtype": "defensive", "team_id": 1, "player_id": 15},
            {"event_index": 12, "event_type": "2pt", "team_id": 1, "shot_result": "Missed"},
            {"event_index": 13, "event_type": "period"},
        ]
    )
    possessions = _possessions(
        [
            {"possession_index": 0, "start_event_index": 0, "end_event_index": 1, "terminal_reason": "made_field_goal", "offense_team_id": 1, "defense_team_id": 2},
            {"possession_index": 1, "start_event_index": 2, "end_event_index": 3, "terminal_reason": "made_field_goal", "offense_team_id": 2, "defense_team_id": 1},
            {"possession_index": 2, "start_event_index": 4, "end_event_index": 5, "terminal_reason": "made_final_free_throw", "offense_team_id": 1, "defense_team_id": 2},
            {"possession_index": 3, "start_event_index": 6, "end_event_index": 7, "terminal_reason": "turnover", "offense_team_id": 2, "defense_team_id": 1},
            {"possession_index": 4, "start_event_index": 8, "end_event_index": 9, "terminal_reason": "turnover", "offense_team_id": 1, "defense_team_id": 2},
            {"possession_index": 5, "start_event_index": 10, "end_event_index": 11, "terminal_reason": "defensive_rebound", "offense_team_id": 2, "defense_team_id": 1},
            {"possession_index": 6, "start_event_index": 12, "end_event_index": 13, "terminal_reason": "period_end", "offense_team_id": 1, "defense_team_id": 2},
        ]
    )

    frame = build_possession_start_state_frame(possessions, events)

    assert tuple(frame.columns) == MART_COLUMNS
    frame = frame.set_index("possession_index")
    assert frame["start_state"].to_dict() == {
        0: "period_start",
        1: "after_made_2pt",
        2: "after_made_3pt",
        3: "after_final_free_throw",
        4: "after_live_ball_turnover_steal",
        5: "after_dead_ball_turnover",
        6: "after_defensive_rebound",
    }
    assert frame.loc[4, "state_trigger_player_id"] == 13
    assert frame.loc[4, "turnover_has_credited_steal"]
    assert frame.loc[5, "is_dead_ball"]
    assert frame.loc[6, "is_live_ball"]
    assert {"offense_points", "points_home", "points_away"}.isdisjoint(frame.columns)


def test_start_state_mart_records_stoppage_modifiers_between_possessions() -> None:
    events = _events(
        [
            {"event_index": 0, "event_type": "period"},
            {"event_index": 1, "event_type": "2pt", "team_id": 1, "shot_result": "Made"},
            {"event_index": 2, "event_type": "timeout", "description": "Timeout"},
            {"event_index": 3, "event_type": "substitution"},
            {"event_index": 4, "event_type": "2pt", "team_id": 2, "shot_result": "Missed"},
            {"event_index": 5, "event_type": "period"},
        ]
    )
    possessions = _possessions(
        [
            {"possession_index": 0, "start_event_index": 0, "end_event_index": 1, "terminal_reason": "made_field_goal", "offense_team_id": 1, "defense_team_id": 2},
            {"possession_index": 1, "start_event_index": 4, "end_event_index": 5, "terminal_reason": "period_end", "offense_team_id": 2, "defense_team_id": 1},
        ]
    )

    row = build_possession_start_state_frame(possessions, events).set_index("possession_index").loc[1]

    assert row["start_state"] == "after_made_2pt"
    assert row["intervening_event_count"] == 2
    assert row["has_intervening_timeout_or_review"]
    assert row["has_intervening_substitution"]


def test_start_state_mart_audits_unknown_turnovers_instead_of_guessing() -> None:
    events = _events(
        [
            {"event_index": 0, "event_type": "period"},
            {"event_index": 1, "event_type": "turnover", "team_id": 1},
            {"event_index": 2, "event_type": "2pt", "team_id": 2, "shot_result": "Missed"},
            {"event_index": 3, "event_type": "period"},
        ]
    )
    possessions = _possessions(
        [
            {"possession_index": 0, "start_event_index": 0, "end_event_index": 1, "terminal_reason": "turnover", "offense_team_id": 1, "defense_team_id": 2},
            {"possession_index": 1, "start_event_index": 2, "end_event_index": 3, "terminal_reason": "period_end", "offense_team_id": 2, "defense_team_id": 1},
        ]
    )

    row = build_possession_start_state_frame(possessions, events).set_index("possession_index").loc[1]

    assert row["start_state"] == "audited_fallback"
    assert row["coarse_start_state"] == "unknown"
    assert row["is_audited_fallback"]
    assert row["fallback_reason"] == "unclassified_turnover_without_credited_steal"
