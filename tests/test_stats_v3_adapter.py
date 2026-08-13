from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from nba_lineup_model.events import canonical_events
from nba_lineup_model.ingest.nba_cdn import (
    CachedResponse,
    NbaCdnEndpoint,
    RawJsonCache,
)
from nba_lineup_model.ingest.nba_stats import (
    NbaStatsEndpoint,
    NbaStatsRawCache,
    StatsCachedResponse,
)
from nba_lineup_model.normalize.stats_v3 import adapt_stats_v3_game
from nba_lineup_model.season.source import load_game_source_documents

GAME_ID = "0021900001"
HOME_TEAM_ID = 1610612701
AWAY_TEAM_ID = 1610612702


def test_adapts_stats_v3_boxscore_and_event_vocabulary() -> None:
    play_by_play, boxscore = adapt_stats_v3_game(
        stats_play_by_play(),
        stats_boxscore(),
    )

    home_players = boxscore["game"]["homeTeam"]["players"]
    assert [player["starter"] for player in home_players] == [
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
    ]
    assert home_players[0]["statistics"]["minutes"] == "PT30M05.00S"

    actions = play_by_play["game"]["actions"]
    assert [action["actionType"] for action in actions] == [
        "period",
        "jumpball",
        "2pt",
        "block",
        "rebound",
        "turnover",
        "steal",
        "substitution",
        "substitution",
        "period",
    ]
    assert actions[4]["subType"] == "defensive"
    assert actions[4]["teamId"] == AWAY_TEAM_ID
    assert actions[4]["personId"] == 0
    assert actions[7]["subType"] == "out"
    assert actions[7]["personId"] == 101
    assert actions[8]["subType"] == "in"
    assert actions[8]["personId"] == 106

    events = canonical_events(play_by_play)
    assert len({event.source_order_number for event in events}) == len(events)
    turnover, steal = events[5:7]
    assert turnover.source_action_number == steal.source_action_number == 20
    assert turnover.source_order_number != steal.source_order_number


def test_adapts_historical_stats_v3_boxscore_starter_order_and_minutes() -> None:
    boxscore = stats_boxscore()
    for side in ("homeTeam", "awayTeam"):
        players = boxscore["boxScoreTraditional"][side]["players"]
        for player in players:
            player["position"] = "G"
        players[0]["statistics"]["minutes"] = "10"

    _, adapted_boxscore = adapt_stats_v3_game(stats_play_by_play(), boxscore)

    home_players = adapted_boxscore["game"]["homeTeam"]["players"]
    assert [player["starter"] for player in home_players] == [
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
    ]
    assert home_players[0]["statistics"]["minutes"] == "PT10M00.00S"


def test_forward_fills_historical_zero_score_placeholders() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"][2]["scoreHome"] = "2"

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())
    events = canonical_events(adapted)

    assert events[2].score_home == 2
    assert events[3].score_home == 2
    assert "negative_score_delta" not in events[3].validation_flags


def test_forward_fills_historical_score_regressions() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"][2]["scoreHome"] = "7"
    play_by_play["game"]["actions"][3]["scoreHome"] = "2"

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())
    events = canonical_events(adapted)

    assert events[2].score_home == 7
    assert events[3].score_home == 7
    assert "negative_score_delta" not in events[3].validation_flags


def test_supplies_missing_historical_period_start() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"] = play_by_play["game"]["actions"][1:]

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())

    first_action = adapted["game"]["actions"][0]
    assert first_action["actionType"] == "period"
    assert first_action["subType"] == "start"
    assert first_action["period"] == 1


def test_labels_stats_v3_overtime_periods() -> None:
    play_by_play = stats_play_by_play()
    overtime = action(
        action_id=10,
        action_number=30,
        period=5,
        clock="PT05M00.00S",
        action_type="period",
        subtype="start",
        description="Start of OT1",
    )
    play_by_play["game"]["actions"].append(overtime)

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())

    assert adapted["game"]["actions"][-1]["periodType"] == "OVERTIME"


def test_orders_late_insertions_by_clock_and_period_end_last() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"] = [
        action(
            action_id=1,
            action_number=2,
            action_type="period",
            subtype="start",
            description="Start of 1st Period",
        ),
        action(
            action_id=2,
            action_number=20,
            clock="PT10M00.00S",
            team_id=HOME_TEAM_ID,
            person_id=101,
            action_type="Made Shot",
            description="Homeone 2' Layup",
            is_field_goal=1,
            shot_value=2,
            shot_result="Made",
        ),
        action(
            action_id=3,
            action_number=10,
            clock="PT11M00.00S",
            team_id=AWAY_TEAM_ID,
            person_id=201,
            action_type="Missed Shot",
            description="MISS Awayone 3' Layup",
            is_field_goal=1,
            shot_value=2,
            shot_result="Missed",
        ),
        action(
            action_id=4,
            action_number=30,
            clock="PT00M00.00S",
            action_type="period",
            subtype="end",
            description="End of 1st Period",
        ),
        action(
            action_id=5,
            action_number=31,
            clock="PT00M00.00S",
            team_id=HOME_TEAM_ID,
            person_id=102,
            action_type="Missed Shot",
            description="MISS Hometwo 50' Jump Shot",
            is_field_goal=1,
            shot_value=3,
            shot_result="Missed",
        ),
    ]

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())
    events = canonical_events(adapted)

    assert [event.source_action_number for event in events] == [2, 10, 20, 31, 30]
    assert [event.clock for event in events] == [
        "12:00.00",
        "11:00.00",
        "10:00.00",
        "00:00.00",
        "00:00.00",
    ]
    assert all("nonmonotonic_source_clock" not in event.validation_flags for event in events)


def test_assigns_team_heave_from_home_away_location() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"] = [
        play_by_play["game"]["actions"][0],
        action(
            action_id=2,
            action_number=4,
            clock="PT00M00.20S",
            location="v",
            action_type="Heave",
            subtype="Team Field Goal Attempt",
            description="AWAY Heave",
        ),
        play_by_play["game"]["actions"][-1],
    ]

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())
    heave = canonical_events(adapted)[1]

    assert heave.event_type == "heave"
    assert heave.team_id == AWAY_TEAM_ID
    assert heave.source_possession_team_id == AWAY_TEAM_ID


def test_assigns_jump_ball_possession_to_tip_recipient() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"] = [
        play_by_play["game"]["actions"][0],
        action(
            action_id=2,
            action_number=4,
            clock="PT12M00.00S",
            team_id=HOME_TEAM_ID,
            person_id=101,
            action_type="Jump Ball",
            description="HOME Jumper vs. AWAY Jumper: Tip to Awayone",
        ),
        play_by_play["game"]["actions"][-1],
    ]

    adapted, _ = adapt_stats_v3_game(play_by_play, stats_boxscore())
    jump_ball = canonical_events(adapted)[1]

    assert jump_ball.team_id == AWAY_TEAM_ID
    assert jump_ball.source_possession_team_id == AWAY_TEAM_ID


def test_source_loader_uses_stats_v3_when_live_data_is_missing(tmp_path) -> None:
    stats_cache = NbaStatsRawCache(tmp_path / "stats")
    write_stats(
        stats_cache,
        NbaStatsEndpoint.PLAY_BY_PLAY_V3,
        stats_play_by_play(),
    )
    write_stats(
        stats_cache,
        NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
        stats_boxscore(),
    )

    documents = load_game_source_documents(GAME_ID, raw_dir=tmp_path)

    assert documents.play_by_play.source == "stats_v3"
    assert documents.boxscore.source == "stats_v3"
    assert documents.play_by_play.payload["game"]["actions"][1]["actionType"] == "jumpball"
    assert documents.game_rotation is None


def test_game_rotation_supplies_period_start_lineups() -> None:
    play_by_play = stats_play_by_play()
    play_by_play["game"]["actions"].extend(
        [
            action(
                action_id=10,
                action_number=30,
                period=2,
                action_type="period",
                subtype="start",
                description="Start of 2nd Period",
            ),
            action(
                action_id=11,
                action_number=31,
                period=2,
                clock="PT00M00.00S",
                action_type="period",
                subtype="end",
                description="End of 2nd Period",
            ),
        ]
    )

    adapted, _ = adapt_stats_v3_game(
        play_by_play,
        stats_boxscore(),
        game_rotation_payload=rotation_payload(),
    )

    rotation_substitutions = [
        action
        for action in adapted["game"]["actions"]
        if action.get("descriptor") == "stats_v3_period_lineup"
    ]
    assert [(action["subType"], action["personId"]) for action in rotation_substitutions] == [
        ("out", 106),
        ("in", 101),
    ]


def test_source_loader_retains_game_rotation_provenance(tmp_path) -> None:
    stats_cache = NbaStatsRawCache(tmp_path / "stats")
    write_stats(stats_cache, NbaStatsEndpoint.PLAY_BY_PLAY_V3, stats_play_by_play())
    write_stats(stats_cache, NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3, stats_boxscore())
    write_stats(stats_cache, NbaStatsEndpoint.GAME_ROTATION, rotation_payload())

    documents = load_game_source_documents(GAME_ID, raw_dir=tmp_path)

    assert documents.game_rotation is not None
    assert documents.game_rotation.source == "stats_v3"
    assert len(documents.game_rotation.sha256) == 64


def test_source_loader_prefers_stats_v3_per_endpoint(tmp_path) -> None:
    stats_cache = NbaStatsRawCache(tmp_path / "stats")
    write_stats(
        stats_cache,
        NbaStatsEndpoint.PLAY_BY_PLAY_V3,
        stats_play_by_play(),
    )
    write_stats(
        stats_cache,
        NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
        stats_boxscore(),
    )
    live_payload = {
        "game": {
            "gameId": GAME_ID,
            "actions": [],
        }
    }
    RawJsonCache(tmp_path).write(
        CachedResponse(
            endpoint=NbaCdnEndpoint.PLAY_BY_PLAY,
            game_id=GAME_ID,
            url="https://cdn.nba.test/playbyplay",
            payload=live_payload,
        )
    )

    documents = load_game_source_documents(GAME_ID, raw_dir=tmp_path)

    assert documents.play_by_play.source == "stats_v3"
    assert documents.boxscore.source == "stats_v3"


def test_source_loader_expands_legacy_live_data_substitution(tmp_path) -> None:
    stats_cache = NbaStatsRawCache(tmp_path / "stats")
    write_stats(
        stats_cache,
        NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
        stats_boxscore(),
    )
    live_payload = {
        "game": {
            "gameId": GAME_ID,
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 100,
                    "period": 1,
                    "clock": "PT10M00.00S",
                    "actionType": "substitution",
                    "subType": "out",
                    "personId": 101,
                    "personIdsFilter": [101, 106],
                }
            ],
        }
    }
    RawJsonCache(tmp_path).write(
        CachedResponse(
            endpoint=NbaCdnEndpoint.PLAY_BY_PLAY,
            game_id=GAME_ID,
            url="https://cdn.nba.test/playbyplay",
            payload=live_payload,
        )
    )

    documents = load_game_source_documents(GAME_ID, raw_dir=tmp_path)

    actions = documents.play_by_play.payload["game"]["actions"]
    assert [(action["subType"], action["personId"]) for action in actions] == [
        ("out", 101),
        ("in", 106),
    ]


def stats_play_by_play() -> dict[str, Any]:
    return {
        "game": {
            "gameId": GAME_ID,
            "actions": [
                action(
                    action_id=1,
                    action_number=2,
                    action_type="period",
                    subtype="start",
                    description="Start of 1st Period",
                ),
                action(
                    action_id=2,
                    action_number=4,
                    team_id=HOME_TEAM_ID,
                    person_id=101,
                    player_name="Homeone",
                    action_type="Jump Ball",
                    description="Jump Ball Homeone vs. Awayone: Tip to Hometwo",
                ),
                action(
                    action_id=3,
                    action_number=10,
                    clock="PT11M30.00S",
                    team_id=HOME_TEAM_ID,
                    person_id=101,
                    player_name="Homeone",
                    action_type="Missed Shot",
                    subtype="Jump Shot",
                    description="MISS Homeone 12' Jump Shot",
                    is_field_goal=1,
                    shot_value=2,
                    shot_result="Missed",
                ),
                action(
                    action_id=4,
                    action_number=10,
                    clock="PT11M30.00S",
                    team_id=AWAY_TEAM_ID,
                    person_id=201,
                    player_name="Awayone",
                    action_type="",
                    description="Awayone BLOCK (1 BLK)",
                ),
                action(
                    action_id=5,
                    action_number=12,
                    clock="PT11M28.00S",
                    person_id=AWAY_TEAM_ID,
                    action_type="Rebound",
                    subtype="Unknown",
                    description="AWAY Rebound",
                ),
                action(
                    action_id=6,
                    action_number=20,
                    clock="PT10M00.00S",
                    team_id=AWAY_TEAM_ID,
                    person_id=201,
                    player_name="Awayone",
                    action_type="Turnover",
                    subtype="Bad Pass",
                    description="Awayone Bad Pass Turnover",
                ),
                action(
                    action_id=7,
                    action_number=20,
                    clock="PT10M00.00S",
                    team_id=HOME_TEAM_ID,
                    person_id=102,
                    player_name="Hometwo",
                    action_type="",
                    description="Hometwo STEAL (1 STL)",
                ),
                action(
                    action_id=8,
                    action_number=24,
                    clock="PT09M00.00S",
                    team_id=HOME_TEAM_ID,
                    person_id=101,
                    player_name="Homeone",
                    action_type="Substitution",
                    description="SUB: Homebench FOR Homeone",
                ),
                action(
                    action_id=9,
                    action_number=26,
                    clock="PT00M00.00S",
                    action_type="period",
                    subtype="end",
                    description="End of 1st Period",
                ),
            ],
        }
    }


def stats_boxscore() -> dict[str, Any]:
    return {
        "boxScoreTraditional": {
            "gameId": GAME_ID,
            "homeTeam": team(
                HOME_TEAM_ID,
                "HOM",
                [
                    (101, "Homeone"),
                    (102, "Hometwo"),
                    (103, "Homethree"),
                    (104, "Homefour"),
                    (105, "Homefive"),
                    (106, "Homebench"),
                ],
            ),
            "awayTeam": team(
                AWAY_TEAM_ID,
                "AW",
                [
                    (201, "Awayone"),
                    (202, "Awaytwo"),
                    (203, "Awaythree"),
                    (204, "Awayfour"),
                    (205, "Awayfive"),
                    (206, "Awaybench"),
                ],
            ),
        }
    }


def rotation_payload() -> dict[str, Any]:
    headers = ["TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"]
    home_rows = [
        [HOME_TEAM_ID, player_id, 0.0, 14400.0]
        for player_id in (102, 103, 104, 105)
    ] + [
        [HOME_TEAM_ID, 106, 0.0, 7200.0],
        [HOME_TEAM_ID, 101, 7200.0, 14400.0],
    ]
    away_rows = [
        [AWAY_TEAM_ID, player_id, 0.0, 14400.0]
        for player_id in (201, 202, 203, 204, 205)
    ]
    return {
        "parameters": {"GameID": GAME_ID, "LeagueID": "00"},
        "resultSets": [
            {"name": "AwayTeam", "headers": headers, "rowSet": away_rows},
            {"name": "HomeTeam", "headers": headers, "rowSet": home_rows},
        ]
    }


def team(
    team_id: int,
    tricode: str,
    player_values: list[tuple[int, str]],
) -> dict[str, Any]:
    return {
        "teamId": team_id,
        "teamTricode": tricode,
        "teamName": tricode,
        "teamCity": "Test",
        "statistics": {"minutes": "240:00", "points": 0},
        "players": [
            {
                "personId": player_id,
                "firstName": family_name,
                "familyName": family_name,
                "nameI": f"T. {family_name}",
                "playerSlug": family_name.casefold(),
                "position": "G" if index < 5 else "",
                "comment": "",
                "jerseyNum": str(index),
                "statistics": {"minutes": "30:05", "points": 0},
            }
            for index, (player_id, family_name) in enumerate(player_values)
        ],
    }


def action(
    *,
    action_id: int,
    action_number: int,
    period: int = 1,
    clock: str = "PT12M00.00S",
    team_id: int = 0,
    person_id: int = 0,
    player_name: str = "",
    action_type: str,
    subtype: str = "",
    description: str,
    is_field_goal: int = 0,
    shot_value: int = 0,
    shot_result: str = "",
    location: str = "",
) -> dict[str, Any]:
    return {
        "actionNumber": action_number,
        "actionId": action_id,
        "clock": clock,
        "period": period,
        "teamId": team_id,
        "teamTricode": "",
        "location": location,
        "personId": person_id,
        "playerName": player_name,
        "playerNameI": player_name,
        "description": description,
        "actionType": action_type,
        "subType": subtype,
        "isFieldGoal": is_field_goal,
        "shotValue": shot_value,
        "shotResult": shot_result,
        "scoreHome": "0",
        "scoreAway": "0",
    }


def write_stats(
    cache: NbaStatsRawCache,
    endpoint: NbaStatsEndpoint,
    payload: dict[str, Any],
) -> None:
    raw_body = json.dumps(payload, separators=(",", ":")).encode()
    cache.write(
        StatsCachedResponse(
            endpoint=endpoint,
            game_id=GAME_ID,
            url=f"https://stats.nba.test/{endpoint.value}",
            fetched_at=datetime(2026, 7, 30, tzinfo=UTC),
            payload=payload,
            raw_body=raw_body,
        )
    )
