from __future__ import annotations

import json

import pandas as pd

from nba_lineup_model.players.availability import (
    STATE_AVAILABLE_DNP_ACTIVE,
    STATE_AVAILABLE_DNP_COACH,
    STATE_AVAILABLE_G_LEAGUE_ASSIGNMENT,
    STATE_PLAYED,
    STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED,
    STATE_UNAVAILABLE_INELIGIBLE,
    STATE_UNAVAILABLE_INJURY,
    STATE_UNAVAILABLE_PERSONAL,
    STATE_UNAVAILABLE_REST,
    STATE_UNAVAILABLE_SUSPENSION,
    STATE_UNKNOWN_DNP,
    _correct_isolated_coach_dnp_with_medical_neighbors,
    availability_label,
    availability_rows_from_payload,
    availability_rows_from_stats_v3_and_summary,
    build_player_availability_mart,
    classify_availability_state,
)


def test_classifies_structured_modern_reason_states() -> None:
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="INACTIVE",
            raw_reason="INACTIVE_COACH",
            raw_comment=None,
        )
        == STATE_AVAILABLE_DNP_COACH
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="INACTIVE",
            raw_reason="INACTIVE_INJURY",
            raw_comment=None,
        )
        == STATE_UNAVAILABLE_INJURY
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="INACTIVE",
            raw_reason="INACTIVE_REST",
            raw_comment=None,
        )
        == STATE_UNAVAILABLE_REST
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="INACTIVE",
            raw_reason="INACTIVE_GLEAGUE_TWOWAY",
            raw_comment=None,
        )
        == STATE_AVAILABLE_G_LEAGUE_ASSIGNMENT
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="INACTIVE",
            raw_reason="INACTIVE_LEAGUE_SUSPENSION",
            raw_comment=None,
        )
        == STATE_UNAVAILABLE_SUSPENSION
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="INACTIVE",
            raw_reason="INELIGIBLE_TO_PLAY",
            raw_comment=None,
        )
        == STATE_UNAVAILABLE_INELIGIBLE
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=False,
            raw_status="ACTIVE",
            raw_reason=None,
            raw_comment=None,
        )
        == STATE_AVAILABLE_DNP_ACTIVE
    )


def test_classifies_historical_comment_states_conservatively() -> None:
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment="DNP - Coach's Decision",
        )
        == STATE_AVAILABLE_DNP_COACH
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment="DND - Sprained Left Ankle",
        )
        == STATE_UNAVAILABLE_INJURY
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment="NWT - Personal Reasons",
        )
        == STATE_UNAVAILABLE_PERSONAL
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment="NWT - Left Knee Surgery",
        )
        == STATE_UNAVAILABLE_INJURY
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment="NWT",
        )
        == STATE_UNKNOWN_DNP
    )
    assert (
        classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment=None,
        )
        == STATE_UNKNOWN_DNP
    )


def test_played_state_wins_over_any_raw_reason() -> None:
    assert (
        classify_availability_state(
            minutes=12.5,
            raw_played=True,
            raw_status="ACTIVE",
            raw_reason=None,
            raw_comment=None,
        )
        == STATE_PLAYED
    )


def test_corrects_single_coach_dnp_bracketed_by_medical_absences() -> None:
    games = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 1],
            "player_id": [7, 7, 7, 8],
            "game_id": ["a", "b", "c", "b"],
            "game_date": pd.to_datetime(
                ["2026-03-06", "2026-03-08", "2026-03-10", "2026-03-08"], utc=True
            ),
            "availability_state": [
                STATE_UNAVAILABLE_INJURY,
                STATE_AVAILABLE_DNP_COACH,
                STATE_UNAVAILABLE_INJURY,
                STATE_AVAILABLE_DNP_COACH,
            ],
            "available": [False, True, False, True],
        }
    )

    corrected = _correct_isolated_coach_dnp_with_medical_neighbors(games)

    assert corrected.loc[1, "availability_state"] == STATE_UNAVAILABLE_INJURY
    assert not corrected.loc[1, "available"]
    assert corrected.loc[3, "availability_state"] == STATE_AVAILABLE_DNP_COACH
    assert corrected.loc[3, "available"]


def test_gamebook_reason_wording_maps_to_binary_target() -> None:
    examples = {
        "DNP - Coach's Decision": True,
        "G League - Two-Way": True,
        "Injury/Illness - Right Second Toe: Sprain": False,
        "Rest": False,
        "Personal": False,
        "Not With Team": False,
        "Team Suspension - Team Suspension": False,
        "Ineligible To Play": False,
        None: None,
    }
    for raw_comment, expected in examples.items():
        state = classify_availability_state(
            minutes=0.0,
            raw_played=None,
            raw_status=None,
            raw_reason=None,
            raw_comment=raw_comment,
        )
        assert availability_label(state) is expected


def test_explicit_historical_inactive_without_reason_is_unavailable() -> None:
    state = classify_availability_state(
        minutes=0.0,
        raw_played=None,
        raw_status=None,
        raw_reason=None,
        raw_comment=None,
        historically_inactive=True,
    )
    assert state == STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED
    assert availability_label(state) is False


def test_blank_historical_boxscore_row_is_not_an_inactive_designation() -> None:
    state = classify_availability_state(
        minutes=0.0,
        raw_played=None,
        raw_status=None,
        raw_reason=None,
        raw_comment=None,
    )
    assert state == STATE_UNKNOWN_DNP
    assert availability_label(state) is None


def test_normalizes_live_and_stats_v3_player_rows() -> None:
    live = {
        "game": {
            "homeTeam": {
                "teamId": 1,
                "teamTricode": "MIN",
                "players": [
                    {
                        "personId": 7,
                        "name": "Coach DNP",
                        "played": 0,
                        "status": "INACTIVE",
                        "notPlayingReason": "INACTIVE_COACH",
                        "statistics": {"minutes": "PT00M00.00S"},
                    }
                ],
            },
            "awayTeam": {"teamId": 2, "teamTricode": "HOU", "players": []},
        }
    }
    stats = {
        "boxScoreTraditional": {
            "homeTeam": {
                "teamId": 1,
                "teamTricode": "MIN",
                "players": [
                    {
                        "personId": 8,
                        "firstName": "Injured",
                        "familyName": "Player",
                        "comment": "DND - Sprained Left Ankle",
                        "statistics": {"minutes": ""},
                    }
                ],
            },
            "awayTeam": {"teamId": 2, "teamTricode": "HOU", "players": []},
        }
    }
    game_date = pd.Timestamp("2025-10-21", tz="UTC")
    live_frame = availability_rows_from_payload(
        live,
        source_kind="live_boxscore",
        season="2025-26",
        game_id="0022500001",
        game_date=game_date,
        source_path="live.json",
    )
    stats_frame = availability_rows_from_payload(
        stats,
        source_kind="stats_v3",
        season="2025-26",
        game_id="0022500002",
        game_date=game_date,
        source_path="stats.json",
    )
    assert live_frame.loc[0, "availability_state"] == STATE_AVAILABLE_DNP_COACH
    assert live_frame.loc[0, "raw_not_playing_reason"] == "INACTIVE_COACH"
    assert live_frame.loc[0, "source_player_table_contract"] == "full_roster_status"
    assert stats_frame.loc[0, "availability_state"] == STATE_UNAVAILABLE_INJURY
    assert stats_frame.loc[0, "player_name"] == "Injured Player"
    assert stats_frame.loc[0, "source_player_table_contract"] == "boxscore_listed_players"


def test_merges_summary_inactive_membership_into_historical_panel() -> None:
    stats = {
        "boxScoreTraditional": {
            "homeTeam": {
                "teamId": 1,
                "teamTricode": "MIN",
                "players": [
                    {
                        "personId": 7,
                        "firstName": "Coach",
                        "familyName": "DNP",
                        "comment": "DNP - Coach's Decision",
                        "statistics": {"minutes": ""},
                    }
                ],
            },
            "awayTeam": {"teamId": 2, "teamTricode": "HOU", "players": []},
        }
    }
    summary = {
        "resultSets": [
            {
                "name": "InactivePlayers",
                "headers": [
                    "PLAYER_ID",
                    "FIRST_NAME",
                    "LAST_NAME",
                    "TEAM_ID",
                    "TEAM_ABBREVIATION",
                ],
                "rowSet": [[9, "Inactive", "Player", 1, "MIN"]],
            }
        ]
    }

    frame = availability_rows_from_stats_v3_and_summary(
        stats,
        summary,
        season="2021-22",
        game_id="0022100001",
        game_date=pd.Timestamp("2021-10-19", tz="UTC"),
        stats_path="traditional.json",
        summary_path="summary.json",
    ).set_index("player_id")

    assert set(frame.index) == {7, 9}
    assert frame.loc[7, "available"]
    assert not frame.loc[9, "available"]
    assert frame.loc[9, "availability_state"] == STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED
    assert frame.loc[9, "source_player_table_contract"] == "full_roster_membership"


def test_builds_season_mart_with_live_source_preferred(tmp_path) -> None:
    raw = tmp_path / "raw"
    schedule = {
        "leagueSchedule": {
            "gameDates": [
                {"games": [{"gameId": "0022500001", "gameDateEst": "2025-10-21T00:00:00Z"}]}
            ]
        }
    }
    schedule_path = raw / "scheduleleaguev2" / "2025-26.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(json.dumps(schedule))
    payload = {
        "game": {
            "homeTeam": {
                "teamId": 1,
                "teamTricode": "MIN",
                "players": [
                    {
                        "personId": 7,
                        "name": "Coach DNP",
                        "played": 0,
                        "status": "INACTIVE",
                        "notPlayingReason": "INACTIVE_COACH",
                        "statistics": {"minutes": "PT00M00.00S"},
                    }
                ],
            },
            "awayTeam": {
                "teamId": 2,
                "teamTricode": "HOU",
                "players": [
                    {
                        "personId": 8,
                        "name": "Available Player",
                        "played": 1,
                        "status": "ACTIVE",
                        "statistics": {"minutes": "PT12M00.00S"},
                    }
                ],
            },
        }
    }
    source = raw / "boxscore" / "0022500001.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(payload))

    output = build_player_availability_mart(
        "2025-26", raw_dir=raw, curated_dir=tmp_path / "curated"
    )

    mart = pd.read_parquet(output)
    coverage = pd.read_parquet(output.parent / "coverage.parquet")
    assert set(mart["availability_state"]) == {STATE_AVAILABLE_DNP_COACH, STATE_PLAYED}
    assert mart.set_index("player_id").loc[7, "available"]
    assert mart.set_index("player_id").loc[8, "available"]
    assert coverage.loc[0, "live_boxscore_games"] == 1
    assert coverage.loc[0, "missing_source_games"] == 0
