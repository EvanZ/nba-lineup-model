from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import nba_lineup_model.season.daily_data_replay as replay
from nba_lineup_model.modeling.in_season_nail_monitor import (
    _rapm_stints_from_nba_boxscore,
)
from nba_lineup_model.season.daily_data_replay import (
    _load_official_game_catalog,
    _nba_actual_minutes,
    _parse_official_inactive_list,
    _official_final_game,
    _terminal_play_by_play_scores,
)


def _boxscore(*, include_dnp: bool = True) -> dict[str, object]:
    def team(team_id: int, tricode: str, score: int, player_id: int):
        players = [
            {
                "personId": player_id,
                "firstName": "Played",
                "familyName": tricode,
                "starter": "1",
                "status": "ACTIVE",
                "statistics": {"minutes": "PT30M30.00S"},
            }
        ]
        if include_dnp:
            players.append(
                {
                    "personId": player_id + 1,
                    "firstName": "Did Not",
                    "familyName": "Play",
                    "starter": "0",
                    "status": "ACTIVE",
                    "statistics": {"minutes": "PT00M00.00S"},
                }
            )
        return {
            "teamId": team_id,
            "teamTricode": tricode,
            "score": score,
            "players": players,
        }

    return {
        "game": {
            "gameId": "0022600001",
            "gameStatus": 3,
            "gameStatusText": "Final",
            "homeTeam": team(1610612765, "DET", 110, 1001),
            "awayTeam": team(1610612738, "BOS", 105, 2001),
        }
    }


def _catalog_game() -> dict[str, object]:
    return {
        "game_id": "0022600001",
        "season": "2026-27",
        "game_time_utc": pd.Timestamp("2026-10-20T23:00:00Z"),
        "home_team_tricode": "DET",
        "away_team_tricode": "BOS",
    }


def test_nba_person_ids_resolve_played_and_dnp_players_without_name_matching() -> None:
    game_record = _official_final_game(
        catalog_game=_catalog_game(),
        boxscore=_boxscore(),
        game_date_pacific=date(2026, 10, 20),
    )

    actual, identity, roster = _nba_actual_minutes(
        game_record=game_record,
        boxscore=_boxscore(),
    )

    assert set(actual["player_id"]) == {1001, 2001}
    assert set(identity["player_id"]) == {1001, 1002, 2001, 2002}
    assert set(identity["resolution"]) == {"nba_person_id"}
    assert identity.loc[identity["player_id"].eq(1002), "played"].item() is False
    assert roster["teams"][0]["players"][1]["status"] == "ACTIVE"


def test_nba_actual_minutes_rejects_a_missing_canonical_person_id() -> None:
    boxscore = _boxscore(include_dnp=False)
    boxscore["game"]["homeTeam"]["players"][0].pop("personId")
    game_record = {
        "game_id": "0022600001",
        "game_date_pacific": "2026-10-20",
    }

    with pytest.raises(ValueError, match="malformed DET player identity"):
        _nba_actual_minutes(game_record=game_record, boxscore=boxscore)


def test_official_inactives_are_added_to_the_canonical_roster() -> None:
    game_record = _official_final_game(
        catalog_game=_catalog_game(),
        boxscore=_boxscore(),
        game_date_pacific=date(2026, 10, 20),
    )
    inactive = {
        "game_id": "0022600001",
        "teams": [
            {
                "team_id": 1610612765,
                "team": "DET",
                "side": "home",
                "players": [{"player_id": 3001, "player_name": "Inactive Player"}],
            },
            {
                "team_id": 1610612738,
                "team": "BOS",
                "side": "away",
                "players": [],
            },
        ],
    }

    actual, identity, roster = _nba_actual_minutes(
        game_record=game_record,
        boxscore=_boxscore(),
        inactive_list=inactive,
    )

    assert 3001 not in set(actual["player_id"])
    row = identity.loc[identity["player_id"].eq(3001)].iloc[0]
    assert row["resolution"] == "nba_inactive_person_id"
    player = roster["teams"][0]["players"][-1]
    assert player == {
        "player_id": 3001,
        "player_name": "Inactive Player",
        "played": False,
        "starter": False,
        "status": "INACTIVE",
        "minutes": 0.0,
        "source": "official_nba_inactive_list",
    }


def test_zero_minute_box_player_may_also_appear_on_official_inactive_list() -> None:
    game_record = _official_final_game(
        catalog_game=_catalog_game(),
        boxscore=_boxscore(),
        game_date_pacific=date(2026, 10, 20),
    )
    inactive = {
        "game_id": "0022600001",
        "teams": [
            {
                "team_id": 1610612765,
                "team": "DET",
                "side": "home",
                "players": [{"player_id": 1002, "player_name": "Did Not Play"}],
            },
            {
                "team_id": 1610612738,
                "team": "BOS",
                "side": "away",
                "players": [],
            },
        ],
    }

    _actual, identity, roster = _nba_actual_minutes(
        game_record=game_record,
        boxscore=_boxscore(),
        inactive_list=inactive,
    )

    assert identity.loc[identity["player_id"].eq(1002), "resolution"].item() == (
        "nba_person_id_and_inactive_person_id"
    )
    player = next(
        row for row in roster["teams"][0]["players"] if row["player_id"] == 1002
    )
    assert player["status"] == "INACTIVE"
    assert player["source"] == "official_nba_box_and_inactive_list"


def test_official_inactive_page_parser_validates_game_and_teams() -> None:
    payload = {
        "props": {
            "pageProps": {
                "game": {
                    "gameId": "0022600001",
                    "homeTeam": {
                        "teamId": 1610612765,
                        "teamTricode": "DET",
                        "inactives": [
                            {
                                "personId": 3001,
                                "firstName": "Inactive",
                                "familyName": "Player",
                            }
                        ],
                    },
                    "awayTeam": {
                        "teamId": 1610612738,
                        "teamTricode": "BOS",
                        "inactives": [],
                    },
                }
            }
        }
    }
    raw = (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(payload)
        + "</script></html>"
    ).encode()

    parsed = _parse_official_inactive_list(
        raw,
        expected_game_id="0022600001",
        expected_home="DET",
        expected_away="BOS",
    )

    assert parsed["teams"][0]["players"] == [
        {"player_id": 3001, "player_name": "Inactive Player"}
    ]
    with pytest.raises(ValueError, match="does not match"):
        _parse_official_inactive_list(
            raw,
            expected_game_id="0022609999",
            expected_home="DET",
            expected_away="BOS",
        )


def test_terminal_play_by_play_score_uses_the_last_scored_action() -> None:
    payload = {
        "game": {
            "actions": [
                {
                    "actionNumber": 1,
                    "period": 1,
                    "orderNumber": 10000,
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
                {"actionNumber": 2},
                {
                    "actionNumber": 3,
                    "period": 4,
                    "orderNumber": 30000,
                    "scoreHome": "110",
                    "scoreAway": "105",
                },
                {"actionNumber": 4},
                {
                    "actionNumber": 2,
                    "period": 1,
                    "orderNumber": 20000,
                    "scoreHome": "2",
                    "scoreAway": "0",
                },
            ]
        }
    }

    assert _terminal_play_by_play_scores(payload) == (110, 105)


def test_nba_boxscore_starters_drive_lineup_reconstruction() -> None:
    fixture_root = Path(__file__).parent / "fixtures"
    boxscore = json.loads((fixture_root / "boxscore_lineup_scenario.json").read_text())
    play_by_play = json.loads(
        (fixture_root / "playbyplay_lineup_scenario.json").read_text()
    )
    boxscore["game"]["homeTeam"]["score"] = 2
    boxscore["game"]["awayTeam"]["score"] = 2

    stints, team_ids, _issue_count = _rapm_stints_from_nba_boxscore(
        boxscore=boxscore,
        play_by_play=play_by_play,
        play_by_play_sha256="a" * 64,
        boxscore_sha256="b" * 64,
        game_date="2026-10-20",
        game_time_utc="2026-10-20T23:00:00Z",
    )

    assert team_ids == {"HOM": 100, "AWY": 200}
    assert not stints.empty
    assert stints["boxscore_sha256"].eq("b" * 64).all()


def test_official_catalog_assigns_games_by_pacific_tip_date(tmp_path: Path) -> None:
    path = tmp_path / "games.parquet"
    pd.DataFrame(
        [
            {
                "game_id": "0022600001",
                "season": "2026-27",
                "season_type": "regular",
                "game_time_utc": pd.Timestamp("2026-10-21T05:30:00Z"),
                "home_team_id": 1,
                "home_team_tricode": "LAL",
                "away_team_id": 2,
                "away_team_tricode": "GSW",
                "source_url": "https://stats.nba.com/stats/scheduleleaguev2",
            }
        ]
    ).to_parquet(path, index=False)

    grouped = _load_official_game_catalog(
        path,
        season="2026-27",
        start_date=date(2026, 10, 20),
        end_date=date(2026, 10, 20),
    )

    assert [row["game_id"] for row in grouped[date(2026, 10, 20)]] == [
        "0022600001"
    ]


def test_official_injury_report_is_content_addressed(monkeypatch) -> None:
    stored: dict[str, bytes] = {}

    class Store:
        def put_bytes(self, uri: str, payload: bytes) -> bool:
            stored[uri] = payload
            return True

    monkeypatch.setattr(
        replay,
        "_fetch_latest_official_injury_report",
        lambda **_kwargs: (
            {
                "source_url": (
                    "https://ak-static.cms.nba.com/referee/injury/"
                    "Injury-Report_2026-10-20_11_45PM.pdf"
                ),
                "game_team_reports": {},
            },
            b"official-nba-pdf",
        ),
    )
    loader = replay._official_injury_report_loader(
        store=Store(),
        season="2026-27",
        s3_prefix="s3://test/lake/raw/nba/injury-reports",
    )

    report = loader(date(2026, 10, 20))

    assert report.immutable_uri.endswith(f"/{report.sha256}.pdf")
    assert stored == {report.immutable_uri: b"official-nba-pdf"}
