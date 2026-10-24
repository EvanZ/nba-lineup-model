from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import nba_lineup_model.web_api.win_projections as win_projections
from nba_lineup_model.web_api.market_win_totals import BETMGM_WIN_TOTALS_2026_27
from nba_lineup_model.web_api.win_projections import (
    _aws_s3_cp_exclusive,
    _build_final_game_archive,
    _completed_team_win_totals,
    _default_live_score_date,
    _espn_team,
    _fetch_polymarket_moneyline_snapshot,
    _game_rotation,
    _parse_official_injury_report,
    _team_win_totals,
    fit_win_probability_scale,
    read_current_season_records,
    read_schedule_day,
    read_schedule_performance,
    score_season_win_forecasts,
    score_win_probabilities,
    simulate_baseline_win_total_intervals,
)


def test_current_season_records_use_immutable_regular_season_finals(tmp_path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    games = [
        ("401", "0022600001", "2026-10-21T02:00:00Z", "BOS", "DET", 119, 121),
        ("402", "0022600002", "2026-10-23T02:30:00Z", "DEN", "OKC", 84, 96),
        ("403", "0012600001", "2026-10-10T02:00:00Z", "LAL", "GSW", 90, 100),
    ]
    for espn_id, game_id, start_time, away, home, away_score, home_score in games:
        (final_dir / f"{espn_id}.summary.json").write_text(
            json.dumps(
                {
                    "season": "2026-27",
                    "game_id": game_id,
                    "archived_at": "2026-10-23T05:00:00Z",
                    "start_time": start_time,
                    "away_team": away,
                    "home_team": home,
                    "final": {
                        "away_score": away_score,
                        "home_score": home_score,
                        "status": {"completed": True},
                    },
                }
            )
        )

    payload = read_current_season_records(state_dir=tmp_path)

    assert payload["as_of"] == "2026-10-22"
    assert payload["teams"] == [
        {"team": "BOS", "games": 1, "wins": 0, "losses": 1, "last_game_date": "2026-10-20"},
        {"team": "DEN", "games": 1, "wins": 0, "losses": 1, "last_game_date": "2026-10-22"},
        {"team": "DET", "games": 1, "wins": 1, "losses": 0, "last_game_date": "2026-10-20"},
        {"team": "OKC", "games": 1, "wins": 1, "losses": 0, "last_game_date": "2026-10-22"},
    ]


def test_schedule_performance_scores_rolling_matched_markets(tmp_path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    games = [
        ("2026-10-23T23:00:00Z", 10.0, 6.0, 0.7, -4.0, -300, 300, 0.6, "win"),
        ("2026-10-20T23:00:00Z", -5.0, 2.0, 0.6, -1.0, 300, -300, 0.55, "loss"),
        ("2026-10-10T23:00:00Z", 3.0, 4.0, 0.55, 2.0, -120, 100, 0.52, "push"),
        ("2026-09-10T23:00:00Z", -2.0, -1.0, 0.45, -3.0, 110, -130, 0.4, "win"),
    ]
    for index, (
        start_time,
        actual_margin,
        model_margin,
        model_probability,
        market_margin,
        home_moneyline,
        away_moneyline,
        polymarket_probability,
        ats_result,
    ) in enumerate(games, start=1):
        (final_dir / f"{index}.summary.json").write_text(
            json.dumps(
                {
                    "season": "2026-27",
                    "game_id": f"002260000{index}",
                    "archived_at": "2026-10-24T05:00:00Z",
                    "start_time": start_time,
                    "final": {
                        "actual_home_margin": actual_margin,
                        "status": {"completed": True},
                    },
                    "grading": {
                        "projected_home_margin": model_margin,
                        "projected_home_win_probability": model_probability,
                        "market": {
                            "market_implied_home_margin": market_margin,
                            "model_ats_result": ats_result,
                        },
                    },
                    "pregame": {
                        "market": {
                            "home_moneyline": home_moneyline,
                            "away_moneyline": away_moneyline,
                        },
                        "prediction_market": {
                            "home_probability": polymarket_probability,
                        },
                    },
                    "retrospective": {
                        "home_margin": [8.0, -3.0, 2.0, 1.0][index - 1],
                        "home_win_probability": [0.8, 0.3, 0.6, 0.55][index - 1],
                    },
                }
            )
        )

    payload = read_schedule_performance(state_dir=tmp_path)

    assert payload["as_of"] == "2026-10-23"
    seven_days, thirty_days, season = payload["windows"]
    assert [seven_days["games"], thirty_days["games"], season["games"]] == [2, 3, 4]
    assert seven_days["probability"]["n"] == 2
    assert seven_days["probability"]["gestalt"]["accuracy"] == pytest.approx(0.5)
    assert seven_days["probability"]["gestalt"]["brier"] == pytest.approx(0.225)
    assert seven_days["probability"]["draftkings"]["accuracy"] == pytest.approx(1.0)
    assert seven_days["probability"]["draftkings"]["brier"] == pytest.approx(0.0625)
    assert seven_days["probability"]["polymarket"]["brier"] == pytest.approx(0.23125)
    assert seven_days["probability"]["actual_minutes"] == {
        "n": 2,
        "accuracy": pytest.approx(1.0),
        "brier": pytest.approx(0.065),
    }
    assert seven_days["spread"] == {
        "n": 2,
        "gestalt_mae": pytest.approx(5.5),
        "draftkings_mae": pytest.approx(9.0),
        "gestalt_advantage": pytest.approx(3.5),
        "actual_minutes_n": 2,
        "actual_minutes_mae": pytest.approx(2.0),
    }
    assert seven_days["ats"]["cover_rate"] == pytest.approx(0.5)
    assert seven_days["ats"]["units_at_minus_110"] == pytest.approx(-1.0 / 11.0)


def test_polymarket_moneyline_snapshot_uses_main_market_midpoints(monkeypatch) -> None:
    token_ids = ["away-token", "home-token"]
    event_slug = "nba-gsw-lal-2026-10-21"
    requested_urls: list[str] = []
    midpoint_requests: list[dict[str, str]] = []

    def fake_fetch_json(url, *, timeout=20):
        assert timeout == 5
        requested_urls.append(url)
        return {
            "id": "event-1",
            "slug": event_slug,
            "markets": [
                {
                    "id": "prop-1",
                    "slug": f"{event_slug}-team-to-score-first",
                    "outcomes": '["Warriors", "Lakers"]',
                    "clobTokenIds": '["wrong-1", "wrong-2"]',
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "moneyline-1",
                    "slug": event_slug,
                    "version": "v1",
                    "outcomes": '["Warriors", "Lakers"]',
                    "outcomePrices": '["0.35", "0.65"]',
                    "clobTokenIds": '["away-token", "home-token"]',
                    "volumeNum": 45978.94,
                    "liquidityNum": 64259.13,
                    "spread": 0.01,
                    "active": True,
                    "closed": False,
                },
            ],
        }

    def fake_post_json(url, payload):
        assert url == win_projections.POLYMARKET_MIDPOINTS_URL
        midpoint_requests.extend(payload)
        return {"away-token": "0.345", "home-token": "0.655"}

    monkeypatch.setattr(win_projections, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(win_projections, "_post_json", fake_post_json)
    espn_payload = {
        "events": [
            {
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "away",
                                "team": {"abbreviation": "GS"},
                            },
                            {
                                "homeAway": "home",
                                "team": {"abbreviation": "LAL"},
                            },
                        ]
                    }
                ]
            }
        ]
    }

    snapshot = _fetch_polymarket_moneyline_snapshot(
        espn_payload,
        game_date="2026-10-21",
        collected_at=datetime(2026, 10, 20, 23, 0, tzinfo=UTC),
    )

    assert requested_urls == [
        f"{win_projections.POLYMARKET_EVENT_URL}/{event_slug}"
    ]
    assert midpoint_requests == [{"token_id": token_id} for token_id in token_ids]
    assert snapshot["available_game_count"] == 1
    market = snapshot["games"]["GSW-LAL"]
    assert market["market_id"] == "moneyline-1"
    assert market["pricing_method"] == "CLOB midpoint"
    assert market["away_probability"] == pytest.approx(0.345)
    assert market["home_probability"] == pytest.approx(0.655)
    assert market["raw_probability_sum"] == pytest.approx(1.0)
    assert market["url"].endswith(event_slug)


def test_schedule_day_attaches_immutable_final_audit(tmp_path) -> None:
    latest = tmp_path / "latest" / "2026-10-20.json"
    frozen = tmp_path / "frozen" / "401909088.json"
    final = tmp_path / "final" / "401909088.summary.json"
    for path in (latest, frozen, final):
        path.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        '{"games":[{"espn_game_id":"401909088","status":{"state":"post"}}]}'
    )
    frozen.write_text('{"projection":{"home_margin":4.5}}')
    final.write_text(
        '{"grading":{"model_favorite_won":true,'
        '"market":{"model_ats_result":"win"}}}'
    )

    payload = read_schedule_day("2026-10-20", state_dir=tmp_path)

    assert payload["games"][0]["pregame"]["projection"]["home_margin"] == 4.5
    assert payload["games"][0]["final_audit"]["grading"]["model_favorite_won"] is True
    assert (
        payload["games"][0]["final_audit"]["grading"]["market"]["model_ats_result"]
        == "win"
    )


def test_live_score_default_keeps_watching_unfinished_previous_date(tmp_path) -> None:
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "2026-10-22.json").write_text(
        '{"games":[{"start_time":"2026-10-23T01:30:00Z",'
        '"status":{"state":"in"}}]}'
    )
    (latest / "2026-10-23.json").write_text(
        '{"games":[{"start_time":"2026-10-23T23:00:00Z",'
        '"status":{"state":"pre"}}]}'
    )

    game_date = _default_live_score_date(
        state_dir=tmp_path,
        now=datetime(2026, 10, 23, 4, 23, tzinfo=UTC),
    )

    assert game_date == "2026-10-22"


def test_live_score_default_advances_after_previous_date_is_final(tmp_path) -> None:
    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "2026-10-22.json").write_text(
        '{"games":[{"start_time":"2026-10-23T01:30:00Z",'
        '"status":{"state":"post"}}]}'
    )
    (latest / "2026-10-23.json").write_text(
        '{"games":[{"start_time":"2026-10-23T23:00:00Z",'
        '"status":{"state":"pre"}}]}'
    )

    game_date = _default_live_score_date(
        state_dir=tmp_path,
        now=datetime(2026, 10, 23, 4, 23, tzinfo=UTC),
    )

    assert game_date == "2026-10-23"


def test_injury_pdf_preserves_team_reports_that_are_not_submitted(monkeypatch) -> None:
    rows = [
        [
            (201.0, "GSW@LAL"),
            (265.2, "Golden State Warriors"),
            (667.1, "NOT"),
            (687.3, "YET"),
            (704.2, "SUBMITTED"),
        ],
        [
            (265.2, "Los Angeles Lakers"),
            (667.1, "NOT"),
            (687.3, "YET"),
            (704.2, "SUBMITTED"),
        ],
        [
            (201.0, "ATL@ORL"),
            (265.2, "Atlanta Hawks"),
            (426.0, "Young,"),
            (470.0, "Trae"),
            (586.7, "Out"),
            (667.1, "Injury/Illness"),
        ],
        [(265.2, "Orlando Magic"), (667.1, "No injuries to report")],
    ]

    class FakePage:
        def extract_text(self, visitor_text):
            for y, row in enumerate(rows, start=100):
                for x, value in row:
                    visitor_text(
                        value,
                        None,
                        [1.0, 0.0, 0.0, 1.0, x, float(y)],
                        None,
                        9.0,
                    )

    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _source: SimpleNamespace(pages=[FakePage()]),
    )

    report = _parse_official_injury_report(b"fake pdf")

    assert report["team_reports"]["GSW"]["status"] == "not_yet_submitted"
    assert report["team_reports"]["LAL"]["status"] == "not_yet_submitted"
    assert report["team_reports"]["ATL"]["status"] == "submitted"
    assert report["team_reports"]["ORL"]["status"] == "submitted"
    assert report["players"][0]["player_name"] == "Trae Young"


def test_injury_pdf_scopes_repeated_team_reports_by_date_and_matchup(monkeypatch) -> None:
    rows = [
        [
            (24.1, "10/23/2026"),
            (201.0, "LAC@LAL"),
            (265.2, "LA"),
            (277.5, "Clippers"),
            (426.0, "Beal,"),
            (448.2, "Bradley"),
            (586.7, "Out"),
            (667.1, "Injury/Illness"),
        ],
        [(265.2, "Los"), (280.9, "Angeles"), (315.1, "Lakers"), (667.1, "No injuries to report")],
        [
            (24.1, "10/24/2026"),
            (201.0, "GSW@LAC"),
            (265.2, "Golden State Warriors"),
            (667.1, "NOT YET SUBMITTED"),
        ],
        [(265.2, "LA Clippers"), (667.1, "NOT YET SUBMITTED")],
    ]

    class FakePage:
        def extract_text(self, visitor_text):
            for y, row in enumerate(rows, start=100):
                for x, value in row:
                    visitor_text(
                        value,
                        None,
                        [1.0, 0.0, 0.0, 1.0, x, float(y)],
                        None,
                        9.0,
                    )

    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _source: SimpleNamespace(pages=[FakePage()]),
    )

    report = _parse_official_injury_report(b"fake pdf")

    assert report["team_reports"]["LAC"]["status"] == "submitted"
    assert report["team_reports"]["LAC"]["matchup"] == "LAC@LAL"
    assert report["game_team_reports"]["2026-10-23|LAC@LAL"]["LAC"]["status"] == "submitted"
    assert report["game_team_reports"]["2026-10-24|GSW@LAC"]["LAC"]["status"] == "not_yet_submitted"
    assert report["players"][0]["game_date"] == "2026-10-23"
    assert report["players"][0]["matchup"] == "LAC@LAL"
    assert report["players"][0]["team"] == "LAC"


def test_injury_pdf_treats_omitted_zero_injury_team_as_submitted(monkeypatch) -> None:
    rows = [
        [
            (24.1, "10/23/2026"),
            (201.0, "CHI@MIL"),
            (265.2, "Chicago Bulls"),
            (426.0, "Claxton,"),
            (470.0, "Nic"),
            (586.7, "Out"),
            (667.1, "Injury/Illness"),
        ],
        [
            (201.0, "ATL@CHA"),
            (265.2, "Atlanta Hawks"),
            (426.0, "Young,"),
            (470.0, "Trae"),
            (586.7, "Available"),
            (667.1, "-"),
        ],
    ]

    class FakePage:
        def extract_text(self, visitor_text):
            for y, row in enumerate(rows, start=100):
                for x, value in row:
                    visitor_text(
                        value,
                        None,
                        [1.0, 0.0, 0.0, 1.0, x, float(y)],
                        None,
                        9.0,
                    )

    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _source: SimpleNamespace(pages=[FakePage()]),
    )

    report = _parse_official_injury_report(b"fake pdf")

    game_reports = report["game_team_reports"]["2026-10-23|CHI@MIL"]
    assert game_reports["CHI"]["status"] == "submitted"
    assert game_reports["MIL"]["status"] == "submitted"


def test_injury_pdf_parses_second_team_after_continuation_page_header(monkeypatch) -> None:
    pages = [
        [
            (
                100.0,
                [
                    (201.0, "ATL@ORL"),
                    (265.2, "Atlanta Hawks"),
                    (426.0, "Young,"),
                    (470.0, "Trae"),
                    (586.7, "Out"),
                    (667.1, "Injury/Illness"),
                ],
            ),
            (110.0, [(265.2, "Orlando Magic"), (667.1, "No injuries to report")]),
            (
                120.0,
                [
                    (201.0, "DEN@OKC"),
                    (265.2, "Denver Nuggets"),
                    (667.1, "NOT"),
                    (687.3, "YET"),
                    (704.2, "SUBMITTED"),
                ],
            ),
        ],
        [
            (
                62.0,
                [
                    (289.1, "Injury"),
                    (355.6, "Report:"),
                    (438.3, "10/21/26"),
                ],
            ),
            (
                90.9,
                [
                    (265.2, "Oklahoma"),
                    (309.1, "City"),
                    (326.8, "Thunder"),
                    (667.1, "NOT"),
                    (687.3, "YET"),
                    (704.2, "SUBMITTED"),
                ],
            ),
        ],
    ]

    class FakePage:
        def __init__(self, rows):
            self.rows = rows

        def extract_text(self, visitor_text):
            for y, row in self.rows:
                for x, value in row:
                    visitor_text(
                        value,
                        None,
                        [1.0, 0.0, 0.0, 1.0, x, y],
                        None,
                        9.0,
                    )

    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _source: SimpleNamespace(pages=[FakePage(rows) for rows in pages]),
    )

    report = _parse_official_injury_report(b"fake pdf")

    assert report["team_reports"]["DEN"]["status"] == "not_yet_submitted"
    assert report["team_reports"]["OKC"]["status"] == "not_yet_submitted"


def test_injury_pdf_parses_suffixes_before_name_delimiter(monkeypatch) -> None:
    names = [
        ("Hardaway", "Jr.,", "Tim", "Tim Hardaway Jr."),
        ("Nance", "Sr.,", "Larry", "Larry Nance Sr."),
        ("Lively", "II,", "Dereck", "Dereck Lively II"),
        ("Butler", "III,", "Jimmy", "Jimmy Butler III"),
        ("Walker", "IV,", "Lonnie", "Lonnie Walker IV"),
        ("Brown", "V,", "Charlie", "Charlie Brown V"),
    ]
    rows = [
        [
            (426.0, last_name),
            (453.0, suffix),
            (465.4, first_name),
            (586.7, "Out"),
            (667.1, "Injury/Illness"),
            (721.8, "-"),
            (727.1, "Surgery"),
        ]
        for last_name, suffix, first_name, _expected in names
    ]

    class FakePage:
        def extract_text(self, visitor_text):
            for y, row in enumerate(rows, start=100):
                for x, value in row:
                    visitor_text(value, None, [1.0, 0.0, 0.0, 1.0, x, float(y)], None, 9.0)

    monkeypatch.setattr("pypdf.PdfReader", lambda _source: SimpleNamespace(pages=[FakePage()]))

    report = _parse_official_injury_report(b"fake pdf")

    assert [row["player_name"] for row in report["players"]] == [
        expected for _last, _suffix, _first, expected in names
    ]
    assert all(row["status"] == "Out" for row in report["players"])


def test_injury_pdf_groups_wrapped_and_page_continuation_reasons(monkeypatch) -> None:
    pages = [
        [
            (
                500.0,
                [
                    (426.0, "Bassey,"),
                    (465.4, "Charles"),
                    (586.7, "Questionable"),
                    (667.1, "Injury/Illness"),
                    (721.8, "-"),
                    (727.1, "Right Patellar"),
                ],
            )
        ],
        [
            (88.9, [(667.1, "Tendinopathy")]),
            (
                111.0,
                [
                    (667.1, "Injury/Illness"),
                    (721.8, "-"),
                    (727.1, "Right ACL Repair /"),
                ],
            ),
            (
                118.0,
                [
                    (426.0, "Butler"),
                    (453.0, "III,"),
                    (465.4, "Jimmy"),
                    (586.7, "Out"),
                ],
            ),
            (125.0, [(667.1, "Surgery")]),
            (
                147.0,
                [
                    (426.0, "Curry,"),
                    (465.4, "Stephen"),
                    (586.7, "Probable"),
                    (667.1, "Injury/Illness"),
                    (721.8, "-"),
                    (727.1, "Right Ankle Soreness"),
                ],
            ),
        ],
    ]

    class FakePage:
        def __init__(self, rows):
            self.rows = rows

        def extract_text(self, visitor_text):
            for y, row in self.rows:
                for x, value in row:
                    visitor_text(
                        value,
                        None,
                        [1.0, 0.0, 0.0, 1.0, x, y],
                        None,
                        9.0,
                    )

    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda _source: SimpleNamespace(pages=[FakePage(rows) for rows in pages]),
    )

    report = _parse_official_injury_report(b"fake pdf")

    assert [row["reason"] for row in report["players"]] == [
        "Injury/Illness - Right Patellar Tendinopathy",
        "Injury/Illness - Right ACL Repair / Surgery",
        "Injury/Illness - Right Ankle Soreness",
    ]


def test_suffix_injury_status_matches_roster_player() -> None:
    rotation = _game_rotation(
        "GSW",
        players=[
            {
                "player_id": 1,
                "player_name": "Jimmy Butler III",
                "team": "GSW",
                "conditional_minutes_per_game": 36.0,
                "projected_nail": 3.0,
            },
            {
                "player_id": 2,
                "player_name": "Stephen Curry",
                "team": "GSW",
                "conditional_minutes_per_game": 36.0,
                "projected_nail": 4.0,
            },
        ],
        injury_report={
            "players": [
                {
                    "player_name": "Jimmy Butler III",
                    "status": "Out",
                    "reason": "Injury/Illness - Right ACL Repair / Surgery",
                    "play_probability": 0.0,
                }
            ],
            "team_reports": {"GSW": {"status": "submitted"}},
        },
        rotation_size=2,
    )

    butler = next(
        player for player in rotation["players"] if player["player_name"] == "Jimmy Butler III"
    )
    assert butler["status"] == "Out"
    assert butler["play_probability"] == 0.0
    assert butler["projected_minutes"] == 0.0


def test_pending_team_report_does_not_claim_players_are_available() -> None:
    rotation = _game_rotation(
        "GSW",
        players=[
            {
                "player_id": 1,
                "player_name": "Jimmy Butler",
                "team": "GSW",
                "conditional_minutes_per_game": 36.0,
                "projected_nail": 2.0,
            },
            {
                "player_id": 2,
                "player_name": "Stephen Curry",
                "team": "GSW",
                "conditional_minutes_per_game": 34.0,
                "projected_nail": 3.0,
            },
        ],
        injury_report={
            "players": [],
            "team_reports": {"GSW": {"status": "not_yet_submitted"}},
        },
        rotation_size=2,
    )

    assert rotation["report_status"] == "not_yet_submitted"
    assert rotation["injury_assumptions"] == []
    assert {player["status"] for player in rotation["players"]} == {"Report pending"}
    assert all(player["play_probability"] is None for player in rotation["players"])
    assert sum(player["projected_minutes"] for player in rotation["players"]) == pytest.approx(
        240.0
    )


def test_game_rotation_uses_only_the_exact_date_and_matchup_report() -> None:
    players = [
        {
            "player_id": 1,
            "player_name": "Jimmy Butler III",
            "team": "GSW",
            "conditional_minutes_per_game": 36.0,
            "projected_nail": 3.0,
        },
        {
            "player_id": 2,
            "player_name": "Stephen Curry",
            "team": "GSW",
            "conditional_minutes_per_game": 36.0,
            "projected_nail": 4.0,
        },
    ]
    injury_report = {
        "players": [
            {
                "player_name": "Jimmy Butler III",
                "team": "GSW",
                "game_date": "2026-10-23",
                "matchup": "MEM@GSW",
                "status": "Out",
                "reason": "Injury/Illness - Right ACL Repair / Surgery",
                "play_probability": 0.0,
            },
            {
                "player_name": "Jimmy Butler III",
                "team": "GSW",
                "game_date": "2026-10-24",
                "matchup": "GSW@PHX",
                "status": "Available",
                "reason": "-",
                "play_probability": 1.0,
            },
        ],
        "team_reports": {"GSW": {"status": "submitted"}},
        "game_team_reports": {
            "2026-10-23|MEM@GSW": {"GSW": {"status": "submitted"}},
            "2026-10-24|GSW@PHX": {"GSW": {"status": "submitted"}},
        },
    }

    rotation = _game_rotation(
        "GSW",
        players=players,
        injury_report=injury_report,
        rotation_size=2,
        game_date="2026-10-23",
        matchup="MEM@GSW",
    )

    butler = next(
        player for player in rotation["players"] if player["player_name"] == "Jimmy Butler III"
    )
    assert butler["status"] == "Out"
    assert butler["projected_minutes"] == 0.0

    unreported_future_rotation = _game_rotation(
        "GSW",
        players=players,
        injury_report=injury_report,
        rotation_size=2,
        game_date="2026-10-25",
        matchup="GSW@LAL",
    )
    future_players = {
        player["player_name"]: player for player in unreported_future_rotation["players"]
    }
    assert unreported_future_rotation["report_status"] == "not_in_report"
    assert future_players["Jimmy Butler III"]["status"] == "Out"
    assert future_players["Stephen Curry"]["status"] == "Report pending"


@pytest.mark.parametrize(
    ("espn_code", "projection_code"),
    [("WSH", "WAS"), ("UTAH", "UTA")],
)
def test_espn_team_alias_matches_projection_roster_code(
    espn_code: str, projection_code: str
) -> None:
    assert _espn_team({"team": {"abbreviation": espn_code}}) == projection_code


def test_schedule_window_ingests_available_future_boards_with_current_report(
    tmp_path, monkeypatch
) -> None:
    now = datetime(2026, 10, 20, 22, 30, tzinfo=UTC)
    injury_report = {
        "players": [],
        "published_at": "2026-10-20T18:15:00-04:00",
    }
    injury_pdf = b"injury report"
    fetched_urls: list[str] = []
    archived_calls: list[dict[str, object]] = []

    def fake_injury_fetch(*, report_date, now):
        assert report_date == "2026-10-20"
        return injury_report, injury_pdf

    def fake_fetch_json(url):
        fetched_urls.append(url)
        if url == "http://projections.test":
            return {"win_projection": {"season": "2026-27"}, "minutes": {}}
        if "20261021" in url:
            return {"events": [{"id": "tomorrow"}]}
        if "20261022" in url:
            return {"events": [{"id": "following-day"}]}
        return {"events": []}

    def fake_refresh_schedule_archive(**kwargs):
        archived_calls.append(kwargs)
        return {"games": [{"id": kwargs["game_date"]}], "newly_frozen_game_count": 0}

    monkeypatch.setattr(
        win_projections,
        "_fetch_latest_official_injury_report",
        fake_injury_fetch,
    )
    monkeypatch.setattr(win_projections, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        win_projections,
        "refresh_schedule_archive",
        fake_refresh_schedule_archive,
    )

    result = win_projections.refresh_schedule_window(
        start_date="2026-10-20",
        days_ahead=2,
        state_dir=tmp_path,
        projection_url="http://projections.test",
        s3_uri=None,
        now=now,
    )

    assert result["archived_day_count"] == 2
    assert result["game_count"] == 2
    assert [call["game_date"] for call in archived_calls] == ["2026-10-21", "2026-10-22"]
    assert all(call["injury_report"] is injury_report for call in archived_calls)
    assert all(call["injury_pdf"] == injury_pdf for call in archived_calls)
    assert all("forecast_response" in call for call in archived_calls)
    assert any("dates=20261020" in url for url in fetched_urls)
    assert any("dates=20261022" in url for url in fetched_urls)


def test_immutable_s3_upload_accepts_only_byte_identical_retry(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "frozen.json"
    source.write_bytes(b'{"projection": 4.5}\n')
    calls: list[list[str]] = []

    def identical_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=source.read_bytes())

    monkeypatch.setattr(
        "nba_lineup_model.web_api.win_projections.subprocess.run",
        identical_run,
    )
    _aws_s3_cp_exclusive(source, "s3://archive/frozen/game.json")

    assert "--no-overwrite" in calls[0]
    assert calls[1][1:3] == ["s3", "cp"]

    def changed_run(command, **_kwargs):
        return SimpleNamespace(stdout=b'{"projection": -1.5}\n')

    monkeypatch.setattr(
        "nba_lineup_model.web_api.win_projections.subprocess.run",
        changed_run,
    )
    with pytest.raises(RuntimeError, match="Refusing to overwrite immutable S3 object"):
        _aws_s3_cp_exclusive(source, "s3://archive/frozen/game.json")


def test_final_game_archive_preserves_pregame_projection_and_grades_market() -> None:
    pregame = {
        "frozen_at": "2026-10-20T18:30:04Z",
        "projection": {
            "model": "W0 minute-weighted NAIL",
            "home_margin": 4.48,
            "home_win_probability": 0.616,
        },
        "market": {
            "provider": "Draft Kings",
            "home_spread": 1.5,
            "away_spread": -1.5,
            "total": 220.5,
            "home_moneyline": 100,
            "away_moneyline": -120,
        },
    }
    game = {
        "game_id": "0022600001",
        "espn_game_id": "401909088",
        "start_time": "2026-10-20T19:00:00Z",
        "away_team": "BOS",
        "home_team": "DET",
        "away_score": 101,
        "home_score": 112,
        "status": {"state": "post", "completed": True},
    }

    result = _build_final_game_archive(
        game,
        pregame=pregame,
        retrospective={"home_margin": 3.25},
        archived_at=datetime(2026, 10, 20, 21, 30, tzinfo=UTC),
    )

    assert result["pregame"] == pregame
    assert result["final"]["actual_home_margin"] == 11
    assert result["grading"]["model_favorite"] == "DET"
    assert result["grading"]["model_fair_home_spread"] == pytest.approx(-4.48)
    assert result["grading"]["model_fair_line_team"] == "DET"
    assert result["grading"]["model_fair_line"] == pytest.approx(-4.48)
    assert result["grading"]["model_favorite_won"] is True
    market = result["grading"]["market"]
    assert market["market_favorite"] == "BOS"
    assert market["market_favorite_won"] is False
    assert market["model_edge_points"] == pytest.approx(5.98)
    assert market["model_ats_selection"] == "DET"
    assert market["model_ats_selection_spread"] == pytest.approx(1.5)
    assert market["model_ats_cover_margin"] == pytest.approx(12.5)
    assert market["model_ats_result"] == "win"


def test_betmgm_win_total_snapshot_covers_every_nba_team() -> None:
    assert len(BETMGM_WIN_TOTALS_2026_27) == 30
    assert BETMGM_WIN_TOTALS_2026_27["OKC"] == pytest.approx(62.5)
    assert BETMGM_WIN_TOTALS_2026_27["SAC"] == pytest.approx(21.5)


def test_win_probability_scale_is_positive_for_aligned_edges() -> None:
    beta = fit_win_probability_scale(
        pd.Series([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert beta > 0.0
    metrics = score_win_probabilities(
        pd.Series([-2.0, 2.0]), pd.Series([0.0, 1.0]), beta=beta
    )
    assert metrics["brier"] < 0.25
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_completed_team_win_totals_and_scores() -> None:
    games = pd.DataFrame(
        {
            "home_team_tricode": ["AAA", "BBB"],
            "away_team_tricode": ["BBB", "AAA"],
            "edge": [0.0, 0.0],
            "home_win": [1.0, 0.0],
        }
    )
    totals = _completed_team_win_totals(games, beta=0.1)

    assert totals.set_index("team")["projected_wins"].to_dict() == {
        "AAA": pytest.approx(1.0),
        "BBB": pytest.approx(1.0),
    }
    assert score_season_win_forecasts(totals, prediction_column="projected_wins") == {
        "mae": pytest.approx(1.0),
        "rmse": pytest.approx(1.0),
        "bias": pytest.approx(0.0),
    }


def test_baseline_win_total_intervals_are_deterministic_and_ordered() -> None:
    games = pd.DataFrame(
        {
            "home_team_tricode": ["AAA"] * 40 + ["BBB"] * 40,
            "away_team_tricode": ["BBB"] * 40 + ["AAA"] * 40,
            "home_win_probability": [0.5] * 80,
        }
    )
    strength = pd.Series({"AAA": 0.0, "BBB": 0.0})

    first = simulate_baseline_win_total_intervals(
        games, team_strength=strength, beta=0.1, home_court=0.0, trials=1_000, seed=7
    )
    second = simulate_baseline_win_total_intervals(
        games, team_strength=strength, beta=0.1, home_court=0.0, trials=1_000, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["win_total_p1"].le(first["win_total_p5"]).all()
    assert first["win_total_p5"].le(first["win_total_p50"]).all()
    assert first["win_total_p50"].le(first["win_total_p95"]).all()
    assert first["win_total_p95"].le(first["win_total_p99"]).all()


def test_unassigned_cup_placeholder_preserves_the_league_win_total() -> None:
    games = pd.DataFrame(
        {
            "home_team_tricode": ["AAA"] * 40 + ["BBB"] * 40,
            "away_team_tricode": ["BBB"] * 40 + ["AAA"] * 40,
            "home_win_probability": [0.6] * 40 + [0.4] * 40,
        }
    )

    totals = _team_win_totals(
        games,
        team_strength=pd.Series({"AAA": 2.0, "BBB": -2.0}),
        beta=0.1,
        home_court=2.0,
    )

    assert totals["scheduled_wins"].sum() == pytest.approx(80.0)
    assert totals["unassigned_wins"].sum() == pytest.approx(2.0)
    assert totals["projected_wins"].sum() == pytest.approx(82.0)
