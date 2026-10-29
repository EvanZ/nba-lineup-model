from __future__ import annotations

import hashlib
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
    build_current_season_minutes_payload,
    build_current_season_win_projection_payload,
    enrich_player_game_trajectory,
    fit_win_probability_scale,
    mirror_missing_frozen_pregame_snapshots,
    read_current_season_records,
    read_schedule_day,
    read_schedule_performance,
    reconcile_final_game_artifacts_to_s3,
    refresh_live_score_state,
    score_season_win_forecasts,
    score_win_probabilities,
    simulate_baseline_win_total_intervals,
)


def test_current_minutes_payload_uses_promoted_ratings_and_roles() -> None:
    baseline = {
        "season": "2026-27",
        "completed_season": "2025-26",
        "model": "preseason",
        "initial_rotation_size": 2,
        "regulation_team_minutes": 240.0,
        "contract": "frozen",
        "teams": ["AAA"],
        "players": [
            {
                "player_id": 1,
                "player_name": "One",
                "position": "G",
                "age": 25.0,
                "team": "AAA",
                "prior_season_minutes": 1000.0,
                "availability_probability": 0.9,
                "conditional_minutes_per_game": 30.0,
                "projected_nail": 1.0,
            },
            {
                "player_id": 2,
                "player_name": "Two",
                "position": "F",
                "age": 24.0,
                "team": "AAA",
                "prior_season_minutes": 800.0,
                "availability_probability": 0.8,
                "conditional_minutes_per_game": 20.0,
                "projected_nail": 0.0,
            },
            {
                "player_id": 3,
                "player_name": "Three",
                "position": "C",
                "age": 23.0,
                "team": "AAA",
                "prior_season_minutes": 400.0,
                "availability_probability": 0.7,
                "conditional_minutes_per_game": 10.0,
                "projected_nail": -1.0,
            },
        ],
    }
    evaluator = SimpleNamespace(
        current_season_rankings=pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "player_name": ["One", "Two", "Three"],
                "team": ["AAA", "AAA", "AAA"],
                "position": ["G", "F", "C"],
                "age": [25.0, 24.0, 23.0],
                "rapm": [1.5, 0.25, -0.5],
                "preseason_nail": [1.0, 0.0, -1.0],
                "current_expected_minutes_if_available": [34.0, 24.0, 12.0],
                "preseason_expected_minutes_if_available": [30.0, 20.0, 10.0],
                "current_participation_estimate": [1.0, 1.0, 1.0],
            }
        ),
        current_season_metadata={
            "evidence_through_date": "2026-10-26",
            "bundle_sha256": "a" * 64,
        },
    )

    payload = build_current_season_minutes_payload(
        evaluator=evaluator,
        preseason_minutes_payload=baseline,
    )

    by_id = {row["player_id"]: row for row in payload["players"]}
    assert payload["state_kind"] == "in_season_updated"
    assert payload["as_of"] == "2026-10-26"
    assert by_id[1]["projected_nail"] == 1.5
    assert by_id[1]["conditional_minutes_per_game"] == 34.0
    assert by_id[3]["baseline_minutes_per_game"] == 0.0
    assert sum(row["baseline_minutes_per_game"] for row in payload["players"]) == pytest.approx(
        240.0
    )


def test_current_win_projection_fixes_actual_wins_and_scores_remaining_schedule(
    tmp_path, monkeypatch
) -> None:
    catalog = pd.DataFrame(
        {
            "game_id": ["0022600001", "0022600002"],
            "season": ["2026-27", "2026-27"],
            "season_type": ["regular", "regular"],
            "game_date": pd.to_datetime(["2026-10-21", "2026-10-23"]),
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_team_tricode": ["AAA", "BBB"],
            "away_team_tricode": ["BBB", "AAA"],
        }
    )
    catalog_path = tmp_path / "games.parquet"
    catalog.to_parquet(catalog_path, index=False)
    final_dir = tmp_path / "schedule" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "401.summary.json").write_text(
        json.dumps(
            {
                "season": "2026-27",
                "game_id": "0022600001",
                "archived_at": "2026-10-22T06:00:00Z",
                "start_time": "2026-10-21T23:00:00Z",
                "away_team": "BBB",
                "home_team": "AAA",
                "final": {
                    "away_score": 100,
                    "home_score": 110,
                    "status": {"completed": True},
                },
            }
        )
    )
    monkeypatch.setattr(
        win_projections,
        "simulate_current_win_total_intervals",
        lambda _games, *, team_strength, **_kwargs: pd.DataFrame(
            {
                "team": team_strength.index,
                "win_total_p1": [30.0, 20.0],
                "win_total_p5": [32.0, 22.0],
                "win_total_p50": [40.0, 30.0],
                "win_total_p95": [48.0, 38.0],
                "win_total_p99": [50.0, 40.0],
            }
        ),
    )
    evaluator = SimpleNamespace(
        current_season_metadata={
            "evidence_through_date": "2026-10-22",
            "bundle_sha256": "b" * 64,
        }
    )
    minutes = {
        "teams": ["AAA", "BBB"],
        "players": [
            {"team": "AAA", "baseline_minutes_per_game": 48.0, "projected_nail": 1.0},
            {"team": "BBB", "baseline_minutes_per_game": 48.0, "projected_nail": -1.0},
        ],
    }
    preseason = {
        "model": "preseason",
        "season": "2026-27",
        "win_probability_scale": 0.2,
        "home_court": 2.0,
        "back_to_back": -1.0,
        "unassigned_regular_games_per_team": 2,
        "calibration": {},
        "teams": [
            {"team": "AAA", "betmgm_win_total": 45.5},
            {"team": "BBB", "betmgm_win_total": 35.5},
        ],
        "scheduled_games": [],
        "contract": "frozen",
    }

    payload = build_current_season_win_projection_payload(
        evaluator=evaluator,
        minutes_payload=minutes,
        preseason_forecast=preseason,
        state_dir=tmp_path / "schedule",
        catalog_path=catalog_path,
    )

    by_team = {row["team"]: row for row in payload["teams"]}
    assert payload["state_kind"] == "in_season_updated"
    assert payload["model_track"] == "updated_ratings_updated_minutes"
    assert payload["serving_bundle_sha256"] == "b" * 64
    assert [game["game_id"] for game in payload["scheduled_games"]] == [
        "0022600001",
        "0022600002",
    ]
    assert [game["game_id"] for game in payload["remaining_scheduled_games"]] == [
        "0022600002"
    ]
    assert len(payload["scheduled_games"]) == payload["scheduled_game_count"]
    assert by_team["AAA"]["actual_wins"] == 1
    assert by_team["AAA"]["completed_games"] == 1
    assert by_team["AAA"]["projected_wins"] > by_team["AAA"]["remaining_scheduled_wins"]


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


def test_schedule_performance_scores_rolling_market_consensus(tmp_path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    games = [
        ("2026-10-23T23:00:00Z", 10.0, 6.0, 0.7, -4.0, -300, 300, 0.75, "win", 4.0),
        ("2026-10-20T23:00:00Z", -5.0, 2.0, 0.6, -1.0, 300, -300, 0.25, "loss", 3.999),
        ("2026-10-10T23:00:00Z", 3.0, 4.0, 0.55, 2.0, -120, 100, 0.52, "push", 2.0),
        ("2026-09-10T23:00:00Z", -2.0, -1.0, 0.45, -3.0, 110, -130, 0.4, "win", 1.999),
    ]
    for index, (
        start_time,
        actual_margin,
        model_margin,
        model_probability,
        market_margin,
        home_moneyline,
        away_moneyline,
        market_probability,
        ats_result,
        model_edge_points,
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
                            "model_edge_points": model_edge_points,
                        },
                    },
                    "pregame": {
                        "market": {
                            "provider": "Market consensus",
                            "provider_key": "market_consensus",
                            "home_moneyline": home_moneyline,
                            "away_moneyline": away_moneyline,
                            "home_win_probability": market_probability,
                            "away_win_probability": 1.0 - market_probability,
                        },
                        "prediction_market": {
                            "home_probability": market_probability,
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
    assert seven_days["probability"]["gestalt"]["n"] == 2
    assert seven_days["probability"]["market_consensus"]["n"] == 2
    assert seven_days["probability"]["gestalt"]["accuracy"] == pytest.approx(0.5)
    assert seven_days["probability"]["gestalt"]["brier"] == pytest.approx(0.225)
    assert seven_days["probability"]["market_consensus"]["accuracy"] == pytest.approx(
        1.0
    )
    assert seven_days["probability"]["market_consensus"]["brier"] == pytest.approx(
        0.0625
    )
    assert seven_days["probability"]["polymarket"] == {
        "n": 2,
        "accuracy": pytest.approx(1.0),
        "brier": pytest.approx(0.0625),
    }
    assert seven_days["probability"]["actual_minutes"] == {
        "n": 2,
        "accuracy": pytest.approx(1.0),
        "brier": pytest.approx(0.065),
    }
    assert seven_days["spread"] == {
        "n": 2,
        "gestalt_mae": pytest.approx(5.5),
        "market_consensus_mae": pytest.approx(9.0),
        "gestalt_advantage": pytest.approx(3.5),
        "actual_minutes_n": 2,
        "actual_minutes_mae": pytest.approx(2.0),
    }
    assert seven_days["ats"]["cover_rate"] == pytest.approx(0.5)
    assert seven_days["ats"]["units_at_minus_110"] == pytest.approx(-1.0 / 11.0)
    assert season["ats"]["edge_buckets"] == [
        {
            "key": "under_2",
            "label": "<2 pts",
            "minimum_edge": 0.0,
            "maximum_edge": 2.0,
            "n": 1,
            "wins": 1,
            "losses": 0,
            "pushes": 0,
            "cover_rate": pytest.approx(1.0),
            "units_at_minus_110": pytest.approx(10.0 / 11.0),
        },
        {
            "key": "2_to_under_4",
            "label": "2–<4 pts",
            "minimum_edge": 2.0,
            "maximum_edge": 4.0,
            "n": 2,
            "wins": 0,
            "losses": 1,
            "pushes": 1,
            "cover_rate": pytest.approx(0.0),
            "units_at_minus_110": pytest.approx(-1.0),
        },
        {
            "key": "4_plus",
            "label": "4+ pts",
            "minimum_edge": 4.0,
            "maximum_edge": None,
            "n": 1,
            "wins": 1,
            "losses": 0,
            "pushes": 0,
            "cover_rate": pytest.approx(1.0),
            "units_at_minus_110": pytest.approx(10.0 / 11.0),
        },
    ]


def test_updated_track_performance_includes_new_live_final_audits(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"model_track": "updated_ratings_updated_minutes"})
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    common = {
        "season": "2026-27",
        "archived_at": "2026-10-28T06:30:00Z",
        "start_time": "2026-10-28T03:00:00Z",
        "final": {
            "actual_home_margin": -7.0,
            "status": {"completed": True},
        },
        "grading": {
            "projected_home_margin": 5.8,
            "projected_home_win_probability": 0.65,
            "market": {
                "market_implied_home_margin": 1.5,
                "model_ats_result": "loss",
                "model_edge_points": 4.3,
            },
        },
    }
    (final_dir / "current.summary.json").write_text(
        json.dumps(
            {
                **common,
                "record_type": "immutable_final_game_audit",
                "game_id": "0022600131",
                "pregame": {
                    "forecast_version": {
                        "model_track": "updated_ratings_updated_minutes"
                    },
                    "market": {
                        "provider": "Market consensus",
                        "provider_key": "market_consensus",
                        "home_win_probability": 0.54,
                    },
                },
            }
        )
    )
    (final_dir / "old-track.summary.json").write_text(
        json.dumps(
            {
                **common,
                "record_type": "immutable_final_game_audit",
                "game_id": "0022600132",
                "pregame": {"market": {"provider": "Draft Kings"}},
            }
        )
    )
    (final_dir / "current-state.summary.json").write_text(
        json.dumps(
            {
                **common,
                "record_type": "immutable_final_game_audit",
                "game_id": "0022600133",
                "pregame": {
                    "projection": {
                        "state_kind": "in_season_updated",
                        "serving_bundle_sha256": "b" * 64,
                    },
                    "market": {
                        "provider": "Market consensus",
                        "provider_key": "market_consensus",
                        "home_win_probability": 0.54,
                    },
                },
            }
        )
    )

    payload = read_schedule_performance(state_dir=tmp_path)
    season = payload["windows"][-1]

    assert season["games"] == 2
    assert season["probability"]["gestalt"]["n"] == 2
    assert season["probability"]["market_consensus"]["n"] == 2
    assert season["spread"]["n"] == 2
    assert season["ats"]["n"] == 2
    assert season["ats"]["losses"] == 2
    assert season["ats"]["edge_buckets"][-1]["losses"] == 2


def test_schedule_performance_includes_legacy_draftkings_games_as_market_benchmark(
    tmp_path,
) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "legacy.summary.json").write_text(
        json.dumps(
            {
                "season": "2026-27",
                "game_id": "0022600001",
                "archived_at": "2026-10-24T05:00:00Z",
                "start_time": "2026-10-23T23:00:00Z",
                "final": {
                    "actual_home_margin": 5.0,
                    "status": {"completed": True},
                },
                "grading": {
                    "projected_home_margin": 3.0,
                    "projected_home_win_probability": 0.6,
                    "market": {
                        "market_implied_home_margin": 2.5,
                        "model_ats_result": "win",
                        "model_edge_points": 0.5,
                    },
                },
                "pregame": {
                    "market": {
                        "provider": "Draft Kings",
                        "home_spread": -2.5,
                        "away_spread": 2.5,
                        "home_moneyline": -140,
                        "away_moneyline": 120,
                    },
                },
            }
        )
    )

    payload = read_schedule_performance(state_dir=tmp_path)
    season = payload["windows"][-1]

    assert season["games"] == 1
    assert season["market_sources"] == {"draftkings": 1, "market_consensus": 0}
    assert season["probability"]["n"] == 1
    assert season["probability"]["gestalt"]["n"] == 1
    assert season["probability"]["market_consensus"]["n"] == 1
    assert season["probability"]["gestalt"]["brier"] == pytest.approx(0.16)
    assert season["probability"]["market_consensus"]["brier"] == pytest.approx(
        0.19180563695455263
    )
    assert season["probability"]["polymarket"] == {
        "n": 0,
        "accuracy": None,
        "brier": None,
    }
    assert season["spread"]["n"] == 1
    assert season["spread"]["gestalt_mae"] == pytest.approx(2.0)
    assert season["spread"]["market_consensus_mae"] == pytest.approx(2.5)
    assert season["ats"]["n"] == 1
    assert season["ats"]["wins"] == 1


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


def test_player_game_trajectory_uses_frozen_minutes_and_archived_box_score(
    tmp_path,
) -> None:
    final = tmp_path / "final"
    final.mkdir()
    (final / "401.summary.json").write_text(
        json.dumps(
            {
                "game_id": "0022600001",
                "espn_game_id": "401",
                "away_team": "GSW",
                "home_team": "LAL",
                "final": {"away_score": 105, "home_score": 127},
                "pregame": {
                    "projected_minutes": {
                        "away": [
                            {
                                "player_id": 201939,
                                "player_name": "Stephen Curry",
                                "projected_minutes": 31.9,
                                "status": "Available",
                                "reason": None,
                            }
                        ],
                        "home": [],
                    }
                },
                "retrospective": {
                    "actual_minutes": {
                        "away": [
                            {
                                "player_id": 201939,
                                "player_name": "Stephen Curry",
                                "actual_minutes": 21.0,
                            }
                        ],
                        "home": [],
                    }
                },
            }
        )
    )
    (final / "401.espn.json").write_text(
        json.dumps(
            {
                "boxscore": {
                    "players": [
                        {
                            "statistics": [
                                {
                                    "labels": ["MIN", "FG", "3PT", "FT", "PTS", "REB", "AST", "TO", "STL", "BLK", "+/-"],
                                    "athletes": [
                                        {
                                            "athlete": {"displayName": "Stephen Curry"},
                                            "stats": ["21", "9-17", "4-9", "1-1", "23", "2", "1", "4", "2", "1", "-22"],
                                            "didNotPlay": False,
                                        }
                                    ],
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )

    points = enrich_player_game_trajectory(
        [
            {"source": "preseason", "rating": 1.45},
            {
                "source": "in_season",
                "game_id": "0022600001",
                "team": "GSW",
                "opponent": "LAL",
                "played": True,
                "rating": 1.20,
            },
        ],
        player_id=201939,
        player_name="Stephen Curry",
        state_dir=tmp_path,
    )

    game = points[1]
    assert game["home_away"] == "away"
    assert game["result"] == "L"
    assert game["team_score"] == 105
    assert game["opponent_score"] == 127
    assert game["projected_minutes"] == pytest.approx(31.9)
    assert game["actual_minutes"] == pytest.approx(21.0)
    assert game["box_score"] == {
        "points": 23,
        "rebounds": 2,
        "assists": 1,
        "turnovers": 4,
        "steals": 2,
        "blocks": 1,
        "two_points_made": 5,
        "two_points_attempted": 8,
        "three_points_made": 4,
        "three_points_attempted": 9,
        "free_throws_made": 1,
        "free_throws_attempted": 1,
        "plus_minus": -22,
    }


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
        now=datetime(2026, 10, 23, 12, 23, tzinfo=UTC),
    )

    assert game_date == "2026-10-23"


def test_live_score_records_first_final_observation(monkeypatch, tmp_path) -> None:
    latest = tmp_path / "latest" / "2026-10-24.json"
    latest.parent.mkdir()
    latest.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "espn_game_id": "401909860",
                        "start_time": "2026-10-25T02:00:00Z",
                        "retrospective": {},
                        "status": {"state": "in", "completed": False},
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        win_projections,
        "_fetch_json",
        lambda _url: {
            "events": [
                {
                    "id": "401909860",
                    "competitions": [
                        {
                            "competitors": [
                                {"homeAway": "home", "score": "110"},
                                {"homeAway": "away", "score": "105"},
                            ]
                        }
                    ],
                    "status": {
                        "displayClock": "0.0",
                        "period": 4,
                        "type": {
                            "state": "post",
                            "completed": True,
                            "description": "Final",
                            "detail": "Final",
                        },
                    },
                }
            ]
        },
    )

    refresh_live_score_state(game_date="2026-10-24", state_dir=tmp_path)

    game = json.loads(latest.read_text())["games"][0]
    assert game["status"]["completed"] is True
    assert datetime.fromisoformat(game["final_observed_at"].replace("Z", "+00:00"))


def _write_unfinalized_live_game(tmp_path) -> None:
    latest = tmp_path / "latest" / "2026-10-24.json"
    latest.parent.mkdir()
    latest.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "espn_game_id": "401909860",
                        "game_id": "0022600030",
                        "start_time": "2026-10-25T02:00:00Z",
                        "away_team": "GSW",
                        "home_team": "LAL",
                        "status": {"state": "in", "completed": False},
                    }
                ]
            }
        )
    )
    frozen = tmp_path / "frozen" / "401909860.json"
    frozen.parent.mkdir()
    frozen.write_text(
        json.dumps(
            {
                "frozen_at": "2026-10-25T01:30:00Z",
                "projection": {
                    "model": "NAIL-RAPM v1.2.1.3",
                    "home_margin": 2.5,
                    "home_win_probability": 0.58,
                },
                "market": None,
            }
        )
    )


def _final_scoreboard_payload() -> dict[str, object]:
    return {
        "events": [
            {
                "id": "401909860",
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "score": "110"},
                            {"homeAway": "away", "score": "105"},
                        ]
                    }
                ],
                "status": {
                    "displayClock": "0.0",
                    "period": 4,
                    "type": {
                        "state": "post",
                        "completed": True,
                        "description": "Final",
                        "detail": "Final",
                    },
                },
            }
        ]
    }


def test_live_finalizer_immediately_publishes_immutable_artifacts(
    monkeypatch, tmp_path
) -> None:
    _write_unfinalized_live_game(tmp_path)
    monkeypatch.setattr(
        win_projections,
        "_fetch_json",
        lambda url: (
            _final_scoreboard_payload()
            if "scoreboard" in url
            else ({"boxscore": {}} if "summary" in url else {})
        ),
    )
    monkeypatch.setattr(
        win_projections,
        "_build_actual_minutes_retrospective",
        lambda *_args, **_kwargs: {"home_win_probability": 0.61},
    )
    uploads: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win_projections,
        "_aws_s3_cp_exclusive",
        lambda source, destination: uploads.append((source.name, destination)),
    )

    result = refresh_live_score_state(
        game_date="2026-10-24",
        state_dir=tmp_path,
        s3_uri="s3://archive/schedule",
    )

    assert result["newly_finalized_game_count"] == 1
    assert result["uploaded_final_artifact_count"] == 2
    assert uploads == [
        (
            "401909860.summary.json",
            "s3://archive/schedule/season=2026-27/game_date=2026-10-24/"
            "final/401909860.summary.json",
        ),
        (
            "401909860.espn.json",
            "s3://archive/schedule/season=2026-27/game_date=2026-10-24/"
            "final/401909860.espn.json",
        ),
    ]
    assert (tmp_path / "final" / "401909860.summary.json").is_file()
    assert (tmp_path / "final" / "401909860.espn.json").is_file()


def test_live_finalizer_recovers_audit_after_board_already_marked_final(
    monkeypatch, tmp_path
) -> None:
    _write_unfinalized_live_game(tmp_path)
    latest = tmp_path / "latest" / "2026-10-24.json"
    payload = json.loads(latest.read_text())
    payload["games"][0]["status"] = {"state": "post", "completed": True}
    latest.write_text(json.dumps(payload))
    monkeypatch.setattr(
        win_projections,
        "_fetch_json",
        lambda url: (
            _final_scoreboard_payload()
            if "scoreboard" in url
            else ({"boxscore": {}} if "summary" in url else {})
        ),
    )
    monkeypatch.setattr(
        win_projections,
        "_build_actual_minutes_retrospective",
        lambda *_args, **_kwargs: {"home_win_probability": 0.61},
    )

    result = refresh_live_score_state(
        game_date="2026-10-24",
        state_dir=tmp_path,
    )

    assert result["newly_finalized_game_count"] == 1
    assert (tmp_path / "final" / "401909860.summary.json").is_file()
    assert (tmp_path / "final" / "401909860.espn.json").is_file()


def test_live_finalizer_s3_failure_leaves_latest_state_retryable(
    monkeypatch, tmp_path
) -> None:
    _write_unfinalized_live_game(tmp_path)
    original_latest = (tmp_path / "latest" / "2026-10-24.json").read_bytes()
    monkeypatch.setattr(
        win_projections,
        "_fetch_json",
        lambda url: (
            _final_scoreboard_payload()
            if "scoreboard" in url
            else ({"boxscore": {}} if "summary" in url else {})
        ),
    )
    monkeypatch.setattr(
        win_projections,
        "_build_actual_minutes_retrospective",
        lambda *_args, **_kwargs: {"home_win_probability": 0.61},
    )
    monkeypatch.setattr(
        win_projections,
        "_aws_s3_cp_exclusive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("S3 unavailable")),
    )

    with pytest.raises(RuntimeError, match="S3 unavailable"):
        refresh_live_score_state(
            game_date="2026-10-24",
            state_dir=tmp_path,
            s3_uri="s3://archive/schedule",
        )

    assert (tmp_path / "latest" / "2026-10-24.json").read_bytes() == original_latest
    assert (tmp_path / "final" / "401909860.summary.json").is_file()
    assert (tmp_path / "final" / "401909860.espn.json").is_file()


def test_final_artifact_reconciliation_verifies_completed_games(
    monkeypatch, tmp_path
) -> None:
    latest = tmp_path / "latest" / "2026-10-24.json"
    latest.parent.mkdir()
    latest.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "espn_game_id": "401909860",
                        "status": {"state": "post", "completed": True},
                    },
                    {
                        "espn_game_id": "401909861",
                        "status": {"state": "pre", "completed": False},
                    },
                ]
            }
        )
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    for suffix in ("summary.json", "espn.json"):
        (final_dir / f"401909860.{suffix}").write_text("{}\n")
    uploads: list[str] = []
    monkeypatch.setattr(
        win_projections,
        "_aws_s3_cp_exclusive",
        lambda _source, destination: uploads.append(destination),
    )

    result = reconcile_final_game_artifacts_to_s3(
        game_date="2026-10-24",
        state_dir=tmp_path,
        s3_uri="s3://archive/schedule",
    )

    assert result == {
        "game_date": "2026-10-24",
        "final_game_count": 1,
        "verified_artifact_count": 2,
    }
    assert uploads == [
        "s3://archive/schedule/season=2026-27/game_date=2026-10-24/"
        "final/401909860.summary.json",
        "s3://archive/schedule/season=2026-27/game_date=2026-10-24/"
        "final/401909860.espn.json",
    ]


def test_final_artifact_reconciliation_fails_on_missing_local_artifact(
    monkeypatch, tmp_path
) -> None:
    latest = tmp_path / "latest" / "2026-10-24.json"
    latest.parent.mkdir()
    latest.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "espn_game_id": "401909860",
                        "status": {"state": "post", "completed": True},
                    }
                ]
            }
        )
    )
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401909860.summary.json").write_text("{}\n")
    monkeypatch.setattr(win_projections, "_aws_s3_cp_exclusive", lambda *_args: None)

    with pytest.raises(FileNotFoundError, match="401909860.espn.json"):
        reconcile_final_game_artifacts_to_s3(
            game_date="2026-10-24",
            state_dir=tmp_path,
            s3_uri="s3://archive/schedule",
        )


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


def test_injury_pdf_is_uploaded_once_by_checksum(tmp_path, monkeypatch) -> None:
    injury_pdf = b"same official injury report"
    uploads: list[tuple[bytes, str]] = []

    def fake_upload(source, destination):
        uploads.append((source.read_bytes(), destination))

    monkeypatch.setattr(win_projections, "_aws_s3_cp_exclusive", fake_upload)

    first = win_projections._ensure_s3_injury_report(
        injury_pdf=injury_pdf,
        state_root=tmp_path,
        s3_uri="s3://archive/schedule",
    )
    second = win_projections._ensure_s3_injury_report(
        injury_pdf=injury_pdf,
        state_root=tmp_path,
        s3_uri="s3://archive/schedule",
    )

    digest = hashlib.sha256(injury_pdf).hexdigest()
    assert first == second == {
        "sha256": digest,
        "s3_uri": (
            "s3://archive/schedule/season=2026-27/"
            f"injury-reports/sha256={digest}.pdf"
        ),
    }
    assert uploads == [(injury_pdf, first["s3_uri"])]
    assert not (tmp_path / "pending" / "injury-reports" / f"{digest}.pdf").exists()
    assert (tmp_path / "artifacts" / "injury-reports" / f"{digest}.json").is_file()


def test_mirror_missing_frozen_pregame_snapshots_is_validated_and_idempotent(
    tmp_path,
) -> None:
    source = tmp_path / "serving"
    target = tmp_path / "model-input"
    (source / "latest").mkdir(parents=True)
    (source / "frozen").mkdir()
    (source / "latest" / "2026-10-28.json").write_text(
        json.dumps({"games": [{"espn_game_id": "401909888"}]})
    )
    frozen = {
        "game_id": "0022600139",
        "espn_game_id": "401909888",
        "status": {"state": "pre", "completed": False},
        "late_freeze": False,
        "projection": {
            "state_kind": "in_season_updated",
            "serving_bundle_sha256": "b" * 64,
        },
    }
    (source / "frozen" / "401909888.json").write_text(json.dumps(frozen))

    assert mirror_missing_frozen_pregame_snapshots(
        game_date="2026-10-28",
        source_state_dir=source,
        target_state_dir=target,
    ) == ["401909888"]
    assert json.loads((target / "frozen" / "401909888.json").read_text()) == frozen
    assert mirror_missing_frozen_pregame_snapshots(
        game_date="2026-10-28",
        source_state_dir=source,
        target_state_dir=target,
    ) == []

    (target / "frozen" / "401909888.json").unlink()
    frozen["status"] = {"state": "post", "completed": True}
    frozen["late_freeze"] = True
    (source / "frozen" / "401909888.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="non-pregame"):
        mirror_missing_frozen_pregame_snapshots(
            game_date="2026-10-28",
            source_state_dir=source,
            target_state_dir=target,
        )


def test_s3_schedule_snapshot_does_not_retain_local_archive(tmp_path, monkeypatch) -> None:
    immutable_uploads: list[str] = []
    mutable_uploads: list[str] = []
    artifact = {
        "sha256": "abc123",
        "s3_uri": "s3://archive/schedule/season=2026-27/injury-reports/sha256=abc123.pdf",
    }

    monkeypatch.setattr(
        win_projections,
        "_normalize_espn_schedule",
        lambda *_args, **_kwargs: {"game_date": "2026-10-24", "games": []},
    )
    monkeypatch.setattr(
        win_projections,
        "_aws_s3_cp_exclusive",
        lambda _source, destination: immutable_uploads.append(destination),
    )
    monkeypatch.setattr(
        win_projections,
        "_aws_s3_cp",
        lambda _source, destination: mutable_uploads.append(destination),
    )

    result = win_projections.refresh_schedule_archive(
        game_date="2026-10-24",
        state_dir=tmp_path,
        s3_uri="s3://archive/schedule",
        source_payload={"events": []},
        injury_report={"players": []},
        injury_pdf=b"official report",
        injury_artifact=artifact,
        forecast_response={"win_projection": {}, "minutes": {}},
        prediction_market_snapshot={"games": {}},
        now=datetime(2026, 10, 24, 20, 30, tzinfo=UTC),
    )

    assert result["injury_report_artifact"] == artifact
    assert not (tmp_path / "archive").exists()
    assert not any((tmp_path / "pending" / "2026-10-24").glob("*"))
    assert all(not destination.endswith(".injury.pdf") for destination in immutable_uploads)
    assert len(immutable_uploads) == 3
    assert mutable_uploads == [
        "s3://archive/schedule/season=2026-27/game_date=2026-10-24/latest.json"
    ]
    latest = json.loads((tmp_path / "latest" / "2026-10-24.json").read_text())
    assert latest["injury_report_artifact"] == artifact


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
