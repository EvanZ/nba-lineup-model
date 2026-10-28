from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.modeling.current_season_schedule_materialization import (
    _reforecast_pregame_from_current_state,
    _schedule_source_sha256,
    _validate_forecast_schedule_contract,
    _validate_player_evaluation,
    materialize_current_season_schedule,
)
from nba_lineup_model.web_api.win_projections import read_schedule_performance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_materializes_strict_forward_schedule_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "production-schedule"
    ratings = tmp_path / "ratings"
    minutes = tmp_path / "minutes"
    output = tmp_path / "dev-schedule"
    game_id = "0022600001"
    espn_game_id = "401999001"
    pregame = {
        "game_id": game_id,
        "espn_game_id": espn_game_id,
        "start_time": "2026-10-21T02:00:00Z",
        "away_team": "BOS",
        "home_team": "DET",
        "status": {"state": "pre", "completed": False},
        "projection": {
            "model": "NAIL preseason",
            "home_margin": -4.5,
            "home_win_probability": 0.38,
        },
        "market": {
            "provider": "DraftKings",
            "provider_key": "draftkings",
            "home_spread": -1.5,
            "away_spread": 1.5,
        },
        "injury_report": {
            "published_at": "2026-10-20T20:00:00Z",
            "assumptions": [
                {
                    "player_id": 1,
                    "player_name": "Away Player",
                    "team": "BOS",
                    "status": "Available",
                    "reason": None,
                    "play_probability": 1.0,
                    "projected_minutes": 30.0,
                }
            ],
        },
        "projected_minutes": {
            "away": [
                {
                    "player_id": 1,
                    "player_name": "Away Player",
                    "status": "Available",
                    "play_probability": 1.0,
                    "conditional_minutes": 30.0,
                    "projected_minutes": 30.0,
                    "reason": None,
                }
            ],
            "home": [
                {
                    "player_id": 2,
                    "player_name": "Home Player",
                    "status": "Available",
                    "play_probability": 1.0,
                    "conditional_minutes": 32.0,
                    "projected_minutes": 32.0,
                    "reason": None,
                }
            ],
        },
        "frozen_at": "2026-10-21T01:30:00Z",
    }
    summary = {
        "schema_version": 1,
        "record_type": "immutable_final_game_audit",
        "archived_at": "2026-10-21T05:00:00Z",
        "season": "2026-27",
        "game_date": "2026-10-20",
        "game_id": game_id,
        "espn_game_id": espn_game_id,
        "start_time": "2026-10-21T02:00:00Z",
        "away_team": "BOS",
        "home_team": "DET",
        "final": {
            "away_score": 110,
            "home_score": 118,
            "actual_home_margin": 8,
            "winner": "DET",
            "status": {"state": "post", "completed": True},
        },
        "pregame": pregame,
        "grading": {},
        "retrospective": {"home_margin": 2.0, "home_win_probability": 0.55},
    }
    latest = {
        "available": True,
        "game_date": "2026-10-20",
        "games": [
            {
                "game_id": game_id,
                "espn_game_id": espn_game_id,
                "away_team": "BOS",
                "home_team": "DET",
                "start_time": "2026-10-21T02:00:00Z",
                "status": {"state": "post", "completed": True},
            }
        ],
    }
    _write_json(source / "latest" / "2026-10-20.json", latest)
    _write_json(source / "frozen" / f"{espn_game_id}.json", pregame)
    source_summary = source / "final" / f"{espn_game_id}.summary.json"
    _write_json(source_summary, summary)
    _write_json(source / "final" / f"{espn_game_id}.espn.json", {"boxscore": {}})
    original_source_bytes = source_summary.read_bytes()

    run_id = "test-run"
    through_date = "2026-10-20"
    run_root = minutes / "runs" / f"through_date={through_date}" / run_id
    run_root.mkdir(parents=True)
    game_evaluation = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "game_date": "2026-10-20",
                "away_team": "BOS",
                "home_team": "DET",
                "updated_updated_margin": 2.5,
                "updated_updated_probability": 0.57,
            }
        ]
    )
    game_path = run_root / "four_cell_game_evaluation.csv"
    game_evaluation.to_csv(game_path, index=False)
    player_evaluation = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "team": "BOS",
                "player_id": 1,
                "player_name": "Away Player",
                "updated_conditional_minutes": 35.0,
                "updated_projected_minutes": 35.0,
            },
            {
                "game_id": game_id,
                "team": "DET",
                "player_id": 2,
                "player_name": "Home Player",
                "updated_conditional_minutes": 36.0,
                "updated_projected_minutes": 36.0,
            },
        ]
    )
    player_path = run_root / "player_game_evaluation.parquet"
    player_evaluation.to_parquet(player_path, index=False)
    common = {
        "schema_version": 1,
        "season": "2026-27",
        "through_date": through_date,
        "generated_at": "2026-10-21T12:00:00Z",
        "coverage": {"games": 1, "excluded": 0},
    }
    _write_json(
        ratings / "latest.json",
        {
            **common,
            "dagster_run_id": "ratings-run",
            "artifacts": {"player_state": "s3://test/rating-state.parquet"},
        },
    )
    _write_json(
        minutes / "latest.json",
        {
            **common,
            "dagster_run_id": run_id,
            "artifacts": {
                "four_cell_game_evaluation": (
                    f"s3://test/four_cell_game_evaluation-{_sha256(game_path)}.csv"
                ),
                "player_game_evaluation": (
                    f"s3://test/player_game_evaluation-{_sha256(player_path)}.parquet"
                ),
            },
        },
    )

    release = materialize_current_season_schedule(
        source_root=source,
        rating_root=ratings,
        minutes_root=minutes,
        output_root=output,
        current_forecast_response={
            "players": [],
            "win_projection": {
                "as_of": through_date,
                "serving_bundle_sha256": "test-serving-bundle",
                "win_probability_scale": 0.1,
                "home_court": 2.0,
                "back_to_back": -1.0,
                "scheduled_game_count": 1,
                "scheduled_games": [
                    {
                        "game_id": game_id,
                        "game_date": "2026-10-20",
                        "away_team": "BOS",
                        "home_team": "DET",
                        "home_back_to_back": 0,
                        "away_back_to_back": 0,
                    }
                ],
            },
        },
    )

    assert (output / "current").resolve() == release.resolve()
    replay = json.loads((release / "frozen" / f"{espn_game_id}.json").read_text())
    assert replay["projection"]["home_margin"] == 2.5
    assert replay["projection"]["home_win_probability"] == 0.57
    assert replay["projection"]["model_track"] == "updated_ratings_updated_minutes"
    assert replay["projected_minutes"]["away"][0]["projected_minutes"] == 35.0
    assert replay["injury_report"]["assumptions"][0]["projected_minutes"] == 35.0
    replay_summary = json.loads(
        (release / "final" / f"{espn_game_id}.summary.json").read_text()
    )
    assert replay_summary["grading"]["projected_home_margin"] == 2.5
    assert replay_summary["grading"]["model_favorite_won"] is True
    assert replay_summary["grading"]["market"]["model_ats_result"] == "win"
    assert replay_summary["record_type"] == "strict_forward_replay_final_game_audit"
    original = json.loads(
        (release / "as-published" / "final" / f"{espn_game_id}.summary.json").read_text()
    )
    assert original["pregame"]["projection"]["home_margin"] == -4.5
    assert source_summary.read_bytes() == original_source_bytes
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["coverage"]["evaluation_games"] == 1
    assert manifest["coverage"]["replayed_games"] == 1
    assert manifest["coverage"]["cutoff_state_reforecast_games"] == 0
    assert manifest["coverage"]["replayed_final_games"] == 1
    assert manifest["coverage"]["skipped_games"] == 0
    performance = read_schedule_performance(state_dir=output / "current")
    season = next(row for row in performance["windows"] if row["key"] == "season")
    assert performance["model_track"] == "updated_ratings_updated_minutes"
    assert season["games"] == 1
    assert season["probability"]["gestalt"]["brier"] == (0.57 - 1.0) ** 2


def test_rejects_truncated_public_schedule_before_promotion() -> None:
    with pytest.raises(ValueError, match="calendar is incomplete"):
        _validate_forecast_schedule_contract(
            {
                "scheduled_game_count": 2,
                "scheduled_games": [{"game_id": "0022600002"}],
            }
        )


def test_rejects_impossible_player_minutes_before_materialization() -> None:
    players = pd.DataFrame(
        [
            {
                "game_id": "0022600001",
                "team": "PHX",
                "player_id": 1642863,
                "player_name": "Khaman Maluach",
                "updated_projected_minutes": 122.9,
            }
        ]
    )

    with pytest.raises(ValueError, match="physically impossible"):
        _validate_player_evaluation(players)


def test_schedule_source_identity_changes_when_a_final_is_added(tmp_path: Path) -> None:
    source = tmp_path / "schedule"
    _write_json(source / "final" / "one.summary.json", {"game_id": "1"})
    before = _schedule_source_sha256(source)

    _write_json(source / "final" / "two.summary.json", {"game_id": "2"})

    assert _schedule_source_sha256(source) != before


def test_reforecasts_boundary_game_with_current_minutes_and_frozen_injuries() -> None:
    source = {
        "game_id": "0022600053",
        "espn_game_id": "401999053",
        "game_date": "2026-10-27",
        "away_team": "BOS",
        "home_team": "DET",
        "projection": {"home_margin": -2.0, "home_win_probability": 0.45},
        "injury_report": {
            "team_reports": {
                "BOS": {"status": "submitted"},
                "DET": {"status": "submitted"},
            },
            "assumptions": [
                {
                    "player_id": 2,
                    "player_name": "Home Player",
                    "team": "DET",
                    "status": "Out",
                    "reason": "Ankle",
                    "play_probability": 0.0,
                }
            ],
        },
        "projected_minutes": {"away": [], "home": []},
    }
    players = [
        {
            "player_id": 1,
            "player_name": "Away Player",
            "team": "BOS",
            "conditional_minutes_per_game": 40.0,
            "projected_nail": 1.0,
        },
        {
            "player_id": 2,
            "player_name": "Home Player",
            "team": "DET",
            "conditional_minutes_per_game": 40.0,
            "projected_nail": 3.0,
        },
        {
            "player_id": 3,
            "player_name": "Home Reserve",
            "team": "DET",
            "conditional_minutes_per_game": 20.0,
            "projected_nail": 0.0,
        },
    ]
    replay = _reforecast_pregame_from_current_state(
        source,
        forecast={"model": "updated", "serving_bundle_sha256": "abc"},
        minute_players=players,
        scheduled={
            "game_date": "2026-10-27",
            "away_team": "BOS",
            "home_team": "DET",
            "home_back_to_back": 0,
            "away_back_to_back": 0,
        },
        rotation_size=15,
        beta=0.1,
        home_court=2.0,
        back_to_back=-1.0,
        through_date="2026-10-26",
    )

    assert replay["projection"]["model_track"] == "updated_ratings_updated_minutes"
    assert replay["forecast_version"]["mode"] == "cutoff_state_reforecast"
    assert replay["projected_minutes"]["away"][0]["projected_minutes"] == 240.0
    assert replay["projected_minutes"]["home"][0]["player_id"] == 3
    assert replay["projected_minutes"]["home"][0]["projected_minutes"] == 240.0
    assert replay["injury_report"]["assumptions"][0]["status"] == "Out"
    assert replay["injury_report"]["assumptions"][0]["projected_minutes"] == 0.0
