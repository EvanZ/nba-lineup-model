from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from nba_lineup_model.season.official_schedule_runtime import (
    _apply_frozen_pregame,
    _build_official_actual_minutes_retrospective,
    _normalize_official_board,
    _publish_board,
    _require_or_attach_frozen_attribution,
    refresh_official_live_score_state,
)
from nba_lineup_model.web_api.win_projections import read_schedule_day

GAME_DATE = "2026-11-01"
GAME_ID = "0022600100"
MATCHUP = "AAA@BBB"
START = "2026-11-01T02:00:00Z"


def _forecast(*, home_star_rating: float = 2.0) -> dict[str, object]:
    players = [
        (1, "Away One", "AAA", 30.0, 1.0),
        (2, "Away Two", "AAA", 20.0, 0.0),
        (3, "Away Three", "AAA", 10.0, -1.0),
        (4, "Home Star", "BBB", 30.0, home_star_rating),
        (5, "Home Two", "BBB", 20.0, 1.0),
        (6, "Home Three", "BBB", 10.0, 0.0),
    ]
    return {
        "win_projection": {
            "model": "test-model",
            "model_track": "updated_ratings_updated_minutes",
            "state_kind": "in_season_updated",
            "as_of": "2026-10-31",
            "serving_bundle_sha256": "f" * 64,
            "home_court": 2.0,
            "back_to_back": -1.0,
            "win_probability_scale": 0.2,
            "scheduled_games": [
                {
                    "game_id": GAME_ID,
                    "home_back_to_back": 1,
                    "away_back_to_back": 0,
                }
            ],
        },
        "minutes": {
            "initial_rotation_size": 3,
            "projection_provenance": {
                "schema_version": 1,
                "serving_bundle_sha256": "f" * 64,
                "rating_state_sha256": "a" * 64,
                "minutes_state_sha256": "b" * 64,
                "rating_as_of": "2026-10-31",
                "as_of_cutoff_time": "2026-11-01T07:00:00Z",
            },
            "players": [
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team": team,
                    "conditional_minutes_per_game": conditional_minutes,
                    "projected_nail": rating,
                }
                for player_id, player_name, team, conditional_minutes, rating in players
            ],
        },
    }


def _injury_report(
    *,
    home_star_status: str | None = None,
    submitted: bool = True,
    published_at: str = "2026-11-01T20:00:00-04:00",
) -> dict[str, object]:
    report_status = "submitted" if submitted else "not_yet_submitted"
    players: list[dict[str, object]] = []
    if home_star_status is not None:
        players.append(
            {
                "player_name": "Home Star",
                "team": "BBB",
                "status": home_star_status,
                "reason": "Test injury",
                "play_probability": {
                    "Out": 0.0,
                    "Questionable": 0.5,
                }[home_star_status],
                "game_date": GAME_DATE,
                "matchup": MATCHUP,
            }
        )
    scoped = {
        "AAA": {"status": report_status},
        "BBB": {"status": report_status},
    }
    return {
        "source_url": "https://official.example/injury.pdf",
        "published_at": published_at,
        "players": players,
        "team_reports": scoped,
        "game_team_reports": {f"{GAME_DATE}|{MATCHUP}": scoped},
    }


def _board(
    *,
    collected_at: datetime,
    home_star_status: str | None = None,
    submitted: bool = True,
    artifact_character: str = "c",
    home_star_rating: float = 2.0,
) -> dict[str, object]:
    return _normalize_official_board(
        games=[
            {
                "game_id": GAME_ID,
                "start_time": START,
                "away_team": "AAA",
                "home_team": "BBB",
            }
        ],
        scoreboard={},
        forecast_response=_forecast(home_star_rating=home_star_rating),
        injury_report=_injury_report(
            home_star_status=home_star_status,
            submitted=submitted,
        ),
        bookmaker_lines={},
        prediction_markets={},
        game_date=GAME_DATE,
        collected_at=collected_at,
        catalog_sha256="d" * 64,
        injury_artifact={"sha256": artifact_character * 64},
    )


def test_player_ledger_exactly_reproduces_existing_spread_math() -> None:
    payload = _board(collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC))
    game = payload["games"][0]
    projection = game["projection"]
    ledger = projection["ledger"]

    for side in ("away", "home"):
        team = ledger[side]
        assert team["projected_minutes"] == pytest.approx(240.0)
        assert sum(
            row["expected_contribution"] for row in team["players"]
        ) == pytest.approx(team["team_strength"], abs=1e-12)
        for row in team["players"]:
            assert row["expected_contribution"] == pytest.approx(
                row["projected_minutes"] / 48.0 * row["frozen_nail"],
                abs=1e-12,
            )

    expected_margin = (
        ledger["home"]["team_strength"]
        - ledger["away"]["team_strength"]
        + 2.0
        - 1.0
    )
    assert ledger["home_margin"] == pytest.approx(expected_margin, abs=1e-12)
    assert projection["home_margin"] == pytest.approx(expected_margin, abs=1e-12)
    assert projection["fair_home_spread"] == pytest.approx(-expected_margin)
    assert ledger["provenance"]["rating_state_sha256"] == "a" * 64
    assert ledger["provenance"]["minutes_state_sha256"] == "b" * 64
    assert ledger["provenance"]["injury_report_sha256"] == "c" * 64


def test_final_minutes_create_a_second_auditable_ledger() -> None:
    game = _board(collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC))["games"][0]
    game["status"] = {"state": "post", "completed": True}
    boxscore = {
        "game": {
            "gameId": GAME_ID,
            "awayTeam": {
                "teamTricode": "AAA",
                "players": [
                    {
                        "personId": player_id,
                        "firstName": "Away",
                        "familyName": str(player_id),
                        "statistics": {"minutes": minutes},
                    }
                    for player_id, minutes in ((1, "PT100M"), (2, "PT80M"), (3, "PT60M"))
                ],
            },
            "homeTeam": {
                "teamTricode": "BBB",
                "players": [
                    {
                        "personId": player_id,
                        "firstName": "Home",
                        "familyName": str(player_id),
                        "statistics": {"minutes": minutes},
                    }
                    for player_id, minutes in ((4, "PT120M"), (5, "PT80M"), (6, "PT40M"))
                ],
            },
        }
    }

    retrospective = _build_official_actual_minutes_retrospective(
        game,
        pregame=game,
        boxscore=boxscore,
        projection_payload=_forecast(),
    )

    ledger = retrospective["ledger"]
    assert ledger["artifact_kind"] == "retrospective_actual_minutes_attribution"
    assert ledger["home_margin"] == pytest.approx(retrospective["home_margin"])
    assert ledger["away"]["actual_minutes"] == pytest.approx(240.0)
    assert ledger["home"]["actual_minutes"] == pytest.approx(240.0)
    assert sum(
        row["actual_contribution"] for row in ledger["home"]["players"]
    ) == pytest.approx(ledger["home"]["team_strength"])


def test_injury_probability_reallocates_the_whole_team_and_remains_auditable() -> None:
    questionable = _board(
        collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC),
        home_star_status="Questionable",
    )["games"][0]
    out = _board(
        collected_at=datetime(2026, 11, 1, 1, 5, tzinfo=UTC),
        home_star_status="Out",
        artifact_character="e",
    )["games"][0]

    questionable_players = {
        row["player_name"]: row
        for row in questionable["projection"]["ledger"]["home"]["players"]
    }
    out_players = {
        row["player_name"]: row
        for row in out["projection"]["ledger"]["home"]["players"]
    }
    assert questionable_players["Home Star"]["play_probability"] == 0.5
    assert out_players["Home Star"]["play_probability"] == 0.0
    assert out_players["Home Star"]["projected_minutes"] == 0.0
    assert out_players["Home Two"]["projected_minutes"] > questionable_players[
        "Home Two"
    ]["projected_minutes"]
    assert sum(row["projected_minutes"] for row in out_players.values()) == pytest.approx(
        240.0
    )
    assert out["projection"]["home_margin"] != questionable["projection"][
        "home_margin"
    ]


def test_report_pending_is_unknown_not_available() -> None:
    game = _board(
        collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC),
        submitted=False,
    )["games"][0]

    players = game["projection"]["ledger"]["home"]["players"]
    assert {row["status"] for row in players} == {"Report pending"}
    assert {row["play_probability"] for row in players} == {None}
    assert sum(row["projected_minutes"] for row in players) == pytest.approx(240.0)


def test_latest_valid_pre_tip_candidate_freezes_and_post_tip_update_cannot_mutate(
    tmp_path,
) -> None:
    first = _board(
        collected_at=datetime(2026, 11, 1, 1, 20, tzinfo=UTC),
        home_star_status="Questionable",
    )
    first_result = _publish_board(
        first,
        state_root=tmp_path,
        s3_uri=None,
        collected_at=datetime(2026, 11, 1, 1, 20, tzinfo=UTC),
    )
    assert first_result["newly_published_candidate_count"] == 1
    assert first_result["newly_frozen_game_count"] == 0
    assert not (tmp_path / "frozen" / f"{GAME_ID}.json").exists()

    final_pregame = _board(
        collected_at=datetime(2026, 11, 1, 1, 55, tzinfo=UTC),
        home_star_status="Out",
        artifact_character="e",
        home_star_rating=2.5,
    )
    _publish_board(
        final_pregame,
        state_root=tmp_path,
        s3_uri=None,
        collected_at=datetime(2026, 11, 1, 1, 55, tzinfo=UTC),
    )

    live = copy.deepcopy(final_pregame)
    live["games"][0]["status"] = {
        "state": "in",
        "completed": False,
        "description": "Q1",
    }
    freeze_result = _publish_board(
        live,
        state_root=tmp_path,
        s3_uri=None,
        collected_at=datetime(2026, 11, 1, 2, 1, tzinfo=UTC),
    )
    assert freeze_result["newly_frozen_game_count"] == 1
    frozen_path = tmp_path / "frozen" / f"{GAME_ID}.json"
    frozen = json.loads(frozen_path.read_text())
    frozen_bytes = frozen_path.read_bytes()
    assert frozen["freeze_policy"] == "latest valid pre-tip candidate"
    assert frozen["pregame_captured_at"] == "2026-11-01T01:55:00Z"
    assert frozen["injury_report"]["artifact_sha256"] == "e" * 64
    frozen_home = {
        row["player_name"]: row
        for row in frozen["projection"]["ledger"]["home"]["players"]
    }
    assert frozen_home["Home Star"]["frozen_nail"] == pytest.approx(2.5)

    changed_after_tip = _board(
        collected_at=datetime(2026, 11, 1, 2, 10, tzinfo=UTC),
        home_star_status=None,
        artifact_character="9",
        home_star_rating=9.0,
    )
    changed_after_tip["games"][0]["status"] = live["games"][0]["status"]
    _publish_board(
        changed_after_tip,
        state_root=tmp_path,
        s3_uri=None,
        collected_at=datetime(2026, 11, 1, 2, 10, tzinfo=UTC),
    )

    assert frozen_path.read_bytes() == frozen_bytes
    latest = json.loads((tmp_path / "latest" / f"{GAME_DATE}.json").read_text())
    assert latest["games"][0]["projection"] == frozen["projection"]
    assert latest["games"][0]["injury_report"] == frozen["injury_report"]


def test_candidate_is_archived_two_hours_before_tip_for_operational_resilience(
    tmp_path,
) -> None:
    payload = _board(
        collected_at=datetime(2026, 11, 1, 0, 40, tzinfo=UTC)
    )

    result = _publish_board(
        payload,
        state_root=tmp_path,
        s3_uri=None,
        collected_at=datetime(2026, 11, 1, 0, 40, tzinfo=UTC),
    )

    assert result["newly_published_candidate_count"] == 1
    candidates = list((tmp_path / "pregame-candidates" / GAME_ID).glob("*.json"))
    assert len(candidates) == 1
    candidate = json.loads(candidates[0].read_text())
    assert candidate["captured_at"] == "2026-11-01T00:40:00Z"
    assert candidate["game"]["projection"]["ledger"]


def _legacy_frozen(game: dict[str, object]) -> dict[str, object]:
    frozen = copy.deepcopy(game)
    projection = frozen["projection"]
    assert isinstance(projection, dict)
    projection.pop("ledger")
    projection.pop("input_provenance")
    projection.pop("fair_home_spread")
    minutes = frozen["projected_minutes"]
    assert isinstance(minutes, dict)
    for side in ("away", "home"):
        for row in minutes[side]:
            row.pop("projected_nail", None)
            row.pop("expected_contribution", None)
    return frozen


def test_matching_bundle_materializes_legacy_attribution_without_mutating_frozen() -> None:
    generated = _board(
        collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC)
    )["games"][0]
    frozen = _legacy_frozen(generated)
    frozen_before = copy.deepcopy(frozen)
    game = copy.deepcopy(generated)

    _apply_frozen_pregame(game, frozen)
    _require_or_attach_frozen_attribution(
        game,
        frozen=frozen,
        generated=generated,
    )

    assert frozen == frozen_before
    assert game["projection"]["ledger"]["away"] == generated["projection"]["ledger"]["away"]
    assert game["projection"]["ledger"]["home"] == generated["projection"]["ledger"]["home"]
    assert game["projection"]["ledger"]["home_margin"] == pytest.approx(
        generated["projection"]["ledger"]["home_margin"]
    )
    assert game["projected_minutes"] == generated["projected_minutes"]


def test_legacy_attribution_rejects_a_different_serving_bundle() -> None:
    generated = _board(
        collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC)
    )["games"][0]
    frozen = _legacy_frozen(generated)
    frozen["projection"]["serving_bundle_sha256"] = "0" * 64
    game = copy.deepcopy(generated)
    _apply_frozen_pregame(game, frozen)

    with pytest.raises(ValueError, match="serving bundles differ"):
        _require_or_attach_frozen_attribution(
            game,
            frozen=frozen,
            generated=generated,
        )


def test_live_score_watcher_cannot_strip_prediction_attribution(
    tmp_path, monkeypatch
) -> None:
    payload = _board(
        collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC)
    )
    generated = payload["games"][0]
    frozen = _legacy_frozen(generated)
    latest_path = tmp_path / "latest" / f"{GAME_DATE}.json"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text(json.dumps(payload))
    frozen_path = tmp_path / "frozen" / f"{GAME_ID}.json"
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text(json.dumps(frozen))
    monkeypatch.setattr(
        "nba_lineup_model.season.official_schedule_runtime._scoreboard_games",
        lambda _root: {
            GAME_ID: {
                "gameId": GAME_ID,
                "gameStatus": 2,
                "gameStatusText": "Q1 10:00",
                "gameClock": "PT10M00.00S",
                "period": 1,
                "homeTeam": {"teamTricode": "BBB", "score": 4},
                "awayTeam": {"teamTricode": "AAA", "score": 2},
            }
        },
    )

    refresh_official_live_score_state(
        game_date=GAME_DATE,
        state_dir=tmp_path,
        s3_uri=None,
    )

    latest = json.loads(latest_path.read_text())
    game = latest["games"][0]
    assert game["status"]["state"] == "in"
    assert game["home_score"] == 4
    assert game["projection"]["ledger"]["away"] == generated["projection"]["ledger"]["away"]
    assert game["projection"]["ledger"]["home"] == generated["projection"]["ledger"]["home"]
    assert game["projection"]["ledger"]["home_margin"] == pytest.approx(
        generated["projection"]["ledger"]["home_margin"]
    )
    assert game["projected_minutes"] == generated["projected_minutes"]


def test_game_lifecycle_finalizes_every_serving_artifact_and_repairs_partial_final(
    tmp_path, monkeypatch
) -> None:
    payload = _board(
        collected_at=datetime(2026, 11, 1, 1, 55, tzinfo=UTC)
    )
    game = payload["games"][0]
    latest_path = tmp_path / "latest" / f"{GAME_DATE}.json"
    latest_path.parent.mkdir(parents=True)
    latest_path.write_text(json.dumps(payload))
    frozen_path = tmp_path / "frozen" / f"{GAME_ID}.json"
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text(json.dumps(game))
    final_event = {
        "gameId": GAME_ID,
        "gameStatus": 3,
        "gameStatusText": "Final",
        "gameClock": "",
        "period": 4,
        "homeTeam": {"teamTricode": "BBB", "score": 105},
        "awayTeam": {"teamTricode": "AAA", "score": 100},
    }
    boxscore = {
        "game": {
            "gameId": GAME_ID,
            "awayTeam": {
                "teamTricode": "AAA",
                "players": [
                    {
                        "personId": player_id,
                        "firstName": "Away",
                        "familyName": str(player_id),
                        "statistics": {"minutes": minutes},
                    }
                    for player_id, minutes in (
                        (1, "PT100M"),
                        (2, "PT80M"),
                        (3, "PT60M"),
                    )
                ],
            },
            "homeTeam": {
                "teamTricode": "BBB",
                "players": [
                    {
                        "personId": player_id,
                        "firstName": "Home",
                        "familyName": str(player_id),
                        "statistics": {"minutes": minutes},
                    }
                    for player_id, minutes in (
                        (4, "PT120M"),
                        (5, "PT80M"),
                        (6, "PT40M"),
                    )
                ],
            },
        }
    }
    monkeypatch.setattr(
        "nba_lineup_model.season.official_schedule_runtime._scoreboard_games",
        lambda _root: {GAME_ID: final_event},
    )
    monkeypatch.setattr(
        "nba_lineup_model.season.official_schedule_runtime._fetch_json",
        lambda _url: _forecast(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.season.official_schedule_runtime.NbaCdnClient.fetch_boxscore",
        lambda self, game_id, use_cache=False: SimpleNamespace(payload=boxscore),
    )

    result = refresh_official_live_score_state(
        game_date=GAME_DATE,
        state_dir=tmp_path,
        s3_uri=None,
    )

    summary_path = tmp_path / "final" / f"{GAME_ID}.summary.json"
    boxscore_path = tmp_path / "final" / f"{GAME_ID}.nba.json"
    assert result["all_final"] is True
    assert summary_path.is_file()
    assert boxscore_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["grading"]["model_favorite_won"] is not None
    assert summary["retrospective"]["actual_minutes"]
    assert summary["retrospective"]["ledger"]
    served = read_schedule_day(GAME_DATE, state_dir=tmp_path)["games"][0]
    assert served["pregame"]["projection"]["ledger"]
    assert served["final_audit"]["retrospective"]["ledger"]
    assert served["retrospective"]["actual_minutes"]

    immutable_summary = summary_path.read_bytes()
    boxscore_path.unlink()
    repaired = refresh_official_live_score_state(
        game_date=GAME_DATE,
        state_dir=tmp_path,
        s3_uri=None,
    )
    assert repaired["newly_finalized_game_count"] == 1
    assert summary_path.read_bytes() == immutable_summary
    assert json.loads(boxscore_path.read_text()) == boxscore
