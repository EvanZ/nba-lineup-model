from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from test_official_schedule_runtime import GAME_DATE, _board

from nba_lineup_model.season.serving_selection import (
    ObservationSnapshot,
    build_schedule_selection,
    read_selected_day,
    verify_schedule_selection,
)
from nba_lineup_model.web_api.win_projections import read_schedule_day, read_schedule_performance


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def selection(tmp_path, monkeypatch):
    source = tmp_path / "source"
    board = _board(collected_at=datetime(2026, 11, 1, 1, 0, tzinfo=UTC))
    write(source / "latest" / f"{GAME_DATE}.json", board)
    # This migration test isolates exact selection preservation from the legacy
    # reader's own archive selection. The new reader must never invoke it.
    import nba_lineup_model.web_api.win_projections as wp

    monkeypatch.setattr(wp, "read_schedule_day", lambda *args, **kwargs: copy.deepcopy(board))
    root = tmp_path / "selected"
    build_schedule_selection(source, root, edition_id="approved-replay")
    monkeypatch.undo()
    return root, board


def test_migration_preserves_every_input_and_does_not_invent_publication(tmp_path, monkeypatch):
    root, original = selection(tmp_path, monkeypatch)
    result = read_schedule_day(GAME_DATE, state_dir=root)
    game = result["games"][0]
    for key in ("projection", "projected_minutes", "injury_report", "market", "prediction_market"):
        assert game.get(key) == original["games"][0].get(key)
    assert game["forecast_selection"]["original_publication_receipt"] is None
    assert game["forecast_selection"]["provenance"] == "preserved_existing_selection"
    assert result["serving_selection"]["edition_id"] == "approved-replay"
    assert len(verify_schedule_selection(root)["games"]) == 1


def test_reader_rejects_changed_selected_input_without_legacy_fallback(tmp_path, monkeypatch):
    root, _ = selection(tmp_path, monkeypatch)
    path = root / "latest" / f"{GAME_DATE}.json"
    value = json.loads(path.read_text())
    value["games"][0]["projection"]["home_margin"] = 500
    write(path, value)
    with pytest.raises(ValueError, match="checksum|changed"):
        read_schedule_day(GAME_DATE, state_dir=root)


def test_live_scores_and_final_grades_ignore_collector_predictions(tmp_path, monkeypatch):
    root, board = selection(tmp_path, monkeypatch)
    observation_root = tmp_path / "observations"
    live = copy.deepcopy(board)
    game = live["games"][0]
    game["home_score"], game["away_score"] = 110, 100
    game["status"] = {"state": "post", "completed": True}
    game["projection"]["home_margin"] = -50.0
    game["projected_minutes"]["home"][0]["projected_minutes"] = 500
    game["retrospective"] = {"home_margin": -100}
    write(observation_root / "latest" / f"{GAME_DATE}.json", live)
    snapshot = ObservationSnapshot.capture(observation_root, [GAME_DATE])
    result = read_selected_day(GAME_DATE, root=root, observations=snapshot)
    selected = result["games"][0]
    assert selected["home_score"] == 110
    assert selected["projection"] == board["games"][0]["projection"]
    audit = selected["final_audit"]
    assert audit["pregame"]["projection"] == selected["projection"]
    assert audit["grading"]["projected_home_margin"] == selected["projection"]["home_margin"]
    assert audit["grading"]["model_favorite_won"] is True
    assert "missing_official_box_score" in selected["data_quality_errors"]
    assert selected["retrospective"] == {}
    from nba_lineup_model.season.serving_selection import observation_context

    token = observation_context.set(snapshot)
    try:
        metrics = read_schedule_performance(state_dir=root)
    finally:
        observation_context.reset(token)
    assert next(row for row in metrics["windows"] if row["key"] == "season")["games"] == 1
    # Already captured request snapshots are insulated from later collector writes.
    game["home_score"] = 80
    write(observation_root / "latest" / f"{GAME_DATE}.json", live)
    assert (
        read_selected_day(GAME_DATE, root=root, observations=snapshot)["games"][0]["home_score"]
        == 110
    )
    assert (
        ObservationSnapshot.capture(observation_root, [GAME_DATE]).observation_id
        != snapshot.observation_id
    )


def test_observation_identity_excludes_forecasts_and_includes_real_observations(
    tmp_path, monkeypatch
):
    _, board = selection(tmp_path, monkeypatch)
    path = tmp_path / "observations/latest" / f"{GAME_DATE}.json"
    write(path, board)
    first = ObservationSnapshot.capture(path.parent.parent, [GAME_DATE])
    board["games"][0]["projection"]["home_margin"] += 100
    write(path, board)
    assert (
        ObservationSnapshot.capture(path.parent.parent, [GAME_DATE]).observation_id
        == first.observation_id
    )
    board["games"][0]["home_score"] = 11
    write(path, board)
    assert (
        ObservationSnapshot.capture(path.parent.parent, [GAME_DATE]).observation_id
        != first.observation_id
    )


def test_api_rejects_stale_page_pin_without_loading_a_substitute(monkeypatch, tmp_path):
    import nba_lineup_model.web_api.app as app_module

    class Bound:
        root = tmp_path
        release_id = "new-release"
        bindings = {"edition_id": "edition", "components": {"forecast_selection_id": "selection"}}

        def path(self, name):
            return tmp_path

        def metadata(self):
            return {"release_id": self.release_id}

    monkeypatch.setattr(app_module.ReleaseServing, "load", lambda root: Bound())
    monkeypatch.setattr(app_module, "load_process_activation_identity", lambda root: None)
    app = app_module.create_app(evaluator=object())
    client = TestClient(app)
    response = client.get("/api/serving-view", headers={"X-Gestalt-Release": "old-release"})
    assert response.status_code == 409
    assert response.json()["error"] == "release_changed"
    fresh = client.get("/api/serving-view", headers={"X-Gestalt-Release": "new-release"})
    assert fresh.status_code == 200
    assert fresh.headers["X-Gestalt-Edition"] == "edition"
    assert fresh.headers["X-Gestalt-Release"] == "new-release"


def test_official_box_arrival_fills_retrospective_without_changing_forecast(tmp_path, monkeypatch):
    root, board = selection(tmp_path, monkeypatch)
    gid = board["games"][0]["game_id"]
    live = copy.deepcopy(board)
    live["games"][0].update(
        home_score=110, away_score=100, status={"state": "post", "completed": True}
    )
    observation_root = tmp_path / "observations"
    write(observation_root / "latest" / f"{GAME_DATE}.json", live)
    before = read_selected_day(
        GAME_DATE,
        root=root,
        observations=ObservationSnapshot.capture(observation_root, [GAME_DATE]),
    )["games"][0]
    box = {"game": {"gameId": gid}}
    for side, ids in (("home", (4, 5, 6)), ("away", (1, 2, 3))):
        box["game"][side + "Team"] = {
            "teamTricode": board["games"][0][side + "_team"],
            "players": [
                {
                    "personId": pid,
                    "firstName": side,
                    "familyName": str(pid),
                    "statistics": {"minutes": minutes},
                }
                for pid, minutes in zip(ids, ("PT100M", "PT80M", "PT60M"), strict=True)
            ],
        }
    write(observation_root / "final" / f"{gid}.nba.json", box)
    after = read_selected_day(
        GAME_DATE,
        root=root,
        observations=ObservationSnapshot.capture(observation_root, [GAME_DATE]),
    )["games"][0]
    assert after["projection"] == before["projection"]
    assert after["final_audit"]["grading"] == before["final_audit"]["grading"]
    assert after["retrospective"]["ledger"]["home"]["actual_minutes"] == 240.0
    assert (
        after["retrospective"]["ledger"]["home_court_adjustment"]
        == after["projection"]["ledger"]["home_court_adjustment"]
    )


def test_original_live_selection_feeds_same_reader_and_excludes_missing_forecast(
    tmp_path, monkeypatch
):
    from test_forecast_publication import BEFORE, candidate

    from nba_lineup_model.season.forecast_publication import ForecastRegistry
    from nba_lineup_model.season.serving_selection import observation_context
    from nba_lineup_model.web_api.win_projections import read_current_season_records

    registry = ForecastRegistry(tmp_path / "registry")
    first, payload = candidate(registry, "original")
    registry.record_scheduling(first.scheduling)
    board = _board(collected_at=BEFORE)
    source = tmp_path / "source"
    write(source / "latest" / f"{GAME_DATE}.json", board)
    import nba_lineup_model.web_api.win_projections as wp

    with monkeypatch.context() as context:
        context.setattr(wp, "read_schedule_day", lambda *args, **kwargs: copy.deepcopy(board))
        root = tmp_path / "selected"
        index = build_schedule_selection(
            source,
            root,
            edition_id="edition",
            forecast_registry_root=registry.root,
            version_references={first.game_id: "original"},
            environment="dev",
            now=BEFORE,
        )
    assert index["publication_forecast_ids"] == ["original"]
    # No receipt: a late reader keeps the actual game but cannot invent a prediction.
    live = copy.deepcopy(board)
    live["games"][0].update(
        home_score=110, away_score=100, status={"state": "post", "completed": True}
    )
    obs_root = tmp_path / "observations"
    write(obs_root / "latest" / f"{GAME_DATE}.json", live)
    snapshot = ObservationSnapshot.capture(obs_root, [GAME_DATE])
    game = read_selected_day(GAME_DATE, root=root, observations=snapshot)["games"][0]
    assert game["projection"] is None
    assert "missing_original_forecast" in game["data_quality_errors"]
    assert game["home_score"] == 110
    token = observation_context.set(snapshot)
    try:
        assert read_current_season_records(state_dir=root)["completed_game_ids"] == [first.game_id]
        assert read_schedule_performance(state_dir=root)["windows"] == []
    finally:
        observation_context.reset(token)


def test_malformed_observation_date_does_not_drop_other_dates(tmp_path):
    root = tmp_path / "observations"
    write(root / "latest/2026-11-01.json", {"games": "invalid"})
    write(root / "latest/2026-11-02.json", {"games": [], "collected_at": "2026-11-03T06:00:00Z"})
    snapshot = ObservationSnapshot.capture(root, ["2026-11-01", "2026-11-02"])
    assert snapshot.errors["2026-11-01"]
    assert snapshot.boards["2026-11-02"]["games"] == []


def test_schedule_view_serializes_like_existing_endpoints_and_shares_snapshot(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pandas as pd

    import nba_lineup_model.web_api.app as api
    from nba_lineup_model.season.serving_selection import observation_context

    root, board = selection(tmp_path, monkeypatch)
    snapshot = ObservationSnapshot.capture(root, [GAME_DATE])
    captured = []

    class Bound:
        release_id = "test-release"
        bindings = {"edition_id": "edition", "components": {"forecast_selection_id": "selection"}}

        def path(self, name):
            return root

        def metadata(self):
            return {"release_id": self.release_id}

        def observations(self, observation_root):
            captured.append(snapshot)
            return snapshot

    def day(value, **kwargs):
        assert observation_context.get() is snapshot
        return {"date": value, "games": board["games"]}

    def performance(**kwargs):
        assert observation_context.get() is snapshot
        return {"windows": []}

    monkeypatch.setattr(api.ReleaseServing, "load", lambda root: Bound())
    monkeypatch.setattr(api, "load_process_activation_identity", lambda root: None)
    monkeypatch.setattr(api, "read_schedule_day", day)
    monkeypatch.setattr(api, "read_schedule_performance", performance)
    monkeypatch.setattr(api, "read_win_projection_cache", lambda **kwargs: {
        "minutes": {"players": [{"unavailable_stat": float("nan")}]},
        "win_projection": {"teams": []},
    })
    evaluator = SimpleNamespace(run_id="test", current_season_rankings=pd.DataFrame())
    client = TestClient(api.create_app(evaluator=evaluator))
    existing = client.get("/api/win-projections")
    response = client.get(f"/api/schedule-view?date={GAME_DATE}")
    assert existing.status_code == response.status_code == 200
    assert response.json()["forecast"] == existing.json()
    assert response.json()["forecast"]["players"][0]["unavailable_stat"] is None
    assert len(response.json()["calendar"]) == 7
    assert response.json()["day"]["games"] == board["games"]
    assert captured == [snapshot]
    assert response.headers["X-Gestalt-Observations"] == snapshot.observation_id
    assert client.get("/api/schedule-view?date=2026-02-31").status_code == 422
