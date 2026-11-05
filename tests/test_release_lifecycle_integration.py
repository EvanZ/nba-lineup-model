"""Recorded-source lifecycle checks; run explicitly on EC2, never against production.

Set GESTALT_LIFECYCLE_SOURCE_RELEASE to an immutable dev runtime. Tests copy
only the selected slate and official boxes into pytest's private directory.
The publication timestamps are a simulated lifecycle, NOT claims about the
original publication history or about model performance. No real pointers,
collectors, models, or historical forecasts are changed.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.forecast_publication import (
    ForecastRegistry, record_verified_release_forecasts,
)
from nba_lineup_model.season.release_artifact_contracts import (
    ArtifactReference, ForecastVersionManifest, MarketSnapshotManifest,
    SchedulingInstance, SealedCodeIdentity, ValidationRecord,
)
from nba_lineup_model.season.serving_selection import (
    ObservationSnapshot, build_schedule_selection, digest, verify_schedule_selection,
)
from nba_lineup_model.web_api.release_serving import ReleaseServing

SOURCE = os.environ.get("GESTALT_LIFECYCLE_SOURCE_RELEASE")
pytestmark = pytest.mark.skipif(
    not SOURCE, reason="Requires explicitly selected EC2 dev runtime; not a synthetic pass"
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def ref(kind, value):
    checksum = digest(value)
    return ArtifactReference(
        artifact_kind=kind, schema_version=1, content_sha256=checksum,
        uri=f"fixture://{kind}/{checksum}",
    )


class Clock(datetime):
    current = datetime(2026, 1, 1, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.current.astimezone(tz) if tz else cls.current.replace(tzinfo=None)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    root = Path(SOURCE).resolve()
    if not root.is_relative_to(Path("/var/lib/nba-gestalt-dev")):
        pytest.fail("Lifecycle fixtures must come from an explicitly selected dev runtime")
    source = ReleaseServing.load(root)
    assert source is not None
    schedule = source.path("schedule")
    index = verify_schedule_selection(schedule)
    # The entire earliest completed slate is the fixture, not selected winners.
    dates = sorted(index["dates"])
    day = next(day for day in dates if all(
        g.get("status", {}).get("completed")
        for g in json.loads((schedule / index["dates"][day]["path"]).read_text())["games"]
    ))
    board = json.loads((schedule / index["dates"][day]["path"]).read_text())
    assert board["games"], "A zero-game day cannot stand in for a completed fixture slate"
    boxes = {}
    hashes = {}
    for game in board["games"]:
        path = schedule / "final" / f"{game['game_id']}.nba.json"
        boxes[game["game_id"]] = json.loads(path.read_text())
        assert boxes[game["game_id"]]["game"]["gameId"] == game["game_id"]
        hashes[str(path)] = file_sha256(path)
    hashes[str(schedule / index["dates"][day]["path"])] = file_sha256(
        schedule / index["dates"][day]["path"]
    )
    monkeypatch.setattr("nba_lineup_model.season.forecast_publication.datetime", Clock)
    # Block network even if a code path unexpectedly tries to fetch a replacement.
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("Unexpected network"))
    result = SimpleNamespace(
        root=tmp_path, source=source, day=day, board=board, boxes=boxes, hashes=hashes,
        registry=ForecastRegistry(tmp_path / "registry"),
        observations=tmp_path / "observations",
    )
    yield result
    assert all(file_sha256(Path(path)) == checksum for path, checksum in hashes.items())


def stage(recorded, game, *, name, computed_at, scheduling=None, edition="fixture-original"):
    payload = copy.deepcopy(game.get("pregame", game))
    # Isolate archived forecast components from postgame observations and old
    # publication labels. The fixture is explicitly a simulation of publication.
    payload = {key: payload.get(key) for key in (
        "game_id", "home_team", "away_team", "start_time", "projection",
        "projected_minutes", "injury_report", "market", "prediction_market",
    )}
    payload.update(home_score=None, away_score=None, status={"state": "pre", "completed": False})
    payload["forecast_version"] = {"mode": "lifecycle_test_fixture", "edition_id": edition}
    scheduling = scheduling or SchedulingInstance(
        scheduling_instance_id=f"fixture-{game['game_id']}",
        game_id=game["game_id"], official_schedule_revision=ref("nba_schedule", game["start_time"]),
        scheduled_tipoff_at=stamp(game["start_time"]), cutoff_at=stamp(game["start_time"]),
    )
    payload["start_time"] = scheduling.scheduled_tipoff_at.isoformat()
    ledger = payload["projection"]["ledger"]
    parts = {
        "projected_minutes": payload["projected_minutes"],
        "player_ratings": {
            str(row["player_id"]): row["frozen_nail"]
            for side in ("home", "away") for row in ledger[side]["players"]
        },
        "contributions": {side: ledger[side]["players"] for side in ("home", "away")},
        "ledger": ledger,
    }
    version = ForecastVersionManifest(
        forecast_id=name, season="2026-27", game_id=game["game_id"], scheduling=scheduling,
        model_state=ref("daily_model_state", ledger["provenance"]),
        edition_id=edition, injury_revision=ref("injury_revision", payload["injury_report"]),
        roster_revision=ref("roster_revision", payload["projected_minutes"]),
        information_cutoff_at=scheduling.cutoff_at, computed_at=computed_at,
        components={key: ref(key, value) for key, value in parts.items()},
        predicted_home_margin=payload["projection"]["home_margin"],
        predicted_home_win_probability=payload["projection"]["home_win_probability"],
        ledger_home_margin=ledger["home_margin"], numeric_tolerance=1e-8,
        validation=ValidationRecord(status="clean"),
        code=SealedCodeIdentity(
            package_sha256=digest("fixture-code"), dependency_lock_sha256=digest("fixture-lock"),
            git_revision="f"*40, sealed=True,
        ),
    )
    recorded.registry.stage(version, payload)
    return version, payload


def release(recorded, monkeypatch, name, versions, board, now):
    Clock.current = now
    source = recorded.root / name / "source"
    write(source / "latest" / f"{recorded.day}.json", board)
    root = recorded.root / name / "runtime"
    root.mkdir()
    selected = root / "selected-schedule"
    index = build_schedule_selection(
        source, selected, edition_id=versions[0].edition_id,
        forecast_registry_root=recorded.registry.root,
        version_references={v.game_id: v.forecast_id for v in versions},
        now=now,
    )
    # Assemble minimal, checksummed release bindings. API bootstrap is real;
    # no schedule, grading, observation, or metric reader is mocked.
    bindings = {
        "schema_version": 1, "release_id": name, "edition_id": versions[0].edition_id,
        "edition_source": "lifecycle_test_fixture", "evidence_through_date": "2026-10-19",
        "components": {"forecast_selection_id": index["selection_map_sha256"]},
        "references": {"schedule": "selected-schedule"},
        "surface_bindings": {key: {} for key in (
            "player_pages", "rankings", "teams", "lineups", "wins", "schedule"
        )},
        "historical_runtime": str(root), "historical_files": {},
    }
    bindings["bindings_sha256"] = digest(bindings)
    write(root / "serving-bindings.json", bindings)
    write(root / "runtime-manifest.json", {
        "runtime_sha256": digest(name),
        "identity": {"serving_bindings_sha256": bindings["bindings_sha256"],
                     "approved_release_sha256": name},
    })
    pointer = root.parent / "live-release"
    pointer.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("GESTALT_ACTIVE_RELEASE_ROOT", str(pointer))
    monkeypatch.setenv("GESTALT_LIVE_SCORE_STATE_DIR", str(recorded.observations))
    from nba_lineup_model.web_api.app import create_app
    # These endpoints do not need the model evaluator. Supplying it prevents
    # loading unrelated historical training data; all tested endpoints are real.
    client = TestClient(create_app(evaluator=object()))
    return root, client


def response(client, day):
    value = client.get(f"/api/schedule?date={day}")
    assert value.status_code == 200, value.text
    return value.json()


def publish(root, *, at, generation=1):
    record_verified_release_forecasts(
        root, environment="dev", generation=generation,
        operation_id=f"fixture-{root.parent.name}", published_at=at,
    )


def final_board(recorded, *, include=None):
    board = copy.deepcopy(recorded.board)
    include = include if include is not None else set(recorded.boxes)
    for game in board["games"]:
        if game["game_id"] not in include:
            game.update(home_score=None, away_score=None, status={"state": "pre", "completed": False})
    board["collected_at"] = Clock.current.isoformat()
    return board


def test_full_recorded_slate_through_cutoff_scores_boxes_and_metrics(recorded, monkeypatch):
    r = recorded
    earliest = min(stamp(g["start_time"]) for g in r.board["games"])
    latest = max(stamp(g["start_time"]) for g in r.board["games"])
    before = earliest - timedelta(minutes=30)
    versions, payloads = [], []
    for game in r.board["games"]:
        v, p = stage(r, game, name=f"initial-{game['game_id']}", computed_at=before)
        r.registry.validate_scheduling_transition(v.scheduling)
        versions.append(v)
        payloads.append(p)
        for offset, suffix, spread in ((-10, "closing", -2.5), (1, "live", -20.5)):
            r.registry.record_market(MarketSnapshotManifest(
                snapshot_id=f"{game['game_id']}-{suffix}", game_id=game["game_id"],
                scheduling_instance_id=v.scheduling.scheduling_instance_id,
                source="bookmaker_consensus",
                captured_at=v.scheduling.cutoff_at + timedelta(minutes=offset),
                line_id=suffix, valid=True,
            ), {"home_spread": spread, "away_spread": -spread, "provider": "fixture-consensus",
                "home_moneyline": -120, "away_moneyline": 100})
    board = {"game_date": r.day, "games": payloads, "collected_at": before.isoformat()}
    root, client = release(r, monkeypatch, "pregame", versions, board, before)
    initial = response(client, r.day)
    assert {g["game_id"] for g in initial["games"]} == set(r.boxes)
    publish(root, at=before + timedelta(minutes=1))

    # Delayed worker: all games have started. No later/live line can replace
    # the final eligible pre-tip line. No actual box is available to the reader.
    Clock.current = latest + timedelta(hours=3)
    post = final_board(r)
    write(r.observations / "latest" / f"{r.day}.json", post)
    finalized_without_boxes = response(client, r.day)
    for original, game in zip(initial["games"], finalized_without_boxes["games"], strict=True):
        assert game["projection"] == original["projection"]
        assert game["projected_minutes"] == original["projected_minutes"]
        assert game["market"]["home_spread"] == -2.5
        assert "missing_official_box_score" in game["data_quality_errors"]
        assert game["final_audit"]["grading"]["model_favorite_won"] is not None
        assert game["final_audit"]["grading"]["market"]["model_ats_result"] is not None

    # All official boxes arrive without changing ratings, projected minutes, or
    # market selection. Check each observed player's minutes against NBA input.
    for gid, box in r.boxes.items():
        write(r.observations / "final" / f"{gid}.nba.json", box)
    completed = response(client, r.day)
    observed_players = 0
    for original, game in zip(initial["games"], completed["games"], strict=True):
        gid = game["game_id"]
        assert not game.get("data_quality_errors"), game.get("data_quality_errors")
        assert game["projection"] == original["projection"]
        assert game["projected_minutes"] == original["projected_minutes"]
        ledger = game["retrospective"]["ledger"]
        assert ledger["home_court_adjustment"] == game["projection"]["ledger"]["home_court_adjustment"]
        assert ledger["home_margin"] == pytest.approx(game["retrospective"]["home_margin"])
        for side in ("home", "away"):
            actual = {int(p["personId"]): p for p in r.boxes[gid]["game"][side+"Team"]["players"]}
            rows = ledger[side]["players"]
            from nba_lineup_model.season.official_schedule_runtime import _duration_minutes
            positive = {pid for pid, p in actual.items() if _duration_minutes(p["statistics"]["minutes"]) > 0}
            assert {int(row["player_id"]) for row in rows} == positive
            total = sum(_duration_minutes(p["statistics"]["minutes"]) for p in actual.values())
            for row in rows:
                if int(row["player_id"]) in actual:
                    minutes = _duration_minutes(actual[int(row["player_id"])]["statistics"]["minutes"])
                    assert row["actual_minutes"] == pytest.approx(minutes)
                    observed_players += 1
                assert row["actual_contribution"] == pytest.approx(
                    5 * row["actual_minutes"] / total * row["frozen_nail"]
                )
            assert ledger[side]["actual_minutes"] == pytest.approx(_duration_minutes(
                r.boxes[gid]["game"][side+"Team"]["statistics"]["minutes"]
            ), abs=0.1)
    performance = client.get("/api/schedule-performance")
    assert performance.status_code == 200, performance.text
    season = next(w for w in performance.json()["windows"] if w["key"] == "season")
    assert season["games"] == len(r.boxes)
    records = client.get("/api/current-season-records")
    assert records.status_code == 200, records.text
    assert set(records.json()["completed_game_ids"]) == set(r.boxes)

    # Captured requests are stable when a collector advances concurrently.
    snapshot = ObservationSnapshot.capture(r.observations, [r.day])
    frozen_snapshot_id = snapshot.observation_id
    post["games"][0]["home_score"] += 1
    write(r.observations / "latest" / f"{r.day}.json", post)
    assert snapshot.observation_id == frozen_snapshot_id
    assert ObservationSnapshot.capture(r.observations, [r.day]).observation_id != frozen_snapshot_id

    write(r.root / "lifecycle-evidence.json", {
        "source_release": r.source.release_id, "source_files": r.hashes,
        "fixture_date": r.day, "game_ids": sorted(r.boxes),
        "checked_actual_player_rows": observed_players,
        "phases": ["pregame", "delayed_cutoff", "final_without_boxes", "official_boxes", "metrics"],
        "real_model_replay": False, "real_publication_history_changed": False,
    })


def prepared(recorded, monkeypatch, *, omit_publication=False):
    r = recorded
    before = min(stamp(g["start_time"]) for g in r.board["games"]) - timedelta(minutes=30)
    versions, payloads = [], []
    for game in r.board["games"]:
        version, payload = stage(r, game, name=f"first-{game['game_id']}", computed_at=before)
        r.registry.validate_scheduling_transition(version.scheduling)
        versions.append(version)
        payloads.append(payload)
    board = {"games": payloads, "game_date": r.day, "collected_at": before.isoformat()}
    root, client = release(r, monkeypatch, "first", versions, board, before)
    if not omit_publication:
        publish(root, at=before + timedelta(seconds=1))
    return root, client, versions, board, before


def test_missing_original_keeps_all_results_but_never_invents_forecast(recorded, monkeypatch):
    r = recorded
    root, client, versions, _, before = prepared(r, monkeypatch, omit_publication=True)
    Clock.current = max(v.scheduling.cutoff_at for v in versions) + timedelta(hours=3)
    write(r.observations / "latest" / f"{r.day}.json", final_board(r))
    first = response(client, r.day)
    assert {g["game_id"] for g in first["games"]} == set(r.boxes)
    for game in first["games"]:
        assert game["projection"] is None
        assert "missing_original_forecast" in game["data_quality_errors"]
        assert game["status"]["completed"]
        assert game["home_score"] is not None
    # Recovering a stalled publisher cannot backdate publication eligibility.
    publish(root, at=Clock.current)
    after = response(client, r.day)
    assert all(g["projection"] is None for g in after["games"])
    assert set(client.get("/api/current-season-records").json()["completed_game_ids"]) == set(r.boxes)
    assert client.get("/api/schedule-performance").json()["windows"] == []
    assert len(list((r.registry.root / "failures").glob("*.json"))) == len(r.boxes)


@pytest.mark.parametrize("damage", ["missing", "wrong_game", "malformed"])
def test_one_box_failure_is_visible_without_hiding_other_games(recorded, monkeypatch, damage):
    r = recorded
    _, client, versions, _, _ = prepared(r, monkeypatch)
    Clock.current = max(v.scheduling.cutoff_at for v in versions) + timedelta(hours=3)
    write(r.observations / "latest" / f"{r.day}.json", final_board(r))
    broken = sorted(r.boxes)[0]
    for gid, box in r.boxes.items():
        if gid != broken:
            write(r.observations / "final" / f"{gid}.nba.json", box)
    if damage == "wrong_game":
        box = copy.deepcopy(r.boxes[broken])
        box["game"]["gameId"] = "wrong"
        write(r.observations / "final" / f"{broken}.nba.json", box)
    elif damage == "malformed":
        (r.observations / "final" / f"{broken}.nba.json").write_text("{")
    games = response(client, r.day)["games"]
    assert {g["game_id"] for g in games} == set(r.boxes)
    for game in games:
        if game["game_id"] == broken:
            assert game.get("data_quality_errors")
            assert not game.get("retrospective")
        else:
            assert not game.get("data_quality_errors")
            assert game["retrospective"]["ledger"]["game_id"] == game["game_id"]
        assert game["final_audit"]["grading"]["model_favorite_won"] is not None
    # Late repair completes the affected retrospective, no model re-fit required.
    projections = {g["game_id"]: digest(g["projection"]) for g in games}
    write(r.observations / "final" / f"{broken}.nba.json", r.boxes[broken])
    repaired = response(client, r.day)["games"]
    assert all(not g.get("data_quality_errors") for g in repaired)
    assert projections == {g["game_id"]: digest(g["projection"]) for g in repaired}


def test_api_rejects_stale_page_pin_and_changed_selected_file(recorded, monkeypatch):
    r = recorded
    root, client, _, _, _ = prepared(r, monkeypatch)
    fresh = client.get(f"/api/schedule?date={r.day}")
    pin = fresh.headers["X-Gestalt-Release"]
    assert client.get(f"/api/schedule?date={r.day}",
                      headers={"X-Gestalt-Release": pin}).status_code == 200
    stale = client.get(f"/api/schedule?date={r.day}",
                       headers={"X-Gestalt-Release": "another-release"})
    assert stale.status_code == 409
    assert stale.json()["error"] == "release_changed"
    path = root / "selected-schedule" / "latest" / f"{r.day}.json"
    payload = json.loads(path.read_text())
    payload["games"][0]["projection"]["home_margin"] += 10
    write(path, payload)
    rejected = client.get(f"/api/schedule?date={r.day}")
    assert rejected.status_code == 503
    assert "changed after publication" in rejected.json()["detail"]


def test_late_start_cannot_reopen_cutoff_but_official_reschedule_can(recorded, monkeypatch):
    r = recorded
    _, old_client, versions, board, before = prepared(r, monkeypatch)
    first = versions[0]
    Clock.current = first.scheduling.cutoff_at + timedelta(minutes=5)
    initial = response(old_client, r.day)["games"][0]
    original = initial["forecast_selection"]["registry_selection"]
    assert original["forecast"]["selected_forecast_id"] == first.forecast_id

    moved = first.scheduling.model_copy(update={
        "scheduling_instance_id": first.scheduling.scheduling_instance_id + "-moved",
        "scheduled_tipoff_at": first.scheduling.cutoff_at + timedelta(days=1),
        "cutoff_at": first.scheduling.cutoff_at + timedelta(days=1),
    })
    with pytest.raises(ValueError, match="late starts"):
        r.registry.validate_scheduling_transition(moved, previous=first.scheduling)
    moved = moved.model_copy(update={
        "event": "rescheduled",
        "supersedes_scheduling_instance_id": first.scheduling.scheduling_instance_id,
        "official_schedule_revision": ref("nba_schedule", "controlled reschedule fixture"),
    })
    with pytest.raises(ValueError, match="started"):
        r.registry.validate_scheduling_transition(moved, previous=first.scheduling, play_has_started=True)
    r.registry.validate_scheduling_transition(moved, previous=first.scheduling)
    new, payload = stage(
        r, r.board["games"][0], name="rescheduled-version",
        computed_at=Clock.current, scheduling=moved,
    )
    root, client = release(
        r, monkeypatch, "rescheduled", [new],
        {"games": [payload], "collected_at": Clock.current.isoformat()},
        Clock.current,
    )
    opened = response(client, r.day)["games"][0]
    assert opened["forecast_selection"]["registry_selection"]["state"] == "open"
    publish(root, at=Clock.current + timedelta(minutes=1))
    Clock.current = moved.cutoff_at + timedelta(minutes=1)
    locked = response(client, r.day)["games"][0]
    selected = locked["forecast_selection"]["registry_selection"]
    assert selected["forecast"]["selected_forecast_id"] == "rescheduled-version"
    assert original["forecast"]["selected_forecast_id"] == first.forecast_id
    assert r.registry.lock(
        edition_id=first.edition_id, scheduling=first.scheduling, environment="dev"
    ) == original


def test_suspension_retains_original_forecast_and_market(recorded, monkeypatch):
    r = recorded
    _, client, versions, _, _ = prepared(r, monkeypatch)
    first = versions[0]
    Clock.current = first.scheduling.cutoff_at + timedelta(minutes=20)
    initial = response(client, r.day)["games"][0]["forecast_selection"]["registry_selection"]
    suspended = first.scheduling.model_copy(update={
        "scheduling_instance_id": first.scheduling.scheduling_instance_id + "-resumed",
        "event": "suspended",
        "retains_original_selection_id": initial["forecast"]["selection_id"],
        "scheduled_tipoff_at": first.scheduling.cutoff_at + timedelta(days=1),
        "cutoff_at": first.scheduling.cutoff_at + timedelta(days=1),
    })
    r.registry.validate_scheduling_transition(suspended, previous=first.scheduling, play_has_started=True)
    new, payload = stage(
        r, r.board["games"][0], name="resume-candidate",
        computed_at=Clock.current, scheduling=suspended,
    )
    Clock.current = suspended.cutoff_at + timedelta(minutes=1)
    _, resumed = release(
        r, monkeypatch, "resumed", [new],
        {"games": [payload], "collected_at": Clock.current.isoformat()},
        Clock.current,
    )
    selected = response(resumed, r.day)["games"][0]["forecast_selection"]["registry_selection"]
    assert selected["forecast"] == initial["forecast"]
    assert selected["market"] == initial["market"]
    assert selected["retained_for_scheduling_instance"] == suspended.scheduling_instance_id


def test_published_injury_revision_changes_forecast_only_before_cutoff(recorded, monkeypatch):
    import math
    from nba_lineup_model.season.official_schedule_runtime import _normalize_official_board

    r = recorded
    _, old_client, versions, board, before = prepared(r, monkeypatch)
    game = r.board["games"][0]
    source = game["pregame"]
    ledger = source["projection"]["ledger"]
    chosen = next(p for p in source["projected_minutes"]["home"] if p["projected_minutes"] > 0)
    target_id = chosen["player_id"]
    players = [
        {**p, "team": game[side+"_team"], "conditional_minutes_per_game": p["conditional_minutes"]}
        for side in ("away", "home") for p in source["projected_minutes"][side]
    ]
    assumptions = copy.deepcopy(source["injury_report"]["assumptions"])
    assumptions = [p for p in assumptions if p["player_id"] != target_id]
    assumptions.append({
        "player_id": target_id, "player_name": chosen["player_name"], "team": game["home_team"],
        "status": "Out", "play_probability": 0.0, "reason": "Controlled lifecycle test injury",
    })
    for p in assumptions:
        p.update(game_date=r.day, matchup=game["away_team"]+"@"+game["home_team"])
    updated_at = before + timedelta(minutes=5)
    injury = {
        "source_url": source["injury_report"]["source_url"],
        "published_at": updated_at.isoformat(), "players": assumptions,
        "game_team_reports": {
            r.day+"|"+game["away_team"]+"@"+game["home_team"]: {
                game[side+"_team"]: {"status": "submitted"} for side in ("away", "home")
            }
        },
    }
    probability = source["projection"]["home_win_probability"]
    margin = source["projection"]["home_margin"]
    assert margin != 0, "Fixture needs a nonzero margin to preserve its existing calibration"
    forecast = {
        "model": "lifecycle-fixture", "model_track": "lifecycle-fixture",
        "as_of": ledger["provenance"]["rating_as_of"],
        "home_court": ledger["home_court_adjustment"],
        "back_to_back": ledger["back_to_back"]["coefficient"],
        "win_probability_scale": math.log(probability/(1-probability))/margin,
        "scheduled_games": [{
            "game_id": game["game_id"],
            "home_back_to_back": ledger["back_to_back"]["home_team_is_back_to_back"],
            "away_back_to_back": ledger["back_to_back"]["away_team_is_back_to_back"],
        }],
    }
    revised = _normalize_official_board(
        games=[{k: game[k] for k in ("game_id", "home_team", "away_team", "start_time")}],
        scoreboard={}, forecast_response={
            "win_projection": forecast,
            "minutes": {"players": players, "initial_rotation_size": 15,
                        "projection_provenance": ledger["provenance"]},
        }, injury_report=injury, bookmaker_lines={}, prediction_markets={},
        game_date=r.day, collected_at=updated_at, catalog_sha256=digest("fixture-catalog"),
        injury_artifact={"sha256": digest(injury)},
    )["games"][0]
    version, payload = stage(r, revised, name="injury-revision", computed_at=updated_at)
    assert version.scheduling == versions[0].scheduling
    new_versions = [version, *versions[1:]]
    new_board = copy.deepcopy(board)
    new_board["games"][0] = payload
    root, new_client = release(r, monkeypatch, "injury", new_versions, new_board, updated_at)
    initial_game = response(old_client, r.day)["games"][0]
    revised_game = response(new_client, r.day)["games"][0]
    def player(g):
        return next(p for p in g["projected_minutes"]["home"] if p["player_id"] == target_id)
    assert player(initial_game)["projected_minutes"] > 0
    assert player(revised_game)["projected_minutes"] == 0
    assert revised_game["projection"]["home_margin"] != initial_game["projection"]["home_margin"]
    assert {
        p["player_id"]:p["frozen_nail"]
        for side in ("away", "home") for p in revised_game["projection"]["ledger"][side]["players"]
    } == {
        p["player_id"]:p["frozen_nail"]
        for side in ("away", "home") for p in initial_game["projection"]["ledger"][side]["players"]
    }
    publish(root, at=updated_at + timedelta(seconds=1), generation=2)
    Clock.current = versions[0].scheduling.cutoff_at + timedelta(minutes=1)
    # Even a worker still referencing the earlier candidate resolves the same
    # latest successfully published forecast at cutoff.
    locked = response(old_client, r.day)["games"][0]
    assert locked["forecast_selection"]["registry_selection"]["forecast"]["selected_forecast_id"] == version.forecast_id
    assert player(locked)["projected_minutes"] == 0
    # A post-tip injury feed may be shown as an observation, not rewrite the forecast.
    live = copy.deepcopy(new_board)
    live["games"][0]["injury_report"] = {"assumptions": [], "fixture_revision": "late"}
    live["collected_at"] = Clock.current.isoformat()
    write(r.observations / "latest" / f"{r.day}.json", live)
    after = response(new_client, r.day)["games"][0]
    assert after["projection"] == locked["projection"]
    assert after["injury_report"] == locked["injury_report"]
    assert after["latest_observations"]["injury_report"]["fixture_revision"] == "late"


def test_reassembly_after_cutoff_retains_locked_forecast_and_market(recorded, monkeypatch):
    r = recorded
    _, client, versions, _, _ = prepared(r, monkeypatch)
    first = versions[0]
    Clock.current = max(v.scheduling.cutoff_at for v in versions) + timedelta(minutes=1)
    initial = next(g for g in response(client, r.day)["games"]
        if g["game_id"] == first.game_id)["forecast_selection"]["registry_selection"]
    _, payload = r.registry.version(first.forecast_id)
    _, rebuilt = release(r, monkeypatch, "after-cutoff", [first],
        {"games": [payload], "collected_at": Clock.current.isoformat()}, Clock.current)
    selected = response(rebuilt, r.day)["games"][0]["forecast_selection"]
    assert selected["registry_selection"]["forecast"] == initial["forecast"]
    assert selected["registry_selection"]["market"] == initial["market"]
    assert selected["scheduling_instance_id"] == initial["forecast"]["scheduling_instance_id"]
    assert selected["cutoff_at"] == initial["forecast"]["cutoff_at"]
