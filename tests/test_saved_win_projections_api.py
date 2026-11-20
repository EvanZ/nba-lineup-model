"""The preseason page reads one pinned S3 object, with no evaluator or data writes."""

import hashlib
import io
import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import saved_win_projections as api
from nba_lineup_model.web_api.gateway import MarketGateway


@pytest.fixture
def saved(monkeypatch):
    teams = [f"T{i:02d}" for i in range(30)]
    players = [
        {
            "player_id": i + 1,
            "player_name": f"Player {i + 1}",
            "team": teams[i % 30],
            "projected_nail": 1.2345678912345,
            "availability_probability": 0.8,
            "conditional_minutes_per_game": 25.0,
            "baseline_minutes_per_game": 16.0,
            "baseline_minute_share": 16 / 240,
        }
        for i in range(521)
    ]
    rows = [
        {
            "team": t,
            "team_strength": 0.12345678912345,
            "projected_wins": 42.0,
            "projected_losses": 40.0,
            "scheduled_wins": 41.0,
            "unassigned_wins": 1.0,
            "win_total_p1": 20,
            "win_total_p5": 30,
            "win_total_p50": 42,
            "win_total_p95": 55,
            "win_total_p99": 60,
        }
        for t in teams
    ]
    games = [
        {
            "game_id": f"g{n}-{i}",
            "game_date": "2026-10-20",
            "home_team": teams[i],
            "away_team": teams[i + 15],
            "home_back_to_back": 0,
            "away_back_to_back": 1,
        }
        for n in range(80)
        for i in range(15)
    ]
    doc = {
        "minutes": {
            "season": "2026-27",
            "completed_season": "2025-26",
            "model": "Saved model",
            "teams": teams,
            "players": players,
            "initial_rotation_size": 15,
            "regulation_team_minutes": 240.0,
            "contract": "Saved contract",
        },
        "win_projection": {
            "season": "2026-27",
            "teams": rows,
            "scheduled_games": games,
            "scheduled_game_count": 1200,
            "scheduled_games_per_team": 80,
            "unassigned_regular_games_per_team": 2,
            "home_court": 2.0,
            "back_to_back": -0.5,
            "win_probability_scale": 0.2,
            "calibration": {
                "calibration_beta": 0.2,
                "production_beta": 0.2,
                "holdout_brier": 0.2,
                "holdout_accuracy": 0.6,
            },
        },
    }
    reader = Mock()
    source = {**api.SOURCE}
    state = {}

    def publish():
        state["raw"] = json.dumps(doc).encode()
        source.update(sha256=hashlib.sha256(state["raw"]).hexdigest(), size=len(state["raw"]))
        monkeypatch.setattr(
            api,
            "SOURCE_ID",
            hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        )
        api.snapshot.cache_clear()

    def get(**kwargs):
        assert kwargs == {
            "Bucket": api.BUCKET,
            "Key": source["uri"].split(api.BUCKET + "/", 1)[1],
            "VersionId": source["version_id"],
        }
        return {
            "Body": io.BytesIO(state["raw"]),
            "VersionId": source["version_id"],
            "ETag": source["etag"],
        }

    reader.client.get_object = Mock(side_effect=get)
    monkeypatch.setattr(api, "SOURCE", source)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    publish()
    yield doc, state, reader, publish
    api.snapshot.cache_clear()


def test_exact_saved_values_full_coverage_and_one_cached_read(saved):
    doc, _, reader, _ = saved
    legacy = Mock(side_effect=AssertionError("Legacy evaluator must not start"))
    client = TestClient(MarketGateway(legacy))
    r = client.get("/api/win-projections")
    assert r.status_code == 200
    d = r.json()
    assert d["kind"] == "saved_preseason_win_projections"
    assert (
        d["state_kind"] == "preseason_preview"
        and r.headers["cache-control"] == "private, max-age=3600"
    )
    for key, value in doc["minutes"].items():
        assert d[key] == value
    assert d["win_projection"] == doc["win_projection"]
    assert len(d["teams"]) == 30 and len(d["players"]) == 521
    assert len(d["win_projection"]["scheduled_games"]) == 1200
    assert client.get("/api/win-projections").json() == d
    reader.client.get_object.assert_called_once()
    legacy.assert_not_called()
    with pytest.raises(ValueError):
        api.snapshot("different-selection")


def test_changed_bytes_fail_without_caching_or_fallback(saved):
    _, state, reader, _ = saved
    raw = state["raw"]
    state["raw"] += b" "
    client = TestClient(MarketGateway(None))
    r = client.get("/api/win-projections")
    assert r.status_code == 503 and r.headers["cache-control"] == "no-store"
    assert api.snapshot.cache_info().currsize == 0
    state["raw"] = raw
    assert client.get("/api/win-projections").status_code == 200
    assert reader.client.get_object.call_count == 2


@pytest.mark.parametrize(
    "damage", ["season", "team", "player", "schedule", "interval", "nonfinite"]
)
def test_incomplete_or_invalid_snapshots_fail_explicitly(saved, damage):
    doc, _, _, publish = saved
    if damage == "season":
        doc["minutes"]["season"] = "2025-26"
    elif damage == "team":
        doc["win_projection"]["teams"].pop()
    elif damage == "player":
        doc["minutes"]["players"].pop()
    elif damage == "schedule":
        doc["win_projection"]["scheduled_games"].pop()
    elif damage == "interval":
        doc["win_projection"]["teams"][0]["win_total_p95"] = 1
    else:
        doc["minutes"]["players"][0]["projected_nail"] = float("nan")
    publish()
    r = TestClient(MarketGateway(None)).get("/api/win-projections")
    assert r.status_code == 503 and r.headers["cache-control"] == "no-store"
    assert api.snapshot.cache_info().currsize == 0


@pytest.mark.parametrize("field", ["VersionId", "ETag"])
def test_different_object_version_is_not_accepted(saved, field):
    _, _, reader, _ = saved
    get = reader.client.get_object.side_effect

    def wrong(**kwargs):
        response = get(**kwargs)
        response[field] = "different-object"
        return response

    reader.client.get_object.side_effect = wrong
    r = TestClient(MarketGateway(None)).get("/api/win-projections")
    assert r.status_code == 503 and r.headers["cache-control"] == "no-store"
    assert api.snapshot.cache_info().currsize == 0
