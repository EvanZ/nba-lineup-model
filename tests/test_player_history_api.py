"""Producer-selected game histories: scope, provenance, completeness and corrections."""

import hashlib
import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from test_saved_schedule_api import SavedS3

from nba_lineup_model.web_api import player_history as api
from nba_lineup_model.web_api.gateway import MarketGateway

PID = 1629029
ROOT = f"lake/artifacts/player_game_history/season=2026-27/model_version=v1/player_id={PID}/"


def ref(root, sha="a" * 64):
    return {"uri": "s3://nba-gestalt-data/" + root + sha + ".json", "sha256": sha}


def ratings(day):
    return ref(
        f"lake/artifacts/player_ratings/season=2026-27/model_version=v1/game_date_pacific={day}/"
    )


def game(day, n):
    gid = f"002260{n:04d}"
    box = ref(f"lake/raw/cdn/boxscore/season=2026-27/game_id={gid}/")
    box.update(metadata_uri=box["uri"].replace(".json", ".meta.json"), metadata_sha256="b" * 64)
    return {
        "game_id": gid,
        "team_id": 1610612747 + n,
        "opponent_team_id": 1610612756,
        "team": "LAL" if n == 0 else "DAL",
        "opponent": "PHX",
        "home_away": "away",
        "team_score": 131,
        "opponent_score": 123,
        "result": "W",
        "actual_minutes": 35.733333333333334,
        "possessions": 78.5,
        "prediction_purpose": "historical_replay",
        "participation": {"status": "played", "reason": None, "box_score_row_present": True},
        "pregame": {
            "nail": 5.862227928952787,
            "projected_minutes": 36.79362996266019,
            "injury_status": "Available",
            "injury_reason": None,
        },
        "stats": {k: 0.5 if k.endswith("percentage") else 0 for k in api.STAT_FIELDS},
        "inputs": {
            **{
                k: ref(
                    f"lake/artifacts/{v}/season=2026-27/model_version=v1/game_date_pacific={day}/game_id={gid}/"
                )
                for k, v in (
                    ("actual_minutes", "game_predictions_actual_minutes"),
                    ("prediction", "game_predictions"),
                )
            },
            "box_score": box,
        },
    }


def publish(s3, point):
    raw = json.dumps(point, sort_keys=True, separators=(",", ":")).encode()
    sha = hashlib.sha256(raw).hexdigest()
    key = ROOT + f"game_date_pacific={point['game_date_pacific']}/{sha}.json"
    s3.put_fixture(key, point)
    return {
        "uri": "s3://nba-gestalt-data/" + key,
        "sha256": sha,
        "game_date_pacific": point["game_date_pacific"],
        "source_ratings": point["inputs"]["ratings_state"],
    }


def fixture():
    s3 = SavedS3()
    prior = ref("lake/artifacts/player_ratings/season=2026-27/model_version=v1/priors/")
    identity = {
        "schema_version": 1,
        "season": "2026-27",
        "model_version": "v1",
        "player_id": PID,
        "player_name": "Luka Dončić",
        "model_id": "model",
        "model_run_id": "run",
    }
    points = []
    for n, day in enumerate(("2026-11-02", "2026-11-04")):
        points.append(
            {
                **identity,
                "artifact_kind": "player_game_history",
                "game_date_pacific": day,
                "date_contract": "completed_game_date_v1",
                "calculation_contract": "frozen_priors_stints_joint_hca_fixed_b2b_v1",
                "initialized_on": "2026-10-29",
                "initialization": {"kind": "replacement", "input": prior},
                "nail": 5.925721236619028 + n / 10,
                "starting_nail": -5.109488431706889,
                "cumulative_possessions": 78.5 * (n + 1),
                "team_id": 1610612747 + n,
                "inputs": {"ratings_state": ratings(day), "priors": prior},
                "games": [game(day, n)],
            }
        )
    refs = [publish(s3, p) for p in points]
    index = {
        **identity,
        "artifact_kind": "player_game_history_index",
        "through_game_date_pacific": "2026-11-05",
        "initialization_input": prior,
        "source_ratings": ratings("2026-11-05"),
        "points": refs,
        "last_game": refs[-1],
        "current": {
            "game_date_pacific": "2026-11-05",
            "nail": 6.1,
            "cumulative_possessions": 157,
            "team_id": None,
            "source_ratings": ratings("2026-11-05"),
        },
        "starting_point": {
            "kind": "introduced_prior",
            "nail": points[0]["starting_nail"],
            "initialized_on": "2026-10-29",
            "first_state_date": "2026-10-29",
            "initialization": points[0]["initialization"],
            "source_ratings": ratings("2026-10-29"),
        },
    }
    s3.put_fixture(ROOT + "latest.json", index)
    return s3, index, points


def store_index(s3, index):
    s3.put_fixture(ROOT + "latest.json", index)


def replace_point(s3, index, point, n=0):
    index["points"][n] = publish(s3, point)
    index["last_game"] = index["points"][-1]
    store_index(s3, index)


def test_game_only_history_precision_trades_and_off_day_current():
    s3, index, points = fixture()
    result = api.HistoryReader(s3).history(PID)
    assert result["complete"] and result["kind"] == "saved_player_game_history"
    assert [p["game_date_pacific"] for p in result["points"]] == ["2026-11-02", "2026-11-04"]
    assert result["current"] == index["current"]
    assert result["last_game"]["game_date_pacific"] == "2026-11-04"
    assert not any(p["is_current"] for p in result["points"])
    assert result["starting_point"]["initialized_on"] == "2026-10-29"
    assert result["points"][0]["nail"] == points[0]["nail"]
    assert result["points"][1]["games"][0]["team"] == "DAL"
    assert result["points"][0]["games"][0]["team"] == "LAL"
    # No pinned upstream object exists in the fixture: any reconstruction would fail.
    assert len(s3.objects) == 3


@pytest.mark.parametrize("status", ["not_listed", "inactive", "did_not_play", "unknown"])
def test_preserves_participation_and_nullable_stats_without_inference(status):
    s3, index, points = fixture()
    g = points[0]["games"][0]
    g.update(actual_minutes=0, possessions=0, stats=None)
    g["participation"] = {
        "status": status,
        "reason": None,
        "box_score_row_present": status != "not_listed",
    }
    g["pregame"].update(
        projected_minutes=0, injury_status="Out", injury_reason="Right ACL Repair / Surgery"
    )
    replace_point(s3, index, points[0])
    saved = api.HistoryReader(s3).history(PID)["points"][0]["games"][0]
    assert saved == g and saved["stats"] is None


def test_nullable_pregame_and_empty_game_history():
    s3, index, points = fixture()
    points[0]["games"][0]["pregame"] = None
    replace_point(s3, index, points[0])
    assert api.HistoryReader(s3).history(PID)["points"][0]["games"][0]["pregame"] is None
    index.update(points=[], last_game=None)
    store_index(s3, index)
    h = api.HistoryReader(s3).history(PID)
    assert h["complete"] and h["points"] == [] and h["current"] == index["current"]


@pytest.mark.parametrize(
    "problem",
    [
        "player_id",
        "season",
        "model_version",
        "date",
        "checksum",
        "missing",
        "scope",
        "stats",
        "percentage",
        "duplicate",
        "model_run_id",
    ],
)
def test_rejects_corrupt_missing_or_misidentified_point(problem):
    s3, index, points = fixture()
    p = points[0]
    if problem in {"player_id", "season", "model_version", "model_run_id"}:
        p[problem] = PID + 1 if problem == "player_id" else "wrong"
    elif problem == "date":
        p["game_date_pacific"] = "2026-11-01"
    elif problem == "stats":
        p["games"][0]["participation"]["status"] = "inactive"
    elif problem == "percentage":
        p["games"][0]["stats"]["field_goal_percentage"] = 50
    elif problem == "duplicate":
        p["games"].append(deepcopy(p["games"][0]))
    if problem in {"checksum", "missing", "scope"}:
        r = index["points"][0]
        if problem == "scope":
            r["uri"] = r["uri"].replace(f"player_id={PID}", "player_id=1")
        else:
            key = r["uri"].split("nba-gestalt-data/", 1)[1]
            if problem == "missing":
                del s3.objects[key]
            else:
                s3.put_fixture(key, {"tampered": True})
        store_index(s3, index)
    else:
        r = publish(s3, p)
        r["game_date_pacific"] = "2026-11-02"
        index["points"][0] = r
        store_index(s3, index)
    with pytest.raises(api.ERRORS):
        api.HistoryReader(s3).history(PID)


def test_same_date_current_marker_and_consistency():
    s3, index, points = fixture()
    last = points[-1]
    index.update(
        through_game_date_pacific=last["game_date_pacific"],
        source_ratings=last["inputs"]["ratings_state"],
    )
    index["current"] = {
        k: last[k] for k in ("game_date_pacific", "nail", "cumulative_possessions", "team_id")
    }
    index["current"]["source_ratings"] = index["source_ratings"]
    store_index(s3, index)
    assert api.HistoryReader(s3).history(PID)["points"][-1]["is_current"]
    index["current"]["nail"] += 0.1
    store_index(s3, index)
    with pytest.raises(ValueError, match="Current scalar"):
        api.HistoryReader(s3).history(PID)


def test_hourly_revalidation_accepts_corrections_and_caches_immutable_points(monkeypatch):
    s3, index, points = fixture()
    reader = api.HistoryReader(s3)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    api._history.cache_clear()
    before = api._history(PID, 1)
    calls = len(s3.calls)
    points[0]["nail"] += 0.01
    replace_point(s3, index, points[0])
    assert api._history(PID, 1) is before
    after = api._history(PID, 2)
    assert after["points"][0]["nail"] == points[0]["nail"]
    assert len(s3.calls) == calls + 2  # Mutable index + corrected point only.
    assert all(method == "get" and key.startswith(ROOT) for method, key in s3.calls)
    api._history.cache_clear()


def test_endpoint_caches_success_and_never_caches_errors(monkeypatch):
    s3, index, points = fixture()
    reader = api.HistoryReader(s3)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    api._history.cache_clear()
    with TestClient(MarketGateway(None)) as client:
        response = client.get(f"/api/players/{PID}/rating-history")
        assert response.status_code == 200
        assert "max-age=" in response.headers["cache-control"]
        assert response.json()["points"][0]["nail"] == points[0]["nail"]
        api._history.cache_clear()
        index["current"] = index["points"][-1]
        store_index(s3, index)
        response = client.get(f"/api/players/{PID}/rating-history")
        assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
        del s3.objects[ROOT + "latest.json"]
        response = client.get(f"/api/players/{PID}/rating-history")
        assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    api._history.cache_clear()
