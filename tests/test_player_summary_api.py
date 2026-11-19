"""Saved player summaries validate entire states without invoking legacy/model code."""

import hashlib
import io
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import player_summary as api
from nba_lineup_model.web_api.gateway import MarketGateway

DAY = date(2026, 11, 16)
PID = 201939
NEW_PID = 1642923
INDEX = "lake/raw/playerindex/2026-27"


class SavedS3:
    def __init__(self):
        self.objects = {}
        self.calls = []

    def save(self, key, doc):
        raw = json.dumps(doc, sort_keys=True).encode()
        self.objects[key] = raw
        return hashlib.sha256(raw).hexdigest()

    def get_object(self, *, Bucket, Key):
        assert Bucket == api.BUCKET
        self.calls.append(("get", Key))
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix, Delimiter):
        assert Bucket == api.BUCKET and Delimiter == "/"
        self.calls.append(("list", Prefix))
        folders = {
            Prefix + key.removeprefix(Prefix).split("/")[0] + "/"
            for key in self.objects
            if key.startswith(Prefix)
        }
        yield {"CommonPrefixes": [{"Prefix": folder} for folder in sorted(folders)]}


def publish(s3, kind, doc):
    root = (
        f"lake/artifacts/{kind}/season=2026-27/model_version=v1/"
        f"game_date_pacific={doc['game_date_pacific']}/"
    )
    sha = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    s3.save(root + sha + ".json", doc)
    s3.save(
        root + "latest.json",
        {
            "schema_version": 1,
            "season": api.SEASON,
            "model_version": "v1",
            "game_date_pacific": doc["game_date_pacific"],
            "uri": f"s3://{api.BUCKET}/{root}{sha}.json",
            "sha256": sha,
        },
    )
    return root


def fixture():
    s3 = SavedS3()
    common = {
        "schema_version": 1,
        "season": api.SEASON,
        "game_date_pacific": str(DAY),
        "date_contract": "completed_game_date_v1",
        "observed_through_game_date_pacific": str(DAY),
        "effective_from_game_date_pacific": str(DAY + timedelta(days=1)),
    }
    rating = {
        "artifact_kind": "player_ratings",
        "model_version": "v1",
        **common,
        "players": [
            {
                "player_id": PID,
                "player_name": "Stephen Curry",
                "team_id": 1610612744,
                "nail": 2.9935143638813675,
                "starting_nail": 1.4535258170852985,
                "residual": 1.539988546796069,
                "standard_error": 1.490676046339863,
                "on_court_possessions": 549.9583333333334,
                "forecast_source": "saved_prior",
            },
        ],
    }
    rating["players"].append(
        {**rating["players"][0], "player_id": NEW_PID, "player_name": "Otega Oweh"}
    )
    minutes = {
        "artifact_kind": "conditional_minutes_state",
        "model": {"model_version": "v1"},
        **common,
        "players": {
            str(pid): {
                "player_id": pid,
                "current_team_id": 1610612744,
                "estimate": 24.341277387920687,
                "starting_minutes": 30.546436670818814,
                "history": [{}, {}, {}],
            }
            for pid in [PID, NEW_PID]
        },
    }
    index = {
        "resource": "playerindex",
        "parameters": {"Season": api.SEASON},
        "resultSets": [
            {
                "headers": [
                    "PERSON_ID",
                    "PLAYER_FIRST_NAME",
                    "PLAYER_LAST_NAME",
                    "TEAM_ID",
                    "TEAM_ABBREVIATION",
                    "POSITION",
                    "DRAFT_YEAR",
                    "DRAFT_ROUND",
                    "DRAFT_NUMBER",
                ],
                "rowSet": [
                    [PID, "Stephen", "Curry", 1610612744, "GSW", "G", "2009", "1", "7"],
                    [23, "Retired", "Player", 0, "", None, "Undrafted", None, None],
                ],
            }
        ],
    }
    save_index(s3, index)
    publish(s3, "player_ratings", rating)
    publish(s3, "player_minutes", minutes)
    return s3, rating, minutes, index


def save_index(s3, index):
    sha = s3.save(INDEX + ".json", index)
    s3.save(
        INDEX + ".meta.json",
        {
            "season": api.SEASON,
            "endpoint": "playerindex",
            "sha256": sha,
            "url": "https://stats.nba.com/stats/playerindex?Season=2026-27",
            "fetched_at": None,
            "uploaded_at": "2026-11-13T02:54:40+00:00",
            "acquisition_method": "manual_browser_response",
        },
    )


def test_complete_coverage_precision_and_optional_biography():
    s3, ratings, minutes, _ = fixture()
    inputs = api.read_player_inputs(s3, season=api.SEASON, cutoff=DAY)
    assert inputs["coverage"] == {
        "rating_player_count": 2,
        "minutes_player_count": 2,
        "missing_biography_player_ids": [NEW_PID],
    }
    profile = api.player_summary(inputs, PID)
    assert profile["rating"]["nail"] == ratings["players"][0]["nail"]
    assert profile["rating"]["delta"] == ratings["players"][0]["residual"]
    assert profile["minutes"]["conditional_estimate"] == minutes["players"][str(PID)]["estimate"]
    assert profile["sources"]["biography"]["fetched_at"] is None
    assert profile["observed_through"] == str(DAY) and profile["effective_from"] == "2026-11-17"
    assert profile["team"] == "GSW" and profile["draft"]["year"] == "2009"
    newcomer = api.player_summary(inputs, NEW_PID)
    assert newcomer["biography_available"] is False and newcomer["position"] is None
    assert newcomer["player_name"] == "Otega Oweh" and newcomer["rating"] is not None
    retired = api.player_summary(inputs, 23)
    assert retired["rating"] is retired["minutes"] is retired["team"] is None
    assert "age" not in profile and "rating_history" not in profile
    assert all(op in {"get", "list"} for op, key in s3.calls)


@pytest.mark.parametrize(
    "damage",
    [
        "checksum",
        "pointer_escape",
        "schema",
        "season",
        "date",
        "effective_date",
        "observed_date",
        "version",
        "contract",
        "duplicate_rating",
        "missing_minutes",
        "minutes_key",
        "team",
        "nonfinite",
        "negative_minutes",
        "delta",
        "bool_id",
        "invalid_history",
    ],
)
def test_bad_required_state_fails_without_partial_profiles(damage):
    s3, r, m, _ = fixture()
    root = f"lake/artifacts/player_ratings/season=2026-27/model_version=v1/game_date_pacific={DAY}/"
    if damage in {"checksum", "pointer_escape"}:
        p = json.loads(s3.objects[root + "latest.json"])
        if damage == "checksum":
            s3.save(root + p["sha256"] + ".json", {})
        else:
            p["uri"] = p["uri"].replace("2026-11-16", "2026-11-15")
            s3.save(root + "latest.json", p)
    else:
        if damage == "schema":
            r["schema_version"] = True
        if damage == "season":
            r["season"] = "2025-26"
        if damage == "date":
            r["game_date_pacific"] = "2026-11-15"
        if damage == "effective_date":
            r["effective_from_game_date_pacific"] = "2026-11-18"
        if damage == "observed_date":
            r["observed_through_game_date_pacific"] = "2026-11-15"
        if damage == "version":
            m["model"]["model_version"] = "v2"
        if damage == "contract":
            r["date_contract"] = "other"
        if damage == "duplicate_rating":
            r["players"].append(deepcopy(r["players"][0]))
        if damage == "missing_minutes":
            del m["players"][str(NEW_PID)]
        if damage == "minutes_key":
            m["players"][str(PID)]["player_id"] = 42
        if damage == "team":
            m["players"][str(PID)]["current_team_id"] = 999
        if damage == "nonfinite":
            r["players"][0]["nail"] = float("nan")
        if damage == "negative_minutes":
            m["players"][str(PID)]["estimate"] = -1
        if damage == "delta":
            r["players"][0]["residual"] = 0
        if damage == "bool_id":
            r["players"][0]["player_id"] = True
        if damage == "invalid_history":
            m["players"][str(PID)]["history"] = None
        publish(s3, "player_ratings", r)
        publish(s3, "player_minutes", m)
        if damage == "date":
            p = json.loads(s3.objects[root + "latest.json"])
            p["game_date_pacific"] = "2026-11-15"
            s3.save(root + "latest.json", p)
    with pytest.raises((ValueError, KeyError)):
        api.read_player_inputs(s3, season=api.SEASON, cutoff=DAY)


@pytest.mark.parametrize(
    "damage", ["checksum", "source", "season", "duplicate", "columns", "row", "body_identity"]
)
def test_bad_player_index_is_not_silently_ignored(damage):
    s3, _, _, index = fixture()
    if damage == "duplicate":
        index["resultSets"][0]["rowSet"].append(index["resultSets"][0]["rowSet"][0])
    if damage == "columns":
        index["resultSets"][0]["headers"][0] = "OTHER"
    if damage == "row":
        index["resultSets"][0]["rowSet"][0].pop()
    if damage == "body_identity":
        index["resource"] = "other"
    save_index(s3, index)
    meta = json.loads(s3.objects[INDEX + ".meta.json"])
    if damage == "checksum":
        meta["sha256"] = "0" * 64
    if damage == "source":
        meta["url"] = "https://unapproved.example/playerindex?Season=2026-27"
    if damage == "season":
        meta["season"] = "2025-26"
    s3.save(INDEX + ".meta.json", meta)
    with pytest.raises(ValueError):
        api.read_player_inputs(s3, season=api.SEASON, cutoff=DAY)


def test_different_latest_dates_fail_instead_of_silently_using_older_state():
    s3, _, minutes, _ = fixture()
    minutes["game_date_pacific"] = "2026-11-17"
    publish(s3, "player_minutes", minutes)
    with pytest.raises(ValueError, match="not aligned"):
        api.read_player_inputs(s3, season=api.SEASON, cutoff=DAY + timedelta(days=1))
    # The still-future partition must not be selected before that date is complete.
    assert api.read_player_inputs(s3, season=api.SEASON, cutoff=DAY)["day"] == str(DAY)


def test_gateway_and_shared_hourly_cache(monkeypatch):
    s3, _, _, _ = fixture()
    monkeypatch.setattr("boto3.client", lambda name: s3)
    clock = [3600.0]
    monkeypatch.setattr(api, "time", lambda: clock[0])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 17, 18, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(api, "datetime", Clock)
    api._cached_inputs.cache_clear()
    legacy = Mock(side_effect=AssertionError("Legacy evaluator must not initialize"))
    client = TestClient(MarketGateway(legacy))
    try:
        response = client.get(f"/api/players/{PID}")
        assert response.status_code == 200
        assert response.json()["kind"] == "saved_player_summary"
        assert response.headers["cache-control"] == "private, max-age=3600"
        loaded = list(s3.calls)
        assert client.get(f"/api/players/{NEW_PID}").status_code == 200
        assert client.get("/api/players/23").json()["rating"] is None
        assert client.get("/api/players/999999999").status_code == 404
        assert client.get("/api/players/0").status_code == 422
        assert s3.calls == loaded
        assert len([1 for op, key in loaded if op == "get"]) == 6
        clock[0] = 7200
        assert client.get(f"/api/players/{PID}").status_code == 200
        assert len(s3.calls) == len(loaded) * 2
        legacy.assert_not_called()
        s3.objects.clear()
        clock[0] = 10800
        error = client.get(f"/api/players/{PID}")
        assert error.status_code == 503 and error.headers["cache-control"] == "no-store"
    finally:
        api._cached_inputs.cache_clear()
