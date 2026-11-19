"""Saved-team navigation stays independent of ratings and legacy app startup."""

import hashlib
import io
import json
from datetime import date
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import player_discovery as discovery
from nba_lineup_model.web_api import team_navigation as api
from nba_lineup_model.web_api.gateway import MarketGateway


class ReadOnlyS3:
    def __init__(self):
        self.objects = {}

    def put(self, key, doc):
        raw = json.dumps(doc).encode()
        self.objects[key] = raw
        return hashlib.sha256(raw).hexdigest()

    def get_object(self, *, Bucket, Key):
        assert Bucket == api.BUCKET
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def catalog(monkeypatch):
    current = [
        {"team": "LAL", "team_id": 1, "display_name": "Los Angeles Lakers"},
        {"team": "BOS", "team_id": 2, "display_name": "Boston Celtics"},
    ]
    history = [
        {"team": "LAL", "team_id": 1, "season": "2025-26"},
        {"team": "SEA", "team_id": 3, "season": "2007-08"},
    ]
    entries = api.merge_catalog(current, history)
    monkeypatch.setattr(
        api, "catalog", lambda: ((entries, "2026-11-18", {}), "private, max-age=60")
    )
    return entries


def test_navigation_and_search_bypass_legacy(catalog):
    legacy = Mock(side_effect=AssertionError("Legacy must not load"))
    client = TestClient(MarketGateway(legacy))
    listing = client.get("/api/teams")
    assert listing.status_code == 200
    assert listing.json()["teams"] == ["BOS", "LAL"]
    assert listing.json()["available_seasons"] == ["2026-27", "2025-26", "2007-08"]
    assert client.get("/api/teams/lal").json()["team"] == "LAL"
    assert client.get("/api/teams/LAL").json()["season"] == "2026-27"
    retired = client.get("/api/teams/SEA")
    assert retired.json()["season"] == "2007-08"
    assert retired.json()["roster_as_of"] is None
    assert retired.json()["display_name"] == "SEA"
    for code, entry in catalog.items():
        for season in entry["seasons"]:
            response = client.get(f"/api/teams/{code}", params={"season": season})
            assert response.status_code == 200
            assert response.json()["season"] == season
            assert code in [r["team"] for r in response.json()["teams"]]
            assert "lineups" in response.json()["deferred_sections"]
    for path in ["/api/teams/SEA?season=2026-27", "/api/teams/ZZZ", "/api/teams?season=1900-01"]:
        response = client.get(path)
        assert response.status_code == 404
        assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/teams/LAL?season=bad").status_code == 422
    assert client.get("/api/teams/LAL?minimum_possessions=-1").status_code == 422
    assert (
        client.get("/api/teams/LAL?minimum_possessions=5000").json()
        == client.get("/api/teams/LAL").json()
    )
    assert api.search_teams("lakers", 8)[0][0]["team"] == "LAL"
    assert api.search_teams("Los Angeles", 8)[0][0]["latest_season"] == "2026-27"
    assert api.search_teams("sea", 8)[0][0]["latest_season"] == "2007-08"
    legacy.assert_not_called()


def test_catalog_rejects_ambiguous_historical_identity():
    rows = [{"team": "SEA", "team_id": tid, "season": "2007-08"} for tid in (1, 2)]
    with pytest.raises(ValueError):
        api.merge_catalog([], rows)


@pytest.fixture
def saved(monkeypatch):
    s3 = ReadOnlyS3()
    monkeypatch.setattr(api, "_latest_day", lambda client, root, cutoff: date(2026, 11, 18))
    roster_root = f"lake/artifacts/rosters/season={api.SEASON}/game_date_pacific=2026-11-18/"
    identity = {"schema_version": 1, "season": api.SEASON, "game_date_pacific": "2026-11-18"}
    teams = {str(i): [100 + i] for i in range(1, 31)}
    roster = {**identity, "teams": teams}
    # SHA is computed from exact original bytes, as in the producer contract.
    raw = json.dumps(roster).encode()
    sha = hashlib.sha256(raw).hexdigest()
    key = roster_root + sha + ".json"
    s3.objects[key] = raw
    s3.put(
        roster_root + "latest.json", {**identity, "sha256": sha, "uri": f"s3://{api.BUCKET}/{key}"}
    )
    identities = [
        {"teamId": i, "teamTricode": f"T{i:02}", "teamCity": f"City {i}", "teamName": "Club"}
        for i in range(1, 31)
    ]
    identities.append({"teamId": 99, "teamTricode": "TBA", "teamCity": "", "teamName": "TBA"})
    games = [{"homeTeam": team, "awayTeam": identities[0]} for team in identities]
    schedule = {"leagueSchedule": {"seasonYear": api.SEASON, "gameDates": [{"games": games}]}}
    root = f"lake/raw/scheduleleaguev2/season={api.SEASON}/"
    raw = json.dumps(schedule).encode()
    sha = hashlib.sha256(raw).hexdigest()
    schedule_key = root + f"sha256={sha}/schedule.json"
    s3.objects[schedule_key] = raw
    s3.put(root + "latest.json", {"sha256": sha, "uri": f"s3://{api.BUCKET}/{schedule_key}"})
    return s3, roster_root, key, schedule_key


def test_current_navigation_covers_30_official_teams(saved):
    s3, _, _, _ = saved
    rows, day, refs = api.read_current(s3, date(2026, 11, 18))
    assert len(rows) == 30 and day == "2026-11-18"
    assert "TBA" not in [row["team"] for row in rows]
    assert set(refs) == {"roster", "schedule"}


@pytest.mark.parametrize("damage", ["bytes", "scope", "date", "missing", "duplicate", "schedule"])
def test_corrupt_selection_is_rejected(saved, damage):
    s3, root, key, schedule_key = saved
    if damage == "bytes":
        s3.objects[key] += b" "
    elif damage == "scope":
        pointer = json.loads(s3.objects[root + "latest.json"])
        pointer["uri"] = "s3://other-bucket/object.json"
        s3.put(root + "latest.json", pointer)
    elif damage == "schedule":
        s3.objects[schedule_key] += b" "
    else:
        doc = json.loads(s3.objects[key])
        if damage == "date":
            doc["game_date_pacific"] = "2026-11-17"
        elif damage == "missing":
            del doc["teams"]["30"]
        else:
            doc["teams"]["30"] = doc["teams"]["1"]
        raw = json.dumps(doc).encode()
        sha = hashlib.sha256(raw).hexdigest()
        target = root + sha + ".json"
        s3.objects[target] = raw
        pointer = json.loads(s3.objects[root + "latest.json"])
        pointer.update(uri=f"s3://{api.BUCKET}/{target}", sha256=sha)
        s3.put(root + "latest.json", pointer)
    with pytest.raises(ValueError):
        api.read_current(s3, date(2026, 11, 18))


def test_source_failure_is_explicit_and_not_cached(monkeypatch):
    monkeypatch.setattr(api, "catalog", Mock(side_effect=ValueError("bad hash")))
    response = TestClient(MarketGateway(None)).get("/api/teams/LAL")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_hourly_catalog_revalidates_mutable_inputs(monkeypatch):
    clock = [3601]
    monkeypatch.setattr(api, "time", lambda: clock[0])
    monkeypatch.setattr(api, "_reader", lambda: Mock(client=object()))
    read = Mock(
        return_value=([{"team": "LAL", "team_id": 1, "display_name": "Lakers"}], "2026-11-18", {})
    )
    monkeypatch.setattr(api, "read_current", read)
    monkeypatch.setattr(api, "historical_catalog", lambda source: [])
    api._catalog.cache_clear()
    try:
        api.catalog()
        api.catalog()
        assert read.call_count == 1
        clock[0] += 3600
        api.catalog()
        assert read.call_count == 2
    finally:
        api._catalog.cache_clear()


@pytest.mark.parametrize("failed", ["players", "teams", "both"])
def test_search_sources_fail_independently(monkeypatch, failed):
    player = {"player_id": 1, "name": "Test Player"}
    team = {"team": "LAL", "display_name": "Lakers", "latest_season": "2026-27", "season_count": 1}
    monkeypatch.setattr(
        discovery.profiles,
        "shared_inputs",
        Mock(side_effect=ValueError("bad checksum"))
        if failed in ("players", "both")
        else lambda: ({}, "private, max-age=60"),
    )
    monkeypatch.setattr(discovery, "search_players", lambda inputs, q, limit: [player])
    monkeypatch.setattr(
        discovery,
        "search_teams",
        Mock(side_effect=ValueError("bad checksum"))
        if failed in ("teams", "both")
        else lambda q, limit: ([team], "private, max-age=60"),
    )
    result = TestClient(MarketGateway(None)).get("/api/search?q=test")
    assert result.headers["cache-control"] == "no-store"
    if failed == "both":
        assert result.status_code == 503
    else:
        assert result.status_code == 200
        assert result.json()["unavailable_sections"] == [failed]
        assert result.json()["teams" if failed == "players" else "players"] == [
            team if failed == "players" else player
        ]


@pytest.fixture(autouse=True)
def stub_roster(monkeypatch):
    from nba_lineup_model.web_api import team_roster

    monkeypatch.setattr(team_roster, "roster", lambda *args: {"status": "available", "players": []})
