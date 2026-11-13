import json
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlparse

import pytest
from botocore.exceptions import ClientError

from gestalt_pipelines.resources import NBAPlayerNamesResource, player_names
from gestalt_pipelines.rosters.resources import NBARosterArchive


@pytest.fixture
def saved(monkeypatch):
    objects = {}

    def read(*, Bucket, Key):
        assert Bucket == "nba-gestalt-data"
        if Key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = objects[Key]
        return {"Body": BytesIO(raw), "ETag": sha256(raw).hexdigest(),
                "LastModified": datetime(2026, 11, 12, tzinfo=UTC)}

    def paginator(_):
        return SimpleNamespace(paginate=lambda **kw: [{"Contents": [
            {"Key": key} for key in objects if key.startswith(kw["Prefix"])
        ]}])

    monkeypatch.setattr(player_names.boto3, "client", lambda _: SimpleNamespace(
        get_object=read, get_paginator=paginator,
    ))
    return objects


def index(saved, rows):
    raw = json.dumps({"resultSets": [{"name": "PlayerIndex", "headers": [
        "PERSON_ID", "PLAYER_FIRST_NAME", "PLAYER_LAST_NAME",
    ], "rowSet": rows}]}).encode()
    url = "https://stats.nba.com/stats/playerindex?Season=2026-27&Historical=1&LeagueID=00&TeamID=0"
    saved["lake/raw/playerindex/2026-27.json"] = raw
    saved["lake/raw/playerindex/2026-27.meta.json"] = json.dumps({
        "endpoint": "playerindex", "season": "2026-27", "url": url,
        "params": dict(parse_qsl(urlparse(url).query)), "sha256": sha256(raw).hexdigest(),
        "fetched_at": None, "acquisition_method": "manual_browser_response",
    }).encode()


def test_shared_resource_reads_uploaded_index_without_writing_or_fetching(saved):
    index(saved, [[1643416, "Chris", "Cenac Jr."], [1631250, "Pete", "Nance"]])
    before = dict(saved)
    resource = NBAPlayerNamesResource(season="2026-27")
    names, reference = resource.load()
    assert names == {1643416: "Chris Cenac Jr.", 1631250: "Pete Nance"}
    assert reference["sha256"] == sha256(saved["lake/raw/playerindex/2026-27.json"]).hexdigest()
    assert reference["metadata"]["uri"].endswith("2026-27.meta.json")
    assert saved == before
    index(saved, [[1643416, "Christopher", "Cenac Jr."]])
    assert resource.load()[0][1643416] == "Christopher Cenac Jr."


@pytest.mark.parametrize("bad", ["checksum", "source", "season", "empty", "conflict"])
def test_invalid_index_fails(saved, bad):
    rows = [[1, "NBA", "Player"]]
    if bad == "empty":
        rows = []
    if bad == "conflict":
        rows.append([1, "Different", "Name"])
    index(saved, rows)
    key = "lake/raw/playerindex/2026-27.meta.json"
    meta = json.loads(saved[key])
    if bad == "checksum":
        meta["sha256"] = "0"*64
    elif bad == "source":
        meta["url"] = meta["url"].replace("stats.nba.com", "example.com")
    elif bad == "season":
        meta["season"] = "2025-26"
    saved[key] = json.dumps(meta).encode()
    with pytest.raises(ValueError):
        NBAPlayerNamesResource(season="2026-27").load()


def put_roster(saved, day, teams):
    prefix = f"lake/artifacts/rosters/season=2026-27/game_date_pacific={day}/"
    record = {"schema_version": 1, "season": "2026-27", "game_date_pacific": day, "teams": teams}
    raw = json.dumps(record).encode()
    digest = sha256(raw).hexdigest()
    key = prefix+digest+".json"
    saved[key] = raw
    ref = {"uri": "s3://nba-gestalt-data/"+key, "sha256": digest}
    saved[prefix+"latest.json"] = json.dumps({**record, **ref}).encode()
    return ref


def test_roster_requires_own_date_latest_and_verifies_its_hash(saved):
    archive = NBARosterArchive()
    put_roster(saved, "2026-11-11", {"1": [101]})
    with pytest.raises(FileNotFoundError, match="2026-11-12/latest.json"):
        archive.load("2026-27", "2026-11-12")
    ref = put_roster(saved, "2026-11-12", {"1": [102]})
    record, actual = archive.load("2026-27", "2026-11-12")
    assert record["teams"] == {"1": [102]} and actual == ref
    key = ref["uri"].removeprefix("s3://nba-gestalt-data/")
    saved[key] += b" "
    with pytest.raises(ValueError, match="checksum"):
        archive.load("2026-27", "2026-11-12")


def test_latest_roster_uses_previous_date_until_target_exists_and_never_future(saved):
    from datetime import date
    archive = NBARosterArchive()
    put_roster(saved, "2026-11-14", {"1": [999]})
    target = date(2026, 11, 13)
    with pytest.raises(FileNotFoundError, match="No accepted roster"):
        archive.latest_selection("2026-27", target)
    prior = put_roster(saved, "2026-11-12", {"1": [101]})
    assert archive.latest_selection("2026-27", target) == ("2026-11-12", prior)
    same = put_roster(saved, "2026-11-13", {"1": [102]})
    assert archive.latest_selection("2026-27", target) == ("2026-11-13", same)
    assert archive.latest_selection("2026-27", date(2026, 11, 12)) == ("2026-11-12", prior)
