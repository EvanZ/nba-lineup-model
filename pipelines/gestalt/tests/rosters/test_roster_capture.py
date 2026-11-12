import json
from datetime import UTC, date, datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.rosters.resources import archive
from gestalt_pipelines.rosters.utils import NBA_TEAM_IDS

SEASON = "2026-27"
DAY = date(2026, 11, 8)
FIRST, SECOND = NBA_TEAM_IDS[:2]
GAME = {
    "game_id": "0022600200", "game_time_utc": datetime(2026, 11, 8, 3, tzinfo=UTC),
    "home_team_id": FIRST, "away_team_id": SECOND,
    "home_team_tricode": "ATL", "away_team_tricode": "BOS",
}
STAMP = datetime(2026, 11, 11, 5, tzinfo=UTC)


@pytest.fixture
def roster_job(monkeypatch):
    objects, reads, writes = {}, [], []

    def save(key, value):
        objects[key] = json.dumps(value).encode()
        return sha256(objects[key]).hexdigest()

    def players(index):
        return [{"player_id": pid, "player_name": f"Official {pid}"} for pid in (2*index+1, 2*index+2)]

    for index, team in enumerate(NBA_TEAM_IDS):
        prefix = f"lake/raw/commonteamroster/{SEASON}/{team}"
        digest = save(prefix+".json", {"resultSets": [{
            "name": "CommonTeamRoster", "headers": ["TeamID", "PLAYER_ID"],
            "rowSet": [[team, p["player_id"]] for p in players(index)],
        }]})
        save(prefix+".meta.json", {
            "season": SEASON, "team_id": team, "endpoint": "commonteamroster", "sha256": digest,
            "url": f"https://stats.nba.com/stats/commonteamroster?TeamID={team}",
            "fetched_at": "2026-11-07T02:00:00+00:00",
        })
    for day in ("2026-11-07", "2026-11-08"):
        rows = [] if day.endswith("07") else [{
            "Transaction_Type": "Trade", "TRANSACTION_DATE": day+"T00:00:00",
            "PLAYER_ID": 1, "TEAM_ID": SECOND, "first_observed_at": STAMP.isoformat(),
        }]
        save(f"lake/raw/nba/transactions/season={SEASON}/game_date_pacific={day}/latest.json", {
            "season": SEASON, "game_date_pacific": day, "last_polled_at": STAMP.isoformat(),
            "source_url": "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json",
            "transactions": rows,
        })

    body = {"game": {"gameId": GAME["game_id"], "gameStatus": 3}}
    for side, index in (("home", 0), ("away", 1)):
        body["game"][side+"Team"] = {
            "teamId": GAME[side+"_team_id"],
            "players": [{"personId": p["player_id"]} for p in players(index)],
        }
    raw = json.dumps(body).encode()
    digest = sha256(raw).hexdigest()
    raw_key = f"lake/raw/cdn/boxscore/season={SEASON}/game_id={GAME['game_id']}/{digest}.json"
    objects[raw_key] = raw
    save(raw_key.removesuffix(".json")+".meta.json", {
        "season": SEASON, "game_id": GAME["game_id"], "box_score_sha256": digest,
        "home_team_id": FIRST, "away_team_id": SECOND,
        "home_roster_sha256": "a"*64, "away_roster_sha256": "b"*64,
    })

    def get_object(**args):
        key = args["Key"]
        assert key.startswith(("lake/raw/commonteamroster/", "lake/raw/nba/transactions/",
                               "lake/raw/cdn/boxscore/", "lake/artifacts/rosters/"))
        reads.append(key)
        if key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(objects[key]), "LastModified": STAMP, "ETag": sha256(objects[key]).hexdigest()}

    def put_object(**args):
        key = args["Key"]
        assert key.startswith("lake/artifacts/rosters/season=2026-27/game_date_pacific=")
        assert key.endswith("/latest.json") or key.endswith("/"+sha256(args["Body"]).hexdigest()+".json")
        if args.get("IfNoneMatch") == "*" and key in objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if "IfMatch" in args and (key not in objects or sha256(objects[key]).hexdigest() != args["IfMatch"]):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        objects[key] = args["Body"]
        writes.append(key)

    def paginate(**args):
        return [{"Contents": [{"Key": key, "LastModified": STAMP} for key in objects if key.startswith(args["Prefix"])]}]

    client = SimpleNamespace(get_object=get_object, put_object=put_object,
                             get_paginator=lambda _: SimpleNamespace(paginate=paginate))
    monkeypatch.setattr(archive.boto3, "client", lambda _: client)
    monkeypatch.setattr(NBAScheduleResource, "games_between", lambda *args, **kwargs: (
        [GAME], {"uri": "s3://schedule/immutable.json", "sha256": "verified"},
    ))
    return defs.resolve_job_def("build_nba_rosters"), objects, reads, writes, raw_key


def test_manual_job_publishes_all_teams_and_reuses_identical_inputs(roster_job):
    job, objects, reads, writes, raw_key = roster_job
    control_key = "lake/artifacts/rosters/season=2026-27/latest.json"
    objects[control_key] = b'{"unchanged_live_selection":true}'
    first = job.execute_in_process(partition_key=DAY.isoformat())
    assert first.success and len(writes) == 2
    artifact = json.loads(objects[writes[0]])
    pointer = json.loads(objects[writes[1]])
    assert pointer == {
        "schema_version": 1, "season": SEASON, "game_date_pacific": str(DAY),
        "uri": "s3://nba-gestalt-data/"+writes[0], "sha256": sha256(objects[writes[0]]).hexdigest(),
    }
    assert len(artifact["teams"]) == 30
    assert sum(len(ids) for ids in artifact["teams"].values()) == 60
    assert reads.count(raw_key) == 1
    assert set(artifact["inputs"]) == {"opening_rosters", "schedule", "transactions", "box_scores"}
    assert all(type(pid) is int for ids in artifact["teams"].values() for pid in ids)
    assert "manual_name_repairs_used" not in artifact

    repeat = job.execute_in_process(partition_key=DAY.isoformat())
    assert repeat.success and len(writes) == 2
    metadata = repeat.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["artifact_reused"].value is True
    assert objects[control_key] == b'{"unchanged_live_selection":true}'


def test_changed_inputs_create_another_readable_version(roster_job):
    job, objects, _, writes, _ = roster_job
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    first_key = writes[0]
    first = objects[first_key]
    key = f"lake/raw/nba/transactions/season={SEASON}/game_date_pacific={DAY}/latest.json"
    source = json.loads(objects[key])
    source["transactions"][0]["TEAM_ID"] = NBA_TEAM_IDS[2]
    objects[key] = json.dumps(source).encode()
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    assert len(writes) == 4 and writes[0] != writes[2]
    pointer = json.loads(objects[writes[3]])
    assert pointer["uri"] == "s3://nba-gestalt-data/"+writes[2]
    assert objects[first_key] == first


def test_signing_without_a_name_succeeds(roster_job):
    job, objects, _, writes, _ = roster_job
    key = f"lake/raw/nba/transactions/season={SEASON}/game_date_pacific={DAY}/latest.json"
    source = json.loads(objects[key])
    source["transactions"].append({
        "Transaction_Type": "Signing", "TRANSACTION_DATE": "2026-11-08T00:00:00",
        "PLAYER_ID": 999999, "TEAM_ID": SECOND, "PLAYER_SLUG": "unknown-person",
        "first_observed_at": STAMP.isoformat(),
    })
    objects[key] = json.dumps(source).encode()
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert result.success and len(writes) == 2
    record = json.loads(objects[writes[0]])
    assert 999999 in record["teams"][str(SECOND)]


def test_missing_transaction_date_is_not_treated_as_an_empty_day(roster_job):
    job, objects, _, writes, _ = roster_job
    del objects[f"lake/raw/nba/transactions/season={SEASON}/game_date_pacific=2026-11-07/latest.json"]
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success and not writes



def test_name_repair_dictionary_is_not_a_roster_input(roster_job):
    job, objects, reads, writes, _ = roster_job
    key = "lake/artifacts/rosters/season=2026-27/player_repairs.json"
    objects[key] = b'{"1":"Not A Membership Input"}'
    result = job.execute_in_process(partition_key=DAY.isoformat())
    assert result.success and key not in reads
    record = json.loads(objects[writes[0]])
    assert "player_repairs" not in record["inputs"]


def test_stale_run_cannot_replace_another_roster_selection(roster_job):
    from copy import deepcopy
    job, objects, _, writes, _ = roster_job
    assert job.execute_in_process(partition_key=str(DAY)).success
    a = archive.NBARosterArchive()
    ref, old_etag = a.selection(SEASON, str(DAY))
    original = json.loads(objects[ref["uri"].removeprefix("s3://nba-gestalt-data/")])
    revised = deepcopy(original)
    revised["inputs"]["test_revision"] = 2
    a.publish(revised, expected_etag=old_etag)
    pointer_key = f"lake/artifacts/rosters/season={SEASON}/game_date_pacific={DAY}/latest.json"
    accepted = objects[pointer_key]
    with pytest.raises(ValueError, match="changed during"):
        a.publish(original, expected_etag=old_etag)
    assert objects[pointer_key] == accepted


def test_failed_rebuild_keeps_accepted_roster_pointer(roster_job):
    job, objects, _, writes, _ = roster_job
    assert job.execute_in_process(partition_key=str(DAY)).success
    pointer_key = f"lake/artifacts/rosters/season={SEASON}/game_date_pacific={DAY}/latest.json"
    accepted = objects[pointer_key]
    before = list(writes)
    del objects[f"lake/raw/nba/transactions/season={SEASON}/game_date_pacific=2026-11-07/latest.json"]
    assert not job.execute_in_process(partition_key=str(DAY), raise_on_error=False).success
    assert objects[pointer_key] == accepted and writes == before
