import json
from datetime import UTC, date, datetime
from io import BytesIO
from hashlib import sha256
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance, build_schedule_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.play_by_play import schedules
from gestalt_pipelines.play_by_play.resources import NBAPlayByPlayArchive, archive
from gestalt_pipelines.resources import NBAScheduleResource

DAY = date(2026, 11, 9)
GAMES = [
    {"game_id": "0022600203", "home_team_id": 1, "away_team_id": 2,
     "home_team_tricode": "LAL", "away_team_tricode": "ATL",
     "game_time_utc": datetime(2026, 11, 10, 1, tzinfo=UTC)},
    {"game_id": "0022600204", "home_team_id": 3, "away_team_id": 4,
     "home_team_tricode": "BOS", "away_team_tricode": "MIL",
     "game_time_utc": datetime(2026, 11, 10, 2, tzinfo=UTC)},
]


def payload(game, score=100, final=True):
    return json.dumps({"game": {"gameId": game["game_id"], "actions": [{
        "actionNumber": 500, "actionType": "game" if final else "period",
        "subType": "end", "period": 4, "clock": "PT00M00.00S",
        "scoreHome": str(score), "scoreAway": "99",
    }]}}).encode()


@pytest.fixture
def play_by_play_job(monkeypatch):
    objects = {}
    sources = {g["game_id"]: payload(g) for g in GAMES}
    fetched = []

    def get_object(**kwargs):
        key = kwargs["Key"]
        if key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(objects[key]), "ETag": sha256(objects[key]).hexdigest()}

    def put_object(**kwargs):
        key = kwargs["Key"]
        if "IfMatch" in kwargs:
            assert key.endswith("/latest.json") and kwargs["IfMatch"] == sha256(objects[key]).hexdigest()
        else:
            assert kwargs["IfNoneMatch"] == "*"
            if key in objects:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        objects[key] = kwargs["Body"]

    monkeypatch.setattr(archive.boto3, "client", lambda _: SimpleNamespace(
        get_object=get_object, put_object=put_object,
    ))
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda self, day, **kw: (GAMES, {"uri": "s3://schedule/exact.json", "sha256": "verified"}))

    def fetch(self, game_id):
        fetched.append(game_id)
        if isinstance(sources[game_id], Exception):
            raise sources[game_id]
        return sources[game_id], {
            "source_url": f"{archive.CDN_URL}/playbyplay_{game_id}.json",
            "response_headers": {}, "started_at": "2026-11-11T05:00:00+00:00",
            "finished_at": "2026-11-11T05:00:01+00:00", "duration_seconds": 1.0,
        }

    monkeypatch.setattr(NBAPlayByPlayArchive, "_fetch", fetch)
    return defs.resolve_job_def("capture_nba_play_by_play"), objects, sources, fetched


def test_repeated_refresh_preserves_unchanged_files_and_archives_nba_corrections(play_by_play_job):
    job, objects, sources, fetched = play_by_play_job
    first = job.execute_in_process(partition_key=DAY.isoformat())
    assert first.success and len(objects) == 6
    original = {k: v for k, v in objects.items() if not k.endswith("/latest.json")}
    again = job.execute_in_process(partition_key=DAY.isoformat())
    assert again.success and all(objects[k] == v for k, v in original.items())
    assert len(fetched) == 4
    metadata = again.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["captured_count"].value == 0 and metadata["reused_count"].value == 2

    sources[GAMES[0]["game_id"]] = payload(GAMES[0], score=101)
    revised = job.execute_in_process(partition_key=DAY.isoformat())
    assert revised.success and len(objects) == 8
    assert all(objects[key] == body for key, body in original.items())
    metadata = revised.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["captured_count"].value == 1 and metadata["reused_count"].value == 1
    assert metadata["revised_count"].value == 1
    latest_key = f"lake/raw/cdn/playbyplay/season=2026-27/game_id={GAMES[0]['game_id']}/latest.json"
    latest = json.loads(objects[latest_key])
    assert latest["sha256"] == sha256(sources[GAMES[0]["game_id"]]).hexdigest()

    # The NBA can revert to older bytes; acceptance must follow the fresh response.
    sources[GAMES[0]["game_id"]] = payload(GAMES[0])
    reverted = job.execute_in_process(partition_key=DAY.isoformat())
    assert reverted.success
    accepted = json.loads(objects[latest_key])
    assert accepted["sha256"] == sha256(payload(GAMES[0])).hexdigest()
    assert all(objects[key] == body for key, body in original.items())


def test_missing_response_fails_the_date_but_retains_other_downloads(play_by_play_job):
    job, objects, sources, _ = play_by_play_job
    sources[GAMES[1]["game_id"]] = b""
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    raw_objects = {k: v for k, v in objects.items() if not k.startswith(archive.ERROR_PREFIX)}
    assert len(raw_objects) == 3
    assert all(GAMES[0]["game_id"] in key for key in raw_objects)
    [key] = [k for k in objects if k.endswith(".failure.json")]
    event = json.loads(objects[key])
    assert key.startswith(archive.ERROR_PREFIX+"season=2026-27/occurred_date_pacific=")
    assert event["stage"] == "nba_play_by_play" and event["status"] == "failed"
    assert event["dagster_run_id"] == result.run_id
    assert event["game_date_pacific"] == DAY.isoformat()
    assert list(event["failed_games"]) == [GAMES[1]["game_id"]]
    assert event["source_urls"][GAMES[1]["game_id"]].endswith(GAMES[1]["game_id"]+".json")


def test_prohibited_existing_sidecar_fails_without_overwriting_it(play_by_play_job):
    job, objects, _, _ = play_by_play_job
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    key = next(k for k in objects if k.endswith(".meta.json"))
    sidecar = json.loads(objects[key])
    sidecar["espn_game_id"] = "legacy-id"
    objects[key] = json.dumps(sidecar).encode()
    original = dict(objects)
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    assert all(objects[k] == v for k, v in original.items() if not k.endswith("/latest.json"))


def test_three_am_schedule_uses_previous_pacific_date(monkeypatch):
    schedule = defs.get_schedule_def("play_by_play_morning")
    assert schedule.cron_schedule == "0 3 * * *"
    assert schedule.execution_timezone == "America/Los_Angeles"
    tick = datetime(2026, 11, 10, 3, tzinfo=schedules.PACIFIC)
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _: tick.astimezone(UTC)))
    with DagsterInstance.local_temp() as instance:
        with build_schedule_context(
            instance=instance, scheduled_execution_time=tick,
            repository_def=defs.get_repository_def(),
        ) as context:
            [request] = schedule.evaluate_tick(context).run_requests
            assert request.partition_key == DAY.isoformat()
            assert request.run_config == {}


def test_pointer_conflict_preserves_previous_accepted_version(monkeypatch, play_by_play_job):
    job, objects, sources, _ = play_by_play_job
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    latest_key = f"lake/raw/cdn/playbyplay/season=2026-27/game_id={GAMES[0]['game_id']}/latest.json"
    accepted = objects[latest_key]
    sources[GAMES[0]["game_id"]] = payload(GAMES[0], score=101)
    client = archive.boto3.client("s3")
    put = client.put_object

    def conflict(**kwargs):
        if kwargs["Key"].endswith("/latest.json"):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        return put(**kwargs)

    client.put_object = conflict
    monkeypatch.setattr(archive.boto3, "client", lambda _: client)
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    assert objects[latest_key] == accepted


def test_refresh_fills_missing_sidecar_and_keeps_raw_bytes(play_by_play_job):
    job, objects, _, _ = play_by_play_job
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    # Simulate an interrupted archive before its sidecar and selection were saved.
    for key in list(objects):
        if key.endswith((".meta.json", "/latest.json")):
            del objects[key]
    originals = dict(objects)
    result = job.execute_in_process(partition_key=DAY.isoformat())
    assert result.success and all(objects[k] == v for k, v in originals.items())
    metadata = result.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["captured_count"].value == 0
    assert metadata["metadata_created_count"].value == 2


def test_reexecution_links_recovery_to_immutable_failure(play_by_play_job):
    from dagster._core.definitions.job_base import InMemoryJob
    from dagster._core.execution.api import execute_run

    job, objects, sources, _ = play_by_play_job
    sources[GAMES[1]["game_id"]] = b""
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(
            instance=instance, partition_key=DAY.isoformat(), raise_on_error=False,
        )
        assert not failed.success
        [failure_key] = [k for k in objects if k.endswith(".failure.json")]
        original = objects[failure_key]
        sources[GAMES[1]["game_id"]] = payload(GAMES[1])
        parent = instance.get_run_by_id(failed.run_id)
        child = instance.create_run_for_job(
            job, run_config=parent.run_config, tags=parent.tags,
            parent_run_id=parent.run_id, root_run_id=parent.run_id,
        )
        recovered = execute_run(InMemoryJob(job), child, instance)
        assert recovered.success
    [recovery_key] = [k for k in objects if k.endswith(".recovery.json")]
    recovery = json.loads(objects[recovery_key])
    assert objects[failure_key] == original
    assert recovery["original_failure"]["uri"] == "s3://nba-gestalt-data/"+failure_key
    assert recovery["original_failure"]["sha256"] == sha256(original).hexdigest()
    assert recovery["dagster_run_id"] == recovered.run_id
    assert recovery["status"] == "recovered"
    metadata = recovered.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["recoveries"].value[0]["uri"].endswith(recovery_key)


def test_schedule_read_failure_also_creates_error_record(monkeypatch, play_by_play_job):
    job, objects, _, _ = play_by_play_job

    def unavailable(*args, **kwargs):
        raise ValueError("Official schedule checksum mismatch")

    monkeypatch.setattr(NBAScheduleResource, "games_for_date", unavailable)
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    [record] = objects.values()
    event = json.loads(record)
    assert "schedule checksum mismatch" in event["message"]
    assert event["input"]["schedule"] is None


def test_logging_failure_preserves_original_dagster_error(monkeypatch, play_by_play_job):
    job, _, sources, _ = play_by_play_job
    sources[GAMES[1]["game_id"]] = b""
    client = archive.boto3.client("s3")
    put = client.put_object

    def denied(**kwargs):
        if kwargs["Key"].startswith(archive.ERROR_PREFIX):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        return put(**kwargs)

    client.put_object = denied
    monkeypatch.setattr(archive.boto3, "client", lambda _: client)
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    failure = next(e for e in result.all_events if e.event_type_value == "STEP_FAILURE")
    message = failure.event_specific_data.user_failure_data.description
    assert "1 of 2" in message and "Could not save the S3 error record" in message


def test_failure_reader_rejects_wrong_partition_and_changed_bytes(play_by_play_job):
    job, objects, sources, _ = play_by_play_job
    sources[GAMES[1]["game_id"]] = b""
    failed = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    event = next(e for e in failed.all_events if e.event_type_value == "STEP_FAILURE")
    reference = event.event_specific_data.user_failure_data.metadata["play_by_play_failure"].value
    reader = NBAPlayByPlayArchive()
    with pytest.raises(ValueError, match="partition or checksum"):
        reader.read_failure(reference, season="2026-27", day="2026-11-08")
    key = reference["uri"].removeprefix("s3://nba-gestalt-data/")
    objects[key] = objects[key].replace(b"returned no data", b"CHANGED")
    with pytest.raises(ValueError, match="partition or checksum"):
        reader.read_failure(reference, season="2026-27", day=DAY.isoformat())


@pytest.mark.parametrize("raw", [
    # The two formats seen in the Oct 23 and Oct 26 NBA responses.
    json.dumps({"game": {"gameId": "0022600204", "actions": [
        {"actionType": "game", "subType": "end", "period": 4, "clock": "PT00M00.00S"},
        {"actionType": "period", "subType": "end", "period": 1, "clock": "PT00M00.00S"},
    ]}}).encode(),
    json.dumps({"game": {"gameId": "0022600204", "actions": [
        {"actionType": "period", "subType": "end", "period": 4, "clock": "PT00M00.00S"},
    ]}}).encode(),
    # A parser may reject these contents, but capture preserves the exact response.
    b'{"game":{"actions":[{"unexpected":"source format"}]}}',
    b'{"game": {"actions": []}}',
    b'{"unfinished":',
])
def test_download_archives_response_without_parsing_events(play_by_play_job, raw):
    job, objects, sources, _ = play_by_play_job
    sources[GAMES[1]["game_id"]] = raw
    result = job.execute_in_process(partition_key=DAY.isoformat())
    assert result.success
    key = (
        "lake/raw/cdn/playbyplay/season=2026-27/"
        f"game_id={GAMES[1]['game_id']}/{sha256(raw).hexdigest()}.json"
    )
    assert objects[key] == raw
    assert not any(k.endswith(".failure.json") for k in objects)


def test_missing_endpoint_creates_s3_failure_record(play_by_play_job):
    import requests
    job, objects, sources, _ = play_by_play_job
    sources[GAMES[1]["game_id"]] = requests.HTTPError("404 Client Error: Not Found")
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    [key] = [k for k in objects if k.endswith(".failure.json")]
    event = json.loads(objects[key])
    assert "404" in event["failed_games"][GAMES[1]["game_id"]]
