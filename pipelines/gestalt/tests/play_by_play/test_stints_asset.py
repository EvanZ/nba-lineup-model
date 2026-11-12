import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.play_by_play.resources import NBAPlayByPlayArchive, archive
from gestalt_pipelines.box_scores.resources import NBABoxScoreArchive
from gestalt_pipelines.resources import NBAScheduleResource
from test_stints_parser import full_game

DAY = "2026-10-20"


@pytest.fixture
def stints_job(monkeypatch):
    objects, writes, times = {}, [], {}
    pbp, box, game = full_game()
    for side in ("homeTeam","awayTeam"):
        for p in box["game"][side]["players"]:
            p["name"] = f"Player {p['personId']}"
    def put(**kw):
        key=kw["Key"]
        if kw.get("IfNoneMatch")=="*" and key in objects:
            raise ClientError({"Error":{"Code":"PreconditionFailed"}},"PutObject")
        if "IfMatch" in kw and kw["IfMatch"]!=sha256(objects[key]).hexdigest():
            raise ClientError({"Error":{"Code":"PreconditionFailed"}},"PutObject")
        objects[key]=kw["Body"]
        times[key]=datetime(2026,11,11,tzinfo=UTC)+timedelta(seconds=len(writes))
        writes.append(key)
    def get(**kw):
        key=kw["Key"]
        if key not in objects:
            raise ClientError({"Error":{"Code":"NoSuchKey"}},"GetObject")
        return {"Body":BytesIO(objects[key]),"ETag":sha256(objects[key]).hexdigest()}
    def pages(**kw):
        return iter([{"Contents":[{"Key":k,"LastModified":times[k]} for k in objects if k.startswith(kw["Prefix"])]}])
    client=SimpleNamespace(put_object=put,get_object=get,get_paginator=lambda _:SimpleNamespace(paginate=pages))
    monkeypatch.setattr(archive.boto3,"client",lambda _:client)
    monkeypatch.setattr(NBAScheduleResource,"games_for_date",lambda *args,**kw:([game],{"uri":"s3://fixture/schedule.json","sha256":"s"}))
    monkeypatch.setattr(NBAPlayByPlayArchive,"_fetch",lambda _,gid:(json.dumps(pbp).encode(),{
        "source_url":f"{archive.CDN_URL}/playbyplay_{gid}.json","response_headers":{},
        "started_at":"2026-11-11T00:00:00+00:00","finished_at":"2026-11-11T00:00:01+00:00","duration_seconds":1,
    }))
    NBAPlayByPlayArchive().capture_slate("2026-27",[game],run_id="source-test")
    def save_box(value):
        raw=json.dumps(value).encode()
        digest=sha256(raw).hexdigest()
        key=f"lake/raw/cdn/boxscore/season=2026-27/game_id={game['game_id']}/{digest}.json"
        if key not in objects:
            put(Key=key,Body=raw,IfNoneMatch="*")
        NBABoxScoreArchive()._ensure_metadata(client,key,"2026-27",raw,digest)
    save_box(box)
    return defs.resolve_job_def("parse_nba_games"),objects,writes,box,game,save_box


def test_json_publish_reuse_and_full_source_provenance(stints_job):
    job,objects,writes,_,_,_=stints_job
    sources=dict(objects)
    result=job.execute_in_process(partition_key=DAY)
    assert result.success
    [pointer_key]=[k for k in objects if k.startswith("lake/artifacts/stints/") and k.endswith("/latest.json")]
    pointer=json.loads(objects[pointer_key])
    key=pointer["uri"].removeprefix("s3://nba-gestalt-data/")
    raw=objects[key]
    assert key.endswith("/"+sha256(raw).hexdigest()+".json")
    record=json.loads(raw)
    assert record["validation"]["complete"] is True and len(record["rows"])>0
    assert record["processing_version"]=="nba_stints_v1"
    assert set(record["inputs"]["games"][record["rows"][0]["game_id"]])=={"play_by_play","box_score"}
    assert all(objects[k]==v for k,v in sources.items())
    write_count=len(writes)
    again=job.execute_in_process(partition_key=DAY)
    assert again.success and objects[key]==raw
    assert not any(k.endswith("/latest.json") for k in writes[write_count:])
    assert len([k for k in objects if k.startswith("lake/artifacts/stints/")])==2


def test_bad_game_keeps_last_accepted_pointer_and_logs_exact_inputs(stints_job):
    job,objects,_,box,game,save_box=stints_job
    assert job.execute_in_process(partition_key=DAY).success
    pointer_key=next(k for k in objects if k.startswith("lake/artifacts/stints/") and k.endswith("/latest.json"))
    accepted=objects[pointer_key]
    box["game"]["homeTeam"]["score"]+=1
    save_box(box)
    failed=job.execute_in_process(partition_key=DAY,raise_on_error=False)
    assert not failed.success and objects[pointer_key]==accepted
    [failure_key]=[k for k in objects if k.endswith(".failure.json")]
    error=json.loads(objects[failure_key])
    assert error["stage"]=="nba_stints"
    assert "scoring" in error["failed_games"][game["game_id"]]["message"]
    assert set(error["input"]["games"][game["game_id"]])=={"play_by_play","box_score"}


def test_failed_parent_reexecution_writes_linked_recovery(stints_job):
    from dagster._core.definitions.job_base import InMemoryJob
    from dagster._core.execution.api import execute_run
    job,objects,_,_,_,_=stints_job
    source_key=next(k for k in objects if k.startswith("lake/raw/cdn/playbyplay/") and k.endswith("/latest.json"))
    selection=objects.pop(source_key)
    with DagsterInstance.ephemeral() as instance:
        failed=job.execute_in_process(instance=instance,partition_key=DAY,raise_on_error=False)
        assert not failed.success
        [failure_key]=[k for k in objects if k.endswith(".failure.json")]
        failure_bytes=objects[failure_key]
        objects[source_key]=selection
        parent=instance.get_run_by_id(failed.run_id)
        child=instance.create_run_for_job(job,run_config=parent.run_config,tags=parent.tags,parent_run_id=parent.run_id,root_run_id=parent.run_id)
        recovered=execute_run(InMemoryJob(job),child,instance)
        assert recovered.success
    [recovery_key]=[k for k in objects if k.endswith(".recovery.json")]
    recovery=json.loads(objects[recovery_key])
    assert recovery["stage"]=="nba_stints"
    assert recovery["original_failure"]["uri"]=="s3://nba-gestalt-data/"+failure_key
    assert recovery["original_failure"]["sha256"]==sha256(failure_bytes).hexdigest()
    assert objects[failure_key]==failure_bytes


def test_empty_official_date_has_explicit_accepted_empty_result(monkeypatch,stints_job):
    job,objects,_,_,_,_=stints_job
    monkeypatch.setattr(NBAScheduleResource,"games_for_date",lambda *a,**k:([],{"uri":"s3://fixture/schedule.json","sha256":"s"}))
    assert job.execute_in_process(partition_key=DAY).success
    key=next(k for k in objects if k.startswith("lake/artifacts/stints/") and not k.endswith("/latest.json"))
    record=json.loads(objects[key])
    assert record["rows"]==[] and record["validation"]["expected_game_ids"]==[]


def test_processing_revision_preserves_earlier_version(monkeypatch,stints_job):
    from gestalt_pipelines.play_by_play.assets import stints
    job,objects,_,_,_,_=stints_job
    assert job.execute_in_process(partition_key=DAY).success
    old={k:v for k,v in objects.items() if k.startswith("lake/artifacts/stints/") and not k.endswith("/latest.json")}
    monkeypatch.setattr(stints,"PROCESSING_VERSION","nba_stints_v2_test")
    assert job.execute_in_process(partition_key=DAY).success
    assert all(objects[k]==v for k,v in old.items())
    assert len([k for k in objects if k.startswith("lake/artifacts/stints/")])==3


@pytest.mark.parametrize("scheduled",[True,False])
def test_only_daily_capture_success_requests_same_date_stints(stints_job,scheduled):
    from dagster import DagsterEventType,RunRequest,SkipReason,build_run_status_sensor_context
    from gestalt_pipelines.play_by_play.sensors import play_by_play_ready_for_stints
    with DagsterInstance.ephemeral() as instance:
        result=defs.resolve_job_def("capture_nba_play_by_play").execute_in_process(
            instance=instance,partition_key=DAY,
            tags={"dagster/schedule_name":"play_by_play_morning"} if scheduled else {},
        )
        event=next(e for e in result.all_events if e.event_type==DagsterEventType.RUN_SUCCESS)
        with build_run_status_sensor_context(
            sensor_name="play_by_play_ready_for_stints",dagster_event=event,
            dagster_instance=instance,dagster_run=instance.get_run_by_id(result.run_id),
        ) as context:
            request=play_by_play_ready_for_stints(context)
            repeated=play_by_play_ready_for_stints(context)
        if scheduled:
            assert isinstance(request,RunRequest)
            assert request.partition_key==DAY and request.run_key==result.run_id==repeated.run_key
        else:
            assert isinstance(request,SkipReason)


def test_one_missing_game_does_not_publish_an_accepted_partial_date(monkeypatch,stints_job):
    job,objects,_,_,game,_=stints_job
    missing={**game,"game_id":"0020000002"}
    monkeypatch.setattr(NBAScheduleResource,"games_for_date",lambda *a,**k:([game,missing],{"uri":"s3://fixture/schedule.json","sha256":"s"}))
    failed=job.execute_in_process(partition_key=DAY,raise_on_error=False)
    assert not failed.success
    assert not any(k.startswith("lake/artifacts/stints/") for k in objects)
    [key]=[k for k in objects if k.endswith(".failure.json")]
    event=json.loads(objects[key])
    assert list(event["failed_games"])==[missing["game_id"]]
    assert len(event["input"]["games"][game["game_id"]])==2
