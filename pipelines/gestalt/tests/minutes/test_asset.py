import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from dagster import AssetKey, DagsterInstance, TimeWindowPartitionMapping
from dagster._core.execution.api import execute_run
from dagster._core.definitions.job_base import InMemoryJob

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.box_scores.resources import NBABoxScoreArchive
from gestalt_pipelines.minutes.assets import player_minutes
from gestalt_pipelines.minutes.resources import MinutesArchive
from gestalt_pipelines.minutes.resources import archive
from gestalt_pipelines.minutes.utils import PRIOR_REFERENCES, json_bytes
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.rosters.resources import NBARosterArchive

from .helpers import FIRST, SEASON, priors, reference, slate


@pytest.fixture
def minutes_job(monkeypatch):
    data = slate()
    objects, writes, times = {}, [], {}
    def put(**kw):
        key = kw["Key"]
        if kw.get("IfNoneMatch") == "*" and key in objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if "IfMatch" in kw and kw["IfMatch"] != sha256(objects[key]).hexdigest():
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        objects[key] = kw["Body"]
        times[key] = datetime(2026, 11, 12, tzinfo=UTC)+timedelta(seconds=len(writes))
        writes.append(key)
    def get(**kw):
        if kw["Key"] not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = objects[kw["Key"]]
        return {"Body": BytesIO(raw), "ETag": sha256(raw).hexdigest()}
    def pages(**kw):
        return iter([{"Contents": [
            {"Key": key, "LastModified": times[key]} for key in objects if key.startswith(kw["Prefix"])
        ]}])
    client = SimpleNamespace(get_object=get, put_object=put,
                             get_paginator=lambda _: SimpleNamespace(paginate=pages))
    monkeypatch.setattr(archive.boto3, "client", lambda _: client)
    a = MinutesArchive(season=SEASON)
    prior = a.save_priors(priors())
    monkeypatch.setitem(PRIOR_REFERENCES, (SEASON, "v1"), prior)
    def immutable(prefix, doc, suffix=None):
        raw = json_bytes(doc);digest = sha256(raw).hexdigest()
        key = prefix+(suffix or digest+".json")
        put(Key=key, Body=raw, IfNoneMatch="*")
        return {"uri": "s3://nba-gestalt-data/"+key, "sha256": digest}
    gid = data["games"][0]["game_id"]
    box_prefix = f"lake/raw/cdn/boxscore/season={SEASON}/game_id={gid}/"
    box = immutable(box_prefix, data["boxes"][gid])
    meta = NBABoxScoreArchive()._ensure_metadata(
        client, box["uri"].removeprefix("s3://nba-gestalt-data/"), SEASON,
        json_bytes(data["boxes"][gid]), box["sha256"],
    )
    box.update({k: meta[k] for k in ("metadata_uri", "metadata_sha256")})
    data["stints"]["inputs"] = {"games": {gid: {"box_score": box}}}
    sp = f"lake/artifacts/stints/season={SEASON}/game_date_pacific={FIRST}/"
    sr = immutable(sp, data["stints"])
    put(Key=sp+"latest.json", Body=json_bytes(sr), IfNoneMatch="*")
    ip = f"lake/artifacts/injury_status_by_game_date/season={SEASON}/game_date_pacific={FIRST}/"
    ir = immutable(ip, data["injuries"], "selection_sha256="+("c"*64)+"/statuses.json")
    put(Key=ip+"latest.json", Body=json_bytes(ir), IfNoneMatch="*")
    data["rosters"]["schema_version"] = 1
    NBARosterArchive().publish(data["rosters"], expected_etag=None)
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *a, **k: (data["games"], reference("schedule")))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda *a, **k: (b"schedule", reference("schedule")))
    return defs.resolve_job_def("update_player_minutes"), a, objects, writes, sp


def test_date_only_job_publishes_all_players_and_reuses_same_bytes(minutes_job):
    job, a, objects, writes, _ = minutes_job
    assert job.execute_in_process(partition_key=str(FIRST)).success
    record, ref, _ = a.state(str(FIRST))
    assert record["validation"]["complete"] is True
    assert record["model"]["model_version"] == "v1"
    assert record["players"]["15"]["history"][0]["minutes"] == 0
    assert record["players"]["17"]["history"] == []
    key = ref["uri"].removeprefix("s3://nba-gestalt-data/")
    raw = objects[key]
    assert key.endswith("/"+sha256(raw).hexdigest()+".json")
    before = len(writes)
    assert job.execute_in_process(partition_key=str(FIRST)).success
    assert objects[key] == raw
    assert not any(k.endswith("/latest.json") for k in writes[before:])


@pytest.mark.parametrize("backfill_tags", [False, True])
@pytest.mark.parametrize("source_kind", ["stints", "injuries", "rosters"])
def test_missing_upstream_keeps_pointer_and_failed_parent_retry_recovers(minutes_job, backfill_tags, source_kind):
    job, a, objects, _, source_prefix = minutes_job
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance, partition_key=str(FIRST)).success
        pointer_key = a.prefix+f"game_date_pacific={FIRST}/latest.json"
        accepted = objects[pointer_key]
        source_key = source_prefix+"latest.json"
        if source_kind == "injuries":
            source_key = f"lake/artifacts/injury_status_by_game_date/season={SEASON}/game_date_pacific={FIRST}/latest.json"
        elif source_kind == "rosters":
            source_key = f"lake/artifacts/rosters/season={SEASON}/game_date_pacific={FIRST}/latest.json"
        source = objects.pop(source_key)
        failed = job.execute_in_process(instance=instance, partition_key=str(FIRST), raise_on_error=False)
        assert not failed.success and objects[pointer_key] == accepted
        [failure_key] = [k for k in objects if k.endswith(".failure.json")]
        failure_bytes = objects[failure_key]
        failure = json.loads(failure_bytes)
        assert failure["pipeline"] == "minutes" and failure["model_version"] == "v1"
        objects[source_key] = source
        parent = instance.get_run_by_id(failed.run_id)
        if backfill_tags:
            # Real asset backfills carry a range even with one partition per run.
            parent = parent._replace(
                job_name="__ASSET_JOB", asset_selection=frozenset({AssetKey("player_minutes")}),
                tags={"dagster/asset_partition_range_start": str(FIRST),
                      "dagster/asset_partition_range_end": str(FIRST)},
            )
            context = SimpleNamespace(
                run=SimpleNamespace(parent_run_id=parent.run_id),
                instance=SimpleNamespace(
                    get_run_by_id=lambda _: parent,
                    get_records_for_run=instance.get_records_for_run,
                ),
            )
            ref = a.parent_failure(context, str(FIRST))
            assert ref["sha256"] == sha256(failure_bytes).hexdigest()
            assert a.parent_failure(context, "2026-10-21") is None
        child = instance.create_run_for_job(
            job, run_config=parent.run_config, tags=parent.tags,
            parent_run_id=parent.run_id, root_run_id=parent.run_id,
        )
        result = execute_run(InMemoryJob(job), child, instance)
        assert result.success
    [recovery_key] = [k for k in objects if k.endswith(".recovery.json")]
    recovery = json.loads(objects[recovery_key])
    assert recovery["original_failure"]["uri"] == "s3://nba-gestalt-data/"+failure_key
    assert recovery["original_failure"]["sha256"] == sha256(failure_bytes).hexdigest()
    assert objects[failure_key] == failure_bytes and objects[pointer_key] == accepted


def test_missing_predecessor_does_not_restart_and_versions_are_isolated(minutes_job):
    job, a, objects, _, _ = minutes_job
    assert not job.execute_in_process(partition_key="2026-10-21", raise_on_error=False).success
    assert not any(k.endswith("/latest.json") and k.startswith(a.prefix) for k in objects)
    _, ref = a.load_priors()
    v2 = MinutesArchive(season=SEASON, model_version="v2", prior_uri=ref["uri"])
    with pytest.raises(ValueError, match="reference"):
        v2.load_priors()
    assert v2.state(str(FIRST)) == (None, None, None)


def test_dependency_uses_previous_partition_and_one_partition_per_run():
    from dagster import AssetKey
    mapping = player_minutes.get_partition_mapping(AssetKey("player_minutes"))
    assert isinstance(mapping, TimeWindowPartitionMapping)
    assert (mapping.start_offset, mapping.end_offset) == (-1, -1)
    assert mapping.allow_nonexistent_upstream_partitions
    assert player_minutes.backfill_policy.max_partitions_per_run == 1


def test_publishing_another_model_version_cannot_move_v1(minutes_job):
    from copy import deepcopy
    from gestalt_pipelines.minutes.utils import model_metadata
    job, v1, objects, _, _ = minutes_job
    assert job.execute_in_process(partition_key=str(FIRST)).success
    state, original, _ = v1.state(str(FIRST))
    pointer_key = v1.prefix+f"game_date_pacific={FIRST}/latest.json"
    accepted = objects[pointer_key]
    v2 = MinutesArchive(season=SEASON, model_version="v2")
    prior = priors()
    prior["model"] = model_metadata("v2")
    v2_prior = v2.save_priors(prior)
    record = deepcopy(state)
    record["model"] = model_metadata("v2")
    record["inputs"]["priors"] = v2_prior
    saved = v2.save_state(record, prior=None, expected_etag=None)
    assert "/model_version=v2/" in saved["uri"]
    assert objects[pointer_key] == accepted and v1.state(str(FIRST))[1] == original
    assert v2.state(str(FIRST))[1]["uri"] == saved["uri"]
    with pytest.raises(ValueError, match="incompatible"):
        v1.save_state(record, prior=original, expected_etag=None)
    assert objects[pointer_key] == accepted


def test_roster_reader_uses_pointer_instead_of_a_newer_unselected_version(minutes_job):
    from copy import deepcopy
    _, a, objects, writes, _ = minutes_job
    selected = a._roster_ref(str(FIRST))
    raw = objects[selected["uri"].removeprefix("s3://nba-gestalt-data/")]
    record = deepcopy(json.loads(raw))
    record["teams"][next(iter(record["teams"]))].append(999999)
    raw = json_bytes(record)
    key = selected["uri"].rsplit("/", 1)[0].removeprefix("s3://nba-gestalt-data/")+"/"+sha256(raw).hexdigest()+".json"
    archive.boto3.client("s3").put_object(Key=key, Body=raw, IfNoneMatch="*")
    assert a._roster_ref(str(FIRST)) == selected
    assert a.load_day(str(FIRST))[1]["rosters"] == selected
    pointer = key.rsplit("/", 1)[0]+"/latest.json"
    del objects[pointer]
    with pytest.raises(ValueError, match="Missing accepted upstream"):
        a._roster_ref(str(FIRST))
