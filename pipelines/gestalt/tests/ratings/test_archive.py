"""Immutable publication, stale writers, and actual failed-parent recovery."""

import io
import json
from hashlib import sha256

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance, Definitions

from gestalt_pipelines.ratings.assets import player_rating_priors
from gestalt_pipelines.ratings.jobs import initialize_player_ratings
from gestalt_pipelines.ratings.resources import RatingsArchive


class S3:
    def __init__(self):
        self.objects = {}
        self.race = False

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = self.objects[Key]
        return {"Body": io.BytesIO(raw), "ETag": sha256(raw).hexdigest()}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, **kwargs):
        if (IfNoneMatch == "*" and Key in self.objects) or (self.race and Key.endswith("latest.json")):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        return {}


@pytest.fixture
def archive(monkeypatch):
    s3 = S3()
    monkeypatch.setattr("gestalt_pipelines.ratings.resources.archive.boto3.client", lambda service: s3)
    return RatingsArchive(season="2026-27"), s3


def test_identical_rerun_reuses_immutable_selection(archive, record):
    resource, s3 = archive
    saved = resource.save_priors(record, prior=None, expected_etag=None)
    before = dict(s3.objects)
    reference, etag = resource.selection()
    assert reference == {k: saved[k] for k in ("uri", "sha256")}
    again = resource.save_priors(record, prior=reference, expected_etag=etag)
    assert again["reused"] is True
    assert s3.objects == before
    assert json.loads(s3.objects[resource.prefix+saved["sha256"]+".json"])["players"] == record["players"]


def test_changed_initialization_does_not_replace_existing_selection(archive, record):
    resource, s3 = archive
    resource.save_priors(record, prior=None, expected_etag=None)
    before = dict(s3.objects); reference, etag = resource.selection()
    record["players"][0]["rapm"] += 0.1
    with pytest.raises(ValueError, match="already selected"):
        resource.save_priors(record, prior=reference, expected_etag=etag)
    assert s3.objects == before


def test_stale_first_writer_cannot_advance_pointer(archive, record):
    resource, s3 = archive
    s3.race = True
    with pytest.raises(ValueError, match="Another run"):
        resource.save_priors(record, prior=None, expected_etag=None)
    assert resource.prefix+"latest.json" not in s3.objects


def test_bad_data_does_not_advance_pointer(archive, record):
    resource, s3 = archive
    record["profiles"].pop()
    with pytest.raises(ValueError, match="coverage"):
        resource.save_priors(record, prior=None, expected_etag=None)
    assert not s3.objects


def test_missing_input_logs_failure_in_s3_and_dagster(archive):
    resource, s3 = archive
    defs = Definitions(assets=[player_rating_priors], jobs=[initialize_player_ratings],
                       resources={"ratings_archive": resource})
    with DagsterInstance.ephemeral() as instance:
        job = defs.resolve_job_def("initialize_player_ratings")
        result = job.execute_in_process(instance=instance, raise_on_error=False)
        assert not result.success
    failures = [(key, json.loads(raw)) for key, raw in s3.objects.items() if key.endswith(".failure.json")]
    assert len(failures) == 1
    key, event = failures[0]
    assert key.startswith("lake/artifacts/errors/pipeline=ratings/season=2026-27/occurred_date_pacific=")
    assert event["dagster_run_id"] == result.run_id
    assert event["stage"] == "player_rating_priors" and event["status"] == "failed"
    assert resource.prefix+"latest.json" not in s3.objects


def test_only_a_successful_child_of_failed_job_records_recovery(archive, record, monkeypatch):
    from dagster._core.execution.api import execute_run
    from dagster._core.definitions.job_base import InMemoryJob
    resource, s3 = archive
    defs = Definitions(assets=[player_rating_priors], jobs=[initialize_player_ratings],
                       resources={"ratings_archive": resource})
    job = defs.resolve_job_def("initialize_player_ratings")
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(instance=instance, raise_on_error=False)
        assert not failed.success
        # Repair the missing inputs in the fixture, then genuinely re-execute its failed parent.
        monkeypatch.setattr(RatingsArchive, "read_sources", lambda self: {})
        monkeypatch.setattr("gestalt_pipelines.ratings.assets.initialize.prepare_initialization",
                            lambda blobs, season: record)
        fresh = job.execute_in_process(instance=instance)
        assert fresh.success
        assert not any(k.endswith(".recovery.json") for k in s3.objects)
        parent = instance.get_run_by_id(failed.run_id)
        child = instance.create_run_for_job(job, parent_run_id=parent.run_id, root_run_id=parent.run_id)
        recovered = execute_run(InMemoryJob(job), child, instance)
        assert recovered.success
    recoveries = [json.loads(raw) for key, raw in s3.objects.items() if key.endswith(".recovery.json")]
    assert len(recoveries) == 1
    recovery = recoveries[0]
    failure_key = next(k for k in s3.objects if k.endswith(".failure.json"))
    assert recovery["original_failure"]["sha256"] == sha256(s3.objects[failure_key]).hexdigest()
    assert recovery["dagster_run_id"] == recovered.run_id
