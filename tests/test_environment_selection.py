from __future__ import annotations

import base64
import hashlib
import io
import threading
from datetime import UTC, date, datetime

import pytest
from botocore.exceptions import ClientError

from nba_lineup_model.storage.artifact_store import ArtifactReference
from nba_lineup_model.storage.environment_selection import (
    StaleEnvironmentSelection,
    activate_environment_selection,
    read_environment_selection,
    selection_record_uri,
)

LAKE = "s3://test-bucket/lake"
CREATED = datetime(2026, 11, 5, 16, 0, tzinfo=UTC)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()
        self.fail_after_commit_once = False
        self.puts = []

    def head_object(self, *, Bucket, Key, ChecksumMode):
        with self.lock:
            value = self.objects.get((Bucket, Key))
        if value is None:
            raise ClientError(
                {"Error": {"Code": "404"}},
                "HeadObject",
            )
        data, metadata, checksum, etag = value
        return {
            "ContentLength": len(data),
            "Metadata": metadata,
            "ChecksumSHA256": checksum,
            "ETag": etag,
        }

    def get_object(self, *, Bucket, Key, ChecksumMode):
        with self.lock:
            value = self.objects.get((Bucket, Key))
        if value is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}},
                "GetObject",
            )
        data, metadata, checksum, etag = value
        return {
            "Body": io.BytesIO(data),
            "ContentLength": len(data),
            "Metadata": metadata,
            "ChecksumSHA256": checksum,
            "ETag": etag,
        }

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        del ContinuationToken
        with self.lock:
            keys = sorted(
                key
                for bucket, key in self.objects
                if bucket == Bucket and key.startswith(Prefix)
            )
        return {
            "Contents": [{"Key": key} for key in keys],
            "IsTruncated": False,
        }

    def put_object(self, **kwargs):
        assert kwargs.get("IfNoneMatch") == "*"
        assert "IfMatch" not in kwargs
        bucket, key = kwargs["Bucket"], kwargs["Key"]
        body = kwargs["Body"]
        data = body.read() if hasattr(body, "read") else bytes(body)
        checksum = kwargs["ChecksumSHA256"]
        assert checksum == base64.b64encode(
            hashlib.sha256(data).digest()
        ).decode()
        identity = (bucket, key)
        with self.lock:
            if identity in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}},
                    "PutObject",
                )
            etag = (
                '"'
                + hashlib.md5(data, usedforsecurity=False).hexdigest()
                + '"'
            )
            self.objects[identity] = (
                data,
                kwargs["Metadata"],
                checksum,
                etag,
            )
            self.puts.append(key)
            fail = self.fail_after_commit_once
            self.fail_after_commit_once = False
        if fail:
            raise ClientError(
                {"Error": {"Code": "InternalError"}},
                "PutObject",
            )
        return {
            "ChecksumSHA256": checksum,
            "ETag": etag,
        }


def release(digest="a" * 64):
    return ArtifactReference(
        artifact_kind="site_release",
        season="2026-27",
        game_date_pacific=date(2026, 11, 2),
        content_id="runtime_sha256=" + digest,
    )


def activate(client, **overrides):
    values = {
        "environment": "dev",
        "release": release(),
        "operation_id": "daily-work-1",
        "reason": "validated dev release",
        "expected_previous_generation": 0,
        "created_at": CREATED,
        "client": client,
        "lake_uri": LAKE,
    }
    values.update(overrides)
    return activate_environment_selection(**values)


def test_first_selection_and_retry_use_one_immutable_object():
    client = FakeS3()
    first = activate(client)
    assert first.activated and first.generation == 1
    assert first.selection_uri == selection_record_uri(
        "dev",
        1,
        lake_uri=LAKE,
    )
    assert all(not key.endswith("current.json") for key in client.puts)

    selected = read_environment_selection(
        "dev",
        client=client,
        lake_uri=LAKE,
    )
    assert selected["current"]["release"] == release().as_dict()
    assert selected["record"]["reason"] == "validated dev release"
    assert selected["history_length"] == 1

    retried = activate(client)
    assert not retried.activated
    assert retried.selection_sha256 == first.selection_sha256
    assert len(client.puts) == 1

    retried_after_readback = activate(
        client,
        expected_previous_generation=1,
    )
    assert not retried_after_readback.activated
    assert retried_after_readback.generation == 1
    assert len(client.puts) == 1


def test_generations_form_a_verified_chain_and_stale_writer_fails():
    client = FakeS3()
    first = activate(client)
    second = activate(
        client,
        release=release("b" * 64),
        operation_id="daily-work-2",
        reason="next validated dev release",
        expected_previous_generation=1,
    )
    assert second.activated and second.generation == 2
    selected = read_environment_selection(
        "dev",
        client=client,
        lake_uri=LAKE,
    )
    assert selected["record"]["previous"] == {
        "generation": 1,
        "selection_sha256": first.selection_sha256,
    }

    with pytest.raises(
        StaleEnvironmentSelection,
        match="Expected generation 1, found 2",
    ):
        activate(
            client,
            release=release("c" * 64),
            operation_id="daily-work-stale",
            expected_previous_generation=1,
        )


def test_committed_write_with_lost_response_is_recovered():
    client = FakeS3()
    client.fail_after_commit_once = True
    recovered = activate(client)
    assert not recovered.activated
    assert recovered.generation == 1
    assert len(client.puts) == 1
    assert read_environment_selection(
        "dev",
        client=client,
        lake_uri=LAKE,
    )["current"]["generation"] == 1


def test_competing_operation_cannot_claim_the_same_generation():
    client = FakeS3()
    activate(client)
    activate(
        client,
        release=release("b" * 64),
        operation_id="daily-work-2",
        reason="second operation",
        expected_previous_generation=1,
    )
    with pytest.raises(
        StaleEnvironmentSelection,
        match="Expected generation 1, found 2",
    ):
        activate(
            client,
            release=release("c" * 64),
            operation_id="daily-work-3",
            reason="competing operation",
            expected_previous_generation=1,
        )


def test_dev_and_production_are_separate_and_production_requires_approval():
    client = FakeS3()
    activate(client)
    with pytest.raises(ValueError, match="approval reference"):
        activate(
            client,
            environment="production",
            operation_id="production-1",
            reason="production candidate",
        )
    production = activate(
        client,
        environment="production",
        operation_id="production-1",
        reason="approved production candidate",
        approval_reference="User approval 2026-11-05",
    )
    assert production.generation == 1
    dev = read_environment_selection(
        "dev",
        client=client,
        lake_uri=LAKE,
    )
    prod = read_environment_selection(
        "production",
        client=client,
        lake_uri=LAKE,
    )
    assert dev["record"]["approval_reference"] is None
    assert (
        prod["record"]["approval_reference"]
        == "User approval 2026-11-05"
    )


def test_corrupt_generation_fails_visibly():
    client = FakeS3()
    activate(client)
    key = selection_record_uri(
        "dev",
        1,
        lake_uri=LAKE,
    ).removeprefix("s3://test-bucket/")
    data, metadata, checksum, etag = client.objects["test-bucket", key]
    client.objects["test-bucket", key] = (
        b"corrupt",
        metadata,
        checksum,
        etag,
    )
    with pytest.raises(RuntimeError, match="checksum/size mismatch"):
        read_environment_selection(
            "dev",
            client=client,
            lake_uri=LAKE,
        )


def test_generation_gap_fails_visibly():
    client = FakeS3()
    activate(client)
    activate(
        client,
        release=release("b" * 64),
        operation_id="daily-work-2",
        reason="second operation",
        expected_previous_generation=1,
    )
    first_key = selection_record_uri(
        "dev",
        1,
        lake_uri=LAKE,
    ).removeprefix("s3://test-bucket/")
    del client.objects["test-bucket", first_key]
    with pytest.raises(RuntimeError, match="history has a gap"):
        read_environment_selection(
            "dev",
            client=client,
            lake_uri=LAKE,
        )
