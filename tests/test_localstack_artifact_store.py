from __future__ import annotations

import hashlib
import os
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import boto3
import pytest
from botocore.config import Config

from nba_lineup_model.storage.artifact_store import (
    ArtifactReference,
    ObjectDigest,
    publish_artifact,
    publish_storage_manifest,
    resolve_artifact_file,
)
from nba_lineup_model.storage.environment_selection import (
    activate_environment_selection,
    read_environment_selection,
)

ENDPOINT = os.environ.get("GESTALT_LOCALSTACK_ENDPOINT")

@pytest.fixture
def localstack_s3():
    if not ENDPOINT:
        pytest.skip("set GESTALT_LOCALSTACK_ENDPOINT to run LocalStack S3 tests")
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(s3={"addressing_style": "path"}),
    )
    bucket = "gestalt-artifacts-" + uuid4().hex
    client.create_bucket(Bucket=bucket)
    try:
        yield client, f"s3://{bucket}/lake"
    finally:
        response = client.list_objects_v2(Bucket=bucket)
        objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        client.delete_bucket(Bucket=bucket)


def test_real_s3_api_publication_cas_and_fresh_cache(localstack_s3, tmp_path):
    client, lake_uri = localstack_s3
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_bytes(b'{"canonical": true}\n')
    (source / "payload.bin").write_bytes(b"durable bytes")
    inventory = {
        path.name: ObjectDigest.from_file(path)
        for path in source.iterdir()
    }
    reference = ArtifactReference(
        artifact_kind="daily_data",
        season="2026-27",
        game_date_pacific=date(2026, 11, 2),
        content_id="data_sha256="
        + hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
    )
    publish_artifact(
        source=source,
        destination_uri=reference.prefix_uri(lake_uri=lake_uri),
        files=inventory,
        client=client,
        lake_uri=lake_uri,
    )
    publish_storage_manifest(
        reference=reference,
        files=inventory,
        client=client,
        lake_uri=lake_uri,
    )
    restored = resolve_artifact_file(
        reference=reference,
        relative_path="payload.bin",
        cache_root=tmp_path / "fresh-cache",
        client=client,
        lake_uri=lake_uri,
    )
    assert restored.read_bytes() == b"durable bytes"

    first = activate_environment_selection(
        environment="dev",
        release=reference,
        operation_id="localstack-generation-1",
        reason="LocalStack integration test",
        expected_previous_generation=0,
        created_at=datetime(2026, 11, 5, tzinfo=UTC),
        client=client,
        lake_uri=lake_uri,
    )
    second = activate_environment_selection(
        environment="dev",
        release=reference,
        operation_id="localstack-generation-2",
        reason="LocalStack compare-and-swap test",
        expected_previous_generation=1,
        created_at=datetime(2026, 11, 5, 0, 1, tzinfo=UTC),
        client=client,
        lake_uri=lake_uri,
    )
    assert first.generation == 1
    assert second.generation == 2
    selected = read_environment_selection(
        "dev",
        client=client,
        lake_uri=lake_uri,
    )
    assert selected["current"]["generation"] == 2

