from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from nba_lineup_model.storage.s3_lake import (
    build_migration_manifest,
    upload_migration_manifest,
    verify_migration_manifest,
    write_migration_manifest,
)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "destination_base_uri": "s3://test-bucket/nba-gestalt",
                "manifest_prefix": "manifests/migrations",
                "exclude_globs": ["**/*.tmp"],
                "datasets": [
                    {
                        "name": "raw",
                        "source": "data/raw",
                        "destination": "lake/raw",
                        "disposition": "archive",
                    },
                    {
                        "name": "processed",
                        "source": "data/processed",
                        "disposition": "regenerate",
                    },
                ],
            }
        )
    )


def _planned_manifest(tmp_path: Path) -> tuple[dict[str, object], Path]:
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/raw/game.json").write_text('{"game": 1}\n')
    (tmp_path / "data/raw/ignored.tmp").write_text("ignore")
    (tmp_path / "data/processed/game.parquet").write_bytes(b"parquet")
    config = tmp_path / "migration.json"
    _write_config(config)
    manifest = build_migration_manifest(
        config_path=config,
        source_root=tmp_path,
        created_at=datetime(2026, 10, 24, tzinfo=UTC),
    )
    manifest_path = write_migration_manifest(manifest, tmp_path / "manifest.json")
    return manifest, manifest_path


def test_plan_is_deterministic_and_classifies_reproducible_data(tmp_path: Path) -> None:
    first, _ = _planned_manifest(tmp_path)
    config = tmp_path / "migration.json"
    second = build_migration_manifest(
        config_path=config,
        source_root=tmp_path,
        created_at=datetime(2026, 10, 25, tzinfo=UTC),
    )

    assert first["manifest_id"] == second["manifest_id"]
    assert first["created_at"] != second["created_at"]
    assert first["files"] == [
        {
            "dataset": "raw",
            "source_path": "data/raw/game.json",
            "s3_uri": "s3://test-bucket/nba-gestalt/lake/raw/game.json",
            "size_bytes": 12,
            "sha256": hashlib.sha256(b'{"game": 1}\n').hexdigest(),
        }
    ]
    datasets = {item["name"]: item for item in first["datasets"]}
    assert datasets["raw"]["file_count"] == 1
    assert datasets["processed"] == {
        "name": "processed",
        "source": "data/processed",
        "destination": None,
        "disposition": "regenerate",
        "file_count": 1,
        "total_bytes": 7,
    }


def test_repeated_plan_does_not_rewrite_same_manifest_id(tmp_path: Path) -> None:
    first, manifest_path = _planned_manifest(tmp_path)
    original_bytes = manifest_path.read_bytes()
    second = build_migration_manifest(
        config_path=tmp_path / "migration.json",
        source_root=tmp_path,
        created_at=datetime(2026, 10, 25, tzinfo=UTC),
    )

    assert first["manifest_id"] == second["manifest_id"]
    assert write_migration_manifest(second, manifest_path) == manifest_path
    assert manifest_path.read_bytes() == original_bytes


def test_plan_rejects_symlinks(tmp_path: Path) -> None:
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    source = tmp_path / "outside.json"
    source.write_text("{}")
    (tmp_path / "data/raw/link.json").symlink_to(source)
    config = tmp_path / "migration.json"
    _write_config(config)

    with pytest.raises(ValueError, match="symlinks"):
        build_migration_manifest(config_path=config, source_root=tmp_path)


def test_upload_dry_run_makes_no_aws_calls(tmp_path: Path) -> None:
    _, manifest_path = _planned_manifest(tmp_path)

    summary = upload_migration_manifest(manifest_path, client=object())

    assert summary.executed is False
    assert summary.planned == 1
    assert summary.uploaded == 0


class FakeAws:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[int, str]] = {}
        self.uploads: list[str] = []
        self.lock = threading.Lock()

    def head_object(self, *, Bucket, Key, ChecksumMode):
        assert ChecksumMode == "ENABLED"
        uri = f"s3://{Bucket}/{Key}"
        with self.lock:
            if uri not in self.objects:
                raise ClientError(
                    {"Error": {"Code": "404", "Message": "Not Found"}},
                    "HeadObject",
                )
            size, digest = self.objects[uri]
        return {"ContentLength": size, "Metadata": {"sha256": digest}}

    def put_object(
        self,
        *,
        Bucket,
        Key,
        Body,
        IfNoneMatch,
        Metadata,
        ChecksumAlgorithm,
        ChecksumSHA256,
    ):
        assert IfNoneMatch == "*"
        assert ChecksumAlgorithm == "SHA256"
        uri = f"s3://{Bucket}/{Key}"
        payload = Body.read()
        with self.lock:
            if uri in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed", "Message": "exists"}},
                    "PutObject",
                )
            digest = Metadata["sha256"]
            assert hashlib.sha256(payload).hexdigest() == digest
            self.objects[uri] = (len(payload), digest)
            self.uploads.append(uri)
        return {"ChecksumSHA256": ChecksumSHA256}


def test_upload_is_immutable_verified_and_idempotent(tmp_path: Path) -> None:
    manifest, manifest_path = _planned_manifest(tmp_path)
    aws = FakeAws()

    first = upload_migration_manifest(
        manifest_path, execute=True, client=aws, progress=lambda _: None
    )
    second = upload_migration_manifest(
        manifest_path, execute=True, client=aws, progress=lambda _: None
    )
    verified = verify_migration_manifest(manifest_path, client=aws)

    assert first.uploaded == 2
    assert first.verified == 2
    assert aws.uploads[-1] == manifest["manifest_s3_uri"]
    assert second.uploaded == 0
    assert second.existing == 2
    assert verified.verified == 2


def test_upload_rejects_mismatched_existing_object(tmp_path: Path) -> None:
    manifest, manifest_path = _planned_manifest(tmp_path)
    aws = FakeAws()
    entry = manifest["files"][0]
    aws.objects[entry["s3_uri"]] = (entry["size_bytes"], "0" * 64)

    with pytest.raises(RuntimeError, match="immutable S3 mismatch"):
        upload_migration_manifest(
            manifest_path, execute=True, client=aws, progress=lambda _: None
        )


def test_upload_rejects_local_file_changed_after_plan(tmp_path: Path) -> None:
    _, manifest_path = _planned_manifest(tmp_path)
    (tmp_path / "data/raw/game.json").write_text('{"game": 2}\n')

    with pytest.raises(RuntimeError, match="source changed"):
        upload_migration_manifest(
            manifest_path, execute=True, client=FakeAws(), progress=lambda _: None
        )


def test_manifest_rejects_redirected_manifest_destination(tmp_path: Path) -> None:
    manifest, _ = _planned_manifest(tmp_path)
    manifest["manifest_s3_uri"] = "s3://other-bucket/redirected.json"

    with pytest.raises(ValueError, match="does not match its manifest ID"):
        write_migration_manifest(manifest, tmp_path / "redirected.json")


def test_plan_rejects_overlapping_archived_destinations(tmp_path: Path) -> None:
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    config = tmp_path / "migration.json"
    _write_config(config)
    payload = json.loads(config.read_text())
    payload["datasets"][1] = {
        "name": "processed",
        "source": "data/processed",
        "destination": "lake/raw/processed",
        "disposition": "archive",
    }
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="destinations overlap"):
        build_migration_manifest(config_path=config, source_root=tmp_path)
