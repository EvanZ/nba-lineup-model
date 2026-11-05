from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nba_lineup_model.season import durable_release_storage as release_storage
from nba_lineup_model.storage.artifact_store import (
    ArtifactReference,
    ObjectDigest,
    publish_artifact,
    publish_storage_manifest,
)
from test_artifact_store import FakeS3, LAKE


def _reference(kind: str, content_id: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_kind=kind,
        season="2026-27",
        game_date_pacific=date(2026, 11, 2),
        content_id=content_id,
    )


def _seed(client, root: Path, reference: ArtifactReference) -> None:
    source = root / reference.artifact_kind
    source.mkdir(parents=True)
    (source / "manifest.json").write_text('{"canonical": true}\n')
    inventory = {
        "manifest.json": ObjectDigest.from_file(source / "manifest.json")
    }
    publish_artifact(
        source=source,
        destination_uri=reference.prefix_uri(lake_uri=LAKE),
        files=inventory,
        client=client,
        lake_uri=LAKE,
    )
    publish_storage_manifest(
        reference=reference,
        files=inventory,
        client=client,
        lake_uri=LAKE,
    )


def _release(tmp_path: Path):
    release = tmp_path / ("runtime_sha256=" + "c" * 64)
    release.mkdir()
    (release / "runtime-manifest.json").write_bytes(
        b'{ "original_runtime_manifest": true }\n'
    )
    (release / "payload.bin").write_bytes(b"immutable release payload")
    selected = release / "selected-schedule"
    selected.mkdir()
    (selected / "selection.json").write_text('{"selected": true}\n')
    overlay = tmp_path / "mutable-schedule-overlay"
    overlay.mkdir()
    (overlay / "live.json").write_text('{"mutable": true}\n')
    (release / "schedule").symlink_to(overlay, target_is_directory=True)

    data = _reference("daily_data", "data_sha256=" + "a" * 64)
    model = _reference("daily_model", "model_sha256=" + "b" * 64)
    data_path = (
        tmp_path
        / "daily-data"
        / "season=2026-27"
        / "game_date_pacific=2026-11-02"
        / data.content_id
    )
    model_path = (
        tmp_path
        / "daily-model"
        / "season=2026-27"
        / "game_date_pacific=2026-11-02"
        / model.content_id
    )
    manifest = {
        "runtime_sha256": "c" * 64,
        "identity": {
            "artifact_kind": "production_site_release",
            "season": "2026-27",
            "evidence_through_date": "2026-11-02",
            "schedule_overlay_path": str(overlay),
        },
        "approved_manifest": {
            "identity": {
                "data": {"bundle_path": str(data_path)},
                "model": {"bundle_path": str(model_path)},
            }
        },
    }
    return release, manifest, data, model


def test_release_storage_preserves_manifest_and_excludes_mutable_overlay(
    tmp_path, monkeypatch
):
    release, manifest, data, model = _release(tmp_path)
    client = FakeS3()
    _seed(client, tmp_path / "inputs", data)
    _seed(client, tmp_path / "inputs", model)
    monkeypatch.setattr(
        release_storage,
        "verify_production_site_release",
        lambda _path: manifest,
    )

    publication = release_storage.publish_production_release(
        release,
        cache_root=tmp_path / "cache",
        client=client,
        lake_uri=LAKE,
    )
    assert publication.reference.content_id == "runtime_sha256=" + "c" * 64
    descriptor_key = publication.storage_manifest_uri.removeprefix(
        "s3://test-bucket/"
    )
    descriptor = json.loads(client.objects["test-bucket", descriptor_key][0])
    assert descriptor["manifest_path"] == "runtime-manifest.json"
    assert descriptor["dependencies"] == [
        data.as_dict(),
        model.as_dict(),
    ]
    assert "schedule" not in descriptor["objects"]
    assert "selected-schedule/selection.json" in descriptor["objects"]
    manifest_key = next(
        key
        for bucket, key in client.objects
        if bucket == "test-bucket"
        and key.endswith("/runtime-manifest.json")
    )
    assert client.objects["test-bucket", manifest_key][0] == (
        release / "runtime-manifest.json"
    ).read_bytes()

    repeated = release_storage.publish_production_release(
        release,
        cache_root=tmp_path / "cache",
        client=client,
        lake_uri=LAKE,
    )
    assert repeated.objects.uploaded == 0
    assert not repeated.storage_manifest_uploaded


def test_release_storage_requires_inputs_and_rejects_other_symlinks(
    tmp_path, monkeypatch
):
    release, manifest, data, model = _release(tmp_path)
    client = FakeS3()
    monkeypatch.setattr(
        release_storage,
        "verify_production_site_release",
        lambda _path: manifest,
    )
    with pytest.raises(FileNotFoundError, match="storage manifest is missing"):
        release_storage.publish_production_release(
            release,
            cache_root=tmp_path / "cache",
            client=client,
            lake_uri=LAKE,
        )

    _seed(client, tmp_path / "inputs", data)
    _seed(client, tmp_path / "inputs", model)
    (release / "unexpected").symlink_to(release / "payload.bin")
    with pytest.raises(ValueError, match="unexpected symlink"):
        release_storage.publish_production_release(
            release,
            cache_root=tmp_path / "cache",
            client=client,
            lake_uri=LAKE,
        )

