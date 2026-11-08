import hashlib
import json
from pathlib import Path
import pytest
from nba_lineup_model.season import serving_recovery as recovery
from nba_lineup_model.storage.artifact_store import ArtifactReference, ObjectDigest, publish_artifact, publish_storage_manifest
from nba_lineup_model.storage.s3_lake import _ensure_remote_object
from test_artifact_store import FakeS3, LAKE
from datetime import date


def setup(tmp_path):
    client = FakeS3()
    release = tmp_path / "release"
    release.mkdir()
    history = tmp_path / "gestalt-web-history"
    history.mkdir()
    (history / "ratings.json").write_text('{ "rating": 1 }\n')
    digest = ObjectDigest.from_file(history / "ratings.json")
    uri = "s3://test-bucket/nba-gestalt/releases/gestalt-web-history"
    _ensure_remote_object(source=history / "ratings.json", uri=uri + "/ratings.json",
                          sha256=digest.sha256, size_bytes=digest.size_bytes, client=client)
    bindings = {"historical_runtime": str(history), "historical_files": {"ratings.json": digest.sha256}}
    (release / "serving-bindings.json").write_text(json.dumps(bindings))
    daily = {}
    approved = {}
    for label, kind, char in [("data", "daily_data", "a"), ("model", "daily_model", "b")]:
        ref = ArtifactReference(kind, "2026-27", date(2026, 11, 2), f"{label}_sha256=" + char * 64)
        path = tmp_path / kind / "season=2026-27" / "game_date_pacific=2026-11-02" / ref.content_id
        path.mkdir(parents=True)
        (path / "manifest.json").write_text('{ "preserved": true }\n')
        (path / "data.bin").write_bytes(b"original bytes")
        files = {p.name: ObjectDigest.from_file(p) for p in path.iterdir()}
        publish_artifact(source=path, destination_uri=ref.prefix_uri(lake_uri=LAKE), files=files, client=client, lake_uri=LAKE)
        publish_storage_manifest(reference=ref, files=files, client=client, lake_uri=LAKE)
        daily[label] = {"source_path": str(path), "reference": ref.as_dict()}
        approved[label] = {"bundle_path": str(path), "bundle_sha256": char * 64}
    value = {"schema_version": 1, "lake_uri": LAKE,
             "historical": {"source_root": str(history), "uri": uri,
                            "files": {"ratings.json": {"s3_uri": uri + "/ratings.json", "sha256": digest.sha256, "size_bytes": digest.size_bytes}}},
             "registry": None, "daily": daily, "model_pointer": '{ "bundle_sha256": "' + "b" * 64 + '" }\n'}
    (release / recovery.FILENAME).write_text(json.dumps(value))
    runtime = {"identity": {"durable_serving_sha256": recovery._sha(value)}, "runtime_sha256": "c" * 64,
               "approved_manifest": {"identity": approved}}
    (release / "runtime-manifest.json").write_text(json.dumps(runtime))
    (release / "site-release.env").write_text("GESTALT_ARTIFACT_CACHE_ROOT=" + str(tmp_path / "cold") + "\n")
    return release, history, client, value


def test_cold_restore_uses_exact_s3_bytes_without_original_directories(tmp_path):
    release, history, client, value = setup(tmp_path)
    history.rename(tmp_path / "hidden-history")
    for kind in ("daily_data", "daily_model"):
        (tmp_path / kind).rename(tmp_path / ("hidden-" + kind))
    root = recovery.restore_inputs(release, client=client)
    assert (root / "historical/ratings.json").read_bytes() == b'{ "rating": 1 }\n'
    for item in value["daily"].values():
        path = recovery.daily_path(Path(item["source_path"]), release=release)
        assert path.is_relative_to(root)
        assert (path / "manifest.json").read_bytes() == b'{ "preserved": true }\n'
    downloads = len(client.gets)
    recovery._PREPARED.clear()  # A fresh process must verify the cache again.
    recovery.restore_inputs(release, client=client)
    assert len(client.gets) == downloads


def test_missing_object_cannot_fall_back_to_existing_source(tmp_path):
    release, history, client, value = setup(tmp_path)
    del client.objects["test-bucket", "nba-gestalt/releases/gestalt-web-history/ratings.json"]
    assert (history / "ratings.json").exists()
    with pytest.raises(Exception, match="NoSuchKey"):
        recovery.restore_inputs(release, client=client)


def test_changed_inventory_rejected_before_recovery(tmp_path):
    release, history, client, value = setup(tmp_path)
    value["historical"]["files"]["ratings.json"]["sha256"] = "0" * 64
    (release / recovery.FILENAME).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inventory changed"):
        recovery.restore_inputs(release, client=client)


def test_corrupt_materialized_history_fails_on_restart(tmp_path):
    release, history, client, value = setup(tmp_path)
    root = recovery.restore_inputs(release, client=client)
    (root / "historical/ratings.json").write_text("corrupt")
    recovery._PREPARED.clear()
    with pytest.raises(ValueError, match="Recovered file changed"):
        recovery.restore_inputs(release, client=client)
