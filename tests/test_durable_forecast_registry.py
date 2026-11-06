from pathlib import Path
import pytest
from nba_lineup_model.season import durable_forecast_registry as storage
from nba_lineup_model.season.forecast_publication import ForecastRegistry
from test_artifact_store import FakeS3, LAKE


class RegistryS3(FakeS3):
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        yield {"Contents": [{"Key": key} for bucket, key in self.objects
                            if bucket == Bucket and key.startswith(Prefix)]}


def setup(tmp_path, monkeypatch):
    client = RegistryS3()
    monkeypatch.setattr(storage, "_new_s3_client", lambda _: client)
    root = tmp_path / "original"
    root.mkdir()
    uri = storage.registry_uri(root, LAKE)
    storage.configure(root, uri=uri, cache_root=tmp_path / "cache")
    registry = ForecastRegistry.__new__(ForecastRegistry)
    registry.root = root
    return registry, client, uri


def test_upload_failure_cannot_create_local_publication_receipt(tmp_path, monkeypatch):
    registry, client, uri = setup(tmp_path, monkeypatch)
    client.fail_key = uri.split("/", 3)[3] + "/receipts/published.json"
    with pytest.raises(RuntimeError, match="Interrupted"):
        registry._put("receipts", "published", {"outcome": "successful"})
    assert not (registry.root / "receipts/published.json").exists()
    client.fail_key = None
    path = registry._put("receipts", "published", {"outcome": "successful"})
    assert path.is_file()
    assert len(client.objects) == 1


def test_restore_pinned_and_later_records_without_original_root(tmp_path, monkeypatch):
    registry, client, uri = setup(tmp_path, monkeypatch)
    registry._put("versions", "original", {"frozen": True})
    pinned = storage.publish_registry(registry.root, uri=uri, cache_root=tmp_path / "cache", client=client)
    registry._put("receipts", "later", {"outcome": "successful"})
    registry.root.rename(tmp_path / "inaccessible")
    target = tmp_path / "restored"
    storage.restore_registry(target, uri=uri, cache_root=tmp_path / "cold", required=pinned, client=client)
    assert (target / "versions/original.json").read_bytes() == b'{"frozen":true}\n'
    assert (target / "receipts/later.json").is_file()
    downloads = len(client.gets)
    storage.restore_registry(target, uri=uri, cache_root=tmp_path / "cold", required=pinned, client=client)
    assert len(client.gets) == downloads
    del client.objects["test-bucket", uri.split("/", 3)[3] + "/versions/original.json"]
    with pytest.raises(ValueError, match="missing or changed"):
        storage.restore_registry(target, uri=uri, cache_root=tmp_path / "cold", required=pinned, client=client)


def test_changed_version_cannot_overwrite_durable_or_local_record(tmp_path, monkeypatch):
    registry, client, uri = setup(tmp_path, monkeypatch)
    path = registry._put("versions", "original", {"frozen": True})
    before = dict(client.objects)
    with pytest.raises(ValueError, match="Immutable"):
        registry._put("versions", "original", {"frozen": False})
    assert client.objects == before
    assert path.read_bytes() == b'{"frozen":true}\n'


def test_restore_rejects_corrupt_local_copy(tmp_path, monkeypatch):
    registry, client, uri = setup(tmp_path, monkeypatch)
    registry._put("versions", "original", {"frozen": True})
    pinned = storage.publish_registry(registry.root, uri=uri, cache_root=tmp_path / "cache", client=client)
    (registry.root / "versions/original.json").write_text("corrupt")
    with pytest.raises(ValueError, match="changed"):
        storage.restore_registry(registry.root, uri=uri, cache_root=tmp_path / "cache", required=pinned, client=client)
