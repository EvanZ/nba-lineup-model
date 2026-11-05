from __future__ import annotations

import base64
import hashlib
import io
import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from nba_lineup_model.storage.artifact_store import (
    ArtifactReference,
    ObjectDigest,
    artifact_prefix,
    build_storage_manifest,
    parse_legacy_artifact_reference,
    publish_artifact,
    publish_storage_manifest,
    resolve_artifact_file,
    resolve_object,
    storage_manifest_uri,
    verify_storage_manifest,
)

LAKE = "s3://test-bucket/lake"
PREFIX = (
    f"{LAKE}/artifacts/daily_data/season=2026-27/game_date_pacific=2026-10-20/"
    + "data_sha256=" + "a" * 64
)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.gets = []
        self.lock = threading.Lock()
        self.fail_key = None
        self.body_factory = io.BytesIO
        self.last_body = None

    def head_object(self, *, Bucket, Key, ChecksumMode):
        with self.lock:
            if (Bucket, Key) not in self.objects:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            data, metadata, checksum = self.objects[Bucket, Key]
        return {"ContentLength": len(data), "Metadata": metadata, "ChecksumSHA256": checksum}

    def put_object(
        self, *, Bucket, Key, Body, IfNoneMatch, Metadata, ChecksumAlgorithm, ChecksumSHA256
    ):
        assert IfNoneMatch == "*"
        assert ChecksumAlgorithm == "SHA256"
        data = Body.read()
        assert base64.b64encode(hashlib.sha256(data).digest()).decode() == ChecksumSHA256
        with self.lock:
            self.puts.append(Key)
            if Key == self.fail_key:
                raise RuntimeError("Interrupted upload")
            if (Bucket, Key) in self.objects:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.objects[Bucket, Key] = data, Metadata, ChecksumSHA256
        return {"ChecksumSHA256": ChecksumSHA256}

    def get_object(self, *, Bucket, Key, ChecksumMode):
        with self.lock:
            self.gets.append(Key)
            if (Bucket, Key) not in self.objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            data = self.objects[Bucket, Key][0]
        self.last_body = self.body_factory(data)
        return {"Body": self.last_body, "ContentLength": len(data)}


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "files").mkdir()
    (source / "files/a.bin").write_bytes(b"canonical component")
    (source / "files/b.bin").write_bytes(b"second component")
    # Preserve spacing and hashed absolute-path provenance verbatim.
    (source / "manifest.json").write_bytes(
        b'{ "input_data_uri": "/old/ec2/data", "identity": "unchanged" }\n'
    )
    inventory = {
        str(path.relative_to(source)): ObjectDigest.from_file(path)
        for path in source.rglob("*") if path.is_file()
    }
    return source, inventory, FakeS3()


def publish(source, inventory, client):
    return publish_artifact(
        source=source, destination_uri=PREFIX, files=inventory, client=client, lake_uri=LAKE
    )


def resolve(tmp_path, inventory, client, name="files/a.bin", uri=None):
    return resolve_object(uri=uri or f"{PREFIX}/{name}", digest=inventory[name],
                          cache_root=tmp_path / "cache", client=client, lake_uri=LAKE)


def reference(content_id="data_sha256=" + "a" * 64):
    return ArtifactReference(
        artifact_kind="daily_data",
        season="2026-27",
        game_date_pacific=date(2026, 10, 20),
        content_id=content_id,
    )


def _process_resolve_worker(cache_root, count, data, results):
    class ProcessS3:
        def get_object(self, **kwargs):
            with count.get_lock():
                count.value += 1
            return {"ContentLength": len(data), "Body": io.BytesIO(data)}

    try:
        path = resolve_object(
            uri=f"{PREFIX}/shared.bin",
            digest=ObjectDigest(hashlib.sha256(data).hexdigest(), len(data)),
            cache_root=cache_root,
            client=ProcessS3(),
            lake_uri=LAKE,
        )
        results.put(("ok", str(path), path.read_bytes()))
    except Exception as error:
        results.put(("error", str(error)))


def test_manifest_is_last_verbatim_and_retry_is_idempotent(tmp_path):
    source, inventory, client = fixture(tmp_path)
    first = publish(source, inventory, client)
    assert first.uploaded == 3 and first.verified == 3
    assert client.puts[-1].endswith("/manifest.json")
    manifest = next(
        value[0] for (_, key), value in client.objects.items() if key.endswith("/manifest.json")
    )
    assert manifest == (source / "manifest.json").read_bytes()
    second = publish(source, inventory, client)
    assert second.uploaded == 0 and second.existing == 3
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: publish(source, inventory, client), range(4)))
    assert all(result.uploaded == 0 for result in results)


def test_storage_manifest_is_published_last_and_is_idempotent(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    first = publish_storage_manifest(
        reference=reference(), files=inventory, client=client, lake_uri=LAKE
    )
    assert first.uploaded
    assert first.uri == storage_manifest_uri(reference(), lake_uri=LAKE)
    assert client.puts[-1].endswith("/storage-manifest.json")
    key = first.uri.removeprefix("s3://test-bucket/")
    payload = json.loads(client.objects["test-bucket", key][0])
    verified = verify_storage_manifest(payload, expected=reference(), lake_uri=LAKE)
    assert sorted(verified["objects"]) == ["files/a.bin", "files/b.bin", "manifest.json"]
    assert client.objects[
        "test-bucket", PREFIX.removeprefix("s3://test-bucket/") + "/manifest.json"
    ][0] == (source / "manifest.json").read_bytes()

    second = publish_storage_manifest(
        reference=reference(), files=inventory, client=client, lake_uri=LAKE
    )
    assert not second.uploaded
    assert second.digest == first.digest


def test_storage_manifest_preserves_a_nonstandard_canonical_manifest_name(tmp_path):
    source, inventory, client = fixture(tmp_path)
    runtime_manifest = source / "runtime-manifest.json"
    (source / "manifest.json").replace(runtime_manifest)
    inventory.pop("manifest.json")
    inventory["runtime-manifest.json"] = ObjectDigest.from_file(runtime_manifest)
    publish_artifact(
        source=source,
        destination_uri=PREFIX,
        files=inventory,
        manifest_path="runtime-manifest.json",
        client=client,
        lake_uri=LAKE,
    )
    publication = publish_storage_manifest(
        reference=reference(),
        files=inventory,
        manifest_path="runtime-manifest.json",
        client=client,
        lake_uri=LAKE,
    )
    key = publication.uri.removeprefix("s3://test-bucket/")
    payload = json.loads(client.objects["test-bucket", key][0])
    assert payload["manifest_path"] == "runtime-manifest.json"
    assert "manifest.json" not in payload["objects"]


def test_storage_manifest_requires_every_verified_object(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    missing_key = PREFIX.removeprefix("s3://test-bucket/") + "/files/b.bin"
    del client.objects["test-bucket", missing_key]
    with pytest.raises(RuntimeError, match="missing before manifest publication"):
        publish_storage_manifest(
            reference=reference(), files=inventory, client=client, lake_uri=LAKE
        )
    assert not any(key.endswith("/storage-manifest.json") for _, key in client.objects)


def test_storage_manifest_conflict_cannot_overwrite(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    first = publish_storage_manifest(
        reference=reference(), files=inventory, client=client, lake_uri=LAKE
    )
    key = first.uri.removeprefix("s3://test-bucket/")
    original = client.objects["test-bucket", key]
    changed = dict(inventory)
    changed.pop("files/b.bin")
    with pytest.raises(RuntimeError, match="immutable S3 mismatch"):
        publish_storage_manifest(
            reference=reference(), files=changed, client=client, lake_uri=LAKE
        )
    assert client.objects["test-bucket", key] == original


def test_resolve_artifact_file_downloads_only_requested_object(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    publish_storage_manifest(
        reference=reference(), files=inventory, client=client, lake_uri=LAKE
    )
    client.gets.clear()
    cached = resolve_artifact_file(
        reference=reference(),
        relative_path="files/b.bin",
        cache_root=tmp_path / "durable-cache",
        client=client,
        lake_uri=LAKE,
    )
    assert cached.read_bytes() == (source / "files/b.bin").read_bytes()
    assert len(client.gets) == 2
    assert client.gets[0].endswith("/storage-manifest.json")
    assert client.gets[1].endswith("/files/b.bin")
    assert not any(key.endswith("/files/a.bin") for key in client.gets)

    repeated = resolve_artifact_file(
        reference=reference(),
        relative_path="files/b.bin",
        cache_root=tmp_path / "durable-cache",
        client=client,
        lake_uri=LAKE,
    )
    assert repeated == cached
    assert len(client.gets) == 2


def test_storage_manifest_validates_dependencies_and_identity(tmp_path):
    _, inventory, _ = fixture(tmp_path)
    dependency = ArtifactReference(
        artifact_kind="daily_model",
        season="2026-27",
        game_date_pacific=date(2026, 10, 19),
        content_id="model_sha256=" + "b" * 64,
    )
    payload = build_storage_manifest(
        reference=reference(), files=inventory, dependencies=[dependency], lake_uri=LAKE
    )
    assert payload["dependencies"] == [dependency.as_dict()]
    payload["identity"]["content_id"] = "data_sha256=" + "c" * 64
    with pytest.raises(ValueError, match="different artifact identity"):
        verify_storage_manifest(payload, expected=reference(), lake_uri=LAKE)


def test_legacy_path_resolves_existing_identity_without_rewriting():
    path = (
        "/old/ec2/daily-data/season=2026-27/game_date_pacific=2026-10-20/"
        + "data_sha256=" + "a" * 64
    )
    assert parse_legacy_artifact_reference(path) == reference()
    with pytest.raises(ValueError, match="absolute path"):
        parse_legacy_artifact_reference("season=2026-27/game_date_pacific=2026-10-20/x")


def test_partial_upload_is_not_published_and_retry_completes(tmp_path):
    source, inventory, client = fixture(tmp_path)
    client.fail_key = PREFIX.removeprefix("s3://test-bucket/") + "/files/b.bin"
    with pytest.raises(RuntimeError, match="Interrupted"):
        publish(source, inventory, client)
    assert not any(key.endswith("/manifest.json") for _, key in client.objects)
    client.fail_key = None
    result = publish(source, inventory, client)
    assert result.existing == 1 and result.uploaded == 2


def test_changed_source_or_existing_identity_cannot_overwrite(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    original = dict(client.objects)
    (source / "manifest.json").write_text("different manifest")
    with pytest.raises(RuntimeError, match="checksum/size mismatch"):
        publish(source, inventory, client)
    inventory["manifest.json"] = ObjectDigest.from_file(source / "manifest.json")
    with pytest.raises(RuntimeError, match="immutable S3 mismatch"):
        publish(source, inventory, client)
    assert client.objects == original


def test_component_conflict_leaves_no_manifest(tmp_path):
    source, inventory, client = fixture(tmp_path)
    name = "files/a.bin"
    key = PREFIX.removeprefix("s3://test-bucket/") + "/" + name
    client.objects["test-bucket", key] = (b"bad", {"sha256": "0" * 64}, None)
    with pytest.raises(RuntimeError, match="immutable S3 mismatch"):
        publish(source, inventory, client)
    assert not any(key.endswith("/manifest.json") for _, key in client.objects)


def test_cache_is_lazy_verifies_hits_and_coordinates_readers(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: resolve(tmp_path, inventory, client), range(16)))
    assert len(set(paths)) == 1 and len(client.gets) == 1
    assert paths[0].read_bytes() == (source / "files/a.bin").read_bytes()
    assert len(list((tmp_path / "cache").rglob("object"))) == 1
    paths[0].write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum/size mismatch"):
        resolve(tmp_path, inventory, client)
    assert len(client.gets) == 1


@pytest.mark.parametrize("failure", ["corrupt", "truncated", "interrupted"])
def test_bad_download_never_becomes_cache_hit_and_retry_works(tmp_path, failure):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)

    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("Disconnected")
            return super().read(2)

    if failure == "corrupt":
        client.body_factory = lambda data: io.BytesIO(b"x" * len(data))
    elif failure == "truncated":
        client.body_factory = lambda data: io.BytesIO(data[:-1])
    else:
        client.body_factory = Interrupted
    with pytest.raises((RuntimeError, OSError)):
        resolve(tmp_path, inventory, client)
    assert client.last_body.closed
    assert not list((tmp_path / "cache").rglob("object"))
    assert not list((tmp_path / "cache").rglob(".download-*"))
    client.body_factory = io.BytesIO
    cached = resolve(tmp_path, inventory, client)
    assert cached.read_bytes() == (source / "files/a.bin").read_bytes()


def test_missing_uri_cannot_use_a_different_objects_cache(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    resolve(tmp_path, inventory, client)
    with pytest.raises(ClientError, match="NoSuchKey"):
        resolve(tmp_path, inventory, client, uri=f"{PREFIX}/missing.bin")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "files//a.bin", "files/./a.bin"])
def test_inventory_path_escape_is_rejected_before_upload(tmp_path, name):
    source, inventory, client = fixture(tmp_path)
    inventory[name] = inventory["files/a.bin"]
    with pytest.raises(ValueError):
        publish(source, inventory, client)
    assert not client.objects


def test_symlink_component_and_cached_object_are_rejected(tmp_path):
    source, inventory, client = fixture(tmp_path)
    publish(source, inventory, client)
    cached = resolve(tmp_path, inventory, client)
    cached.unlink()
    cached.symlink_to(source / "files/a.bin")
    with pytest.raises(RuntimeError, match="regular file"):
        resolve(tmp_path, inventory, client)
    (source / "files/a.bin").unlink()
    (source / "files/a.bin").symlink_to(source / "files/b.bin")
    with pytest.raises(ValueError, match="symlink"):
        publish(source, inventory, client)


def test_missing_manifest_and_invalid_digests_are_rejected(tmp_path):
    source, inventory, client = fixture(tmp_path)
    inventory.pop("manifest.json")
    with pytest.raises(ValueError, match="original manifest"):
        publish(source, inventory, client)
    for checksum, size in [("x", 1), ("a" * 64, -1), ("a" * 64, True)]:
        with pytest.raises(ValueError):
            ObjectDigest(checksum, size)


def test_namespace_preserves_existing_content_id():
    assert artifact_prefix(artifact_kind="daily_data", season="2026-27",
                           game_date_pacific=date(2026, 10, 20),
                           content_id="data_sha256=" + "a" * 64, lake_uri=LAKE) == PREFIX
    with pytest.raises(ValueError):
        artifact_prefix(artifact_kind="../data", season="2026-27",
                        game_date_pacific=date(2026, 10, 20), content_id="x", lake_uri=LAKE)

def test_cache_coordinates_separate_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    count = context.Value("i", 0)
    data = b"shared exact object"
    results = context.Queue()
    processes = [
        context.Process(
            target=_process_resolve_worker,
            args=(tmp_path / "cache", count, data, results),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    try:
        received = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
        assert all(process.exitcode == 0 for process in processes)
        assert all(result[0] == "ok" and result[2] == data for result in received)
        assert len({result[1] for result in received}) == 1
        assert count.value == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
