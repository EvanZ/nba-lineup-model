"""Immutable, write-through S3 storage for the existing forecast registry.

The local registry is a replaceable cache. Records retain their original names
and bytes; storage configuration is not a forecast or a publication receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from nba_lineup_model.season.daily_artifact_common import canonical_json
from nba_lineup_model.storage.artifact_store import ObjectDigest, resolve_object
from nba_lineup_model.storage.s3_lake import (
    _ensure_remote_object, _head_object, _new_s3_client, _parse_s3_uri,
    _validate_remote_object,
)

CONFIG = "durable-storage.json"
CATEGORIES = frozenset({"versions", "receipts", "selections", "inputs", "failures", "markets"})


def registry_uri(root: Path, lake_uri: str) -> str:
    # Retain the existing registry's namespace independently of a recovery path.
    identity = hashlib.sha256(str(root.absolute()).encode()).hexdigest()
    return f"{lake_uri.rstrip('/')}/forecast-registry/registry_sha256={identity}"


def record_name(name: str) -> str:
    path = Path(name)
    if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] not in CATEGORIES
            or path.suffix != ".json" or ".." in path.parts):
        raise ValueError(f"Invalid forecast registry record: {name}")
    return path.as_posix()


def storage_config(root: Path) -> dict | None:
    path = root / CONFIG
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not value.get("uri", "").startswith("s3://"):
        raise ValueError("Invalid forecast registry storage configuration")
    return value


def configure(root: Path, *, uri: str, cache_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "uri": uri, "cache_root": str(cache_root.absolute())}
    data = canonical_json(value) + b"\n"
    existing = storage_config(root)
    if existing is not None and existing != value:
        raise ValueError("Forecast registry storage configuration changed")
    if existing is None:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        try:
            try:
                os.link(temporary, root / CONFIG)
            except FileExistsError:
                if (root / CONFIG).read_bytes() != data:
                    raise ValueError("Conflicting registry storage configuration") from None
        finally:
            temporary.unlink(missing_ok=True)


def publish_record(root: Path, name: str, source: Path, *, client=None) -> dict | None:
    config = storage_config(root)
    if config is None:
        return None
    name = record_name(name)
    digest = ObjectDigest.from_file(source)
    uri = config["uri"] + "/" + name
    s3 = client or _new_s3_client(1)
    head = _head_object(uri, client=s3)
    if head is None:
        _ensure_remote_object(source=source, uri=uri, size_bytes=digest.size_bytes,
                              sha256=digest.sha256, client=s3)
    else:
        _validate_remote_object(uri=uri, head=head, size_bytes=digest.size_bytes,
                                sha256=digest.sha256)
    return {"s3_uri": uri, "sha256": digest.sha256, "size_bytes": digest.size_bytes}


def publish_registry(root: Path, *, uri: str, cache_root: Path, client=None) -> dict:
    """Bootstrap existing records, then require write-through for subsequent writes.

    Caller holds the existing registry publisher lock. No original record changes.
    """
    from nba_lineup_model.season.source_policy import require_clean_artifact_sources
    require_clean_artifact_sources(root)
    configure(root, uri=uri, cache_root=cache_root)
    s3 = client or _new_s3_client(1)
    entries = {}
    for category in sorted(CATEGORIES):
        for path in sorted((root / category).glob("*.json")):
            name = record_name(path.relative_to(root).as_posix())
            entries[name] = publish_record(root, name, path, client=s3)
    return entries


def restore_registry(root: Path, *, uri: str, cache_root: Path,
                     required: dict, client=None) -> None:
    """Restore pinned records plus later immutable receipts/selections.

    Listing is required: a release snapshot predates its own publication receipt.
    Fail closed if pinned records have disappeared or differ from their inventory.
    """
    from nba_lineup_model.season.forecast_publication import publisher_lock
    s3 = client or _new_s3_client(1)
    bucket, prefix = _parse_s3_uri(uri + "/")
    found = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            name = record_name(item["Key"][len(prefix):])
            object_uri = uri + "/" + name
            head = _head_object(object_uri, client=s3)
            if head is None:
                raise FileNotFoundError(object_uri)
            digest = ObjectDigest(head.get("Metadata", {}).get("sha256"), head["ContentLength"])
            _validate_remote_object(uri=object_uri, head=head, size_bytes=digest.size_bytes,
                                    sha256=digest.sha256)
            found[name] = {"s3_uri": object_uri, "sha256": digest.sha256,
                           "size_bytes": digest.size_bytes}
    if any(found.get(name) != value for name, value in required.items()):
        raise ValueError("Required durable forecast registry record is missing or changed")
    root.mkdir(parents=True, exist_ok=True)
    with publisher_lock(root):
        for name, entry in found.items():
            path = root / name
            digest = ObjectDigest(entry["sha256"], entry["size_bytes"])
            cached = resolve_object(uri=entry["s3_uri"], digest=digest,
                                    cache_root=cache_root, lake_uri=uri, client=s3)
            if path.exists():
                if path.is_symlink() or ObjectDigest.from_file(path) != digest:
                    raise ValueError(f"Recovered registry record changed: {name}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    shutil.copyfile(cached, temporary)
                    temporary.chmod(0o644)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        configure(root, uri=uri, cache_root=cache_root)
