"""Immutable artifact transport; canonical producers still own artifact schemas.

Inventory digests come from validated canonical manifests. Manifest bytes are
published verbatim, last. This module does not select a model or rewrite legacy
references; callers must resolve those through an explicit identity inventory.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI, validate_lake_uri
from nba_lineup_model.storage.s3_lake import (
    S3Client,
    UploadSummary,
    _ensure_remote_object,
    _head_object,
    _join_s3_uri,
    _new_s3_client,
    _parse_s3_uri,
    _sha256_file,
    _validate_remote_object,
)

STORAGE_MANIFEST_FILENAME = "storage-manifest.json"
STORAGE_MANIFEST_SCHEMA_VERSION = 1


class ArtifactClient(S3Client):
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ObjectDigest:
    """Byte identity, distinct from a canonical artifact's semantic identity."""

    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Object checksum must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("Object size must be a nonnegative integer")

    @classmethod
    def from_file(cls, path: Path) -> ObjectDigest:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Object source must be a regular file: {path}")
        return cls(_sha256_file(path), path.stat().st_size)


@dataclass(frozen=True, order=True)
class ArtifactReference:
    artifact_kind: str
    season: str
    game_date_pacific: date
    content_id: str

    def __post_init__(self) -> None:
        artifact_prefix(
            artifact_kind=self.artifact_kind,
            season=self.season,
            game_date_pacific=self.game_date_pacific,
            content_id=self.content_id,
        )
        if not re.fullmatch(r"[a-z][a-z0-9_]*_sha256=[0-9a-f]{64}", self.content_id):
            raise ValueError("Artifact content ID must contain its existing SHA-256 identity")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ArtifactReference:
        if not isinstance(value, Mapping):
            raise ValueError("Artifact reference must be an object")
        try:
            game_date = date.fromisoformat(str(value["game_date_pacific"]))
            return cls(
                artifact_kind=str(value["artifact_kind"]),
                season=str(value["season"]),
                game_date_pacific=game_date,
                content_id=str(value["content_id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Artifact reference is invalid") from error

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "season": self.season,
            "game_date_pacific": self.game_date_pacific.isoformat(),
            "content_id": self.content_id,
        }

    def prefix_uri(self, *, lake_uri: str = DEFAULT_LAKE_URI) -> str:
        return artifact_prefix(
            artifact_kind=self.artifact_kind,
            season=self.season,
            game_date_pacific=self.game_date_pacific,
            content_id=self.content_id,
            lake_uri=lake_uri,
        )


@dataclass(frozen=True)
class StorageManifestPublication:
    uri: str
    digest: ObjectDigest
    uploaded: bool


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Invalid artifact relative path: {value}")
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"Invalid artifact relative path: {value}")
    return value


def artifact_prefix(
    *, artifact_kind: str, season: str, game_date_pacific: date,
    content_id: str, lake_uri: str = DEFAULT_LAKE_URI,
) -> str:
    """Locate an existing content identity without deriving a replacement identity."""
    for value in (artifact_kind, season, content_id):
        if "/" in _relative_path(value):
            raise ValueError("Artifact namespace fields must be single path components")
    return _join_s3_uri(
        lake_uri, "artifacts", artifact_kind, f"season={season}",
        f"game_date_pacific={game_date_pacific.isoformat()}", content_id,
    )


def storage_manifest_uri(
    reference: ArtifactReference, *, lake_uri: str = DEFAULT_LAKE_URI
) -> str:
    return _join_s3_uri(reference.prefix_uri(lake_uri=lake_uri), STORAGE_MANIFEST_FILENAME)


def build_storage_manifest(
    *,
    reference: ArtifactReference,
    files: Mapping[str, ObjectDigest],
    dependencies: Sequence[ArtifactReference] = (),
    manifest_path: str = "manifest.json",
    lake_uri: str = DEFAULT_LAKE_URI,
) -> dict[str, object]:
    """Describe transport bytes without changing the canonical artifact manifest."""
    prefix = reference.prefix_uri(lake_uri=lake_uri)
    objects: dict[str, dict[str, object]] = {}
    for name, digest in sorted(files.items()):
        _relative_path(name)
        if name == STORAGE_MANIFEST_FILENAME:
            raise ValueError("The storage manifest cannot list itself")
        if not isinstance(digest, ObjectDigest):
            raise ValueError(f"Missing exact object digest: {name}")
        objects[name] = {
            "s3_uri": _join_s3_uri(prefix, name),
            "sha256": digest.sha256,
            "size_bytes": digest.size_bytes,
        }
    manifest_path = _relative_path(manifest_path)
    if manifest_path not in objects:
        raise ValueError("Storage manifest must include the original artifact manifest")
    ordered_dependencies = sorted(dependencies)
    if len(set(ordered_dependencies)) != len(ordered_dependencies):
        raise ValueError("Storage manifest dependencies must be unique")
    payload: dict[str, object] = {
        "schema_version": STORAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "artifact_storage_manifest",
        "identity": {
            **reference.as_dict(),
            "artifact_uri": prefix,
        },
        **({"manifest_path": manifest_path} if manifest_path != "manifest.json" else {}),
        "objects": objects,
        "dependencies": [dependency.as_dict() for dependency in ordered_dependencies],
    }
    return verify_storage_manifest(payload, expected=reference, lake_uri=lake_uri)


def verify_storage_manifest(
    payload: Mapping[str, object],
    *,
    expected: ArtifactReference | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("Storage manifest must be an object")
    if payload.get("schema_version") != STORAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported artifact storage manifest schema")
    if payload.get("artifact_kind") != "artifact_storage_manifest":
        raise ValueError("Object is not an artifact storage manifest")
    raw_identity = payload.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("Storage manifest identity is missing")
    reference = ArtifactReference.from_dict(raw_identity)
    if expected is not None and reference != expected:
        raise ValueError("Storage manifest resolves a different artifact identity")
    prefix = reference.prefix_uri(lake_uri=lake_uri)
    if raw_identity.get("artifact_uri") != prefix:
        raise ValueError("Storage manifest artifact URI is not canonical")

    manifest_path = _relative_path(str(payload.get("manifest_path", "manifest.json")))
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, Mapping) or not raw_objects:
        raise ValueError("Storage manifest object inventory is missing")
    objects: dict[str, dict[str, object]] = {}
    for raw_name, raw_entry in sorted(raw_objects.items()):
        name = _relative_path(raw_name)
        if name == STORAGE_MANIFEST_FILENAME:
            raise ValueError("The storage manifest cannot list itself")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Storage object entry is invalid: {name}")
        digest = ObjectDigest(
            sha256=raw_entry.get("sha256"),
            size_bytes=raw_entry.get("size_bytes"),
        )
        uri = raw_entry.get("s3_uri")
        if uri != _join_s3_uri(prefix, name):
            raise ValueError(f"Storage object URI is not canonical: {name}")
        objects[name] = {
            "s3_uri": uri,
            "sha256": digest.sha256,
            "size_bytes": digest.size_bytes,
        }
    if manifest_path not in objects:
        raise ValueError("Storage manifest omits the original artifact manifest")

    raw_dependencies = payload.get("dependencies")
    if not isinstance(raw_dependencies, list):
        raise ValueError("Storage manifest dependencies must be a list")
    dependencies = [ArtifactReference.from_dict(value) for value in raw_dependencies]
    if dependencies != sorted(set(dependencies)):
        raise ValueError("Storage manifest dependencies must be sorted and unique")
    return {
        "schema_version": STORAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "artifact_storage_manifest",
        "identity": {**reference.as_dict(), "artifact_uri": prefix},
        **({"manifest_path": manifest_path} if manifest_path != "manifest.json" else {}),
        "objects": objects,
        "dependencies": [dependency.as_dict() for dependency in dependencies],
    }


def _storage_manifest_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def publish_storage_manifest(
    *,
    reference: ArtifactReference,
    files: Mapping[str, ObjectDigest],
    dependencies: Sequence[ArtifactReference] = (),
    manifest_path: str = "manifest.json",
    client: S3Client | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> StorageManifestPublication:
    """Publish artifact eligibility only after every immutable object is verified."""
    active_client = client if client is not None else _new_s3_client(1)
    payload = build_storage_manifest(
        reference=reference,
        files=files,
        dependencies=dependencies,
        manifest_path=manifest_path,
        lake_uri=lake_uri,
    )
    for name, entry in payload["objects"].items():
        digest = ObjectDigest(entry["sha256"], entry["size_bytes"])
        head = _head_object(entry["s3_uri"], client=active_client)
        if head is None:
            raise RuntimeError(f"Artifact object is missing before manifest publication: {name}")
        _validate_remote_object(
            uri=entry["s3_uri"],
            head=head,
            size_bytes=digest.size_bytes,
            sha256=digest.sha256,
        )
    encoded = _storage_manifest_bytes(payload)
    digest = ObjectDigest(hashlib.sha256(encoded).hexdigest(), len(encoded))
    uri = storage_manifest_uri(reference, lake_uri=lake_uri)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="artifact-storage-manifest-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        uploaded = _ensure_remote_object(
            source=temporary,
            uri=uri,
            size_bytes=digest.size_bytes,
            sha256=digest.sha256,
            client=active_client,
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return StorageManifestPublication(uri=uri, digest=digest, uploaded=uploaded)


def resolve_storage_manifest(
    *,
    reference: ArtifactReference,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> dict[str, object]:
    active_client = client if client is not None else _new_s3_client(1)
    uri = storage_manifest_uri(reference, lake_uri=lake_uri)
    head = _head_object(uri, client=active_client)
    if head is None:
        raise FileNotFoundError(f"Artifact storage manifest is missing: {uri}")
    metadata = head.get("Metadata")
    sha256 = metadata.get("sha256") if isinstance(metadata, Mapping) else None
    digest = ObjectDigest(sha256=sha256, size_bytes=head.get("ContentLength"))
    _validate_remote_object(
        uri=uri,
        head=head,
        size_bytes=digest.size_bytes,
        sha256=digest.sha256,
    )
    path = resolve_object(
        uri=uri,
        digest=digest,
        cache_root=cache_root / "storage-manifests",
        client=active_client,
        lake_uri=lake_uri,
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Artifact storage manifest is unreadable: {uri}") from error
    return verify_storage_manifest(payload, expected=reference, lake_uri=lake_uri)


def resolve_artifact_file(
    *,
    reference: ArtifactReference,
    relative_path: str,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> Path:
    name = _relative_path(relative_path)
    manifest = resolve_storage_manifest(
        reference=reference, cache_root=cache_root, client=client, lake_uri=lake_uri
    )
    entry = manifest["objects"].get(name)
    if not isinstance(entry, Mapping):
        raise FileNotFoundError(f"Artifact storage manifest does not contain: {name}")
    return resolve_object(
        uri=entry["s3_uri"],
        digest=ObjectDigest(entry["sha256"], entry["size_bytes"]),
        cache_root=cache_root / "objects",
        client=client,
        lake_uri=lake_uri,
    )


def parse_legacy_artifact_reference(
    value: Path | str, *, artifact_kind: str | None = None
) -> ArtifactReference:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Legacy artifact reference must be an absolute path")
    if len(path.parts) < 4:
        raise ValueError("Legacy artifact reference is incomplete")
    content_id = path.name
    date_part = path.parent.name
    season_part = path.parent.parent.name
    if not date_part.startswith("game_date_pacific=") or not season_part.startswith("season="):
        raise ValueError("Legacy artifact reference does not contain a season and Pacific date")
    inferred = {
        "data_sha256": "daily_data",
        "model_sha256": "daily_model",
    }.get(content_id.partition("=")[0])
    kind = artifact_kind or inferred
    if kind is None:
        raise ValueError("Artifact kind is required for this legacy content ID")
    return ArtifactReference(
        artifact_kind=kind,
        season=season_part.removeprefix("season="),
        game_date_pacific=date.fromisoformat(
            date_part.removeprefix("game_date_pacific=")
        ),
        content_id=content_id,
    )


def _verify_file(path: Path, digest: ObjectDigest) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Artifact object is not a regular file: {path}")
    if path.stat().st_size != digest.size_bytes or _sha256_file(path) != digest.sha256:
        raise RuntimeError(f"Artifact object checksum/size mismatch: {path}")


def publish_artifact(
    *, source: Path, destination_uri: str, files: Mapping[str, ObjectDigest],
    manifest_path: str = "manifest.json", client: S3Client | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> UploadSummary:
    """Create immutable objects, publishing the original manifest only after verification.

    The caller validates the canonical artifact and supplies its complete inventory,
    including the original manifest and any reports. Dependency artifacts must be
    published before calling this function. No inventory schema is written here.
    """
    validate_lake_uri(destination_uri, lake_uri=lake_uri)
    manifest_path = _relative_path(manifest_path)
    inventory = dict(files)
    if manifest_path not in inventory:
        raise ValueError("Publication inventory must include the original manifest")
    root = source.resolve(strict=True)
    paths: dict[str, Path] = {}
    for name, digest in inventory.items():
        _relative_path(name)
        if not isinstance(digest, ObjectDigest):
            raise ValueError(f"Missing exact object digest: {name}")
        path = root / name
        parents = [part for part in [path, *path.parents] if part != root and root in part.parents]
        if any(part.is_symlink() for part in parents):
            raise ValueError(f"Artifact component contains a symlink: {name}")
        path.resolve().relative_to(root)
        _verify_file(path, digest)
        paths[name] = path
    s3 = client if client is not None else _new_s3_client(1)
    manifest_uri = _join_s3_uri(destination_uri, manifest_path)
    existing = _head_object(manifest_uri, client=s3)
    if existing is not None:
        digest = inventory[manifest_path]
        _validate_remote_object(
            uri=manifest_uri, head=existing, size_bytes=digest.size_bytes, sha256=digest.sha256,
        )
    uploaded = 0
    for name in [*sorted(set(inventory) - {manifest_path}), manifest_path]:
        digest = inventory[name]
        uri = _join_s3_uri(destination_uri, name)
        uploaded += _ensure_remote_object(
            source=paths[name], uri=uri, size_bytes=digest.size_bytes,
            sha256=digest.sha256, client=s3,
        )
        # A completed Put is verified before the manifest may be published.
        head = _head_object(uri, client=s3)
        if head is None:
            raise RuntimeError(f"Published artifact object is missing: {uri}")
        _validate_remote_object(
            uri=uri, head=head, size_bytes=digest.size_bytes, sha256=digest.sha256,
        )
    return UploadSummary(
        planned=len(inventory), uploaded=uploaded, existing=len(inventory) - uploaded,
        verified=len(inventory), total_bytes=sum(d.size_bytes for d in inventory.values()),
        executed=True,
    )


def resolve_object(
    *, uri: str, digest: ObjectDigest, cache_root: Path,
    client: ArtifactClient | None = None, lake_uri: str = DEFAULT_LAKE_URI,
) -> Path:
    """Read one exact object through a checksum-verified, process-safe EC2 cache.

    A cache hit hashes the local bytes and requires no S3 request. Corrupt cached
    bytes fail visibly. A failed download never becomes a usable cache entry.
    """
    validate_lake_uri(uri, lake_uri=lake_uri)
    # Include the location so a different missing URI cannot borrow an unrelated hit.
    location = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    root = cache_root.resolve()
    directory = root / location[:2] / location / digest.sha256
    for part in (root, directory.parent.parent, directory.parent, directory):
        if part.is_symlink():
            raise RuntimeError(f"Artifact cache directory is a symlink: {part}")
        part.mkdir(parents=True, exist_ok=True)
    target = directory / "object"
    lock_path = directory / "lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists() or target.is_symlink():
            _verify_file(target, digest)
            return target
        s3 = client if client is not None else _new_s3_client(1)
        bucket, key = _parse_s3_uri(uri)
        response = s3.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        body = response["Body"]
        temporary: Path | None = None
        try:
            if response.get("ContentLength") != digest.size_bytes:
                raise RuntimeError(f"Artifact download size mismatch: {uri}")
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".download-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                checksum = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: body.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > digest.size_bytes:
                        raise RuntimeError(
                            f"Artifact download exceeded expected size: {uri}"
                        )
                    checksum.update(chunk)
                    handle.write(chunk)
                if size != digest.size_bytes or checksum.hexdigest() != digest.sha256:
                    raise RuntimeError(f"Artifact download checksum/size mismatch: {uri}")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            body.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target
