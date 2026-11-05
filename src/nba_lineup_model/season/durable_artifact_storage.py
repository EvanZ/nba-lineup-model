"""Publish selected canonical daily artifacts to durable storage without changing them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.daily_model_replay import _read_data_chain, _read_model_chain
from nba_lineup_model.storage.artifact_store import (
    ArtifactClient,
    ArtifactReference,
    ObjectDigest,
    build_storage_manifest,
    publish_artifact,
    publish_storage_manifest,
    resolve_storage_manifest,
    storage_manifest_uri,
)
from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
from nba_lineup_model.storage.s3_lake import UploadSummary


@dataclass(frozen=True)
class DurableArtifactPublication:
    reference: ArtifactReference
    objects: UploadSummary
    storage_manifest_uri: str
    storage_manifest_sha256: str
    storage_manifest_uploaded: bool


def artifact_inventory(
    bundle: Path | str, *, manifest_path: str = "manifest.json"
) -> dict[str, ObjectDigest]:
    """Hash every original artifact byte and reject links or non-files."""
    root = Path(bundle).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Artifact bundle must be a regular directory: {root}")
    inventory: dict[str, ObjectDigest] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Artifact bundle contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Artifact bundle contains a non-file: {relative}")
        inventory[relative] = ObjectDigest.from_file(path)
    if manifest_path not in inventory:
        raise ValueError("Artifact bundle has no canonical manifest")
    return inventory


def daily_data_reference(bundle: Path | str) -> ArtifactReference:
    manifest = verify_daily_data_artifact(bundle)
    identity = manifest["identity"]
    return ArtifactReference(
        artifact_kind="daily_data",
        season=identity["season"],
        game_date_pacific=date.fromisoformat(identity["game_date_pacific"]),
        content_id=f"data_sha256={manifest['bundle_sha256']}",
    )


def daily_model_reference(bundle: Path | str) -> ArtifactReference:
    manifest = verify_daily_model_artifact(bundle)
    identity = manifest["identity"]
    return ArtifactReference(
        artifact_kind="daily_model",
        season=identity["season"],
        game_date_pacific=date.fromisoformat(identity["game_date_pacific"]),
        content_id=f"model_sha256={manifest['bundle_sha256']}",
    )


def daily_model_dependencies(bundle: Path | str) -> tuple[ArtifactReference, ...]:
    manifest = verify_daily_model_artifact(bundle)
    identity = manifest["identity"]
    game_date = date.fromisoformat(identity["game_date_pacific"])
    dependencies = [
        ArtifactReference(
            artifact_kind="daily_data",
            season=identity["season"],
            game_date_pacific=game_date,
            content_id=f"data_sha256={identity['input_data_sha256']}",
        )
    ]
    parent_sha = identity["parent_model_sha256"]
    if parent_sha is not None:
        dependencies.append(
            ArtifactReference(
                artifact_kind="daily_model",
                season=identity["season"],
                game_date_pacific=date.fromisoformat(
                    identity["parent_game_date_pacific"]
                ),
                content_id=f"model_sha256={parent_sha}",
            )
        )
    return tuple(sorted(dependencies))


def _manifest_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def publish_verified_bundle(
    *,
    bundle: Path,
    reference: ArtifactReference,
    dependencies: tuple[ArtifactReference, ...],
    cache_root: Path,
    client: ArtifactClient | None,
    lake_uri: str,
    inventory: dict[str, ObjectDigest] | None = None,
    manifest_path: str = "manifest.json",
) -> DurableArtifactPublication:
    inventory = inventory or artifact_inventory(bundle, manifest_path=manifest_path)
    expected = build_storage_manifest(
        reference=reference,
        files=inventory,
        dependencies=dependencies,
        lake_uri=lake_uri,
        manifest_path=manifest_path,
    )
    encoded = _manifest_bytes(expected)
    manifest_sha = hashlib.sha256(encoded).hexdigest()
    manifest_uri = storage_manifest_uri(reference, lake_uri=lake_uri)
    try:
        existing = resolve_storage_manifest(
            reference=reference,
            cache_root=cache_root,
            client=client,
            lake_uri=lake_uri,
        )
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != expected:
            raise RuntimeError(
                "Durable storage already contains a different object inventory for "
                f"{reference.content_id}"
            )
        return DurableArtifactPublication(
            reference=reference,
            objects=UploadSummary(
                planned=len(inventory),
                uploaded=0,
                existing=len(inventory),
                verified=len(inventory),
                total_bytes=sum(item.size_bytes for item in inventory.values()),
                executed=True,
            ),
            storage_manifest_uri=manifest_uri,
            storage_manifest_sha256=manifest_sha,
            storage_manifest_uploaded=False,
        )

    object_summary = publish_artifact(
        source=bundle,
        destination_uri=reference.prefix_uri(lake_uri=lake_uri),
        files=inventory,
        manifest_path=manifest_path,
        client=client,
        lake_uri=lake_uri,
    )
    published_manifest = publish_storage_manifest(
        reference=reference,
        files=inventory,
        dependencies=dependencies,
        manifest_path=manifest_path,
        client=client,
        lake_uri=lake_uri,
    )
    if published_manifest.digest.sha256 != manifest_sha:
        raise RuntimeError("Published storage manifest checksum differs from its inventory")
    return DurableArtifactPublication(
        reference=reference,
        objects=object_summary,
        storage_manifest_uri=published_manifest.uri,
        storage_manifest_sha256=published_manifest.digest.sha256,
        storage_manifest_uploaded=published_manifest.uploaded,
    )


def publish_daily_data_bundle(
    bundle: Path | str,
    *,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> DurableArtifactPublication:
    verified = Path(bundle).resolve(strict=True)
    reference = daily_data_reference(verified)
    return publish_verified_bundle(
        bundle=verified,
        reference=reference,
        dependencies=(),
        cache_root=cache_root,
        client=client,
        lake_uri=lake_uri,
    )


def publish_daily_model_bundle(
    bundle: Path | str,
    *,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> DurableArtifactPublication:
    verified = Path(bundle).resolve(strict=True)
    dependencies = daily_model_dependencies(verified)
    for dependency in dependencies:
        resolve_storage_manifest(
            reference=dependency,
            cache_root=cache_root,
            client=client,
            lake_uri=lake_uri,
        )
    return publish_verified_bundle(
        bundle=verified,
        reference=daily_model_reference(verified),
        dependencies=dependencies,
        cache_root=cache_root,
        client=client,
        lake_uri=lake_uri,
    )


def publish_selected_data_chain(
    root: Path | str,
    *,
    through_date: date,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> list[DurableArtifactPublication]:
    chain = [entry for entry in _read_data_chain(Path(root)) if entry[0] <= through_date]
    if not chain or chain[-1][0] != through_date:
        raise ValueError("Selected daily data chain does not reach the requested date")
    return [
        publish_daily_data_bundle(
            bundle, cache_root=cache_root, client=client, lake_uri=lake_uri
        )
        for _, bundle, _ in chain
    ]


def publish_selected_model_chain(
    root: Path | str,
    *,
    through_date: date,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> list[DurableArtifactPublication]:
    chain = [entry for entry in _read_model_chain(Path(root)) if entry[0] <= through_date]
    if not chain or chain[-1][0] != through_date:
        raise ValueError("Selected daily model chain does not reach the requested date")
    return [
        publish_daily_model_bundle(
            bundle, cache_root=cache_root, client=client, lake_uri=lake_uri
        )
        for _, bundle, _ in chain
    ]
