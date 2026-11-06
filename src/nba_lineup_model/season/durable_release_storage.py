"""Durable storage for one verified immutable site release.

The canonical runtime manifest is preserved byte-for-byte. The mutable schedule
overlay symlink is deliberately excluded; the immutable schedule snapshot and
selected schedule already contained in the release are archived instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from nba_lineup_model.season.durable_artifact_storage import (
    DurableArtifactPublication,
    publish_verified_bundle,
)
from nba_lineup_model.season.production_site_release import (
    verify_production_site_release,
)
from nba_lineup_model.storage.artifact_store import (
    ArtifactClient,
    ArtifactReference,
    ObjectDigest,
    parse_legacy_artifact_reference,
    resolve_storage_manifest,
)
from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI

RUNTIME_MANIFEST_PATH = "runtime-manifest.json"
MUTABLE_SCHEDULE_LINK = "schedule"


def production_release_reference(manifest: dict[str, object]) -> ArtifactReference:
    identity = manifest["identity"]
    return ArtifactReference(
        artifact_kind="site_release",
        season=str(identity["season"]),
        game_date_pacific=date.fromisoformat(
            str(identity["evidence_through_date"])
        ),
        content_id=f"runtime_sha256={manifest['runtime_sha256']}",
    )


def required_release_dependencies(manifest: dict[str, object]) -> tuple[ArtifactReference, ...]:
    approved = manifest['approved_manifest']['identity']
    dependencies = [parse_legacy_artifact_reference(approved[key]['bundle_path'], artifact_kind=kind)
                    for key, kind in [('data', 'daily_data'), ('model', 'daily_model')]]
    initialized = manifest['identity'].get('forecast_initialization')
    reference = initialized['reference'] if initialized else manifest['identity'].get('reconciliation_reference')
    if reference:
        dependencies.append(ArtifactReference.from_dict(reference))
    return tuple(sorted(set(dependencies)))


def production_release_inventory(
    release: Path,
    manifest: dict[str, object],
) -> dict[str, ObjectDigest]:
    """Hash immutable release bytes while refusing unexpected links."""
    identity = manifest["identity"]
    expected_overlay = Path(str(identity["schedule_overlay_path"])).resolve(
        strict=True
    )
    inventory: dict[str, ObjectDigest] = {}
    for path in sorted(release.rglob("*")):
        relative = path.relative_to(release).as_posix()
        if path.is_symlink():
            if (
                relative != MUTABLE_SCHEDULE_LINK
                or path.resolve(strict=True) != expected_overlay
            ):
                raise ValueError(
                    f"Production release contains an unexpected symlink: {relative}"
                )
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"Production release contains a non-file: {relative}"
            )
        inventory[relative] = ObjectDigest.from_file(path)
    if RUNTIME_MANIFEST_PATH not in inventory:
        raise ValueError("Production release has no canonical runtime manifest")
    return inventory


def publish_production_release(
    release: Path | str,
    *,
    cache_root: Path,
    dependencies: Sequence[ArtifactReference] | None = None,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> DurableArtifactPublication:
    """Publish a verified release only after every selected input is durable."""
    root = Path(release).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Production release must be a regular directory")
    manifest = verify_production_site_release(root)
    reference = production_release_reference(manifest)
    required = required_release_dependencies(manifest)
    selected = tuple(sorted(set(dependencies if dependencies is not None else required)))
    if not set(required).issubset(selected):
        raise ValueError("Release storage dependencies omit its selected daily inputs")
    if len(selected) != len(dependencies or selected):
        raise ValueError("Release storage dependencies must be unique")
    for dependency in selected:
        resolve_storage_manifest(
            reference=dependency,
            cache_root=cache_root,
            client=client,
            lake_uri=lake_uri,
        )
    inventory = production_release_inventory(root, manifest)
    return publish_verified_bundle(
        bundle=root,
        reference=reference,
        dependencies=selected,
        cache_root=cache_root,
        client=client,
        lake_uri=lake_uri,
        inventory=inventory,
        manifest_path=RUNTIME_MANIFEST_PATH,
    )



def restore_production_release(*, reference: ArtifactReference, cache_root: Path,
                               client: ArtifactClient | None = None,
                               lake_uri: str = DEFAULT_LAKE_URI) -> Path:
    """Materialize a complete immutable release from its S3 inventory."""
    import json
    import os
    from nba_lineup_model.season.serving_recovery import _copy_verified
    from nba_lineup_model.storage.artifact_store import resolve_object
    from nba_lineup_model.storage.s3_lake import _new_s3_client

    s3 = client or _new_s3_client(1)
    stored = resolve_storage_manifest(reference=reference, cache_root=cache_root,
                                      client=s3, lake_uri=lake_uri)
    root = cache_root / "releases" / reference.content_id
    for name, entry in stored["objects"].items():
        digest = ObjectDigest(entry["sha256"], entry["size_bytes"])
        source = resolve_object(uri=entry["s3_uri"], digest=digest,
                                cache_root=cache_root / "objects", client=s3, lake_uri=lake_uri)
        _copy_verified(source, root / name, digest)
    runtime = json.loads((root / RUNTIME_MANIFEST_PATH).read_text())
    if not runtime["identity"].get("durable_serving_sha256"):
        raise ValueError("Release has no complete durable serving inventory")
    overlay = root / MUTABLE_SCHEDULE_LINK
    target = Path(runtime["identity"]["schedule_overlay_path"])
    if overlay.is_symlink():
        if os.readlink(overlay) != str(target):
            raise ValueError("Restored overlay differs from release")
    elif overlay.exists():
        raise ValueError("Restored overlay is not the declared live observation link")
    else:
        overlay.symlink_to(target, target_is_directory=True)
    if production_release_reference(verify_production_site_release(root)) != reference:
        raise ValueError("Restored site release identity differs")
    return root
