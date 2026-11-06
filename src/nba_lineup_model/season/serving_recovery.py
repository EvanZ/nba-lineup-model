"""Pinned transport references and restoration for complete serving releases.

Original manifests are never rewritten. This inventory is a new release-owned
component; legacy paths remain provenance and resolve to exact verified bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from nba_lineup_model.season.daily_artifact_common import canonical_json
from nba_lineup_model.storage.artifact_store import (
    ArtifactReference, ObjectDigest, parse_legacy_artifact_reference,
    resolve_object, resolve_storage_manifest,
)
from nba_lineup_model.storage.s3_lake import _new_s3_client

FILENAME = "durable-serving.json"
_PREPARED: set[tuple[str, str]] = set()
_REGISTRIES: set[tuple[str, str]] = set()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(value: dict) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe recovery inventory path")
    return path.as_posix()


def build_inventory(release: Path, *, lake_uri: str, cache_root: Path, client=None) -> dict:
    """Verify existing history in S3 and archive only missing registry records."""
    from nba_lineup_model.season.durable_forecast_registry import publish_registry, registry_uri
    from nba_lineup_model.season.forecast_publication import publisher_lock
    from nba_lineup_model.season.source_policy import require_clean_artifact_sources

    s3 = client or _new_s3_client(1)
    bindings = _read(release / "serving-bindings.json")
    historical_root = Path(bindings["historical_runtime"])
    # Existing legacy releases have a stable release directory and S3 namespace.
    if not historical_root.name.startswith("gestalt-web-"):
        raise ValueError("Historical release has no recognized durable release identity")
    historical_uri = lake_uri.rsplit("/", 1)[0] + "/nba-gestalt/releases/" + historical_root.name
    from nba_lineup_model.season.source_policy import audit_artifact_sources
    failures = audit_artifact_sources(historical_root, files=list(bindings["historical_files"]))
    if failures:
        raise ValueError("Forbidden historical source artifacts: " + ", ".join(failures[:20]))
    history = {}
    for name, checksum in bindings["historical_files"].items():
        name = _relative(name)
        source = historical_root / name
        digest = ObjectDigest.from_file(source)
        if digest.sha256 != checksum:
            raise ValueError(f"Historical serving input changed: {name}")
        uri = historical_uri + "/" + name
        # Never upload historical data: absence or mismatch must fail this stage.
        restored = resolve_object(uri=uri, digest=digest, cache_root=cache_root / "objects",
                                  client=s3, lake_uri=historical_uri)
        if ObjectDigest.from_file(restored) != digest:
            raise ValueError(f"Historical S3 input differs: {name}")
        history[name] = {"s3_uri": uri, "sha256": checksum, "size_bytes": digest.size_bytes}
    index = _read(release / "selected-schedule/forecast-selections.json")
    registry = None
    if index.get("forecast_registry_root"):
        root = Path(index["forecast_registry_root"])
        uri = registry_uri(root, lake_uri)
        from nba_lineup_model.season.durable_forecast_registry import storage_config, restore_registry
        if storage_config(root) is not None:
            restore_registry(root, uri=uri, cache_root=cache_root / "objects", required={}, client=s3)
        with publisher_lock(root):
            entries = publish_registry(root, uri=uri, cache_root=cache_root / "objects", client=s3)
        registry = {"source_root": str(root), "uri": uri, "records": entries}
    runtime = _read(release / "runtime-manifest.json")
    approved = runtime["approved_manifest"]["identity"]
    selected = {}
    for label, kind in (("data", "daily_data"), ("model", "daily_model")):
        ref = parse_legacy_artifact_reference(approved[label]["bundle_path"], artifact_kind=kind)
        selected[label] = {"source_path": approved[label]["bundle_path"], "reference": ref.as_dict()}
    model_root = Path(approved["model"]["bundle_path"]).parents[2]
    pointer = (model_root / "latest.json").read_text()
    if json.loads(pointer)["bundle_sha256"] != approved["model"]["bundle_sha256"]:
        raise ValueError("Historical model pointer is not the accepted release tip")
    value = {"schema_version": 1, "lake_uri": lake_uri,
             "historical": {"source_root": str(historical_root), "uri": historical_uri, "files": history},
             "registry": registry, "daily": selected, "model_pointer": pointer}
    # Validate the exact artifact scope, including historical parquet fields.
    require_clean_artifact_sources(release / "selected-schedule")
    return value


def inventory(release: Path) -> dict | None:
    runtime = _read(release / "runtime-manifest.json")
    checksum = runtime["identity"].get("durable_serving_sha256")
    if checksum is None:
        if (release / FILENAME).exists():
            raise ValueError("Unpinned durable serving inventory")
        return None
    value = _read(release / FILENAME)
    if value.get("schema_version") != 1 or _sha(value) != checksum:
        raise ValueError("Durable serving inventory changed")
    bindings = _read(release / "serving-bindings.json")
    history = value["historical"]
    if history["source_root"] != bindings["historical_runtime"] or {
        name: entry["sha256"] for name, entry in history["files"].items()
    } != bindings["historical_files"]:
        raise ValueError("Durable history differs from serving bindings")
    for label in ("data", "model"):
        expected = runtime["approved_manifest"]["identity"][label]
        ref = ArtifactReference.from_dict(value["daily"][label]["reference"])
        if (value["daily"][label]["source_path"] != expected["bundle_path"]
                or ref.content_id.split("=", 1)[1] != expected["bundle_sha256"]):
            raise ValueError("Durable daily selection differs from approved inputs")
    return value


def cache_root(release: Path) -> Path:
    override = os.environ.get("GESTALT_ARTIFACT_CACHE_ROOT")
    if override:
        return Path(override).absolute()
    # Read only the known setting, never execute the environment file.
    for line in (release / "site-release.env").read_text().splitlines():
        if line.startswith("GESTALT_ARTIFACT_CACHE_ROOT="):
            return Path(line.split("=", 1)[1]).absolute()
    raise ValueError("Durable serving release has no configured EC2 cache")


def recovered_root(release: Path) -> Path:
    runtime = _read(release / "runtime-manifest.json")
    return cache_root(release) / "serving" / runtime["runtime_sha256"]


def _copy_verified(source: Path, target: Path, digest: ObjectDigest) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or ObjectDigest.from_file(target) != digest:
            raise ValueError(f"Recovered file changed: {target}")
        return
    if any(parent.is_symlink() for parent in target.parents):
        raise ValueError("Recovered file has a symlink parent")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.chmod(0o644)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def restore_inputs(release: Path, *, client=None) -> Path | None:
    value = inventory(release)
    if value is None:
        return None
    root = recovered_root(release)
    key = (str(release.resolve()), str(root))
    if key in _PREPARED:
        return root
    s3 = client or _new_s3_client(1)
    cache = cache_root(release)
    history = value["historical"]
    for name, entry in history["files"].items():
        name = _relative(name)
        if entry["s3_uri"] != history["uri"] + "/" + name:
            raise ValueError("Historical object URI differs from pinned release")
        digest = ObjectDigest(entry["sha256"], entry["size_bytes"])
        source = resolve_object(uri=entry["s3_uri"], digest=digest, cache_root=cache / "objects",
                                client=s3, lake_uri=history["uri"])
        _copy_verified(source, root / "historical" / name, digest)
    pending = [ArtifactReference.from_dict(v["reference"]) for v in value["daily"].values()]
    seen = set()
    while pending:
        ref = pending.pop()
        if ref in seen:
            continue
        seen.add(ref)
        manifest = resolve_storage_manifest(reference=ref, cache_root=cache, client=s3,
                                            lake_uri=value["lake_uri"])
        destination = root / ref.artifact_kind / f"season={ref.season}" / f"game_date_pacific={ref.game_date_pacific}" / ref.content_id
        for name, entry in manifest["objects"].items():
            digest = ObjectDigest(entry["sha256"], entry["size_bytes"])
            source = resolve_object(uri=entry["s3_uri"], digest=digest, cache_root=cache / "objects",
                                    client=s3, lake_uri=value["lake_uri"])
            _copy_verified(source, destination / _relative(name), digest)
        pending.extend(ArtifactReference.from_dict(d) for d in manifest["dependencies"])
    # Preserve the accepted pointer's exact bytes; it is cache materialization,
    # never an edit to the accepted source collection or model chain.
    pointer = root / "daily_model/latest.json"
    data = value["model_pointer"].encode()
    if pointer.exists() and pointer.read_bytes() != data:
        raise ValueError("Recovered model pointer changed")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    if not pointer.exists():
        pointer.write_bytes(data)
        pointer.chmod(0o644)
    if value["registry"]:
        from nba_lineup_model.season.durable_forecast_registry import restore_registry
        registry = value["registry"]
        destination = cache / "forecast-registry" / registry["uri"].rsplit("/", 1)[1]
        restore_registry(destination, uri=registry["uri"], cache_root=cache / "objects",
                         required=registry["records"], client=s3)
    _PREPARED.add(key)
    return root


def historical_path(release: Path) -> Path:
    root = restore_inputs(release)
    return root / "historical" if root else Path(_read(release / "serving-bindings.json")["historical_runtime"])


def daily_path(path: Path, *, release: Path | None = None) -> Path:
    if release is None:
        selected = os.environ.get("GESTALT_ACTIVE_RELEASE_ROOT")
        if not selected:
            return path
        release = Path(selected)
    value = inventory(release)
    if value is None:
        return path
    root = restore_inputs(release)
    for label, item in value["daily"].items():
        if str(path) == item["source_path"]:
            ref = ArtifactReference.from_dict(item["reference"])
            return root / ref.artifact_kind / f"season={ref.season}" / f"game_date_pacific={ref.game_date_pacific}" / ref.content_id
    raise ValueError("Daily artifact is not selected by the durable release")


def registry_path(path: Path, *, release: Path | None = None) -> Path:
    selected = os.environ.get("GESTALT_ACTIVE_RELEASE_ROOT")
    if release is None and selected:
        release = Path(selected)
    if release is not None:
        value = inventory(release)
        if value is not None and value["registry"]:
            registry = value["registry"]
            destination = cache_root(release) / "forecast-registry" / registry["uri"].rsplit("/", 1)[1]
            if path == destination:
                return destination
            if str(path) != registry["source_root"]:
                raise ValueError("Forecast registry differs from durable release")
            restore_inputs(release)
            return destination
    from nba_lineup_model.season.durable_forecast_registry import storage_config, restore_registry
    config = storage_config(path)
    if config is not None:
        objects = Path(config["cache_root"])
        destination = objects.parent / "forecast-registry" / config["uri"].rsplit("/", 1)[1]
        key = (str(destination), config["uri"])
        if destination != path and key not in _REGISTRIES:
            restore_registry(destination, uri=config["uri"], cache_root=objects, required={})
            _REGISTRIES.add(key)
        return destination
    return path


def attach_inventory(release: Path, *, lake_uri: str, cache: Path) -> dict:
    """Called only on a new staging directory by the canonical producer."""
    value = build_inventory(release, lake_uri=lake_uri, cache_root=cache)
    (release / FILENAME).write_bytes(canonical_json(value) + b"\n")
    runtime = _read(release / "runtime-manifest.json")
    runtime["identity"]["durable_serving_sha256"] = _sha(value)
    runtime["runtime_sha256"] = _sha(runtime["identity"])
    (release / "runtime-manifest.json").write_bytes(canonical_json(runtime) + b"\n")
    env = release / "site-release.env"
    lines = [line for line in env.read_text().splitlines()
             if not line.startswith("GESTALT_ARTIFACT_CACHE_ROOT=")]
    env.write_text("\n".join([*lines, f"GESTALT_ARTIFACT_CACHE_ROOT={cache.absolute()}"]) + "\n")
    return runtime
