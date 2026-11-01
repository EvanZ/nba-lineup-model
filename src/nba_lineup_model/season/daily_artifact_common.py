"""Shared primitives for immutable, content-addressed daily artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LOGICAL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def require_logical_name(value: object, label: str) -> str:
    normalized = str(value).strip()
    if not _LOGICAL_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{label} must start with a letter and contain only lowercase letters, "
            "digits, and underscores"
        )
    return normalized


def copy_components(
    components: Mapping[str, Path | str],
    *,
    staging: Path,
) -> dict[str, dict[str, object]]:
    """Copy named files into a staging directory and return their inventory."""

    if not components:
        raise ValueError("At least one artifact component is required")
    files_dir = staging / "files"
    files_dir.mkdir(parents=True)
    inventory: dict[str, dict[str, object]] = {}
    for raw_name, raw_source in sorted(components.items()):
        name = require_logical_name(raw_name, "Component name")
        source = Path(raw_source)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact component is missing: {source}")
        suffix = "".join(source.suffixes)
        destination = files_dir / f"{name}{suffix}"
        shutil.copyfile(source, destination)
        inventory[name] = {
            "path": destination.relative_to(staging).as_posix(),
            "sha256": file_sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
    return inventory


def verify_components(bundle: Path, inventory: object) -> None:
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Artifact component inventory is missing")
    for raw_name, raw_entry in inventory.items():
        name = require_logical_name(raw_name, "Component name")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Artifact component inventory is invalid: {name}")
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Artifact component path is missing: {name}")
        component_path = (bundle / raw_path).resolve()
        try:
            component_path.relative_to(bundle.resolve())
        except ValueError as error:
            raise ValueError(f"Artifact component escapes its bundle: {name}") from error
        if not component_path.is_file():
            raise FileNotFoundError(f"Artifact component is missing: {name}")
        expected_sha = require_sha256(raw_entry.get("sha256"), f"{name} checksum")
        if file_sha256(component_path) != expected_sha:
            raise ValueError(f"Artifact component checksum changed: {name}")
        if component_path.stat().st_size != int(raw_entry.get("size_bytes", -1)):
            raise ValueError(f"Artifact component size changed: {name}")


def write_manifest(staging: Path, manifest: Mapping[str, Any]) -> None:
    (staging / "manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: Path | str) -> dict[str, Any]:
    payload = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Artifact manifest must be an object")
    return payload
