"""Explicit, approved migration of a recorded unsealed release.

This creates a NEW sealed rollback package. It never edits the old runtime or
live pointer and cannot accept unidentified code or change data/model inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.production_site_release import verify_production_site_release
from nba_lineup_model.season.sealed_code_package import materialize_sealed_code_package


def _digest(value: dict) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _inventory(root: Path) -> dict[str, str]:
    """All non-code bytes and links, including files outside artifact indexes."""
    excluded = {"code", "code-package.json", "runtime-manifest.json"}
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if base == root:
            dirs[:] = [name for name in dirs if name not in excluded]
            files = [name for name in files if name not in excluded]
        for name in dirs + files:
            path = base / name
            key = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[key] = "link:" + os.readlink(path)
            elif path.is_file():
                result[key] = file_sha256(path)
    return result


def _legacy_manifest(active: dict) -> tuple[Path, dict]:
    if not active or active.get("artifact_kind") != "release_activation":
        raise ValueError("Migration requires a recorded activation, not an inferred identity")
    root = Path(active["release_path"]).resolve(strict=True)
    manifest = _read(root / "runtime-manifest.json")
    identity = manifest["identity"]
    if identity.get("code_identity") or (root / "code-package.json").exists():
        raise ValueError("Migration is only for a legacy unsealed runtime")
    if _digest(identity) != manifest["runtime_sha256"]:
        raise ValueError("Legacy runtime checksum is invalid")
    if active["runtime_sha256"] != manifest["runtime_sha256"]:
        raise ValueError("Legacy runtime differs from recorded activation")
    if identity["code_revision"] != active["code"]["git_revision"]:
        raise ValueError("Legacy code revision differs from recorded activation")
    return root, manifest


def prepare_legacy_code_migration(
    *,
    active: dict,
    repository: Path,
    output_root: Path,
    approval_reference: str,
) -> dict:
    """Reproduce the recorded code, copy unchanged artifacts, and verify both."""
    if not approval_reference or not approval_reference.strip():
        raise ValueError("Legacy code migration requires explicit approval")
    source, original = _legacy_manifest(active)
    request = {"active_activation": active, "approval_reference": approval_reference}
    target = output_root / _digest(request)
    receipt_path = target / "migration.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        verify_legacy_code_migration(receipt, active)
        return receipt
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / (".migration.tmp-" + uuid4().hex)
    staging.mkdir()
    runtime = staging / "runtime"
    try:
        before = _inventory(source)
        shutil.copytree(
            source,
            runtime,
            symlinks=True,
            ignore=lambda directory, names: ["code"] if Path(directory) == source else [],
        )
        # A detached worktree is only an export source; the release owns its copy.
        with tempfile.TemporaryDirectory(prefix="gestalt-code-export-") as temporary:
            checkout = Path(temporary) / "checkout"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "worktree",
                    "add",
                    "--detach",
                    str(checkout),
                    active["code"]["git_revision"],
                ],
                check=True,
                capture_output=True,
            )
            try:
                package = materialize_sealed_code_package(
                    checkout,
                    runtime,
                    expected_revision=active["code"]["git_revision"],
                )
            finally:
                subprocess.run(
                    ["git", "-C", str(repository), "worktree", "remove", str(checkout)],
                    check=True,
                    capture_output=True,
                )
        # Old records predate sealing. Preserve every recorded identity field;
        # the new seal is independently verified by the package validator.
        expected_code = {"sealed": True, **active["code"]}
        if package["identity"] != expected_code:
            raise ValueError("Reconstructed code differs from recorded activation checksum")
        identity = {**original["identity"], "code_identity": package["identity"]}
        migrated = {**original, "identity": identity, "runtime_sha256": _digest(identity)}
        (runtime / "runtime-manifest.json").write_text(json.dumps(migrated, indent=2) + "\n")
        verify_production_site_release(runtime)
        if before != _inventory(source) or before != _inventory(runtime):
            raise ValueError("Non-code release content changed during migration")
        receipt = {
            "schema_version": 1,
            "artifact_kind": "legacy_code_migration",
            **request,
            "source_manifest_sha256": file_sha256(source / "runtime-manifest.json"),
            "unchanged_content_sha256": _digest(before),
            "sealed_runtime": str(target / "runtime"),
            "sealed_runtime_sha256": migrated["runtime_sha256"],
        }
        receipt["migration_sha256"] = _digest(receipt)
        (staging / "migration.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(staging, target)
        verify_legacy_code_migration(receipt, active)
        return receipt
    finally:
        if staging.exists():
            for directory, _dirs, _files in os.walk(staging, followlinks=False):
                Path(directory).chmod(0o755)
            shutil.rmtree(staging)


def verify_legacy_code_migration(receipt: dict, active: dict) -> Path:
    """Validate the explicit replacement rollback target without relaxing gates."""
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_kind") != "legacy_code_migration"
    ):
        raise ValueError("Invalid legacy code migration receipt")
    if not receipt.get("approval_reference", "").strip():
        raise ValueError("Legacy code migration requires explicit approval")
    identity = {k: v for k, v in receipt.items() if k != "migration_sha256"}
    if receipt.get("migration_sha256") != _digest(identity):
        raise ValueError("Legacy migration receipt checksum is invalid")
    if receipt["active_activation"] != active:
        raise ValueError("Legacy migration is bound to a different active generation")
    source, original = _legacy_manifest(active)
    if file_sha256(source / "runtime-manifest.json") != receipt["source_manifest_sha256"]:
        raise ValueError("Legacy runtime manifest changed since migration")
    runtime = Path(receipt["sealed_runtime"]).resolve(strict=True)
    migrated = verify_production_site_release(runtime)
    if migrated["runtime_sha256"] != receipt["sealed_runtime_sha256"]:
        raise ValueError("Sealed rollback identity changed")
    expected_code = {"sealed": True, **active["code"]}
    if migrated["identity"] != {**original["identity"], "code_identity": expected_code}:
        raise ValueError("Legacy migration altered more than code sealing")
    if migrated["approved_manifest"] != original["approved_manifest"]:
        raise ValueError("Legacy migration altered the approved artifacts")
    if _digest(_inventory(source)) != receipt["unchanged_content_sha256"]:
        raise ValueError("Legacy non-code content changed")
    if _inventory(source) != _inventory(runtime):
        raise ValueError("Migrated non-code content differs from original")
    return runtime
