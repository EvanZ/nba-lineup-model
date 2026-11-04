"""Materialize and verify a release-owned sealed code package.

Release runtimes must never point at an editable Git checkout. A package is a
byte-for-byte copy of the tracked files at one clean Git revision plus a
manifest binding every copied file, the dependency lock, and the revision.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.release_artifact_contracts import SealedCodeIdentity

SCHEMA_VERSION = 1
ARTIFACT_KIND = "sealed_code_package"
MANIFEST_FILENAME = "code-package.json"
RUNTIME_PATHS = ("src", "deploy", "scripts", "pyproject.toml", "uv.lock")


def _git_output(code_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(code_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _tracked_paths(code_root: Path) -> tuple[str, ...]:
    raw_paths = subprocess.run(
        ["git", "-C", str(code_root), "ls-tree", "-rz", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    paths = tuple(raw.decode("utf-8") for raw in raw_paths if raw)
    if not paths:
        raise ValueError("Release code checkout has no tracked files")
    return paths


def inspect_source_code(code_root: Path | str) -> dict[str, Any]:
    """Return the sealed identity and file inventory for a clean checkout."""

    root = Path(code_root).resolve()
    if _git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Release code checkout has modified tracked files")
    untracked_runtime = _git_output(
        root, "ls-files", "--others", "--exclude-standard", "--", *RUNTIME_PATHS
    )
    if untracked_runtime:
        raise ValueError("Release code checkout has untracked runtime files")
    revision = _git_output(root, "rev-parse", "HEAD").strip()
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for relative in _tracked_paths(root):
        path = root / relative
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Release code must not contain symlinks: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Tracked release file is missing: {relative}")
        checksum = file_sha256(path)
        files[relative] = checksum
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(checksum))
    lock_path = root / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("Release code checkout has no uv.lock")
    identity = SealedCodeIdentity(
        package_sha256=digest.hexdigest(),
        dependency_lock_sha256=file_sha256(lock_path),
        git_revision=revision,
        sealed=True,
    ).model_dump(mode="json")
    return {"identity": identity, "files": files}


def materialize_sealed_code_package(
    code_root: Path | str,
    release_root: Path | str,
    *,
    expected_revision: str,
) -> dict[str, Any]:
    """Copy tracked source into a release; never link to the source checkout."""

    source = Path(code_root).resolve()
    release = Path(release_root)
    package = inspect_source_code(source)
    if package["identity"]["git_revision"] != expected_revision:
        raise ValueError("Code checkout differs from the approved dev release")
    destination = release / "code"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Release code destination already exists")
    destination.mkdir()
    # Stream the committed tree, never the editable working files.
    with subprocess.Popen(
        ["git", "-C", str(source), "archive", "--format=tar", expected_revision],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    if not member.isdir() and not member.isfile():
                        raise ValueError("Release code archive contains a non-regular entry")
                    archive.extract(member, destination, filter="data")
            if process.wait() != 0:
                raise ValueError("Git could not export the selected code revision")
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        **package,
    }
    (release / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verified = verify_sealed_code_package(release)
    # Keep Python bytecode and other generated files outside the sealed tree.
    # Release owners can still deliberately alter permissions; digest checks
    # remain mandatory on every subsequent release validation.
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        elif path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)
    return verified


def verify_sealed_code_package(release_root: Path | str) -> dict[str, Any]:
    """Fail closed if a release-owned code package differs from its manifest."""

    release = Path(release_root).resolve()
    code_root = release / "code"
    if code_root.is_symlink():
        # A publisher generation references an already sealed runtime. It must
        # retain that exact package and manifest, never resolve to a checkout.
        receipt_path = release / "activation.json"
        if not receipt_path.is_file():
            raise ValueError("Release code package must not be a symlink")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        generation = receipt.get("generation")
        if (
            receipt.get("artifact_kind") != "release_activation"
            or not isinstance(generation, int)
            or generation < 1
            or release.name != f"generation={generation:020d}"
        ):
            raise ValueError("Code reference requires a valid publisher generation")
        target = Path(receipt["release_path"]).resolve()
        if (
            target == release
            or (target / "code").is_symlink()
            or code_root.resolve() != target / "code"
            or (release / MANIFEST_FILENAME).resolve() != target / MANIFEST_FILENAME
            or (release / "runtime-manifest.json").resolve() != target / "runtime-manifest.json"
        ):
            raise ValueError("Publisher code reference differs from its sealed runtime")
        manifest = verify_sealed_code_package(target)
        if receipt.get("code") != manifest["identity"]:
            raise ValueError("Publisher code digest differs from its sealed runtime")
        return manifest
    if not code_root.is_dir():
        raise FileNotFoundError("Release code package is missing")
    if (release / MANIFEST_FILENAME).is_symlink():
        raise ValueError("Release must own its sealed code manifest")
    manifest = json.loads((release / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Sealed code manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported sealed code package schema")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Artifact is not a sealed code package")
    identity = SealedCodeIdentity.model_validate(manifest.get("identity")).model_dump(mode="json")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Sealed code package has no file inventory")
    actual_paths = sorted(
        path.relative_to(code_root).as_posix() for path in code_root.rglob("*") if path.is_file()
    )
    if any(path.is_symlink() for path in code_root.rglob("*")):
        raise ValueError("Release code package must not contain symlinks")
    if actual_paths != sorted(expected_files):
        raise ValueError("Sealed code package file inventory changed")
    digest = hashlib.sha256()
    for relative in actual_paths:
        checksum = file_sha256(code_root / relative)
        if checksum != expected_files[relative]:
            raise ValueError(f"Sealed code file changed after staging: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(checksum))
    if digest.hexdigest() != identity["package_sha256"]:
        raise ValueError("Sealed code package checksum is invalid")
    if file_sha256(code_root / "uv.lock") != identity["dependency_lock_sha256"]:
        raise ValueError("Sealed dependency lock changed after staging")
    return manifest
