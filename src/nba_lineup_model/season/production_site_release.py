"""Stage and atomically activate one reviewed GESTALT site release.

The development release is the approval record.  Production staging copies its
verified serving artifacts into an immutable runtime directory, binds the exact
code and frontend, and leaves the mutable Schedule watcher in a separate
overlay.  Only the ``live-release`` symlink is switched during activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from nba_lineup_model.season.current_serving import (
    resolve_latest_bundle,
    verify_serving_bundle,
)
from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.dev_site_release import (
    _tree_sha256,
    resolve_latest_dev_site_release,
    verify_dev_site_release,
)
from nba_lineup_model.web_api.current_season_lineups import (
    verify_current_season_lineup_bundle,
)

SCHEMA_VERSION = 1
ARTIFACT_KIND = "production_site_release"
MANIFEST_FILENAME = "runtime-manifest.json"


def stage_production_site_release(
    *,
    dev_site_release_root: Path | str,
    serving_root: Path | str,
    lineups_root: Path | str,
    schedule_snapshot_root: Path | str,
    schedule_overlay_path: Path | str,
    catalog_path: Path | str,
    preseason_rankings_path: Path | str,
    preseason_profiles_path: Path | str,
    code_root: Path | str,
    frontend_root: Path | str,
    output_root: Path | str,
    live_pointer: Path | str,
) -> Path:
    """Copy a reviewed dev release into an immutable production candidate."""

    dev_release_root = Path(dev_site_release_root).resolve()
    dev_release = resolve_latest_dev_site_release(dev_release_root)
    dev_manifest = verify_dev_site_release(dev_release)
    approved = dev_manifest["identity"]
    approved_sha = str(dev_manifest["release_sha256"])

    serving_source = Path(serving_root).resolve()
    serving_bundle = resolve_latest_bundle(serving_source)
    serving_manifest = verify_serving_bundle(serving_bundle)
    if serving_manifest["bundle_sha256"] != approved["serving"]["bundle_sha256"]:
        raise ValueError("Serving root differs from the approved dev release")

    lineup_source = Path(lineups_root).resolve()
    lineup_pointer = _read_json(lineup_source / "latest.json")
    lineup_bundle = (lineup_source / str(lineup_pointer["bundle_path"])).resolve()
    _require_beneath(lineup_bundle, lineup_source, "Lineup bundle")
    lineup_manifest = verify_current_season_lineup_bundle(lineup_bundle)
    if lineup_manifest["bundle_sha256"] != approved["lineups"]["bundle_sha256"]:
        raise ValueError("Lineup root differs from the approved dev release")

    schedule_source = Path(schedule_snapshot_root).resolve()
    from nba_lineup_model.season.source_policy import require_clean_artifact_sources
    require_clean_artifact_sources(schedule_source)
    schedule_sha = _tree_sha256(schedule_source)
    if schedule_sha != approved["schedule"]["tree_sha256"]:
        raise ValueError("Schedule snapshot differs from the approved dev release")

    catalog_source = Path(catalog_path).resolve()
    rankings_source = Path(preseason_rankings_path).resolve()
    profiles_source = Path(preseason_profiles_path).resolve()
    _require_file_checksum(
        catalog_source,
        str(approved["catalog"]["sha256"]),
        "Game catalog",
    )
    _require_file_checksum(
        rankings_source,
        str(approved["preseason_catalog"]["rankings_sha256"]),
        "Preseason rankings",
    )
    _require_file_checksum(
        profiles_source,
        str(approved["preseason_catalog"]["profiles_sha256"]),
        "Preseason profiles",
    )

    code_source = Path(code_root).resolve()
    code_revision = _git_revision(code_source)
    if code_revision != approved["code_revision"]:
        raise ValueError("Code checkout differs from the approved dev release")
    frontend_source = Path(frontend_root).resolve()
    if not (frontend_source / "index.html").is_file():
        raise FileNotFoundError("Frontend candidate has no index.html")

    output = Path(output_root).resolve()
    target = output / f"release_sha256={approved_sha}"
    runtime_identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "approved_release_sha256": approved_sha,
        "season": approved["season"],
        "evidence_through_date": approved["evidence_through_date"],
        "game_count": approved["game_count"],
        "code_revision": code_revision,
        "serving_bundle_sha256": serving_manifest["bundle_sha256"],
        "lineup_bundle_sha256": lineup_manifest["bundle_sha256"],
        "schedule_snapshot_sha256": schedule_sha,
        "schedule_overlay_path": Path(schedule_overlay_path).resolve().as_posix(),
        "catalog_sha256": file_sha256(catalog_source),
        "preseason_rankings_sha256": file_sha256(rankings_source),
        "preseason_profiles_sha256": file_sha256(profiles_source),
        "frontend_sha256": _tree_sha256(frontend_source),
        "live_pointer": Path(live_pointer).absolute().as_posix(),
    }
    runtime_sha = hashlib.sha256(canonical_json(runtime_identity)).hexdigest()

    output.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = verify_production_site_release(target)
        if existing["runtime_sha256"] != runtime_sha:
            raise ValueError("Existing production candidate has different runtime inputs")
        return target

    staging = output / f".production-release.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        shutil.copytree(serving_source, staging / "current-season-serving")
        shutil.copytree(lineup_source, staging / "current-season-lineups")
        shutil.copytree(schedule_source, staging / "schedule-snapshot")
        shutil.copytree(dev_release_root, staging / "site-releases")
        shutil.copytree(frontend_source, staging / "web")
        (staging / "catalog").mkdir()
        shutil.copy2(catalog_source, staging / "catalog" / "games.parquet")
        (staging / "preseason").mkdir()
        shutil.copy2(rankings_source, staging / "preseason" / "rankings.parquet")
        shutil.copy2(profiles_source, staging / "preseason" / "profiles.parquet")
        (staging / "code").symlink_to(code_source, target_is_directory=True)
        (staging / "schedule").symlink_to(
            Path(schedule_overlay_path).resolve(), target_is_directory=True
        )
        live = Path(live_pointer).absolute()
        (staging / "site-release.env").write_text(
            "GESTALT_PRESEASON_RANKINGS_PATH="
            f"{live / 'preseason' / 'rankings.parquet'}\n"
            "GESTALT_PRESEASON_PROFILES_PATH="
            f"{live / 'preseason' / 'profiles.parquet'}\n",
            encoding="utf-8",
        )
        runtime_manifest = {
            "runtime_sha256": runtime_sha,
            "identity": runtime_identity,
            "approved_manifest": dev_manifest,
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    verify_production_site_release(target)
    return target


def verify_production_site_release(path: Path | str) -> dict[str, Any]:
    """Fail closed if any immutable runtime component has changed."""

    release = Path(path).resolve()
    manifest = _read_json(release / MANIFEST_FILENAME)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Production runtime identity is missing")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported production release schema")
    if identity.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Artifact is not a production site release")
    runtime_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    if manifest.get("runtime_sha256") != runtime_sha:
        raise ValueError("Production runtime checksum is invalid")

    approved_release = resolve_latest_dev_site_release(release / "site-releases")
    approved_manifest = verify_dev_site_release(approved_release)
    if approved_manifest["release_sha256"] != identity["approved_release_sha256"]:
        raise ValueError("Copied approval record has a different release identity")
    if manifest.get("approved_manifest") != approved_manifest:
        raise ValueError("Copied approval manifest changed after staging")

    serving_bundle = resolve_latest_bundle(release / "current-season-serving")
    serving_manifest = verify_serving_bundle(serving_bundle)
    if serving_manifest["bundle_sha256"] != identity["serving_bundle_sha256"]:
        raise ValueError("Production serving bundle changed after staging")

    lineup_root = release / "current-season-lineups"
    lineup_pointer = _read_json(lineup_root / "latest.json")
    lineup_bundle = (lineup_root / str(lineup_pointer["bundle_path"])).resolve()
    _require_beneath(lineup_bundle, lineup_root, "Lineup bundle")
    lineup_manifest = verify_current_season_lineup_bundle(lineup_bundle)
    if lineup_manifest["bundle_sha256"] != identity["lineup_bundle_sha256"]:
        raise ValueError("Production lineup bundle changed after staging")

    checks = (
        (release / "schedule-snapshot", identity["schedule_snapshot_sha256"], "Schedule"),
        (release / "web", identity["frontend_sha256"], "Frontend"),
    )
    for component, expected, label in checks:
        if _tree_sha256(component) != expected:
            raise ValueError(f"{label} tree changed after staging")
    _require_file_checksum(
        release / "catalog" / "games.parquet",
        str(identity["catalog_sha256"]),
        "Game catalog",
    )
    _require_file_checksum(
        release / "preseason" / "rankings.parquet",
        str(identity["preseason_rankings_sha256"]),
        "Preseason rankings",
    )
    _require_file_checksum(
        release / "preseason" / "profiles.parquet",
        str(identity["preseason_profiles_sha256"]),
        "Preseason profiles",
    )
    if _git_revision(release / "code") != identity["code_revision"]:
        raise ValueError("Production code changed after staging")
    schedule_link = release / "schedule"
    if not schedule_link.is_symlink():
        raise ValueError("Production Schedule overlay link is missing")
    if schedule_link.resolve() != Path(str(identity["schedule_overlay_path"])).resolve():
        raise ValueError("Production Schedule overlay target changed")
    if not (release / "web" / "index.html").is_file():
        raise FileNotFoundError("Production frontend has no index.html")
    return manifest


def switch_production_release(
    release: Path | str,
    *,
    live_pointer: Path | str,
) -> Path | None:
    """Atomically switch the single live pointer after full verification."""

    candidate = Path(release).resolve()
    verify_production_site_release(candidate)
    pointer = Path(live_pointer).absolute()
    if pointer.exists() and not pointer.is_symlink():
        raise ValueError("Live release pointer is not a symlink")
    previous = pointer.resolve() if pointer.is_symlink() else None
    temporary = pointer.with_name(f".{pointer.name}.tmp-{uuid4().hex}")
    temporary.symlink_to(candidate, target_is_directory=True)
    os.replace(temporary, pointer)
    return previous


def promote_with_rollback(
    release: Path | str,
    *,
    live_pointer: Path | str,
    health_check: Callable[[], None],
) -> Path | None:
    """Switch, run a caller-supplied health check, and restore on failure."""

    pointer = Path(live_pointer).absolute()
    previous = switch_production_release(release, live_pointer=pointer)
    try:
        health_check()
    except Exception:
        if previous is None:
            pointer.unlink(missing_ok=True)
        else:
            switch_production_release(previous, live_pointer=pointer)
        raise
    return previous


def _require_file_checksum(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} checksum differs from the approved release")


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_beneath(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its root") from error


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--dev-site-release-root", type=Path, required=True)
    stage.add_argument("--serving-root", type=Path, required=True)
    stage.add_argument("--lineups-root", type=Path, required=True)
    stage.add_argument("--schedule-snapshot-root", type=Path, required=True)
    stage.add_argument("--schedule-overlay-path", type=Path, required=True)
    stage.add_argument("--catalog-path", type=Path, required=True)
    stage.add_argument("--preseason-rankings-path", type=Path, required=True)
    stage.add_argument("--preseason-profiles-path", type=Path, required=True)
    stage.add_argument("--code-root", type=Path, required=True)
    stage.add_argument("--frontend-root", type=Path, required=True)
    stage.add_argument("--output-root", type=Path, required=True)
    stage.add_argument("--live-pointer", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("release", type=Path)
    switch = subparsers.add_parser("switch")
    switch.add_argument("release", type=Path)
    switch.add_argument("--live-pointer", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "stage":
        path = stage_production_site_release(
            dev_site_release_root=args.dev_site_release_root,
            serving_root=args.serving_root,
            lineups_root=args.lineups_root,
            schedule_snapshot_root=args.schedule_snapshot_root,
            schedule_overlay_path=args.schedule_overlay_path,
            catalog_path=args.catalog_path,
            preseason_rankings_path=args.preseason_rankings_path,
            preseason_profiles_path=args.preseason_profiles_path,
            code_root=args.code_root,
            frontend_root=args.frontend_root,
            output_root=args.output_root,
            live_pointer=args.live_pointer,
        )
        print(path)
    elif args.command == "verify":
        manifest = verify_production_site_release(args.release)
        print(json.dumps(manifest, sort_keys=True))
    else:
        previous = switch_production_release(
            args.release,
            live_pointer=args.live_pointer,
        )
        print(json.dumps({"previous": previous.as_posix() if previous else None}))


if __name__ == "__main__":
    main()
