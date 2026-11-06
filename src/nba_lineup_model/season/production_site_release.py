"""Stage and verify one reviewed GESTALT site release.

The development release is the approval record.  Production staging copies its
verified serving artifacts into an immutable runtime directory, binds the exact
code and frontend, and leaves the mutable Schedule watcher in a separate
overlay. Activation authority belongs exclusively to ``release_publisher``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
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
from nba_lineup_model.season.sealed_code_package import (
    inspect_source_code,
    materialize_sealed_code_package,
    verify_sealed_code_package,
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
    historical_runtime_root: Path | str | None = None,
    observation_root: Path | str | None = None,
    forecast_registry_root: Path | None = None,
    version_references: dict[str, str] | None = None,
    environment: str = "dev",
    artifact_lake_uri: str | None = None,
    artifact_cache_root: Path | None = None,
    reconciliation_root: Path | None = None,
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
    code_identity = inspect_source_code(code_source)["identity"]
    code_revision = code_identity["git_revision"]
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
        "code_identity": code_identity,
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
    if target.exists() and historical_runtime_root is None:
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
        sealed = materialize_sealed_code_package(
            code_source, staging, expected_revision=code_revision
        )
        if sealed["identity"] != code_identity:
            raise ValueError("Release code changed while staging")
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
        if reconciliation_root is not None:
            from nba_lineup_model.players.reconciliation_artifact import reference, verify
            runtime_identity['reconciliation_reference'] = reference(verify(reconciliation_root)).as_dict()
        if historical_runtime_root is not None:
            from nba_lineup_model.web_api.release_serving import build_serving_bindings

            bindings = build_serving_bindings(
                staging,
                runtime_identity=runtime_identity,
                approved=approved,
                historical_runtime=Path(historical_runtime_root),
                observation_root=Path(observation_root) if observation_root else None,
                forecast_registry_root=forecast_registry_root,
                version_references=version_references,
                environment=environment,
            )
            runtime_identity["serving_bindings_sha256"] = bindings["bindings_sha256"]
            runtime_sha = hashlib.sha256(canonical_json(runtime_identity)).hexdigest()
            target = output / f"runtime_sha256={runtime_sha}"
        runtime_sha = hashlib.sha256(canonical_json(runtime_identity)).hexdigest()
        target = output / f"runtime_sha256={runtime_sha}"
        runtime_manifest = {
            "runtime_sha256": runtime_sha,
            "identity": runtime_identity,
            "approved_manifest": dev_manifest,
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if artifact_lake_uri is not None:
            if artifact_cache_root is None:
                raise ValueError("S3-backed release requires a cache root")
            from nba_lineup_model.season.serving_recovery import attach_inventory
            runtime_manifest = attach_inventory(staging, lake_uri=artifact_lake_uri, cache=artifact_cache_root)
            runtime_sha = runtime_manifest["runtime_sha256"]
            target = output / f"runtime_sha256={runtime_sha}"
        if target.exists():
            existing = verify_production_site_release(target)
            if existing["runtime_sha256"] != runtime_sha:
                raise ValueError(
                    "Existing release has different bound inputs; use a new approval record"
                )
        else:
            verify_production_site_release(staging)
            staging.replace(target)
    finally:
        if staging.exists():
            # Only temporary package directories are made writable for cleanup.
            temporary_code = staging / "code"
            if temporary_code.is_dir() and not temporary_code.is_symlink():
                for directory in [temporary_code, *temporary_code.rglob("*")]:
                    if directory.is_dir() and not directory.is_symlink():
                        directory.chmod(0o755)
            shutil.rmtree(staging)
    verify_production_site_release(target)
    return target


def stage_code_release(*, base_release: Path, code_root: Path, output_root: Path,
                       artifact_lake_uri: str | None = None, artifact_cache_root: Path | None = None) -> Path:
    """Replace code only; retain the original data approval and every data byte.

    The runtime identity records new code; the retained approval describes the
    unchanged data. The code deployment command performs fresh API validation.
    """
    base = base_release.resolve(strict=True)
    original = verify_production_site_release(base)
    code = inspect_source_code(code_root)["identity"]
    identity = {**original["identity"], "code_revision": code["git_revision"],
                "code_identity": code}
    checksum = hashlib.sha256(canonical_json(identity)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"runtime_sha256={checksum}"
    if target.exists() and artifact_lake_uri is None:
        existing = verify_production_site_release(target)
        if existing["identity"] != identity:
            raise ValueError("Code release identity differs from existing release")
        return target
    staging = output_root / f".code-release.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        # No serving builder, model updater, forecast selector or collector runs.
        for name in ("current-season-serving", "current-season-lineups",
                     "schedule-snapshot", "site-releases", "web", "catalog", "preseason"):
            shutil.copytree(base / name, staging / name)
        if (base / "serving-bindings.json").is_file():
            shutil.copytree(base / "selected-schedule", staging / "selected-schedule")
            shutil.copy2(base / "serving-bindings.json", staging / "serving-bindings.json")
        shutil.copy2(base / "site-release.env", staging / "site-release.env")
        # Source artifacts may be 0600 and owned by the service account. A root
        # deployment changes ownership when copying; retain content, not that
        # owner-only restriction. These are non-secret website data components.
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise ValueError("Immutable data component contains a symlink")
            path.chmod(0o755 if path.is_dir() else 0o644)
        (staging / "schedule").symlink_to(
            Path(identity["schedule_overlay_path"]), target_is_directory=True)
        sealed = materialize_sealed_code_package(
            code_root, staging, expected_revision=code["git_revision"])
        if sealed["identity"] != code:
            raise ValueError("Code changed during packaging")
        manifest = {**original, "identity": identity, "runtime_sha256": checksum}
        (staging / MANIFEST_FILENAME).write_bytes(canonical_json(manifest) + b"\n")
        if artifact_lake_uri is not None:
            if artifact_cache_root is None:
                raise ValueError("S3-backed code release requires a cache root")
            from nba_lineup_model.season.serving_recovery import attach_inventory
            manifest = attach_inventory(staging, lake_uri=artifact_lake_uri, cache=artifact_cache_root)
            target = output_root / f"runtime_sha256={manifest['runtime_sha256']}"
        elif (base / "durable-serving.json").exists():
            shutil.copy2(base / "durable-serving.json", staging / "durable-serving.json")
        verify_production_site_release(staging)
        if verify_production_site_release(base) != original:
            raise ValueError("Base release changed during code packaging")
        if target.exists():
            if verify_production_site_release(target) != manifest:
                raise ValueError("Existing code release differs")
        else:
            staging.replace(target)
    finally:
        if staging.exists():
            for directory in (staging / "code", *(staging / "code").rglob("*")):
                if directory.is_dir() and not directory.is_symlink():
                    directory.chmod(0o755)
            shutil.rmtree(staging)
    return target


def stage_schedule_release(*, base_release: Path, schedule_root: Path, output_root: Path,
                           initialization: dict | None = None) -> Path:
    """Stage forecasts and optional verified introductions with accepted model/code retained.

    The retained daily approval describes the accepted model/serving evidence.
    The new forecast selection has its own identity and is API-validated by the
    forecast publisher before activation. No model or application build runs.
    """
    from nba_lineup_model.season.serving_selection import verify_schedule_selection
    from nba_lineup_model.season.source_policy import require_clean_artifact_sources
    from nba_lineup_model.web_api.release_serving import build_serving_bindings

    base = base_release.resolve(strict=True)
    original = verify_production_site_release(base)
    selection = verify_schedule_selection(schedule_root)
    previous = verify_schedule_selection(base / "selected-schedule")
    if selection["edition_id"] != previous["edition_id"]:
        raise ValueError("Forecast publication cannot change the accepted model edition")
    if set(selection["games"]) != set(previous["games"]) or set(selection["dates"]) != set(previous["dates"]):
        raise ValueError("Forecast publication must preserve complete schedule coverage")
    require_clean_artifact_sources(schedule_root)
    initialized = None
    if initialization is not None:
        from nba_lineup_model.season.forecast_initialization import validate_publication
        initialized = validate_publication(base=base, **initialization)
    identity = {**original["identity"], "schedule_snapshot_sha256": _tree_sha256(schedule_root)}
    if initialized is not None:
        identity.update(serving_bundle_sha256=initialized['serving_bundle_sha256'],
            preseason_rankings_sha256=file_sha256(initialization['rankings_path']),
            preseason_profiles_sha256=file_sha256(initialization['profiles_path']),
            forecast_initialization=initialized)
    identity.pop("serving_bindings_sha256", None)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".schedule-release.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        for name in ("current-season-serving", "current-season-lineups", "site-releases",
                     "web", "catalog", "preseason"):
            shutil.copytree(base / name, staging / name)
        if initialized is not None:
            shutil.rmtree(staging / 'current-season-serving')
            shutil.copytree(initialization['serving_root'], staging / 'current-season-serving')
            shutil.copyfile(initialization['rankings_path'], staging / 'preseason/rankings.parquet')
            shutil.copyfile(initialization['profiles_path'], staging / 'preseason/profiles.parquet')
        shutil.copytree(schedule_root, staging / "schedule-snapshot")
        shutil.copy2(base / "site-release.env", staging / "site-release.env")
        (staging / "schedule").symlink_to(Path(identity["schedule_overlay_path"]), target_is_directory=True)
        sealed = materialize_sealed_code_package(base / "code", staging,
            expected_revision=identity["code_revision"])
        if sealed["identity"] != identity["code_identity"]:
            raise ValueError("Forecast publication changed deployed application code")
        bindings = _read_json(base / "serving-bindings.json")
        rebound = build_serving_bindings(staging, runtime_identity=identity,
            approved=original["approved_manifest"]["identity"],
            historical_runtime=Path(bindings["historical_runtime"]))
        identity["serving_bindings_sha256"] = rebound["bindings_sha256"]
        # All page pins except their forecast-selection dependency must match.
        for component, checksum in bindings["components"].items():
            allowed = {'forecast_selection_id'} | ({'serving_bundle_id', 'reconciliation_id'} if initialized else set())
            if component not in allowed and rebound["components"].get(component) != checksum:
                raise ValueError(f"Forecast publication changed another component: {component}")
        checksum = hashlib.sha256(canonical_json(identity)).hexdigest()
        target = output_root / f"runtime_sha256={checksum}"
        manifest = {**original, "identity": identity, "runtime_sha256": checksum}
        (staging / MANIFEST_FILENAME).write_bytes(canonical_json(manifest) + b"\n")
        if (base / "durable-serving.json").exists():
            from nba_lineup_model.season.serving_recovery import attach_inventory, inventory, cache_root
            storage = inventory(base)
            manifest = attach_inventory(staging, lake_uri=storage["lake_uri"], cache=cache_root(base))
            target = output_root / f"runtime_sha256={manifest['runtime_sha256']}"
        # These components contain no secrets; the API service must read them.
        for path in staging.rglob("*"):
            if path.is_symlink() or path.is_relative_to(staging / "code"):
                continue
            path.chmod(0o755 if path.is_dir() else 0o644)
        verify_production_site_release(staging)
        if verify_production_site_release(base) != original:
            raise ValueError("Active source changed while packaging forecast data")
        if target.exists():
            if verify_production_site_release(target) != manifest:
                raise ValueError("Existing forecast release differs from requested inputs")
        else:
            staging.replace(target)
        return target
    finally:
        if staging.exists():
            for directory in (staging / "code", *(staging / "code").rglob("*")):
                if directory.is_dir() and not directory.is_symlink():
                    directory.chmod(0o755)
            shutil.rmtree(staging)


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

    if identity.get("serving_bindings_sha256"):
        from nba_lineup_model.web_api.release_serving import verify_serving_bindings

        bindings = verify_serving_bindings(release)
        if identity["serving_bindings_sha256"] != bindings["bindings_sha256"]:
            raise ValueError("Serving bindings changed after staging")
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
    if not identity.get("code_identity"):
        raise ValueError(
            "Release lacks a sealed code identity; explicit legacy migration is required"
        )
    code = verify_sealed_code_package(release)["identity"]
    if code != identity["code_identity"] or code["git_revision"] != identity["code_revision"]:
        raise ValueError("Release code identity differs from its runtime manifest")
    schedule_link = release / "schedule"
    if not schedule_link.is_symlink():
        raise ValueError("Production Schedule overlay link is missing")
    if schedule_link.resolve() != Path(str(identity["schedule_overlay_path"])).resolve():
        raise ValueError("Production Schedule overlay target changed")
    if not (release / "web" / "index.html").is_file():
        raise FileNotFoundError("Production frontend has no index.html")
    return manifest


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
    else:
        manifest = verify_production_site_release(args.release)
        print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
