"""Fenced, journaled activation of one immutable GESTALT release.

Artifact builders may stage candidates, but only this publisher advances an
environment's live pointer.  Every activation is generation-fenced,
idempotent by operation ID, and recoverable after interruption.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.production_site_release import (
    verify_production_site_release,
)

PUBLISHER_SCHEMA_VERSION = 1
ACTIVATION_FILENAME = "activation.json"
Environment = Literal["dev", "prod"]
ReloadCallback = Callable[[dict[str, Any]], None]
VerifyCallback = Callable[[dict[str, Any]], None]


class StalePublisherError(RuntimeError):
    """The caller's expected generation no longer matches the active one."""


class ActivationSupersededError(RuntimeError):
    """An interrupted operation was superseded by a newer activation."""


@dataclass(frozen=True)
class ActivationResult:
    operation_id: str
    generation: int
    release_id: str
    status: Literal["activated", "recovered", "already_activated"]
    activation_root: Path


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_symlink(target: Path, pointer: Path) -> None:
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.tmp-{uuid4().hex}")
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, pointer)
    _fsync_directory(pointer.parent)


@contextmanager
def publisher_lock(publisher_root: Path | str) -> Iterator[None]:
    root = Path(publisher_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".publisher.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _operation_events(publisher_root: Path, operation_id: str) -> list[dict[str, Any]]:
    operation_root = publisher_root / "journal" / operation_id
    if not operation_root.is_dir():
        return []
    return [_read_json(path) for path in sorted(operation_root.glob("*.json"))]


def _append_event(
    publisher_root: Path,
    operation_id: str,
    state: str,
    **fields: Any,
) -> dict[str, Any]:
    events = _operation_events(publisher_root, operation_id)
    payload = {
        "schema_version": PUBLISHER_SCHEMA_VERSION,
        "artifact_kind": "release_activation_event",
        "operation_id": operation_id,
        "sequence": len(events) + 1,
        "state": state,
        "recorded_at": _timestamp(),
        **fields,
    }
    path = (
        publisher_root
        / "journal"
        / operation_id
        / f"{payload['sequence']:04d}-{state}.json"
    )
    _atomic_write_json(path, payload)
    return payload


def _last_event(
    publisher_root: Path, operation_id: str, state: str
) -> dict[str, Any] | None:
    for event in reversed(_operation_events(publisher_root, operation_id)):
        if event.get("state") == state:
            return event
    return None


def _git_output(code_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(code_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _sealed_code_identity(code_root: Path) -> dict[str, str]:
    tracked_status = _git_output(
        code_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise ValueError("Release code package has modified tracked files")
    untracked_runtime = _git_output(
        code_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "deploy",
        "scripts",
        "pyproject.toml",
        "uv.lock",
    )
    if untracked_runtime:
        raise ValueError("Release code package has untracked runtime files")
    revision = _git_output(code_root, "rev-parse", "HEAD").strip()
    tracked = subprocess.run(
        ["git", "-C", str(code_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    digest = hashlib.sha256()
    for raw in tracked:
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Tracked release file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    lock_path = code_root / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("Release code package has no uv.lock")
    return {
        "git_revision": revision,
        "package_sha256": digest.hexdigest(),
        "dependency_lock_sha256": file_sha256(lock_path),
    }


def _validated_release_identity(release: Path) -> dict[str, Any]:
    manifest = verify_production_site_release(release)
    identity = manifest["identity"]
    code = _sealed_code_identity((release / "code").resolve())
    if code["git_revision"] != identity["code_revision"]:
        raise ValueError("Release code revision differs from its runtime manifest")
    approved_identity = manifest["approved_manifest"]["identity"]
    return {
        "release_path": release.as_posix(),
        "release_id": str(identity["approved_release_sha256"]),
        "runtime_sha256": str(manifest["runtime_sha256"]),
        "binding_map_sha256": hashlib.sha256(
            canonical_json(approved_identity)
        ).hexdigest(),
        "edition_id": str(
            approved_identity.get("edition_id")
            or approved_identity.get("season")
            or "legacy-original"
        ),
        "code": code,
    }


def _validate_activation_record(record: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_kind",
        "environment",
        "generation",
        "operation_id",
        "release_path",
        "release_id",
        "runtime_sha256",
        "binding_map_sha256",
        "edition_id",
        "code",
        "created_at",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Activation record is incomplete: {sorted(missing)}")
    if record["schema_version"] != PUBLISHER_SCHEMA_VERSION:
        raise ValueError("Unsupported activation record schema")
    if record["artifact_kind"] != "release_activation":
        raise ValueError("Artifact is not a release activation")
    if record["environment"] not in {"dev", "prod"}:
        raise ValueError("Activation environment is invalid")
    if not isinstance(record["generation"], int) or record["generation"] < 1:
        raise ValueError("Activation generation is invalid")
    if record["environment"] == "prod" and not record.get("approval_reference"):
        raise ValueError("Production activation requires an approval reference")
    return record


def read_active_activation(live_pointer: Path | str) -> dict[str, Any] | None:
    pointer = Path(live_pointer).absolute()
    if not pointer.is_symlink():
        if pointer.exists():
            raise ValueError("Live release pointer is not a symlink")
        return None
    target = pointer.resolve()
    record_path = target / ACTIVATION_FILENAME
    if not record_path.is_file():
        return {
            "schema_version": 0,
            "artifact_kind": "legacy_release_activation",
            "environment": None,
            "generation": 0,
            "operation_id": "legacy-bootstrap",
            "release_path": target.as_posix(),
            "release_id": None,
            "runtime_sha256": None,
            "binding_map_sha256": None,
            "edition_id": None,
            "code": None,
            "created_at": None,
            "previous_generation": None,
            "previous_release_path": None,
        }
    record = _validate_activation_record(_read_json(record_path))
    if target.name != f"generation={int(record['generation']):020d}":
        raise ValueError("Active generation directory name disagrees with its record")
    return record


def read_active_generation(live_pointer: Path | str) -> int:
    active = read_active_activation(live_pointer)
    return int(active["generation"]) if active is not None else 0


def _verify_rollback_target(active: dict[str, Any] | None) -> None:
    if active is None:
        raise RuntimeError(
            "Activation requires an existing verified release for rollback"
        )
    identity = _validated_release_identity(
        Path(str(active["release_path"])).resolve()
    )
    if int(active["generation"]) == 0:
        return
    for key in (
        "release_id",
        "runtime_sha256",
        "binding_map_sha256",
        "edition_id",
        "code",
    ):
        if active.get(key) != identity[key]:
            raise ValueError(
                f"Active rollback release identity changed after activation: {key}"
            )


def _stage_generation(
    *,
    publisher_root: Path,
    environment: Environment,
    generation: int,
    operation_id: str,
    release: Path,
    previous: dict[str, Any] | None,
    reason: str,
    approval_reference: str | None,
) -> tuple[Path, dict[str, Any]]:
    target = (
        publisher_root
        / "generations"
        / f"generation={generation:020d}"
    )
    identity = _validated_release_identity(release)
    record = {
        "schema_version": PUBLISHER_SCHEMA_VERSION,
        "artifact_kind": "release_activation",
        "environment": environment,
        "generation": generation,
        "operation_id": operation_id,
        **identity,
        "previous_generation": int(previous["generation"]) if previous else 0,
        "previous_release_path": previous.get("release_path") if previous else None,
        "reason": reason,
        "approval_reference": approval_reference,
        "created_at": _timestamp(),
    }
    _validate_activation_record(record)
    if target.exists():
        existing = _validate_activation_record(_read_json(target / ACTIVATION_FILENAME))
        comparable = {key: value for key, value in record.items() if key != "created_at"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "created_at"
        }
        if comparable != existing_comparable:
            raise ValueError("Existing activation generation has different inputs")
        return target, existing

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".generation.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        for source in sorted(release.iterdir(), key=lambda path: path.name):
            if source.name == ACTIVATION_FILENAME:
                raise ValueError("Release candidate contains reserved activation metadata")
            (staging / source.name).symlink_to(
                source,
                target_is_directory=source.is_dir(),
            )
        _atomic_write_json(staging / ACTIVATION_FILENAME, record)
        _fsync_directory(staging)
        os.replace(staging, target)
        _fsync_directory(target.parent)
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
    verify_production_site_release(target)
    return target, record


def _reload_and_verify(
    record: dict[str, Any],
    *,
    reload_api: ReloadCallback,
    verify_api: VerifyCallback,
) -> None:
    reload_api(record)
    verify_api(record)


def _rollback_locked(
    *,
    publisher_root: Path,
    live_pointer: Path,
    failed_record: dict[str, Any],
    operation_id: str,
    reload_api: ReloadCallback,
    verify_api: VerifyCallback,
) -> dict[str, Any]:
    active = read_active_activation(live_pointer)
    if (
        active is None
        or active.get("generation") != failed_record["generation"]
        or active.get("operation_id") != failed_record["operation_id"]
    ):
        raise ActivationSupersededError(
            "A newer activation superseded the failed operation; rollback refused"
        )
    previous_path = failed_record.get("previous_release_path")
    if not previous_path:
        raise RuntimeError("First activation has no verified release to restore")
    rollback_generation = int(failed_record["generation"]) + 1
    rollback_operation = f"{operation_id}:rollback:{rollback_generation}"
    rollback_root, rollback_record = _stage_generation(
        publisher_root=publisher_root,
        environment=failed_record["environment"],
        generation=rollback_generation,
        operation_id=rollback_operation,
        release=Path(str(previous_path)).resolve(),
        previous=failed_record,
        reason=f"rollback after failed activation {operation_id}",
        approval_reference=failed_record.get("approval_reference"),
    )
    _atomic_symlink(rollback_root, live_pointer)
    _append_event(
        publisher_root,
        operation_id,
        "rollback_pointer_switched",
        generation=rollback_generation,
        release_id=rollback_record["release_id"],
    )
    _reload_and_verify(
        rollback_record,
        reload_api=reload_api,
        verify_api=verify_api,
    )
    _append_event(
        publisher_root,
        operation_id,
        "rolled_back",
        generation=rollback_generation,
        release_id=rollback_record["release_id"],
    )
    return rollback_record


def activate_release(
    *,
    release: Path | str,
    live_pointer: Path | str,
    publisher_root: Path | str,
    environment: Environment,
    operation_id: str,
    expected_previous_generation: int,
    reason: str,
    reload_api: ReloadCallback,
    verify_api: VerifyCallback,
    approval_reference: str | None = None,
    after_pointer_switch: Callable[[dict[str, Any]], None] | None = None,
) -> ActivationResult:
    """Activate or recover one idempotent, generation-fenced operation."""

    if not operation_id or "/" in operation_id or operation_id in {".", ".."}:
        raise ValueError("Operation ID must be a non-empty path-safe value")
    if environment == "prod" and not approval_reference:
        raise ValueError("Production activation requires an approval reference")
    candidate = Path(release).resolve()
    pointer = Path(live_pointer).absolute()
    root = Path(publisher_root).absolute()
    root.mkdir(parents=True, exist_ok=True)

    with publisher_lock(root):
        events = _operation_events(root, operation_id)
        requested = next(
            (event for event in events if event.get("state") == "requested"),
            None,
        )
        if requested is not None:
            expected_request = {
                "environment": environment,
                "expected_previous_generation": expected_previous_generation,
                "release_path": candidate.as_posix(),
                "reason": reason,
                "approval_reference": approval_reference,
            }
            actual_request = {
                key: requested.get(key) for key in expected_request
            }
            if actual_request != expected_request:
                raise ValueError(
                    "Operation ID was already used with different activation inputs"
                )
        activated = next(
            (event for event in reversed(events) if event.get("state") == "activated"),
            None,
        )
        active = read_active_activation(pointer)
        if activated is not None:
            if (
                active is None
                or active.get("operation_id") != operation_id
                or int(active["generation"]) != int(activated["generation"])
            ):
                raise ActivationSupersededError(
                    "A newer activation superseded this completed operation"
                )
            generation_root = (
                root
                / "generations"
                / f"generation={int(activated['generation']):020d}"
            )
            record = _validate_activation_record(
                _read_json(generation_root / ACTIVATION_FILENAME)
            )
            return ActivationResult(
                operation_id=operation_id,
                generation=int(record["generation"]),
                release_id=str(record["release_id"]),
                status="already_activated",
                activation_root=generation_root,
            )
        if (
            active is not None
            and active.get("operation_id") == operation_id
            and int(active["generation"]) == expected_previous_generation + 1
        ):
            _append_event(
                root,
                operation_id,
                "recovery_started",
                generation=active["generation"],
                release_id=active["release_id"],
            )
            try:
                _reload_and_verify(
                    active,
                    reload_api=reload_api,
                    verify_api=verify_api,
                )
            except Exception as error:
                _append_event(
                    root,
                    operation_id,
                    "verification_failed",
                    generation=active["generation"],
                    error=f"{type(error).__name__}: {error}",
                )
                try:
                    _rollback_locked(
                        publisher_root=root,
                        live_pointer=pointer,
                        failed_record=active,
                        operation_id=operation_id,
                        reload_api=reload_api,
                        verify_api=verify_api,
                    )
                except Exception as rollback_error:
                    _append_event(
                        root,
                        operation_id,
                        "rollback_failed",
                        error=f"{type(rollback_error).__name__}: {rollback_error}",
                    )
                    raise ExceptionGroup(
                        "Activation recovery and rollback failed",
                        [error, rollback_error],
                    ) from rollback_error
                raise
            _append_event(
                root,
                operation_id,
                "activated",
                generation=active["generation"],
                release_id=active["release_id"],
                recovered=True,
            )
            return ActivationResult(
                operation_id=operation_id,
                generation=int(active["generation"]),
                release_id=str(active["release_id"]),
                status="recovered",
                activation_root=pointer.resolve(),
            )

        _verify_rollback_target(active)
        current_generation = int(active["generation"])
        if (
            requested is not None
            and _last_event(root, operation_id, "pointer_switched") is not None
            and (active is None or active.get("operation_id") != operation_id)
        ):
            _append_event(
                root,
                operation_id,
                "superseded",
                current_generation=current_generation,
                current_operation_id=active.get("operation_id") if active else None,
            )
            raise ActivationSupersededError(
                "A newer activation superseded this interrupted operation"
            )
        if current_generation != expected_previous_generation:
            _append_event(
                root,
                operation_id,
                "rejected_stale",
                expected_previous_generation=expected_previous_generation,
                current_generation=current_generation,
            )
            raise StalePublisherError(
                "Stale publisher: expected generation "
                f"{expected_previous_generation}, active generation {current_generation}"
            )

        _append_event(
            root,
            operation_id,
            "requested",
            environment=environment,
            expected_previous_generation=expected_previous_generation,
            release_path=candidate.as_posix(),
            reason=reason,
            approval_reference=approval_reference,
        )
        generation = expected_previous_generation + 1
        generation_root, record = _stage_generation(
            publisher_root=root,
            environment=environment,
            generation=generation,
            operation_id=operation_id,
            release=candidate,
            previous=active,
            reason=reason,
            approval_reference=approval_reference,
        )
        _append_event(
            root,
            operation_id,
            "validated",
            generation=generation,
            release_id=record["release_id"],
            runtime_sha256=record["runtime_sha256"],
        )
        _atomic_symlink(generation_root, pointer)
        _append_event(
            root,
            operation_id,
            "pointer_switched",
            generation=generation,
            release_id=record["release_id"],
        )
        if after_pointer_switch is not None:
            after_pointer_switch(record)
        try:
            _reload_and_verify(
                record,
                reload_api=reload_api,
                verify_api=verify_api,
            )
        except Exception as error:
            _append_event(
                root,
                operation_id,
                "verification_failed",
                generation=generation,
                error=f"{type(error).__name__}: {error}",
            )
            try:
                _rollback_locked(
                    publisher_root=root,
                    live_pointer=pointer,
                    failed_record=record,
                    operation_id=operation_id,
                    reload_api=reload_api,
                    verify_api=verify_api,
                )
            except Exception as rollback_error:
                _append_event(
                    root,
                    operation_id,
                    "rollback_failed",
                    error=f"{type(rollback_error).__name__}: {rollback_error}",
                )
                raise ExceptionGroup(
                    "Activation verification and rollback failed",
                    [error, rollback_error],
                ) from rollback_error
            raise

        _append_event(
            root,
            operation_id,
            "activated",
            generation=generation,
            release_id=record["release_id"],
            recovered=False,
        )
        return ActivationResult(
            operation_id=operation_id,
            generation=generation,
            release_id=str(record["release_id"]),
            status="activated",
            activation_root=generation_root,
        )


def load_process_activation_identity(
    active_release_root: Path | str | None,
) -> dict[str, Any] | None:
    """Load activation identity once at API process construction time."""

    if not active_release_root:
        return None
    active = read_active_activation(Path(active_release_root))
    if active is None:
        return None
    return {
        "schema_version": active["schema_version"],
        "environment": active["environment"],
        "generation": active["generation"],
        "operation_id": active["operation_id"],
        "release_id": active["release_id"],
        "runtime_sha256": active["runtime_sha256"],
        "binding_map_sha256": active["binding_map_sha256"],
        "edition_id": active["edition_id"],
        "code": active["code"],
    }


def verify_loaded_identity(
    api_root: str,
    record: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.25,
) -> None:
    """Wait until the API proves it loaded the requested activation."""

    expected = {
        "environment": record["environment"],
        "generation": record["generation"],
        "operation_id": record["operation_id"],
        "release_id": record["release_id"],
        "runtime_sha256": record["runtime_sha256"],
        "binding_map_sha256": record["binding_map_sha256"],
        "edition_id": record["edition_id"],
        "code": record["code"],
    }
    health_url = f"{api_root.rstrip('/')}/api/health"
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as response:
                payload = json.load(response)
            loaded = payload.get("loaded_release")
            if loaded is not None:
                actual = {key: loaded.get(key) for key in expected}
                if actual == expected:
                    return
                last_error = ValueError(
                    f"API loaded release identity mismatch: {actual!r}"
                )
            else:
                last_error = ValueError("API health has no loaded release identity")
        except Exception as error:
            last_error = error
        time.sleep(poll_interval_seconds)
    raise RuntimeError(
        f"API did not load activation generation {record['generation']}: {last_error}"
    )


def _reload_service(service: str) -> ReloadCallback:
    def reload_api(_record: dict[str, Any]) -> None:
        subprocess.run(
            ["systemctl", "restart", service],
            check=True,
            timeout=60,
        )

    return reload_api


def _verify_api(api_root: str, timeout_seconds: float) -> VerifyCallback:
    def verify_api(record: dict[str, Any]) -> None:
        verify_loaded_identity(
            api_root,
            record,
            timeout_seconds=timeout_seconds,
        )

    return verify_api


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--release", type=Path, required=True)
    activate.add_argument("--live-pointer", type=Path, required=True)
    activate.add_argument("--publisher-root", type=Path, required=True)
    activate.add_argument("--environment", choices=("dev", "prod"), required=True)
    activate.add_argument("--operation-id", required=True)
    activate.add_argument("--expected-previous-generation", type=int, required=True)
    activate.add_argument("--reason", required=True)
    activate.add_argument("--service", required=True)
    activate.add_argument("--api-root", required=True)
    activate.add_argument("--approval-reference")
    activate.add_argument("--timeout-seconds", type=float, default=30.0)

    status = subparsers.add_parser("status")
    status.add_argument("--live-pointer", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        print(json.dumps(read_active_activation(args.live_pointer), indent=2))
        return 0
    result = activate_release(
        release=args.release,
        live_pointer=args.live_pointer,
        publisher_root=args.publisher_root,
        environment=args.environment,
        operation_id=args.operation_id,
        expected_previous_generation=args.expected_previous_generation,
        reason=args.reason,
        reload_api=_reload_service(args.service),
        verify_api=_verify_api(args.api_root, args.timeout_seconds),
        approval_reference=args.approval_reference,
    )
    print(
        json.dumps(
            {
                "operation_id": result.operation_id,
                "generation": result.generation,
                "release_id": result.release_id,
                "status": result.status,
                "activation_root": result.activation_root.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
