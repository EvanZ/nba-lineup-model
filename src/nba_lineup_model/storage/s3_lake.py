from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

SCHEMA_VERSION = 1
DISPOSITIONS = {"archive", "regenerate", "already_archived", "defer"}
DEFAULT_CONFIG = Path("config/s3_lake_migration.json")
DEFAULT_MANIFEST_DIR = Path("data/manifests/s3_migration")


@dataclass(frozen=True)
class UploadSummary:
    planned: int
    uploaded: int
    existing: int
    verified: int
    total_bytes: int
    executed: bool


class S3Client(Protocol):
    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


Result = TypeVar("Result")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _normalized_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must not escape its root: {value}")
    return path


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"S3 URI is invalid: {uri}")
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"S3 URI is invalid: {uri}")
    return bucket, key


def _join_s3_uri(base_uri: str, *parts: str) -> str:
    bucket, base_key = _parse_s3_uri(base_uri.rstrip("/"))
    clean_parts = [str(_normalized_relative_path(part, label="S3 key part")) for part in parts]
    return f"s3://{bucket}/{'/'.join([base_key, *clean_parts])}"


def _excluded(path: PurePosixPath, patterns: Sequence[str]) -> bool:
    value = path.as_posix()
    return any(
        fnmatch.fnmatch(value, pattern)
        or (pattern.startswith("**/") and fnmatch.fnmatch(value, pattern.removeprefix("**/")))
        for pattern in patterns
    )


def _dataset_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset source does not exist: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _excluded(relative, patterns):
            continue
        if path.is_symlink():
            raise ValueError(f"Dataset sources cannot contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _validate_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Migration config must use schema version {SCHEMA_VERSION}")
    destination = payload.get("destination_base_uri")
    if not isinstance(destination, str):
        raise ValueError("Migration config requires destination_base_uri")
    _parse_s3_uri(destination)
    _normalized_relative_path(payload.get("manifest_prefix"), label="manifest_prefix")
    if not isinstance(payload.get("datasets"), list) or not payload["datasets"]:
        raise ValueError("Migration config requires at least one dataset")


def build_migration_manifest(
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    source_root: Path | str = Path("."),
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Inventory configured datasets and hash every object selected for archival."""

    config = json.loads(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Migration config must contain a JSON object")
    _validate_config(config)
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Migration source root does not exist: {root}")
    patterns = config.get("exclude_globs", [])
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ValueError("exclude_globs must be a list of strings")

    datasets: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    archived_uris: set[str] = set()
    source_paths: list[PurePosixPath] = []
    destination_paths: list[PurePosixPath] = []

    for raw_dataset in config["datasets"]:
        if not isinstance(raw_dataset, dict):
            raise ValueError("Each dataset config must be an object")
        name = raw_dataset.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Dataset name must be non-empty and unique: {name}")
        names.add(name)
        source = _normalized_relative_path(raw_dataset.get("source"), label=f"{name}.source")
        for previous in source_paths:
            if (
                source == previous
                or source.is_relative_to(previous)
                or previous.is_relative_to(source)
            ):
                raise ValueError(f"Dataset sources overlap: {previous} and {source}")
        source_paths.append(source)
        disposition = raw_dataset.get("disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"Unsupported disposition for {name}: {disposition}")
        destination_value = raw_dataset.get("destination")
        destination: PurePosixPath | None = None
        if disposition == "archive":
            destination = _normalized_relative_path(
                destination_value, label=f"{name}.destination"
            )
            for previous in destination_paths:
                if (
                    destination == previous
                    or destination.is_relative_to(previous)
                    or previous.is_relative_to(destination)
                ):
                    raise ValueError(
                        f"Archived dataset destinations overlap: {previous} and {destination}"
                    )
            destination_paths.append(destination)
        elif destination_value is not None:
            raise ValueError(f"Only archived datasets may set a destination: {name}")

        dataset_root = root.joinpath(*source.parts)
        files = _dataset_files(dataset_root, patterns)
        total_bytes = sum(path.stat().st_size for path in files)
        datasets.append(
            {
                "name": name,
                "source": source.as_posix(),
                "destination": destination.as_posix() if destination else None,
                "disposition": disposition,
                "file_count": len(files),
                "total_bytes": total_bytes,
            }
        )
        if destination is None:
            continue
        for path in files:
            relative = PurePosixPath(path.relative_to(dataset_root).as_posix())
            source_path = source / relative
            s3_uri = _join_s3_uri(
                config["destination_base_uri"], destination.as_posix(), relative.as_posix()
            )
            if s3_uri in archived_uris:
                raise ValueError(f"Multiple source files map to the same S3 URI: {s3_uri}")
            archived_uris.add(s3_uri)
            entries.append(
                {
                    "dataset": name,
                    "source_path": source_path.as_posix(),
                    "s3_uri": s3_uri,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "destination_base_uri": config["destination_base_uri"].rstrip("/"),
        "manifest_prefix": str(
            _normalized_relative_path(config["manifest_prefix"], label="manifest_prefix")
        ),
        "datasets": datasets,
        "files": entries,
    }
    manifest_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    manifest_s3_uri = _join_s3_uri(
        identity["destination_base_uri"],
        identity["manifest_prefix"],
        f"{manifest_id}.json",
    )
    return {
        **identity,
        "manifest_id": manifest_id,
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "source_root": str(root),
        "manifest_s3_uri": manifest_s3_uri,
    }


def _identity_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "destination_base_uri": manifest.get("destination_base_uri"),
        "manifest_prefix": manifest.get("manifest_prefix"),
        "datasets": manifest.get("datasets"),
        "files": manifest.get("files"),
    }


def validate_migration_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Migration manifest must use schema version {SCHEMA_VERSION}")
    expected = hashlib.sha256(_canonical_json(_identity_from_manifest(manifest))).hexdigest()
    if manifest.get("manifest_id") != expected:
        raise ValueError("Migration manifest ID does not match its contents")
    source_root = manifest.get("source_root")
    if not isinstance(source_root, str) or not Path(source_root).is_absolute():
        raise ValueError("Migration manifest source_root must be absolute")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Migration manifest files must be a list")
    destinations: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Migration manifest file entry must be an object")
        _normalized_relative_path(entry.get("source_path"), label="source_path")
        uri = entry.get("s3_uri")
        if not isinstance(uri, str):
            raise ValueError("Migration manifest file entry requires s3_uri")
        _parse_s3_uri(uri)
        if uri in destinations:
            raise ValueError(f"Migration manifest has a duplicate destination: {uri}")
        destinations.add(uri)
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise ValueError("Migration manifest file size is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Migration manifest file SHA-256 is invalid")
    manifest_uri = manifest.get("manifest_s3_uri")
    if not isinstance(manifest_uri, str):
        raise ValueError("Migration manifest requires manifest_s3_uri")
    _parse_s3_uri(manifest_uri)
    expected_manifest_uri = _join_s3_uri(
        manifest["destination_base_uri"],
        manifest["manifest_prefix"],
        f"{manifest['manifest_id']}.json",
    )
    if manifest_uri != expected_manifest_uri:
        raise ValueError("Migration manifest S3 URI does not match its manifest ID")


def write_migration_manifest(
    manifest: Mapping[str, Any], output_path: Path | str | None = None
) -> Path:
    validate_migration_manifest(manifest)
    path = (
        Path(output_path)
        if output_path is not None
        else DEFAULT_MANIFEST_DIR / f"{manifest['manifest_id']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_migration_manifest(path)
        if existing["manifest_id"] != manifest["manifest_id"]:
            raise FileExistsError(f"Refusing to replace a different migration manifest: {path}")
        if _identity_from_manifest(existing) != _identity_from_manifest(manifest):
            raise RuntimeError(f"Existing migration manifest identity is inconsistent: {path}")
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def read_migration_manifest(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Migration manifest must contain a JSON object")
    validate_migration_manifest(payload)
    return payload


def _new_s3_client(max_workers: int) -> S3Client:
    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=max(10, max_workers * 2),
            retries={"mode": "adaptive", "max_attempts": 10},
        ),
    )


def _client_error_code(error: ClientError) -> str:
    response = error.response if isinstance(error.response, dict) else {}
    details = response.get("Error")
    return str(details.get("Code", "")) if isinstance(details, dict) else ""


def _head_object(uri: str, *, client: S3Client) -> dict[str, Any] | None:
    bucket, key = _parse_s3_uri(uri)
    try:
        result = client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as error:
        if _client_error_code(error) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise RuntimeError(f"Unable to inspect immutable S3 object {uri}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"S3 head-object returned an invalid response for {uri}")
    return result


def _validate_remote_object(
    *, uri: str, head: Mapping[str, Any], size_bytes: int, sha256: str
) -> None:
    metadata = head.get("Metadata")
    remote_digest = metadata.get("sha256") if isinstance(metadata, dict) else None
    expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
    remote_checksum = head.get("ChecksumSHA256")
    if head.get("ContentLength") != size_bytes or remote_digest != sha256:
        raise RuntimeError(
            "Refusing immutable S3 mismatch for "
            f"{uri}: expected bytes={size_bytes}, sha256={sha256}; "
            f"found bytes={head.get('ContentLength')}, sha256={remote_digest}"
        )
    if remote_checksum is not None and remote_checksum != expected_checksum:
        raise RuntimeError(
            f"Refusing immutable S3 checksum mismatch for {uri}: "
            f"expected {expected_checksum}, found {remote_checksum}"
        )


def _ensure_remote_object(
    *,
    source: Path,
    uri: str,
    size_bytes: int,
    sha256: str,
    client: S3Client,
) -> bool:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Migration source is no longer a regular file: {source}")
    if source.stat().st_size != size_bytes or _sha256_file(source) != sha256:
        raise RuntimeError(f"Migration source changed after the manifest was created: {source}")
    bucket, key = _parse_s3_uri(uri)
    expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
    try:
        with source.open("rb") as handle:
            result = client.put_object(
                Bucket=bucket,
                Key=key,
                Body=handle,
                IfNoneMatch="*",
                Metadata={"sha256": sha256},
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=expected_checksum,
            )
    except ClientError as error:
        if _client_error_code(error) not in {"412", "PreconditionFailed"}:
            raise RuntimeError(f"Unable to upload immutable S3 object {uri}") from error
        existing = _head_object(uri, client=client)
        if existing is None:
            raise RuntimeError(f"S3 precondition failed but object is missing: {uri}") from error
        _validate_remote_object(
            uri=uri, head=existing, size_bytes=size_bytes, sha256=sha256
        )
        return False
    stored_checksum = result.get("ChecksumSHA256") if isinstance(result, dict) else None
    if stored_checksum is not None and stored_checksum != expected_checksum:
        raise RuntimeError(
            f"Uploaded S3 checksum mismatch for {uri}: "
            f"expected {expected_checksum}, found {stored_checksum}"
        )
    if stored_checksum is None:
        uploaded = _head_object(uri, client=client)
        if uploaded is None:
            raise RuntimeError(f"Uploaded S3 object cannot be verified: {uri}")
        _validate_remote_object(
            uri=uri, head=uploaded, size_bytes=size_bytes, sha256=sha256
        )
    return True


def _bounded_parallel_results(
    items: Sequence[Mapping[str, Any]],
    *,
    worker: Callable[[Mapping[str, Any]], Result],
    max_workers: int,
) -> Iterator[Result]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: set[Future[Result]] = set()
        for _ in range(max_workers):
            try:
                pending.add(executor.submit(worker, next(iterator)))
            except StopIteration:
                break
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    pending.add(executor.submit(worker, next(iterator)))
                except StopIteration:
                    pass


def upload_migration_manifest(
    manifest_path: Path | str,
    *,
    execute: bool = False,
    client: S3Client | None = None,
    max_workers: int = 4,
    progress_every: int = 1_000,
    progress: Callable[[str], None] = print,
) -> UploadSummary:
    """Upload selected files immutably, then publish their manifest last."""

    path = Path(manifest_path).resolve()
    manifest = read_migration_manifest(path)
    files = manifest["files"]
    total_bytes = sum(entry["size_bytes"] for entry in files)
    if not execute:
        return UploadSummary(
            planned=len(files),
            uploaded=0,
            existing=0,
            verified=0,
            total_bytes=total_bytes,
            executed=False,
        )
    root = Path(manifest["source_root"])
    active_client = client or _new_s3_client(max_workers)

    def upload_entry(entry: Mapping[str, Any]) -> bool:
        source = root.joinpath(*PurePosixPath(entry["source_path"]).parts)
        return _ensure_remote_object(
            source=source,
            uri=entry["s3_uri"],
            size_bytes=entry["size_bytes"],
            sha256=entry["sha256"],
            client=active_client,
        )

    uploaded_count = 0
    existing_count = 0
    for completed_count, uploaded in enumerate(
        _bounded_parallel_results(files, worker=upload_entry, max_workers=max_workers),
        start=1,
    ):
        uploaded_count += int(uploaded)
        existing_count += int(not uploaded)
        if progress_every > 0 and (
            completed_count % progress_every == 0 or completed_count == len(files)
        ):
            progress(
                f"Migration progress: {completed_count}/{len(files)} files; "
                f"uploaded={uploaded_count}; existing={existing_count}"
            )

    manifest_digest = _sha256_file(path)
    manifest_uploaded = _ensure_remote_object(
        source=path,
        uri=manifest["manifest_s3_uri"],
        size_bytes=path.stat().st_size,
        sha256=manifest_digest,
        client=active_client,
    )
    uploaded_count += int(manifest_uploaded)
    existing_count += int(not manifest_uploaded)
    return UploadSummary(
        planned=len(files),
        uploaded=uploaded_count,
        existing=existing_count,
        verified=len(files) + 1,
        total_bytes=total_bytes,
        executed=True,
    )


def verify_migration_manifest(
    manifest_path: Path | str,
    *,
    client: S3Client | None = None,
    max_workers: int = 4,
) -> UploadSummary:
    path = Path(manifest_path).resolve()
    manifest = read_migration_manifest(path)
    files = manifest["files"]
    active_client = client or _new_s3_client(max_workers)

    def verify_entry(entry: Mapping[str, Any]) -> None:
        head = _head_object(entry["s3_uri"], client=active_client)
        if head is None:
            raise FileNotFoundError(f"Archived S3 object is missing: {entry['s3_uri']}")
        _validate_remote_object(
            uri=entry["s3_uri"],
            head=head,
            size_bytes=entry["size_bytes"],
            sha256=entry["sha256"],
        )
    _bounded_parallel_results(files, worker=verify_entry, max_workers=max_workers)
    manifest_head = _head_object(manifest["manifest_s3_uri"], client=active_client)
    if manifest_head is None:
        raise FileNotFoundError(
            f"Archived migration manifest is missing: {manifest['manifest_s3_uri']}"
        )
    manifest_digest = _sha256_file(path)
    _validate_remote_object(
        uri=manifest["manifest_s3_uri"],
        head=manifest_head,
        size_bytes=path.stat().st_size,
        sha256=manifest_digest,
    )
    return UploadSummary(
        planned=len(files),
        uploaded=0,
        existing=len(files) + 1,
        verified=len(files) + 1,
        total_bytes=sum(entry["size_bytes"] for entry in files),
        executed=True,
    )


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and execute immutable S3 dataset migrations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Inventory and hash configured datasets")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--source-root", type=Path, default=Path("."))
    plan.add_argument("--output", type=Path)
    upload = subparsers.add_parser("upload", help="Upload a completed migration manifest")
    upload.add_argument("--manifest", type=Path, required=True)
    upload.add_argument(
        "--execute",
        action="store_true",
        help="Perform immutable AWS writes; omission is a no-network dry run",
    )
    upload.add_argument("--workers", type=int, default=4)
    verify = subparsers.add_parser("verify", help="Verify every object in S3")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        manifest = build_migration_manifest(
            config_path=args.config,
            source_root=args.source_root,
        )
        path = write_migration_manifest(manifest, args.output)
        archive_bytes = sum(entry["size_bytes"] for entry in manifest["files"])
        print(
            f"Migration plan {manifest['manifest_id']}: files={len(manifest['files'])}; "
            f"bytes={_format_bytes(archive_bytes)}; manifest={path}"
        )
        for dataset in manifest["datasets"]:
            print(
                f"{dataset['name']}: {dataset['disposition']}; "
                f"files={dataset['file_count']}; bytes={_format_bytes(dataset['total_bytes'])}"
            )
        return
    if args.command == "upload":
        summary = upload_migration_manifest(
            args.manifest,
            execute=args.execute,
            max_workers=args.workers,
            progress=lambda message: print(message, flush=True),
        )
        mode = "executed" if summary.executed else "dry-run"
        print(
            f"Upload {mode}: files={summary.planned}; bytes={_format_bytes(summary.total_bytes)}; "
            f"uploaded={summary.uploaded}; existing={summary.existing}; verified={summary.verified}"
        )
        return
    summary = verify_migration_manifest(args.manifest, max_workers=args.workers)
    print(
        f"Verification complete: objects={summary.verified}; "
        f"bytes={_format_bytes(summary.total_bytes)}"
    )


if __name__ == "__main__":
    main()
