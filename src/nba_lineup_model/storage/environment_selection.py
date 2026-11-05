"""Immutable generation-fenced environment selections stored separately from artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from nba_lineup_model.storage.artifact_store import ArtifactReference
from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
from nba_lineup_model.storage.s3_lake import (
    S3Client,
    _ensure_remote_object,
    _join_s3_uri,
    _new_s3_client,
    _parse_s3_uri,
)

SCHEMA_VERSION = 1
SELECTION_KIND = "environment_selection"


class SelectionClient(S3Client):
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...


class StaleEnvironmentSelection(RuntimeError):
    """Another immutable generation won the activation race."""


@dataclass(frozen=True)
class SelectionResult:
    generation: int
    selection_uri: str
    selection_sha256: str
    activated: bool


def _environment(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z][a-z0-9_-]*", value
    ):
        raise ValueError("Environment must be a lowercase path-safe name")
    return value


def _timestamp(value: datetime | None) -> str:
    selected = value or datetime.now(UTC)
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("Selection time must be timezone-aware")
    return selected.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def selection_record_uri(
    environment: str,
    generation: int,
    *,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> str:
    environment = _environment(environment)
    if type(generation) is not int or generation < 1:
        raise ValueError("Selection generation must be a positive integer")
    return _join_s3_uri(
        lake_uri,
        "environment-selections",
        environment,
        "generations",
        f"generation={generation:020d}",
        "selection.json",
    )


def _generation_prefix(
    environment: str, *, lake_uri: str = DEFAULT_LAKE_URI
) -> str:
    return _join_s3_uri(
        lake_uri,
        "environment-selections",
        _environment(environment),
        "generations",
    )


def _client_error_code(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


def _read_json_object(
    uri: str, *, client: SelectionClient
) -> tuple[dict[str, object], str, str] | None:
    bucket, key = _parse_s3_uri(uri)
    try:
        response = client.get_object(
            Bucket=bucket,
            Key=key,
            ChecksumMode="ENABLED",
        )
    except ClientError as error:
        if _client_error_code(error) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise RuntimeError(
            f"Unable to read durable environment selection: {uri}"
        ) from error
    body = response["Body"]
    try:
        encoded = body.read()
    finally:
        body.close()
    metadata = response.get("Metadata")
    expected_sha = metadata.get("sha256") if isinstance(metadata, Mapping) else None
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha
    ):
        raise RuntimeError(
            f"Durable environment selection lacks a valid checksum: {uri}"
        )
    if response.get("ContentLength") != len(encoded) or _sha256(encoded) != expected_sha:
        raise RuntimeError(
            f"Durable environment selection checksum/size mismatch: {uri}"
        )
    etag = response.get("ETag")
    if not isinstance(etag, str) or not etag:
        raise RuntimeError(
            f"Durable environment selection lacks an ETag: {uri}"
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Durable environment selection is unreadable: {uri}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Durable environment selection is not an object: {uri}"
        )
    return payload, etag, expected_sha


def _put_immutable_json(
    uri: str, payload: Mapping[str, object], *, client: S3Client
) -> str:
    encoded = _json_bytes(payload)
    checksum = _sha256(encoded)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="environment-selection-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _ensure_remote_object(
            source=temporary,
            uri=uri,
            size_bytes=len(encoded),
            sha256=checksum,
            client=client,
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return checksum


def _validate_record(
    payload: Mapping[str, object],
    *,
    environment: str,
    generation: int,
) -> dict[str, object]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported environment selection schema")
    if payload.get("artifact_kind") != SELECTION_KIND:
        raise ValueError("Object is not an environment selection")
    if (
        payload.get("environment") != environment
        or payload.get("generation") != generation
    ):
        raise ValueError(
            "Environment selection location disagrees with its identity"
        )
    release = ArtifactReference.from_dict(payload.get("release", {}))
    previous = payload.get("previous")
    if generation == 1:
        if previous is not None:
            raise ValueError(
                "First environment selection cannot have a predecessor"
            )
    else:
        if not isinstance(previous, Mapping):
            raise ValueError(
                "Environment selection predecessor is missing"
            )
        if previous.get("generation") != generation - 1:
            raise ValueError(
                "Environment selection predecessor generation is invalid"
            )
        checksum = previous.get("selection_sha256")
        if not isinstance(checksum, str) or not re.fullmatch(
            r"[0-9a-f]{64}", checksum
        ):
            raise ValueError(
                "Environment selection predecessor checksum is invalid"
            )
    for field in ("operation_id", "reason", "created_at"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"Environment selection {field} is missing")
    if (
        environment in {"prod", "production"}
        and not payload.get("approval_reference")
    ):
        raise ValueError(
            "Production selection requires an approval reference"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": SELECTION_KIND,
        "environment": environment,
        "generation": generation,
        "previous": dict(previous) if isinstance(previous, Mapping) else None,
        "release": release.as_dict(),
        "operation_id": payload["operation_id"],
        "reason": payload["reason"],
        "approval_reference": payload.get("approval_reference"),
        "created_at": payload["created_at"],
    }


def _list_generation_uris(
    environment: str,
    *,
    client: SelectionClient,
    lake_uri: str,
) -> list[tuple[int, str]]:
    prefix_uri = _generation_prefix(environment, lake_uri=lake_uri)
    bucket, prefix = _parse_s3_uri(prefix_uri)
    prefix += "/"
    pattern = re.compile(
        re.escape(prefix)
        + r"generation=([0-9]{20})/selection[.]json"
    )
    results: list[tuple[int, str]] = []
    token: str | None = None
    while True:
        parameters: dict[str, object] = {
            "Bucket": bucket,
            "Prefix": prefix,
        }
        if token is not None:
            parameters["ContinuationToken"] = token
        response = client.list_objects_v2(**parameters)
        for item in response.get("Contents", []):
            key = item.get("Key") if isinstance(item, Mapping) else None
            match = pattern.fullmatch(str(key))
            if match:
                results.append(
                    (int(match.group(1)), f"s3://{bucket}/{key}")
                )
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not isinstance(token, str) or not token:
            raise RuntimeError(
                "Environment selection listing omitted its continuation token"
            )
    results.sort()
    if len({generation for generation, _ in results}) != len(results):
        raise RuntimeError(
            "Environment selection contains duplicate generations"
        )
    return results


def read_environment_selection(
    environment: str,
    *,
    client: SelectionClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> dict[str, object] | None:
    """Read and verify the complete immutable generation chain."""
    environment = _environment(environment)
    active_client = client if client is not None else _new_s3_client(1)
    entries = _list_generation_uris(
        environment,
        client=active_client,
        lake_uri=lake_uri,
    )
    if not entries:
        return None
    if [generation for generation, _ in entries] != list(
        range(1, entries[-1][0] + 1)
    ):
        raise RuntimeError(
            "Environment selection generation history has a gap"
        )

    previous_sha: str | None = None
    latest: dict[str, object] | None = None
    latest_sha = ""
    latest_uri = ""
    for generation, uri in entries:
        result = _read_json_object(uri, client=active_client)
        if result is None:
            raise RuntimeError(
                "Listed environment selection record is missing"
            )
        record = _validate_record(
            result[0],
            environment=environment,
            generation=generation,
        )
        record_sha = _sha256(_json_bytes(record))
        if record_sha != result[2]:
            raise RuntimeError(
                "Environment selection record is not canonical"
            )
        if generation > 1 and record["previous"] != {
            "generation": generation - 1,
            "selection_sha256": previous_sha,
        }:
            raise RuntimeError(
                "Environment selection predecessor chain is invalid"
            )
        previous_sha = record_sha
        latest = record
        latest_sha = record_sha
        latest_uri = uri
    assert latest is not None
    current = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "environment_current_selection",
        "environment": environment,
        "generation": latest["generation"],
        "selection_uri": latest_uri,
        "selection_sha256": latest_sha,
        "release": latest["release"],
        "operation_id": latest["operation_id"],
    }
    return {
        "current": current,
        "record": latest,
        "history_length": len(entries),
    }


def activate_environment_selection(
    *,
    environment: str,
    release: ArtifactReference,
    operation_id: str,
    reason: str,
    expected_previous_generation: int,
    approval_reference: str | None = None,
    created_at: datetime | None = None,
    client: SelectionClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> SelectionResult:
    """Select a release by creating exactly one immutable generation object."""
    environment = _environment(environment)
    if not operation_id.strip() or not reason.strip():
        raise ValueError(
            "Selection operation ID and reason are required"
        )
    if (
        type(expected_previous_generation) is not int
        or expected_previous_generation < 0
    ):
        raise ValueError(
            "Expected previous generation must be a nonnegative integer"
        )
    if environment in {"prod", "production"} and not approval_reference:
        raise ValueError(
            "Production selection requires an approval reference"
        )
    active_client = client if client is not None else _new_s3_client(1)
    active = read_environment_selection(
        environment,
        client=active_client,
        lake_uri=lake_uri,
    )
    current = active["current"] if active else None
    current_generation = current["generation"] if current else 0
    if current is not None and (
        current["operation_id"] == operation_id
        and current["release"] == release.as_dict()
    ):
        return SelectionResult(
            generation=current_generation,
            selection_uri=current["selection_uri"],
            selection_sha256=current["selection_sha256"],
            activated=False,
        )
    if current_generation != expected_previous_generation:
        raise StaleEnvironmentSelection(
            f"Expected generation {expected_previous_generation}, "
            f"found {current_generation}"
        )

    generation = expected_previous_generation + 1
    previous = None
    if current is not None:
        previous = {
            "generation": current["generation"],
            "selection_sha256": current["selection_sha256"],
        }
    record = _validate_record(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": SELECTION_KIND,
            "environment": environment,
            "generation": generation,
            "previous": previous,
            "release": release.as_dict(),
            "operation_id": operation_id,
            "reason": reason,
            "approval_reference": approval_reference,
            "created_at": _timestamp(created_at),
        },
        environment=environment,
        generation=generation,
    )
    record_uri = selection_record_uri(
        environment,
        generation,
        lake_uri=lake_uri,
    )
    try:
        record_sha = _put_immutable_json(
            record_uri,
            record,
            client=active_client,
        )
    except RuntimeError as error:
        winner = read_environment_selection(
            environment,
            client=active_client,
            lake_uri=lake_uri,
        )
        if (
            winner is not None
            and winner["current"]["generation"] == generation
            and winner["record"] == record
        ):
            return SelectionResult(
                generation,
                record_uri,
                winner["current"]["selection_sha256"],
                False,
            )
        raise StaleEnvironmentSelection(
            f"Generation {generation} was created by another operation"
        ) from error

    confirmed = read_environment_selection(
        environment,
        client=active_client,
        lake_uri=lake_uri,
    )
    if (
        confirmed is None
        or confirmed["current"]["generation"] != generation
        or confirmed["record"] != record
    ):
        raise StaleEnvironmentSelection(
            f"Generation {generation} was immediately superseded"
        )
    if confirmed["current"]["selection_sha256"] != record_sha:
        raise RuntimeError(
            "Environment selection readback checksum changed"
        )
    return SelectionResult(
        generation,
        record_uri,
        record_sha,
        True,
    )
