from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import boto3
from botocore.exceptions import ClientError


class S3Client(Protocol):
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BackfillSummary:
    legacy_objects: int
    unique_legacy_payloads: int
    checksum_objects_created: int
    checksum_objects_existing: int
    source_bytes: int
    unique_bytes: int
    executed: bool


def _legacy_pdf_objects(client: S3Client, *, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    continuation_token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix.rstrip("/") + "/"}
        if continuation_token is not None:
            request["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**request)
        for item in response.get("Contents", []):
            key = str(item.get("Key", ""))
            if key.endswith(".injury.pdf") and "/injury-reports/" not in key:
                objects.append(item)
        if not response.get("IsTruncated"):
            break
        continuation_token = str(response["NextContinuationToken"])
    return sorted(objects, key=lambda item: str(item["Key"]))


def _is_missing(error: ClientError) -> bool:
    return str(error.response.get("Error", {}).get("Code")) in {"404", "NoSuchKey"}


def _verified_checksum_object(
    client: S3Client,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    sha256: str,
) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if _is_missing(error):
            return False
        raise
    if int(head.get("ContentLength", -1)) != len(payload):
        raise RuntimeError(f"Checksum object has the wrong size: s3://{bucket}/{key}")
    metadata = head.get("Metadata", {})
    if isinstance(metadata, dict) and metadata.get("sha256") == sha256:
        return True
    existing = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if existing != payload:
        raise RuntimeError(f"Checksum object has different bytes: s3://{bucket}/{key}")
    return True


def backfill_legacy_injury_pdfs(
    *,
    bucket: str,
    prefix: str,
    execute: bool = False,
    client: S3Client | None = None,
) -> BackfillSummary:
    """Backfill legacy per-snapshot injury PDFs into content-addressed S3 keys."""

    s3 = client or boto3.client("s3")
    legacy = _legacy_pdf_objects(s3, bucket=bucket, prefix=prefix)
    fingerprints: dict[tuple[str, int], tuple[str, bytes]] = {}
    payloads: dict[str, bytes] = {}
    source_bytes = 0

    for item in legacy:
        key = str(item["Key"])
        size = int(item.get("Size", 0))
        source_bytes += size
        fingerprint = (str(item.get("ETag", "")).strip('"'), size)
        cached = fingerprints.get(fingerprint)
        if cached is None:
            payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            if len(payload) != size:
                raise RuntimeError(f"Legacy object changed while reading: s3://{bucket}/{key}")
            digest = hashlib.sha256(payload).hexdigest()
            cached = (digest, payload)
            fingerprints[fingerprint] = cached
        digest, payload = cached
        payloads.setdefault(digest, payload)

    created = 0
    existing = 0
    checksum_prefix = prefix.rstrip("/") + "/injury-reports"
    for digest, payload in sorted(payloads.items()):
        target = f"{checksum_prefix}/sha256={digest}.pdf"
        if _verified_checksum_object(
            s3,
            bucket=bucket,
            key=target,
            payload=payload,
            sha256=digest,
        ):
            existing += 1
            continue
        if not execute:
            continue
        try:
            s3.put_object(
                Bucket=bucket,
                Key=target,
                Body=payload,
                Metadata={"sha256": digest},
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
                ContentType="application/pdf",
            )
            if not _verified_checksum_object(
                s3,
                bucket=bucket,
                key=target,
                payload=payload,
                sha256=digest,
            ):
                raise RuntimeError(f"Unable to verify s3://{bucket}/{target}")
            created += 1
        except ClientError as error:
            if str(error.response.get("Error", {}).get("Code")) != "PreconditionFailed":
                raise
            if not _verified_checksum_object(
                s3,
                bucket=bucket,
                key=target,
                payload=payload,
                sha256=digest,
            ):
                raise RuntimeError(f"Unable to verify s3://{bucket}/{target}") from error
            existing += 1

    return BackfillSummary(
        legacy_objects=len(legacy),
        unique_legacy_payloads=len(payloads),
        checksum_objects_created=created,
        checksum_objects_existing=existing,
        source_bytes=source_bytes,
        unique_bytes=sum(len(payload) for payload in payloads.values()),
        executed=execute,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill schedule injury PDFs into checksum-addressed S3 objects."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    summary = backfill_legacy_injury_pdfs(
        bucket=args.bucket,
        prefix=args.prefix,
        execute=args.execute,
    )
    print(
        json.dumps(
            {
                **asdict(summary),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
