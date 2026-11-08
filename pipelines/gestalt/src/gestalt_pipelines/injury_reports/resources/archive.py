"""Verify the archived PDF before publishing its immutable capture record."""

import base64
import json
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from .official_nba import EASTERN, OfficialInjuryReport

BUCKET = "nba-gestalt-data"
PREFIX = "lake/raw/nba/injury-reports"


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _put_once(client, key: str, body: bytes, content_type: str, *, metadata=None) -> bool:
    try:
        client.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=body,
            ContentType=content_type,
            Metadata=metadata or {},
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=base64.b64encode(sha256(body).digest()).decode("ascii"),
            IfNoneMatch="*",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"412", "PreconditionFailed"}:
            raise
        return False
    return True


def _verify_bytes(client, key: str, expected: bytes) -> dict:
    response = client.get_object(Bucket=BUCKET, Key=key)
    with closing(response["Body"]) as body:
        if body.read(len(expected) + 1) != expected:
            raise ValueError(f"Stored object does not match the expected bytes: s3://{BUCKET}/{key}")
    return response


class InjuryReportArchive(ConfigurableResource):
    def save(self, report: OfficialInjuryReport, expected_sha256: str) -> dict:
        digest = sha256(report.pdf).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"The report PDF checksum changed: expected {expected_sha256}, got {digest}.")

        day = report.published_at.astimezone(EASTERN).date().isoformat()
        instant = report.published_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"{PREFIX}/publication_date={day}/published_at={instant}/sha256={digest}"
        pdf_key = f"{prefix}/report.pdf"
        capture_key = f"{prefix}/capture.json"
        client = boto3.client("s3")
        _put_once(
            client,
            pdf_key,
            report.pdf,
            "application/pdf",
            metadata={"sha256": digest, "first-observed-at": _utc_text(report.first_observed_at)},
        )
        stored_pdf = _verify_bytes(client, pdf_key, report.pdf)
        observed = stored_pdf.get("Metadata", {}).get("first-observed-at")
        if not observed or datetime.fromisoformat(observed).tzinfo is None:
            raise ValueError(f"Stored PDF is missing a valid first observation time: {pdf_key}")

        record = {
            "schema_version": 1,
            "source_url": report.source_url,
            "immutable_uri": f"s3://{BUCKET}/{pdf_key}",
            "sha256": digest,
            "published_at": _utc_text(report.published_at),
            "first_observed_at": observed,
            "captured_at": _utc_text(stored_pdf["LastModified"]),
        }
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        created = _put_once(client, capture_key, payload, "application/json")
        _verify_bytes(client, capture_key, payload)
        return {
            "capture_uri": f"s3://{BUCKET}/{capture_key}",
            "pdf_uri": record["immutable_uri"],
            "sha256": digest,
            "published_at": record["published_at"],
            "created": created,
        }
