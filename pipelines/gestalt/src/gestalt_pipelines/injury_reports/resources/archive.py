"""Verify the archived PDF before publishing its immutable capture record."""

import base64
import json
import re
from contextlib import closing
from datetime import UTC, date, datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from .official_nba import EASTERN, MAX_PDF_BYTES, OfficialInjuryReport, publication_time

BUCKET = "nba-gestalt-data"
PREFIX = "lake/raw/nba/injury-reports"
EXTRACTION_PREFIX = "lake/artifacts/injury_report_extraction"
STATUSES_PREFIX = "lake/artifacts/injury_status_by_game_date"
_PUBLICATION_KEY_FORMAT = "%Y%m%dT%H%M%SZ"
CAPTURE_URI = re.compile(
    rf"s3://{BUCKET}/(?P<key>{PREFIX}/season=(?P<season>[0-9]{{4}}-[0-9]{{2}})/publication_date=[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}/"
    r"published_at=(?P<published_at>[0-9]{8}T[0-9]{6}Z)/sha256=(?P<digest>[0-9a-f]{64})/capture\.json)"
)


def _report_prefix(season: str, published_at: datetime, digest: str) -> str:
    day = published_at.astimezone(EASTERN).date().isoformat()
    instant = published_at.astimezone(UTC).strftime(_PUBLICATION_KEY_FORMAT)
    return f"{PREFIX}/season={season}/publication_date={day}/published_at={instant}/sha256={digest}"


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
    season: str

    def save(self, report: OfficialInjuryReport) -> dict:
        digest = sha256(report.pdf).hexdigest()

        prefix = _report_prefix(self.season, report.published_at, digest)
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
            "season": self.season,
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

    def captures_for_date(self, publication_date: date) -> list[tuple[datetime, datetime, str]]:
        prefix = f"{PREFIX}/season={self.season}/publication_date={publication_date}/"
        client = boto3.client("s3")
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
        captures = []
        for page in pages:
            for obj in page.get("Contents", []):
                uri = f"s3://{BUCKET}/{obj['Key']}"
                match = CAPTURE_URI.fullmatch(uri)
                if match:
                    instant = datetime.strptime(match["published_at"], _PUBLICATION_KEY_FORMAT).replace(tzinfo=UTC)
                    captures.append((instant, obj["LastModified"], uri))
        return sorted(captures, reverse=True)

    def captured_publication_times(self, publication_date: date) -> set[datetime]:
        return {published_at for published_at, _, _ in self.captures_for_date(publication_date)}

    def extraction_uri(self, capture_uri: str, parser_version: str) -> str:
        capture = CAPTURE_URI.fullmatch(capture_uri)
        if not capture or capture["season"] != self.season:
            raise ValueError("Expected an exact canonical S3 injury-report capture URI.")
        revision = capture["key"].removeprefix(f"{PREFIX}/").removesuffix("/capture.json")
        return f"s3://{BUCKET}/{EXTRACTION_PREFIX}/{revision}/parser_version={parser_version}/report.json"

    def extraction_uris(self, publication_date: date, parser_version: str) -> set[str]:
        prefix = f"{EXTRACTION_PREFIX}/season={self.season}/publication_date={publication_date}/"
        suffix = f"/parser_version={parser_version}/report.json"
        client = boto3.client("s3")
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
        return {
            f"s3://{BUCKET}/{obj['Key']}"
            for page in pages for obj in page.get("Contents", [])
            if obj["Key"].endswith(suffix)
        }

    def latest_capture_uri(self, publication_date: date) -> str:
        captures = self.captures_for_date(publication_date)
        if not captures:
            raise FileNotFoundError(f"No saved NBA injury report capture for {publication_date}.")
        newest = captures[0]
        if len(captures) > 1 and newest[:2] == captures[1][:2]:
            raise ValueError(f"Multiple latest captures for {publication_date}. Select an exact capture_uri.")
        return newest[2]

    def load(self, capture_uri: str) -> tuple[dict, bytes]:
        match = CAPTURE_URI.fullmatch(capture_uri)
        if match is None or match["season"] != self.season:
            raise ValueError("Expected an exact canonical S3 injury-report capture URI.")
        capture_key, digest = match["key"], match["digest"]
        client = boto3.client("s3")
        with closing(client.get_object(Bucket=BUCKET, Key=capture_key)["Body"]) as body:
            record = json.load(body)

        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("season") != self.season:
            raise ValueError(f"Unsupported injury-report capture schema: {capture_uri}")
        source_url = record.get("source_url")
        if not isinstance(source_url, str):
            raise ValueError(f"Missing official source URL in capture: {capture_uri}")
        published_at = datetime.strptime(match["published_at"], _PUBLICATION_KEY_FORMAT).replace(tzinfo=UTC)
        prefix = _report_prefix(self.season, published_at, digest)
        pdf_key = f"{prefix}/report.pdf"
        if (
            capture_key != f"{prefix}/capture.json"
            or record.get("immutable_uri") != f"s3://{BUCKET}/{pdf_key}"
            or record.get("sha256") != digest
            or record.get("published_at") != _utc_text(published_at)
        ):
            raise ValueError(f"Capture metadata does not identify the archived PDF: {capture_uri}")

        with closing(client.get_object(Bucket=BUCKET, Key=pdf_key)["Body"]) as body:
            pdf = body.read(MAX_PDF_BYTES + 1)
        if len(pdf) > MAX_PDF_BYTES or sha256(pdf).hexdigest() != digest:
            raise ValueError(f"Archived injury report does not match its capture: {capture_uri}")
        if publication_time(source_url, pdf) != published_at:
            raise ValueError(f"Archived PDF publication time does not match its capture: {capture_uri}")
        return record, pdf

    def save_extraction(self, record: dict) -> tuple[str, bool]:
        uri = self.extraction_uri(record["capture_uri"], record["parser_version"])
        key = uri.removeprefix(f"s3://{BUCKET}/")
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        client = boto3.client("s3")
        created = _put_once(client, key, payload, "application/json")
        _verify_bytes(client, key, payload)
        return uri, created

    def read_pinned(self, uri: str, expected_sha256: str, *, prefix: str) -> bytes:
        key = uri.removeprefix(f"s3://{BUCKET}/")
        if (
            not uri.startswith(f"s3://{BUCKET}/")
            or not key.startswith(prefix)
            or any(part in {"", ".", ".."} for part in key.split("/"))
            or "espn" in key.casefold()
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            raise ValueError(f"Expected a pinned S3 object under s3://{BUCKET}/{prefix}")
        client = boto3.client("s3")
        with closing(client.get_object(Bucket=BUCKET, Key=key)["Body"]) as body:
            payload = body.read()
        if sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f"S3 object checksum differs from the selected input: {uri}")
        return payload

    def read_player_repairs(self) -> dict:
        key = f"{STATUSES_PREFIX}/season={self.season}/player_repairs.json"
        client = boto3.client("s3")
        with closing(client.get_object(Bucket=BUCKET, Key=key)["Body"]) as body:
            payload = body.read()
        repairs = json.loads(payload)
        if not isinstance(repairs, dict) or any(
            not name.strip() or type(player_id) is not int or player_id <= 0
            for name, player_id in repairs.items()
        ):
            raise ValueError("The repair dictionary must map reported names to positive NBA player IDs.")
        if b"espn" in payload.lower():
            raise ValueError("Prohibited source in player repairs.")
        return {
            "uri": f"s3://{BUCKET}/{key}",
            "sha256": sha256(payload).hexdigest(),
            "entries": repairs,
        }

    def save_statuses(self, record: dict) -> dict:
        day = record["game_date_pacific"]
        selection = record["selection_sha256"]
        key = (
            f"{STATUSES_PREFIX}/season={self.season}/game_date_pacific={day}/"
            f"selection_sha256={selection}/statuses.json"
        )
        return self.save_json(key, record)

    def save_json(self, key: str, record: dict) -> dict:
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        client = boto3.client("s3")
        _put_once(client, key, payload, "application/json")
        _verify_bytes(client, key, payload)
        return {"uri": f"s3://{BUCKET}/{key}", "sha256": sha256(payload).hexdigest()}
