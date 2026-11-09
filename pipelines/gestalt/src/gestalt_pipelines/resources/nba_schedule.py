"""Publish and load the official NBA schedule shared by the pipelines."""

import base64
import json
import re
from contextlib import closing
from datetime import datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

SOURCE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"


class NBAScheduleResource(ConfigurableResource):
    season: str
    bucket: str = "nba-gestalt-data"

    @property
    def _prefix(self) -> str:
        return f"lake/raw/scheduleleaguev2/season={self.season}"

    def publish(self, payload: bytes, *, fetched_at: datetime) -> dict[str, str]:
        """Publish validated NBA JSON and return its exact S3 reference."""
        client = boto3.client("s3")
        checksum = sha256(payload)
        digest = checksum.hexdigest()
        key = f"{self._prefix}/sha256={digest}/schedule.json"
        reference = {"uri": f"s3://{self.bucket}/{key}", "sha256": digest}
        try:
            client.put_object(
                Bucket=self.bucket, Key=key, Body=payload,
                ContentType="application/json", IfNoneMatch="*",
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
                Metadata={
                    "source-url": SOURCE_URL,
                    "first-fetched-at": fetched_at.isoformat(),
                    "sha256": digest,
                },
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
        self._load_version(client, reference)
        client.put_object(
            Bucket=self.bucket, Key=f"{self._prefix}/latest.json",
            Body=json.dumps(reference).encode(), ContentType="application/json",
        )
        return reference

    def load(self, reference: dict[str, str] | None = None) -> tuple[bytes, dict[str, str]]:
        """Load the current season schedule or an exact reference for replay."""
        client = boto3.client("s3")
        if reference is None:
            response = client.get_object(Bucket=self.bucket, Key=f"{self._prefix}/latest.json")
            with closing(response["Body"]) as body:
                reference = json.load(body)
        return self._load_version(client, reference), reference

    def _load_version(self, client, reference: dict[str, str]) -> bytes:
        digest = reference.get("sha256") if isinstance(reference, dict) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Expected a SHA-256 checksum in the NBA schedule reference.")
        key = f"{self._prefix}/sha256={digest}/schedule.json"
        if reference.get("uri") != f"s3://{self.bucket}/{key}":
            raise ValueError(f"Expected a versioned NBA schedule reference for {self.season}.")
        response = client.get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as body:
            payload = body.read()
        if sha256(payload).hexdigest() != digest:
            raise ValueError(f"NBA schedule bytes do not match their checksum at s3://{self.bucket}/{key}.")
        return payload
