"""Fetch and publish the official NBA CDN scoreboard."""

import base64
from contextlib import closing
from datetime import date, datetime
from hashlib import sha256

import boto3
import requests
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ..utils import parse_scoreboard

SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"


class NBAScoreboardArchive(ConfigurableResource):
    bucket: str = "nba-gestalt-data"

    def fetch(self) -> bytes:
        with requests.get(
            SCOREBOARD_URL,
            headers={"User-Agent": "nba-gestalt/1.0", "Referer": "https://www.nba.com/"},
            timeout=30, allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"Unexpected NBA CDN scoreboard status {response.status_code}.")
            return response.content

    def publish(self, season: str, game_date: date, payload: bytes, fetched_at: datetime) -> dict[str, str]:
        key = f"lake/raw/cdn/scoreboard/season={season}/game_date_pacific={game_date}/latest.json"
        checksum = sha256(payload)
        boto3.client("s3").put_object(
            Bucket=self.bucket, Key=key, Body=payload, ContentType="application/json",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
            Metadata={"source-url": SCOREBOARD_URL, "fetched-at": fetched_at.isoformat(), "sha256": checksum.hexdigest()},
        )
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": checksum.hexdigest()}

    def read_latest(self, season: str, game_date: date) -> dict | None:
        key = f"lake/raw/cdn/scoreboard/season={season}/game_date_pacific={game_date}/latest.json"
        try:
            response = boto3.client("s3").get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None
            raise
        with closing(response["Body"]) as stream:
            payload = stream.read()
        if sha256(payload).hexdigest() != response.get("Metadata", {}).get("sha256"):
            raise ValueError(f"Scoreboard checksum failed at s3://{self.bucket}/{key}.")
        board = parse_scoreboard(payload)
        if board["gameDate"] != game_date.isoformat():
            raise ValueError("Stored scoreboard belongs to a different date.")
        return board
