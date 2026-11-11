"""Fetch NBA transactions and maintain each date's first-observation history."""

import base64
import json
from contextlib import closing
from datetime import date, datetime
from hashlib import sha256

import boto3
import requests
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ..utils import merge_transactions, official_json, transactions_for_date

SOURCE_URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"


class NBATransactions(ConfigurableResource):
    season: str
    bucket: str = "nba-gestalt-data"

    def fetch(self) -> bytes:
        with requests.get(
            SOURCE_URL,
            headers={
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.nba.com/", "User-Agent": "Mozilla/5.0",
            },
            timeout=(5, 30), allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"Unexpected NBA transaction response {response.status_code}.")
            return response.content

    def publish(self, day: date, payload: bytes, *, observed_at: datetime) -> dict:
        incoming = transactions_for_date(payload, day)
        key = f"lake/raw/nba/transactions/season={self.season}/game_date_pacific={day}/latest.json"
        client = boto3.client("s3")
        etag = None
        existing = []
        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise
        else:
            etag = response["ETag"]
            with closing(response["Body"]) as body:
                saved = official_json(body.read())
            if (
                saved["season"] != self.season or saved["game_date_pacific"] != day.isoformat()
                or saved["source_url"] != SOURCE_URL
            ):
                raise ValueError("Saved transaction history has different date, season, or source.")
            existing = saved["transactions"]

        rows, added = merge_transactions(existing, incoming, day=day, observed_at=observed_at)
        document = {
            "season": self.season, "game_date_pacific": day.isoformat(),
            "source_url": SOURCE_URL, "last_polled_at": observed_at.isoformat(),
            "source_sha256": sha256(payload).hexdigest(), "transactions": rows,
        }
        encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        checksum = sha256(encoded)
        # Refuse to overwrite another run's intervening additions.
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        client.put_object(
            Bucket=self.bucket, Key=key, Body=encoded, ContentType="application/json",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
            **condition,
        )
        return {
            "uri": f"s3://{self.bucket}/{key}", "sha256": checksum.hexdigest(),
            "transaction_count": len(rows), "new_transaction_count": added,
            "source_date_row_count": len(incoming),
        }
