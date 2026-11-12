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

from ..utils import merge_transactions, official_json, transaction_rows_by_date, transactions_for_date

from ...utils import DATA_BUCKET

SOURCE_URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"


class NBATransactions(ConfigurableResource):
    season: str
    bucket: str = DATA_BUCKET

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
        return self._publish(
            boto3.client("s3"), day, transactions_for_date(payload, day),
            observed_at=observed_at, source_sha256=sha256(payload).hexdigest(),
        )

    def poll(self, first: date, last: date, payload: bytes, *, observed_at: datetime) -> dict:
        """Check the poll window without rewriting unchanged historical files."""
        grouped = transaction_rows_by_date(payload, first, last)
        client = boto3.client("s3")
        source_sha256 = sha256(payload).hexdigest()
        return {
            day: self._publish(
                client, day, grouped.get(day, []), observed_at=observed_at,
                source_sha256=source_sha256, skip_unchanged=day != last,
            )
            for day in sorted(grouped.keys() | {last})
        }

    def _publish(
        self, client, day: date, incoming: list[dict], *, observed_at: datetime,
        source_sha256: str, skip_unchanged: bool = False,
    ) -> dict:
        key = f"lake/raw/nba/transactions/season={self.season}/game_date_pacific={day}/latest.json"
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
                saved_raw = body.read()
                saved = official_json(saved_raw)
            if (
                saved["season"] != self.season or saved["game_date_pacific"] != day.isoformat()
                or saved["source_url"] != SOURCE_URL
            ):
                raise ValueError("Saved transaction history has different date, season, or source.")
            existing = saved["transactions"]

        rows, added = merge_transactions(existing, incoming, day=day, observed_at=observed_at)
        if skip_unchanged and etag is not None and added == 0:
            return {
                "uri": f"s3://{self.bucket}/{key}", "sha256": sha256(saved_raw).hexdigest(),
                "transaction_count": len(rows), "new_transaction_count": 0,
                "source_date_row_count": len(incoming),
            }
        document = {
            "season": self.season, "game_date_pacific": day.isoformat(),
            "source_url": SOURCE_URL, "last_polled_at": observed_at.isoformat(),
            "source_sha256": source_sha256, "transactions": rows,
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
