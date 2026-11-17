"""Fetch and publish a quota-protected NBA odds snapshot."""

import json
from hashlib import sha256
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ...utils import DATA_BUCKET, PACIFIC

SPORTS_URL = "https://api.the-odds-api.com/v4/sports"
from .. import BOOKMAKERS, MARKETS, SPORT_KEY

MINIMUM_REMAINING_CREDITS = 50


class OddsApi(ConfigurableResource):
    api_key: str
    bucket: str = DATA_BUCKET

    def fetch(self, season: str) -> dict | None:
        # /sports reports the current quota without consuming credits.
        _, quota_headers = self._get(SPORTS_URL)
        remaining = int(quota_headers["x-requests-remaining"])
        cost = len(MARKETS) * ((len(BOOKMAKERS) + 9) // 10)
        if remaining - cost < MINIMUM_REMAINING_CREDITS:
            return None

        collected_at = datetime.now(UTC)
        events, headers = self._get(
            f"{SPORTS_URL}/{SPORT_KEY}/odds",
            bookmakers=",".join(BOOKMAKERS),
            markets=",".join(MARKETS),
            oddsFormat="american",
            dateFormat="iso",
        )
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise ValueError("The Odds API did not return an event list.")

        return {
            "schema_version": 1,
            "provider": "The Odds API",
            "sport_key": SPORT_KEY,
            "season": season,
            "collected_at": collected_at.isoformat().replace("+00:00", "Z"),
            "request": {
                "bookmakers": list(BOOKMAKERS),
                "markets": list(MARKETS),
                "odds_format": "american",
            },
            "quota": {
                "requests_used": int(headers["x-requests-used"]),
                "requests_remaining": int(headers["x-requests-remaining"]),
                "request_cost": int(headers["x-requests-last"]),
                "minimum_remaining_credits": MINIMUM_REMAINING_CREDITS,
            },
            "events": events,
        }

    def archived_snapshots(self, season: str, day: date) -> list[dict]:
        """Validate an archived capture date without polling or publishing anything."""
        client = boto3.client("s3")
        start = datetime.combine(day, time.min, PACIFIC).astimezone(UTC)
        end = datetime.combine(day+timedelta(days=1), time.min, PACIFIC).astimezone(UTC)
        root = f"lake/raw/odds-api/season={season}/snapshots/"
        references = []
        for utc_day in {start.date(), end.date()}:
            for page in client.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=root+utc_day.strftime("%Y%m%d"),
            ):
                for entry in page.get("Contents", []):
                    key = entry["Key"]
                    if not key.endswith(".json"):
                        continue
                    instant = datetime.strptime(key.removeprefix(root), "%Y%m%dT%H%M%SZ.json").replace(tzinfo=UTC)
                    if not start <= instant < end:
                        continue
                    with closing(client.get_object(Bucket=self.bucket, Key=key)["Body"]) as body:
                        payload = body.read()
                    record = json.loads(payload)
                    collected = datetime.fromisoformat(record["collected_at"])
                    if (
                        record.get("provider") != "The Odds API" or record.get("season") != season
                        or record.get("sport_key") != SPORT_KEY or collected.tzinfo is None
                        or collected.astimezone(PACIFIC).date() != day
                        or collected.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ.json") != key.removeprefix(root)
                        or not isinstance(record.get("events"), list)
                    ):
                        raise ValueError(f"Invalid archived Odds API capture at s3://{self.bucket}/{key}.")
                    references.append({
                        "uri": f"s3://{self.bucket}/{key}", "sha256": sha256(payload).hexdigest(),
                        "collected_at": record["collected_at"], "event_count": len(record["events"]),
                    })
        if not references:
            raise FileNotFoundError(f"No genuine archived odds captures for Pacific date {day}.")
        return sorted(references, key=lambda row: row["collected_at"])

    def _get(self, url: str, **params: str) -> tuple[object, object]:
        if not self.api_key.strip():
            raise ValueError("GESTALT_ODDS_API_KEY is not configured.")
        query = urlencode({"apiKey": self.api_key, **params})
        request = Request(f"{url}?{query}", headers={"User-Agent": "nba-gestalt/1.0"})
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response), response.headers
        except HTTPError as error:
            raise RuntimeError(f"The Odds API returned HTTP {error.code}.") from None
        except (URLError, TimeoutError):
            raise RuntimeError("The Odds API request failed.") from None

    def publish(self, season: str, payload: bytes) -> str:
        snapshot = json.loads(payload)
        if snapshot["season"] != season or snapshot.get("provider") != "The Odds API":
            raise ValueError("The staged odds snapshot failed season or source validation.")

        collected_at = datetime.fromisoformat(snapshot["collected_at"])
        timestamp = collected_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"lake/raw/odds-api/season={season}"
        archive_key = f"{prefix}/snapshots/{timestamp}.json"
        latest_key = f"{prefix}/latest.json"
        uri = f"s3://{self.bucket}/{archive_key}"
        client = boto3.client("s3")
        try:
            client.put_object(
                Bucket=self.bucket, Key=archive_key, Body=payload,
                ContentType="application/json", IfNoneMatch="*",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            response = client.get_object(Bucket=self.bucket, Key=archive_key)
            with closing(response["Body"]) as body:
                if body.read() != payload:
                    raise ValueError(f"Different bytes already exist at {uri}.") from exc

        try:
            response = client.get_object(Bucket=self.bucket, Key=latest_key)
        except client.exceptions.NoSuchKey:
            pass
        else:
            with closing(response["Body"]) as body:
                latest_at = datetime.fromisoformat(json.load(body)["collected_at"])
            if latest_at >= collected_at:
                return uri

        client.put_object(
            Bucket=self.bucket, Key=latest_key, Body=payload, ContentType="application/json",
        )
        return uri
