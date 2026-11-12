"""Fetch and publish a quota-protected NBA odds snapshot."""

import json
from contextlib import closing
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ...utils import DATA_BUCKET

SPORTS_URL = "https://api.the-odds-api.com/v4/sports"
SPORT_KEY = "basketball_nba"
BOOKMAKERS = (
    "draftkings", "fanduel", "betmgm", "betrivers", "hardrockbet",
    "betparx", "ballybet", "betonlineag", "polymarket",
)
MARKETS = ("spreads", "h2h")
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
        for event in events:
            if "bookmakers" in event:
                event["bookmakers"] = [
                    bookmaker for bookmaker in event["bookmakers"]
                    if "espn" not in bookmaker["key"].casefold()
                    and "espn" not in bookmaker["title"].casefold()
                ]

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
        if snapshot["season"] != season or b"espn" in payload.lower():
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
