"""Publish and load the official NBA schedule shared by the pipelines."""

import base64
import json
import re
from contextlib import closing
from datetime import date, datetime
from hashlib import sha256
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

PACIFIC = ZoneInfo("America/Los_Angeles")
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

    def games_for_date(
        self, day: date, reference: dict[str, str] | None = None,
    ) -> tuple[list[dict], dict[str, str]]:
        """Load the published schedule and return the Pacific slate and its reference."""
        payload, reference = self.load(reference)
        # The NBA response includes broadcast listings. Read only game identities and times.
        schedule = json.loads(payload, object_pairs_hook=self._check_schedule_fields)["leagueSchedule"]
        if str(schedule["seasonYear"]) != self.season or str(schedule["leagueId"]) != "00":
            raise ValueError("The official schedule belongs to a different season or league.")

        games = []
        for group in schedule["gameDates"]:
            for game in group["games"]:
                if (
                    str(game.get("gameStatus")) == "1"
                    and str(game["homeTeam"]["teamId"]) == str(game["awayTeam"]["teamId"]) == "0"
                ):
                    continue
                timestamp = (game.get("gameDateTimeUTC") or "").strip()
                raw_date = (game.get("gameDateEst") or group["gameDate"])[:10]
                reported_date = (
                    datetime.strptime(raw_date, "%m/%d/%Y").date()
                    if "/" in raw_date else date.fromisoformat(raw_date)
                )
                start = datetime.fromisoformat(timestamp) if timestamp.upper() not in {"", "TBD", "TBA"} else None
                if start is None or start.tzinfo is None:
                    if reported_date == day:
                        raise ValueError(f"No verified UTC schedule time for game {game['gameId']}.")
                    continue
                if start.astimezone(PACIFIC).date() != day:
                    continue
                home, away = game["homeTeam"], game["awayTeam"]
                home_id, away_id = int(home["teamId"]), int(away["teamId"])
                if (
                    not game["gameId"] or not home["teamTricode"] or not away["teamTricode"]
                    or min(home_id, away_id) <= 0 or home_id == away_id
                ):
                    raise ValueError("Invalid game or team identity in the selected NBA schedule.")
                games.append({
                    "game_id": str(game["gameId"]),
                    "game_date": reported_date,
                    "game_time_utc": start,
                    "away_team_id": away_id,
                    "away_team_tricode": away["teamTricode"].upper(),
                    "home_team_id": home_id,
                    "home_team_tricode": home["teamTricode"].upper(),
                })
        if "espn" in json.dumps(games, default=str).casefold():
            raise ValueError("Prohibited source in the selected NBA game identities.")
        if not games or len({row["game_id"] for row in games}) != len(games):
            raise ValueError(f"Missing or duplicate official games for {day}.")
        return sorted(games, key=lambda row: row["game_id"]), reference

    @staticmethod
    def _check_schedule_fields(pairs: list[tuple]) -> dict:
        if any("espn" in key.casefold() for key, _ in pairs):
            raise ValueError("Prohibited field in the official schedule archive.")
        return dict(pairs)

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
