"""Read immutable captures and publish only a direct per-game S3 pointer."""
import json
from contextlib import closing
from datetime import UTC, datetime, time, timedelta
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC
from .. import SPORT_KEY
from ..utils import eligible_event, instant


class GameOddsArchive(ConfigurableResource):
    season: str
    bucket: str = DATA_BUCKET

    def raw_root(self):
        return f"lake/raw/odds-api/season={self.season}"

    def key(self, day, gid):
        return f"lake/artifacts/game_odds/season={self.season}/game_date_pacific={day}/game_id={gid}/latest.json"

    def _read(self, key):
        try:
            response = boto3.client("s3").get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None, None
            raise
        with closing(response["Body"]) as body:
            return body.read(), response["ETag"]

    def selection(self, day, gid):
        raw, etag = self._read(self.key(day, gid))
        return (json.loads(raw) if raw is not None else None), etag

    def source_identity(self):
        try:
            return boto3.client("s3").head_object(Bucket=self.bucket, Key=self.raw_root()+"/latest.json")["ETag"]
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return None
            raise

    def snapshot(self, key, expected=None):
        prefix = self.raw_root()+"/snapshots/"
        if not key.startswith(prefix):
            raise ValueError("Expected an immutable canonical Odds API snapshot.")
        stamp = datetime.strptime(key.removeprefix(prefix), "%Y%m%dT%H%M%SZ.json").replace(tzinfo=UTC)
        raw, _ = self._read(key)
        if raw is None:
            raise FileNotFoundError(f"Missing archived capture: {key}.")
        digest = sha256(raw).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError("Pinned odds snapshot checksum mismatch.")
        record = json.loads(raw)
        collected = instant(record["collected_at"])
        if (record.get("provider") != "The Odds API" or record.get("season") != self.season
                or record.get("sport_key") != SPORT_KEY or not isinstance(record.get("events"), list)
                or collected.astimezone(UTC).replace(microsecond=0) != stamp):
            raise ValueError("Invalid archived odds source, season or capture timestamp.")
        return record, {"uri": f"s3://{self.bucket}/{key}", "sha256": digest, "collected_at": record["collected_at"]}

    def snapshots(self, day, before):
        # The collector requests games through seven dates ahead. List only the
        # possible capture days; consume bodies newest first until all games match.
        root = self.raw_root()+"/snapshots/"
        first = datetime.combine(day-timedelta(days=7), time.min, PACIFIC).astimezone(UTC).date()
        last = before.astimezone(UTC).date()
        client, keys = boto3.client("s3"), []
        for offset in range((last-first).days+1):
            for page in client.get_paginator("list_objects_v2").paginate(
                    Bucket=self.bucket, Prefix=root+(first+timedelta(days=offset)).strftime("%Y%m%d")):
                for row in page.get("Contents", []):
                    key = row["Key"]
                    stamp = datetime.strptime(key.removeprefix(root), "%Y%m%dT%H%M%SZ.json").replace(tzinfo=UTC)
                    if stamp < before:
                        keys.append(key)
        for key in sorted(keys, reverse=True):
            yield self.snapshot(key)

    def validate(self, pointer, game, names):
        if (pointer.get("schema_version") != 1 or pointer.get("season") != self.season
                or pointer.get("game_date_pacific") != str(game["game_time_utc"].astimezone(PACIFIC).date())
                or pointer.get("game_id") != game["game_id"]
                or instant(pointer["scheduled_start_utc"]) != game["game_time_utc"]
                or any(pointer[side+"_team_id"] != game[side+"_team_id"] for side in ("home", "away"))):
            raise ValueError("Saved game-odds identity differs from the official schedule.")
        ref = pointer["snapshot"]
        prefix = f"s3://{self.bucket}/"
        if not ref["uri"].startswith(prefix):
            raise ValueError("Pinned snapshot belongs to a different bucket.")
        record, actual = self.snapshot(ref["uri"].removeprefix(prefix), ref["sha256"])
        result = eligible_event(record, game, names)
        if (actual != ref or result is None or result[0]["id"] != pointer["provider_event_id"]
                or result[1] != instant(pointer["cutoff_utc"])):
            raise ValueError("Saved game odds do not identify a valid pregame source.")

    def save(self, pointer, expected_etag):
        key = self.key(pointer["game_date_pacific"], pointer["game_id"])
        raw = json.dumps(pointer, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        condition = {"IfMatch": expected_etag} if expected_etag is not None else {"IfNoneMatch": "*"}
        boto3.client("s3").put_object(Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json", **condition)
        actual, _ = self._read(key)
        if actual != raw:
            raise ValueError("Game-odds pointer changed during publication; rerun this date.")
        return f"s3://{self.bucket}/{key}"

    def save_missing_odds_warning(self, *, game, names, schedule, run_id, occurred_at):
        """Persist an immutable per-run warning after an unquoted game's tipoff."""
        if occurred_at < game["game_time_utc"]:
            raise ValueError("Missing-odds warnings are only published at or after tipoff.")
        gid = game["game_id"]
        day = str(game["game_time_utc"].astimezone(PACIFIC).date())
        event_id = sha256(f"{run_id}:{gid}:missing_pregame_odds_at_tipoff".encode()).hexdigest()
        key = (f"{ERRORS_PREFIX}/pipeline=odds/season={self.season}/"
               f"occurred_date_pacific={occurred_at.astimezone(PACIFIC).date()}/{event_id}.warning.json")
        event = {
            "schema_version": 1, "event_id": event_id, "pipeline": "odds", "stage": "game_odds",
            "status": "warning", "warning_type": "missing_pregame_odds_at_tipoff",
            "season": self.season, "game_date_pacific": day, "game_id": gid,
            "timestamp": occurred_at.isoformat(), "dagster_run_id": run_id,
            "message": "No saved pregame odds were available when the game reached its scheduled start.",
            "details": {"scheduled_start_utc": game["game_time_utc"].isoformat(),
                "home_team": names["home"], "away_team": names["away"],
                "home_team_id": game["home_team_id"], "away_team_id": game["away_team_id"],
                "expected_pointer_uri": f"s3://{self.bucket}/{self.key(day, gid)}"},
            "inputs": {"schedule": schedule, "odds_snapshots_prefix": f"s3://{self.bucket}/{self.raw_root()}/snapshots/"},
            "artifact_path": None,
        }
        raw = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        try:
            boto3.client("s3").put_object(Bucket=self.bucket, Key=key, Body=raw,
                ContentType="application/json", IfNoneMatch="*")
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            prior, _ = self._read(key)
            if prior is None or {k: v for k, v in json.loads(prior).items() if k != "timestamp"} != {
                    k: v for k, v in event.items() if k != "timestamp"}:
                raise ValueError("Saved missing-odds warning differs from this run.") from error
            raw = prior
        actual, _ = self._read(key)
        if actual != raw:
            raise ValueError("Saved missing-odds warning failed read-back verification.")
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}
