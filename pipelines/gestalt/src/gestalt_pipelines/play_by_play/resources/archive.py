"""Fetch revised NBA play-by-play and preserve verified source versions."""

import base64
import json
import re
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from dagster import ConfigurableResource

from ..utils import check_download, official_json

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC

CDN_URL = "https://cdn.nba.com/static/json/liveData/playbyplay"
ERROR_PREFIX = f"{ERRORS_PREFIX}/pipeline=play_by_play/"
STAGE = "nba_play_by_play"


def _json_bytes(value: dict | list) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _error_event_key(event: dict) -> str:
    occurred = datetime.fromisoformat(event["timestamp"]).astimezone(PACIFIC).date()
    suffix = {"failed": "failure", "recovered": "recovery"}[event["status"]]
    return (
        f"{ERROR_PREFIX}season={event['season']}/occurred_date_pacific={occurred}/"
        f"{event['event_id']}.{suffix}.json"
    )


class NBAPlayByPlayArchive(ConfigurableResource):
    bucket: str = DATA_BUCKET

    def save_error_event(
        self, *, season: str, day: str, run_id: str, message: str,
        schedule: dict | None, failed_games: dict, original_failure: dict | None = None,
        stage: str = STAGE, inputs: dict | None = None,
    ) -> dict:
        """Persist the same failure/recovery envelope used by injury processing."""
        status = "recovered" if original_failure else "failed"
        identity = [run_id, stage, status, original_failure]
        event = {
            "schema_version": 1,
            "event_id": sha256(_json_bytes(identity)).hexdigest(),
            "stage": stage, "season": season, "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
            "message": message, "dagster_run_id": run_id,
            "game_date_pacific": day,
            "input": inputs if inputs is not None else {"schedule": schedule},
            "artifact_path": (f"s3://{self.bucket}/lake/artifacts/stints/season={season}/game_date_pacific={day}/"
                              if stage == "nba_stints" else f"s3://{self.bucket}/lake/raw/cdn/playbyplay/season={season}/"),
            "failed_games": failed_games,
            "source_urls": {
                game_id: f"{CDN_URL}/playbyplay_{game_id}.json" for game_id in failed_games
            },
        }
        if original_failure:
            event["original_failure"] = original_failure
        key = _error_event_key(event)
        client = boto3.client("s3")
        existing = self._read_optional(client, key)
        if existing is not None:
            prior = official_json(existing)
            if {k: v for k, v in prior.items() if k != "timestamp"} != {
                k: v for k, v in event.items() if k != "timestamp"
            }:
                raise ValueError("Existing play-by-play error event differs from this run.")
            payload = existing
        else:
            payload = _json_bytes(event)
            self._put_immutable(client, key, payload)
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(payload).hexdigest()}

    def read_failure(self, reference: dict, *, season: str, day: str, stage: str = STAGE) -> dict:
        prefix = f"s3://{self.bucket}/{ERROR_PREFIX}season={season}/"
        if not reference["uri"].startswith(prefix):
            raise ValueError("Play-by-play failure is outside the selected season's error path.")
        raw = self._read(boto3.client("s3"), reference["uri"].removeprefix(f"s3://{self.bucket}/"))
        event = official_json(raw)
        if (
            sha256(raw).hexdigest() != reference["sha256"]
            or event.get("schema_version") != 1 or event.get("stage") != stage
            or event.get("status") != "failed" or event.get("season") != season
            or event.get("game_date_pacific") != day
            or reference["uri"] != f"s3://{self.bucket}/{_error_event_key(event)}"
            or event.get("event_id") != sha256(_json_bytes([
                event["dagster_run_id"], stage, "failed", None,
            ])).hexdigest()
        ):
            raise ValueError("Play-by-play failure does not match this partition or checksum.")
        return {**reference, "event_id": event["event_id"]}

    def capture_slate(self, season: str, games: list[dict], *, run_id: str) -> tuple[dict, dict]:
        """Fetch every selected game again so NBA corrections are detected."""
        client = boto3.client("s3")
        references, failures = {}, {}
        for game in games:
            game_id = game["game_id"]
            try:
                latest_key = f"lake/raw/cdn/playbyplay/season={season}/game_id={game_id}/latest.json"
                prior, etag = self._load_latest(client, latest_key, season, game)
                payload, capture = self._fetch(game_id)
                reference = self._archive_version(
                    client, season, game, payload, capture, run_id=run_id,
                    prior_sha256=prior["sha256"] if prior else None,
                )
                pointer = {
                    key: reference[key] for key in ("uri", "sha256", "metadata_uri", "metadata_sha256")
                }
                pointer.update({
                    "season": season, "game_id": game_id,
                    "game_date_pacific": game["game_time_utc"].astimezone(PACIFIC).date().isoformat(),
                    "checked_at": capture["finished_at"], "run_id": run_id,
                })
                encoded = _json_bytes(pointer)
                checksum = sha256(encoded)
                condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
                client.put_object(
                    Bucket=self.bucket, Key=latest_key, Body=encoded, ContentType="application/json",
                    ChecksumAlgorithm="SHA256",
                    ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
                    **condition,
                )
                references[game_id] = {
                    **reference, "latest_uri": f"s3://{self.bucket}/{latest_key}",
                    "revised": bool(prior and prior["sha256"] != reference["sha256"]),
                }
            except (ValueError, requests.RequestException, BotoCoreError, ClientError) as error:
                failures[game_id] = str(error)
        return references, failures

    def _load_latest(self, client, key: str, season: str, game: dict) -> tuple[dict | None, str | None]:
        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None, None
            raise
        with closing(response["Body"]) as body:
            pointer = official_json(body.read())
        digest = pointer.get("sha256")
        if (
            pointer.get("season") != season or pointer.get("game_id") != game["game_id"]
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"Invalid play-by-play selection at s3://{self.bucket}/{key}.")
        raw_key = key.removesuffix("latest.json") + digest + ".json"
        metadata_key = raw_key.removesuffix(".json") + ".meta.json"
        if (
            pointer.get("uri") != f"s3://{self.bucket}/{raw_key}"
            or pointer.get("metadata_uri") != f"s3://{self.bucket}/{metadata_key}"
        ):
            raise ValueError("Play-by-play selection points outside its canonical game prefix.")
        raw = self._read(client, raw_key)
        metadata = self._read(client, metadata_key)
        if sha256(raw).hexdigest() != digest or sha256(metadata).hexdigest() != pointer.get("metadata_sha256"):
            raise ValueError("Accepted play-by-play source or sidecar failed checksum validation.")
        check_download(raw)
        sidecar = official_json(metadata)
        expected = {
            "source": "nba_cdn_live_data", "endpoint": "playbyplay", "status": "succeeded",
            "game_id": game["game_id"], "season": season, "sha256": digest,
            "byte_count": len(raw), "raw_uri": pointer["uri"],
        }
        if any(sidecar.get(field) != value for field, value in expected.items()):
            raise ValueError("Accepted play-by-play sidecar does not match its source.")
        return pointer, response["ETag"]

    def _fetch(self, game_id: str) -> tuple[bytes, dict]:
        if not re.fullmatch(r"[0-9]{10}", game_id):
            raise ValueError("Expected a 10-digit NBA game ID.")
        url = f"{CDN_URL}/playbyplay_{game_id}.json"
        started = datetime.now(UTC)
        with requests.get(
            url, headers={"User-Agent": "nba-gestalt/1.0", "Referer": "https://www.nba.com/"},
            timeout=30, allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(f"Unexpected NBA play-by-play response {response.status_code}.")
            payload, headers = response.content, dict(response.headers)
        finished = datetime.now(UTC)
        return payload, {
            "source_url": url, "response_headers": headers,
            "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "duration_seconds": max(0.0, (finished - started).total_seconds()),
        }

    def _read(self, client, key: str) -> bytes:
        response = client.get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as body:
            return body.read()

    def _read_optional(self, client, key: str) -> bytes | None:
        try:
            return self._read(client, key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None
            raise

    def _put_immutable(self, client, key: str, payload: bytes) -> bool:
        digest = sha256(payload)
        try:
            client.put_object(
                Bucket=self.bucket, Key=key, Body=payload, ContentType="application/json",
                IfNoneMatch="*", ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(digest.digest()).decode("ascii"),
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            if self._read(client, key) != payload:
                raise ValueError(f"Archived bytes differ at s3://{self.bucket}/{key}.") from error
            return False
        return True

    def _archive_version(
        self, client, season: str, game: dict, payload: bytes, capture: dict,
        *, run_id: str, prior_sha256: str | None,
    ) -> dict:
        check_download(payload)
        digest = sha256(payload).hexdigest()
        key = f"lake/raw/cdn/playbyplay/season={season}/game_id={game['game_id']}/{digest}.json"
        uri = f"s3://{self.bucket}/{key}"
        existing = self._read_optional(client, key)
        if existing is not None:
            if existing != payload:
                raise ValueError(f"Archived play-by-play failed checksum verification at {uri}.")
            raw_created = False
        else:
            raw_created = self._put_immutable(client, key, payload)

        metadata_key = key.removesuffix(".json") + ".meta.json"
        saved = self._read_optional(client, metadata_key)
        metadata_created = saved is None
        if saved is None:
            metadata = {
                "schema_version": 1, "source": "nba_cdn_live_data", "endpoint": "playbyplay",
                "run_id": run_id, "attempt_number": 1, "status": "succeeded",
                "game_id": game["game_id"], "season": season,
                "season_type": {
                    "001": "preseason", "002": "regular", "004": "playoffs",
                    "005": "play_in", "006": "nba_cup_final",
                }.get(game["game_id"][:3], "unknown"),
                "game_date": game["game_time_utc"].astimezone(PACIFIC).date().isoformat(),
                "home_team": game["home_team_tricode"], "away_team": game["away_team_tricode"],
                "sha256": digest, "byte_count": len(payload), "raw_uri": uri,
                "prior_sha256": prior_sha256, "error_type": None, "error_message": None,
                **capture,
            }
            encoded = _json_bytes(metadata)
            checksum = sha256(encoded)
            try:
                client.put_object(
                    Bucket=self.bucket, Key=metadata_key, Body=encoded, ContentType="application/json",
                    IfNoneMatch="*", ChecksumAlgorithm="SHA256",
                    ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
                )
                saved = encoded
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                    raise
                saved = self._read(client, metadata_key)
                metadata_created = False

        try:
            metadata = official_json(saved)
        except ValueError as error:
            raise ValueError(f"Invalid play-by-play sidecar at s3://{self.bucket}/{metadata_key}: {error}") from error
        expected = {
            "source": "nba_cdn_live_data", "endpoint": "playbyplay", "status": "succeeded",
            "game_id": game["game_id"], "season": season, "sha256": digest,
            "byte_count": len(payload), "raw_uri": uri,
            "source_url": f"{CDN_URL}/playbyplay_{game['game_id']}.json",
            "home_team": game["home_team_tricode"], "away_team": game["away_team_tricode"],
        }
        if any(metadata.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Play-by-play sidecar differs from its NBA source at s3://{self.bucket}/{metadata_key}.")
        return {
            "uri": uri, "sha256": digest,
            "metadata_uri": f"s3://{self.bucket}/{metadata_key}",
            "metadata_sha256": sha256(saved).hexdigest(),
            "reused": not raw_created, "metadata_reused": not metadata_created,
        }

    def read_game(self, season: str, game: dict) -> tuple[bytes, dict]:
        """Read the current verified capture without fetching or modifying sources."""
        client = boto3.client("s3")
        key = f"lake/raw/cdn/playbyplay/season={season}/game_id={game['game_id']}/latest.json"
        pointer, _ = self._load_latest(client, key, season, game)
        if pointer is None:
            raise FileNotFoundError(f"No saved play-by-play selection for {game['game_id']}.")
        fields = ("uri", "sha256", "metadata_uri", "metadata_sha256")
        reference = {k: pointer[k] for k in fields}
        raw = self._read(client, reference["uri"].removeprefix(f"s3://{self.bucket}/"))
        if sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Selected play-by-play changed during reading.")
        return raw, reference

    def stints_pointer(self, season: str, day: str) -> tuple[dict | None, str | None]:
        client = boto3.client("s3")
        prefix = f"lake/artifacts/stints/season={season}/game_date_pacific={day}/"
        try:
            response = client.get_object(Bucket=self.bucket, Key=prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None, None
            raise
        with closing(response["Body"]) as body:
            pointer = official_json(body.read())
        digest = pointer.get("sha256")
        if (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or pointer.get("uri") != f"s3://{self.bucket}/{prefix}{digest}.json"
            or pointer.get("season") != season or pointer.get("game_date_pacific") != day
        ):
            raise ValueError("Invalid accepted stints selection.")
        raw = self._read(client, prefix+digest+".json")
        record = official_json(raw)
        if (
            sha256(raw).hexdigest() != digest or record.get("season") != season
            or record.get("game_date_pacific") != day
            or record.get("validation", {}).get("complete") is not True
        ):
            raise ValueError("Accepted stints failed checksum or coverage validation.")
        return pointer, response["ETag"]

    def save_stints(self, record: dict, *, prior: dict | None, expected_etag: str | None) -> dict:
        if record["validation"]["complete"] is not True:
            raise ValueError("Cannot accept incomplete stints.")
        raw = _json_bytes(record)
        digest = sha256(raw).hexdigest()
        season, day = record["season"], record["game_date_pacific"]
        prefix = f"lake/artifacts/stints/season={season}/game_date_pacific={day}/"
        client = boto3.client("s3")
        created = self._put_immutable(client, prefix+digest+".json", raw)
        pointer = {
            "schema_version": 1, "season": season, "game_date_pacific": day,
            "processing_version": record["processing_version"],
            "uri": f"s3://{self.bucket}/{prefix}{digest}.json", "sha256": digest,
        }
        if pointer != prior:
            condition = {"IfMatch": expected_etag} if expected_etag else {"IfNoneMatch": "*"}
            client.put_object(
                Bucket=self.bucket, Key=prefix+"latest.json", Body=_json_bytes(pointer),
                ContentType="application/json", **condition,
            )
        return {**pointer, "reused": not created}
