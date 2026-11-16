"""Reuse or archive official final NBA CDN box scores by game ID."""

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

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC

CDN_URL = "https://cdn.nba.com/static/json/liveData/boxscore"
CHECKSUM_FILENAME = re.compile(r"[0-9a-f]{64}\.json")


def _official_fields(pairs: list[tuple]) -> dict:
    for key, value in pairs:
        field = key.casefold()
        if "espn" in field or (
            field in {"source", "source_url", "provider"}
            and isinstance(value, str)
            and "espn" in value.casefold()
        ):
            raise ValueError("Prohibited basketball source in NBA box score.")
    return dict(pairs)


def _validate_box(payload: bytes, scheduled: dict, *, allow_pending: bool = False) -> bool:
    document = json.loads(payload, object_pairs_hook=_official_fields)
    game = document.get("game") if isinstance(document, dict) else None
    if not isinstance(game, dict) or str(game.get("gameId")) != scheduled["game_id"]:
        raise ValueError("Box score game ID differs from the official schedule.")
    status = str(game.get("gameStatus"))
    pending = allow_pending and status in {"1", "2"}
    if status != "3" and not pending:
        raise ValueError(f"NBA box score is not final (game status {status}).")

    for side in ("home", "away"):
        team = game.get(f"{side}Team")
        if (
            not isinstance(team, dict)
            or str(team.get("teamId")) != str(scheduled[f"{side}_team_id"])
            or team.get("teamTricode") != scheduled[f"{side}_team_tricode"]
        ):
            raise ValueError(f"Box score {side} team differs from the official schedule.")
        if pending:
            continue
        players = team.get("players")
        if not isinstance(team.get("score"), (int, float)) or not isinstance(players, list) or not players:
            raise ValueError(f"Final box score is missing the {side} score or players.")
    return not pending


def _roster_hash(team: dict) -> str:
    """Hash NBA player identities, independent of row order or game statistics."""
    roster = []
    seen = set()
    for player in team["players"]:
        if not isinstance(player, dict):
            raise ValueError(f"Invalid roster row for team {team['teamId']}.")
        player_id = player.get("personId")
        name = player.get("name")
        if (
            type(player_id) is not int or player_id <= 0 or player_id in seen
            or not isinstance(name, str) or not name.strip()
        ):
            raise ValueError(f"Invalid or duplicate roster identity for team {team['teamId']}.")
        seen.add(player_id)
        roster.append({"player_id": player_id, "player_name": name.strip()})
    roster.sort(key=lambda row: row["player_id"])
    return sha256(_json_bytes(roster)).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class NBABoxScoreArchive(ConfigurableResource):
    bucket: str = DATA_BUCKET

    def save_event(self, *, season: str, day: str, run_id: str, status: str,
                   message: str, schedule: dict, details: dict, game_id: str | None = None) -> dict:
        """Save a source-refresh event using the shared warning/failure envelope."""
        if status not in {"failed", "warning"}:
            raise ValueError("Unsupported box-score event status.")
        now = datetime.now(UTC)
        event_id = sha256(_json_bytes([run_id, "nba_box_scores", status, game_id])).hexdigest()
        event = {
            "schema_version": 1, "event_id": event_id, "stage": "nba_box_scores",
            "season": season, "game_date_pacific": day, "game_id": game_id,
            "status": status, "timestamp": now.isoformat(), "dagster_run_id": run_id,
            "message": message, "input": {"schedule": schedule}, "details": details,
            "artifact_path": f"s3://{self.bucket}/lake/raw/cdn/boxscore/season={season}/",
            "source_urls": {gid: f"{CDN_URL}/boxscore_{gid}.json"
                            for gid in ([game_id] if game_id else details.get("failed_games", {}))},
        }
        suffix = "failure" if status == "failed" else "warning"
        key = (f"{ERRORS_PREFIX}/pipeline=box_scores/season={season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{event_id}.{suffix}.json")
        payload = _json_bytes(event)
        client = boto3.client("s3")
        try:
            client.put_object(Bucket=self.bucket, Key=key, Body=payload, ContentType="application/json",
                IfNoneMatch="*", ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(sha256(payload).digest()).decode("ascii"))
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            saved = self._read(client, key)
            prior = json.loads(saved)
            if {k:v for k,v in prior.items() if k != "timestamp"} != {k:v for k,v in event.items() if k != "timestamp"}:
                raise ValueError("Saved box-score event differs from this run.") from error
            payload = saved
        if self._read(client, key) != payload:
            raise ValueError("Saved box-score event failed read-back verification.")
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(payload).hexdigest()}

    def capture_slate(self, season: str, games: list[dict], *, allow_pending: bool, refresh_existing: bool = False) -> tuple[dict, dict]:
        """Archive each game independently and report any missing coverage."""
        client = boto3.client("s3")
        references, failures = {}, {}
        for game in games:
            game_id = game["game_id"]
            try:
                reference = self._capture_game(client, season, game, allow_pending=allow_pending, refresh_existing=refresh_existing)
                if reference is not None:
                    references[game_id] = reference
            except (ValueError, requests.RequestException, BotoCoreError, ClientError) as error:
                failures[game_id] = str(error)
        return references, failures

    def _select_saved(self, client, prefix: str):
        candidates = []
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                name = item["Key"].removeprefix(prefix)
                if "espn" in name.casefold():
                    raise ValueError("Prohibited source filename in NBA box-score archive.")
                if CHECKSUM_FILENAME.fullmatch(name):
                    candidates.append(item)
        return max(candidates, key=lambda item: (item["LastModified"], item["Key"]), default=None)


    def read_saved(self, season: str, game: dict, *, allow_missing: bool = False) -> tuple[bytes, dict] | None:
        """Read a saved final box and sidecar, without downloading or writing."""
        if not re.fullmatch(r"[0-9]{10}", game["game_id"]):
            raise ValueError("Expected a 10-digit NBA game ID.")
        client = boto3.client("s3")
        prefix = f"lake/raw/cdn/boxscore/season={season}/game_id={game['game_id']}/"
        selected = self._select_saved(client, prefix)
        if selected is None:
            if allow_missing:
                return None
            raise FileNotFoundError(f"No saved final box score for {game['game_id']}.")
        key = selected["Key"]
        digest = key.removeprefix(prefix).removesuffix(".json")
        raw = self._read(client, key)
        if sha256(raw).hexdigest() != digest:
            raise ValueError("Saved box score failed checksum verification.")
        _validate_box(raw, game)
        metadata = self._ensure_metadata(client, key, season, raw, digest, create=False)
        return raw, {
            "uri": f"s3://{self.bucket}/{key}", "sha256": digest,
            **{k:metadata[k] for k in ("metadata_uri", "metadata_sha256")},
        }

    def _capture_game(self, client, season: str, game: dict, *, allow_pending: bool, refresh_existing: bool = False) -> dict | None:
        game_id = game["game_id"]
        if not re.fullmatch(r"[0-9]{10}", game_id):
            raise ValueError("Expected a 10-digit NBA game ID.")
        prefix = f"lake/raw/cdn/boxscore/season={season}/game_id={game_id}/"
        url = f"{CDN_URL}/boxscore_{game_id}.json"
        saved = self._select_saved(client, prefix)
        reused = saved is not None
        previous = None
        if saved:
            key = saved["Key"]
            digest = key.removeprefix(prefix).removesuffix(".json")
            payload = self._read(client, key)
            if sha256(payload).hexdigest() != digest:
                raise ValueError(f"Archived box score failed checksum validation at s3://{self.bucket}/{key}.")
            _validate_box(payload, game)
            previous = {"uri": f"s3://{self.bucket}/{key}", "sha256": digest}
        if saved is None or refresh_existing:
            with requests.get(
                url, headers={"User-Agent": "nba-gestalt/1.0", "Referer": "https://www.nba.com/"},
                timeout=30, allow_redirects=False,
            ) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f"Unexpected NBA CDN box-score status {response.status_code}.")
                payload = response.content
            if not _validate_box(payload, game, allow_pending=allow_pending):
                return None

            downloaded = json.loads(payload)["game"]
            for side in ("homeTeam", "awayTeam"):
                _roster_hash(downloaded[side])
            checksum = sha256(payload)
            digest = checksum.hexdigest()
            key = f"{prefix}{digest}.json"
            if saved is not None and key == saved["Key"]:
                metadata = self._ensure_metadata(client, key, season, payload, digest)
                return {"uri": f"s3://{self.bucket}/{key}", "sha256": digest,
                        "source_url": url, "reused": True, **metadata}
            reused = False
            try:
                client.put_object(
                    Bucket=self.bucket, Key=key, Body=payload,
                    ContentType="application/json", IfNoneMatch="*",
                    ChecksumAlgorithm="SHA256",
                    ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
                )
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                    raise
                if self._read(client, key) != payload:
                    raise ValueError(f"Different bytes already exist at s3://{self.bucket}/{key}.") from error
                reused = True
            if self._read(client, key) != payload:
                raise ValueError("Saved NBA box-score bytes differ from the downloaded response.")
            if refresh_existing and self._select_saved(client, prefix)["Key"] != key:
                raise ValueError("NBA response matches an older archived version; the latest saved selection differs.")

        metadata = self._ensure_metadata(client, key, season, payload, digest)
        return {
            "uri": f"s3://{self.bucket}/{key}", "sha256": digest, "source_url": url, "reused": reused,
            **metadata,
            **({"previous_box_score": previous} if previous and previous["sha256"] != digest else {}),
        }

    def _ensure_metadata(self, client, key: str, season: str, payload: bytes, digest: str, *, create: bool = True) -> dict:
        game = json.loads(payload, object_pairs_hook=_official_fields)["game"]
        metadata = {
            "season": season, "game_id": game["gameId"], "box_score_sha256": digest,
            "home_team_id": game["homeTeam"]["teamId"],
            "home_roster_sha256": _roster_hash(game["homeTeam"]),
            "away_team_id": game["awayTeam"]["teamId"],
            "away_roster_sha256": _roster_hash(game["awayTeam"]),
        }
        metadata_key = key.removesuffix(".json") + ".meta.json"
        encoded = _json_bytes(metadata)
        reused = True
        try:
            saved = self._read(client, metadata_key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
                raise
            if not create:
                raise FileNotFoundError(f"Missing saved box-score metadata at s3://{self.bucket}/{metadata_key}.") from error
            checksum = sha256(encoded)
            try:
                client.put_object(
                    Bucket=self.bucket, Key=metadata_key, Body=encoded,
                    ContentType="application/json", IfNoneMatch="*",
                    ChecksumAlgorithm="SHA256",
                    ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
                )
                saved = encoded
                reused = False
            except ClientError as conflict:
                if conflict.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                    raise
                saved = self._read(client, metadata_key)

        if json.loads(saved, object_pairs_hook=_official_fields) != metadata:
            raise ValueError(f"Roster metadata differs from its box score at s3://{self.bucket}/{metadata_key}.")
        return {
            "metadata_uri": f"s3://{self.bucket}/{metadata_key}",
            "metadata_sha256": sha256(saved).hexdigest(), "metadata_reused": reused,
        }

    def _read(self, client, key: str) -> bytes:
        response = client.get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as body:
            return body.read()
