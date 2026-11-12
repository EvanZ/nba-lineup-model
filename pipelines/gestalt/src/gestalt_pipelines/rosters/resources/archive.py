"""Read canonical NBA evidence from S3 and publish complete roster versions."""

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timedelta
from hashlib import sha256
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource

from ..utils import NBA_TEAM_IDS, box_roster_ids, nba_id, official_json, opening_ids

from ...utils import DATA_BUCKET, PACIFIC

ROOT = "lake/artifacts/rosters"


class NBARosterArchive(ConfigurableResource):
    bucket: str = DATA_BUCKET

    def _read(self, client, key: str) -> tuple[bytes, dict]:
        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                raise FileNotFoundError(f"Missing roster input s3://{self.bucket}/{key}.") from error
            raise
        with closing(response["Body"]) as body:
            raw = body.read()
        digest = sha256(raw).hexdigest()
        if response.get("Metadata", {}).get("sha256", digest) != digest:
            raise ValueError(f"Roster input failed checksum verification: s3://{self.bucket}/{key}.")
        return raw, {
            "uri": f"s3://{self.bucket}/{key}", "sha256": digest,
            "archived_at": response["LastModified"].isoformat(),
        }

    def _read_many(self, client, keys: list[str]) -> dict:
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(lambda key: self._read(client, key), keys))
        return dict(zip(keys, values))

    def opening_rosters(self, season: str) -> dict:
        client = boto3.client("s3")
        prefix = f"lake/raw/commonteamroster/{season}/"
        keys = [prefix + str(team) + suffix for team in NBA_TEAM_IDS for suffix in (".json", ".meta.json")]
        saved = self._read_many(client, keys)
        teams, references, dates = {}, {}, set()
        for team in NBA_TEAM_IDS:
            raw, raw_ref = saved[prefix + str(team) + ".json"]
            meta_raw, meta_ref = saved[prefix + str(team) + ".meta.json"]
            meta = official_json(meta_raw)
            source = urlsplit(meta["url"])
            observed = datetime.fromisoformat(meta["fetched_at"])
            if (
                meta["season"] != season or meta["endpoint"] != "commonteamroster"
                or nba_id(meta["team_id"]) != team or meta["sha256"] != raw_ref["sha256"]
                or observed.tzinfo is None
                or (source.scheme, source.netloc, source.path) != ("https", "stats.nba.com", "/stats/commonteamroster")
            ):
                raise ValueError(f"Invalid opening roster metadata for NBA team {team}.")
            teams[team] = opening_ids(raw, team)
            dates.add(observed.astimezone(PACIFIC).date())
            references[str(team)] = {"metadata": meta_ref, "roster": raw_ref, "observed_at": meta["fetched_at"]}
        if len(dates) != 1:
            raise ValueError("Opening rosters do not describe a single Pacific baseline date.")
        return {"teams": teams, "date": next(iter(dates)), "references": references}

    def transactions(self, client, season: str, first: date, last: date) -> tuple[list, dict]:
        days = [first + timedelta(days=i) for i in range((last-first).days+1)]
        keys = [f"lake/raw/nba/transactions/season={season}/game_date_pacific={day}/latest.json" for day in days]
        saved = self._read_many(client, keys)
        rows, references = [], {}
        for day, key in zip(days, keys):
            raw, reference = saved[key]
            record = official_json(raw)
            if (
                record["season"] != season or record["game_date_pacific"] != day.isoformat()
                or record["source_url"] != "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
                or not isinstance(record["transactions"], list)
            ):
                raise ValueError(f"Invalid transaction evidence for {day}.")
            for row in record["transactions"]:
                if date.fromisoformat(row["TRANSACTION_DATE"][:10]) != day:
                    raise ValueError(f"Transaction lies outside its date partition {day}.")
                if datetime.fromisoformat(row["first_observed_at"]).tzinfo is None:
                    raise ValueError("Transaction lacks a timezone-aware first observation.")
                rows.append(row)
            references[day.isoformat()] = {**reference, "last_polled_at": record["last_polled_at"]}
        return rows, references

    def latest_box_rosters(
        self, client, season: str, games: list[dict], opening: dict, *, today: date,
    ) -> tuple[list, dict, int]:
        prefix = f"lake/raw/cdn/boxscore/season={season}/"
        inventory = [
            item for page in client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
            for item in page.get("Contents", [])
        ]
        keys = {item["Key"] for item in inventory}
        versions = {}
        for item in inventory:
            relative = item["Key"].removeprefix(prefix)
            match = re.fullmatch(r"game_id=([0-9]{10})/([0-9a-f]{64})\.json", relative)
            if match:
                versions.setdefault(match[1], []).append(item)
        saved = {gid: max(items, key=lambda o: (o["LastModified"], o["Key"])) for gid, items in versions.items()}
        latest, missing = {}, []
        for game in sorted(games, key=lambda g: g["game_time_utc"]):
            obj = saved.get(game["game_id"])
            if obj is None:
                if game["game_time_utc"].astimezone(PACIFIC).date() < today:
                    missing.append(game["game_id"])
                continue
            if obj["Key"].removesuffix(".json") + ".meta.json" not in keys:
                missing.append(game["game_id"])
                continue
            for side in ("home", "away"):
                latest[game[side + "_team_id"]] = game
        if missing:
            raise FileNotFoundError(f"Backfill nba_box_scores and its sidecars first. Missing games: {sorted(set(missing))}")

        cache = {}
        selected = {game["game_id"]: game for game in latest.values()}
        evidence, references, bodies_read = [], {}, 0
        for gid, game in sorted(selected.items()):
            obj = saved[gid]
            raw_key = obj["Key"]
            digest = raw_key.rsplit("/", 1)[-1].removesuffix(".json")
            meta_raw, meta_ref = self._read(client, raw_key.removesuffix(".json") + ".meta.json")
            meta = official_json(meta_raw)
            if meta["season"] != season or meta["game_id"] != gid or meta["box_score_sha256"] != digest:
                raise ValueError(f"Box-score sidecar does not match its source for {gid}.")
            needed = []
            for side in ("home", "away"):
                team = game[side + "_team_id"]
                checksum = meta[side + "_roster_sha256"]
                if meta[side + "_team_id"] != team or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                    raise ValueError(f"Invalid box-score team or roster hash for {gid}.")
                if latest[team]["game_id"] == gid:
                    needed.append((team, checksum))
            raw_ref = {
                "uri": f"s3://{self.bucket}/{raw_key}", "sha256": digest,
                "archived_at": obj["LastModified"].isoformat(),
            }
            if any(checksum not in cache for _, checksum in needed):
                raw, raw_ref = self._read(client, raw_key)
                bodies_read += 1
                if raw_ref["sha256"] != digest:
                    raise ValueError(f"Box-score bytes do not match their filename for {gid}.")
                parsed = box_roster_ids(raw, game)
                for side in ("home", "away"):
                    players = parsed[game[side + "_team_id"]]
                    # Treat the upstream producer's roster hash as an opaque cache key.
                    checksum = meta[side + "_roster_sha256"]
                    if checksum in cache and cache[checksum] != players:
                        raise ValueError(f"Box-score metadata hash refers to inconsistent memberships for {gid}.")
                    cache[checksum] = players
            for team, checksum in needed:
                evidence.append({
                    "team_id": team, "player_ids": cache[checksum],
                    "game_time_utc": game["game_time_utc"], "game_id": gid,
                })
            references[gid] = {"box_score": raw_ref, "metadata": meta_ref}
        return evidence, references, bodies_read

    def daily_evidence(self, season: str, day: date, opening: dict, games: list[dict], *, today: date) -> dict:
        client = boto3.client("s3")
        rows, transaction_refs = self.transactions(client, season, opening["date"] + timedelta(days=1), day)
        boxes, box_refs, bodies_read = self.latest_box_rosters(client, season, games, opening, today=today)
        references = {"transactions": transaction_refs, "box_scores": box_refs}
        return {
            "transactions": rows, "boxes": boxes, "box_bodies_read": bodies_read,
            "references": references,
        }

    def selection(self, season: str, day: str) -> tuple[dict | None, str | None]:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", season) or date.fromisoformat(day).isoformat() != day:
            raise ValueError("Invalid roster season or date.")
        prefix = f"{ROOT}/season={season}/game_date_pacific={day}/"
        try:
            response = boto3.client("s3").get_object(Bucket=self.bucket, Key=prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None
            raise
        with closing(response["Body"]) as body:
            pointer = official_json(body.read())
        digest = pointer.get("sha256", "")
        if (
            pointer.get("schema_version") != 1 or pointer.get("season") != season
            or pointer.get("game_date_pacific") != day
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or pointer.get("uri") != f"s3://{self.bucket}/{prefix}{digest}.json"
        ):
            raise ValueError("Invalid accepted roster selection.")
        return {"uri": pointer["uri"], "sha256": digest}, response["ETag"]

    def publish(self, record: dict, *, expected_etag: str | None) -> dict:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        official_json(encoded)
        season, day = record["season"], record["game_date_pacific"]
        if record.get("schema_version") != 1 or set(record["teams"]) != set(map(str, NBA_TEAM_IDS)):
            raise ValueError("A roster must cover the selected date and all 30 NBA teams.")
        players = [nba_id(pid) for ids in record["teams"].values() for pid in ids]
        if len(players) != len(set(players)) or any(not ids for ids in record["teams"].values()):
            raise ValueError("Roster memberships must be unique and each team must be nonempty.")
        selected, current_etag = self.selection(season, day)
        if current_etag != expected_etag:
            raise ValueError("The accepted roster changed during this run; rerun the date.")
        checksum = sha256(encoded)
        key = f"{ROOT}/season={record['season']}/game_date_pacific={record['game_date_pacific']}/{checksum.hexdigest()}.json"
        client = boto3.client("s3")
        reused = False
        try:
            client.put_object(
                Bucket=self.bucket, Key=key, Body=encoded, ContentType="application/json",
                IfNoneMatch="*", ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(checksum.digest()).decode("ascii"),
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            raw, _ = self._read(client, key)
            if raw != encoded:
                raise ValueError(f"Existing roster artifact differs at s3://{self.bucket}/{key}.") from error
            reused = True
        reference = {"uri": f"s3://{self.bucket}/{key}", "sha256": checksum.hexdigest()}
        pointer_key = key.rsplit("/", 1)[0]+"/latest.json"
        if reference != selected:
            pointer = {
                "schema_version": 1, "season": season, "game_date_pacific": day, **reference,
            }
            raw = json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode()
            condition = {"IfMatch": expected_etag} if expected_etag else {"IfNoneMatch": "*"}
            try:
                client.put_object(
                    Bucket=self.bucket, Key=pointer_key, Body=raw,
                    ContentType="application/json", **condition,
                )
            except ClientError as error:
                if error.response["Error"]["Code"] in {"412", "PreconditionFailed", "ConditionalRequestConflict"}:
                    raise ValueError("The accepted roster changed during publication; rerun the date.") from error
                raise
        return {**reference, "reused": reused, "latest_uri": f"s3://{self.bucket}/{pointer_key}"}
