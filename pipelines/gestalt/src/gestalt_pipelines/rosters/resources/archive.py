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

from ..utils import NBA_TEAM_IDS, nba_id, official_json, opening_ids

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC

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

    def transactions(self, season: str, first: date, last: date) -> tuple[list, dict]:
        days = [first + timedelta(days=i) for i in range((last-first).days+1)]
        keys = [f"lake/raw/nba/transactions/season={season}/game_date_pacific={day}/latest.json" for day in days]
        saved = self._read_many(boto3.client("s3"), keys)
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

    def repair_transactions(self, season: str) -> tuple[list[dict], dict | None]:
        """Read the operator's dated membership repairs on every roster run."""
        key = f"{ROOT}/season={season}/repair_transactions.json"
        try:
            raw, reference = self._read(boto3.client("s3"), key)
        except FileNotFoundError:
            return [], None
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError("Roster repair transactions must be a JSON list.")
        return rows, reference

    def save_roster_size_warning(
        self, *, record: dict, artifact: dict, counts: dict,
        run_id: str, occurred_at: datetime,
    ) -> dict:
        event_id = sha256(f"{run_id}:roster_size".encode()).hexdigest()
        occurred = occurred_at.astimezone(PACIFIC).date()
        key = (
            f"{ERRORS_PREFIX}/pipeline=rosters/season={record['season']}/"
            f"occurred_date_pacific={occurred}/{event_id}.warning.json"
        )
        event = {
            "schema_version": 1, "event_id": event_id, "pipeline": "rosters",
            "stage": "rosters_by_date", "season": record["season"], "status": "warning",
            "timestamp": occurred_at.isoformat(), "dagster_run_id": run_id,
            "game_date_pacific": record["game_date_pacific"],
            "message": "One or more NBA rosters exceed the 18-player warning threshold.",
            "details": {"warning_threshold": 18, "team_player_counts": counts,
                        "team_player_ids": {team: record["teams"][team] for team in counts}},
            "inputs": record["inputs"], "artifact_path": artifact["uri"],
        }
        raw = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        official_json(raw)
        client = boto3.client("s3")
        try:
            client.put_object(Bucket=self.bucket, Key=key, Body=raw,
                              ContentType="application/json", IfNoneMatch="*")
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            raw, _ = self._read(client, key)
            if {k: v for k, v in official_json(raw).items() if k != "timestamp"} != {
                k: v for k, v in event.items() if k != "timestamp"
            }:
                raise ValueError("Existing roster warning differs from this run.") from error
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

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

    def load(self, season: str, day: str) -> tuple[dict, dict]:
        """Read this date's accepted roster; never substitute another date."""
        reference, _ = self.selection(season, day)
        if reference is None:
            raise FileNotFoundError(
                f"Build rosters for {day} first: missing s3://{self.bucket}/{ROOT}/"
                f"season={season}/game_date_pacific={day}/latest.json."
            )
        key = reference["uri"].removeprefix(f"s3://{self.bucket}/")
        raw, actual = self._read(boto3.client("s3"), key)
        record = official_json(raw)
        if (actual["sha256"] != reference["sha256"] or record["season"] != season
                or record["game_date_pacific"] != day or record["schema_version"] != 1):
            raise ValueError("Accepted roster bytes do not match the selected season/date/checksum.")
        return record, reference

    def latest_selection(self, season: str, through_date: date) -> tuple[str, dict]:
        """Read the newest accepted dated snapshot on or before the requested date."""
        prefix = f"{ROOT}/season={season}/"
        days = set()
        for page in boto3.client("s3").get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"].removeprefix(prefix)
                match = re.fullmatch(r"game_date_pacific=([0-9-]+)/latest\.json", name)
                if match and date.fromisoformat(match[1]) <= through_date:
                    days.add(match[1])
        if not days:
            raise FileNotFoundError("No accepted roster snapshot is available on or before the requested date.")
        day = max(days)
        ref, _ = self.selection(season, day)
        if ref is None:
            raise FileNotFoundError("The selected roster pointer disappeared.")
        return day, ref

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
