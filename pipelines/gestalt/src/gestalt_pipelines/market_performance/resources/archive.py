"""Read immutable Odds API captures and publish versioned daily performance."""
import json
import re
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import AssetKey, ConfigurableResource, DagsterEventType

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC
from ..utils import CONTRACT, daily_totals, instant, json_bytes


class MarketPerformanceArchive(ConfigurableResource):
    season: str
    model_version: str = "v1"
    bucket: str = DATA_BUCKET

    @property
    def prefix(self):
        return f"lake/artifacts/market_performance/season={self.season}/model_version={self.model_version}/"

    def _day_prefix(self, day):
        datetime.strptime(day, "%Y-%m-%d")
        return f"{self.prefix}game_date_pacific={day}/"

    def _read(self, key):
        response = boto3.client("s3").get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as body:
            return body.read(), response["ETag"]

    def snapshot_inventory(self, before):
        root = f"lake/raw/odds-api/season={self.season}/snapshots/"
        result = []
        for page in boto3.client("s3").get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=root):
            for item in page.get("Contents", []):
                name = item["Key"].removeprefix(root)
                if re.fullmatch(r"\d{8}T\d{6}Z.json", name):
                    timestamp = datetime.strptime(name, "%Y%m%dT%H%M%SZ.json").replace(tzinfo=UTC)
                    if timestamp < before:
                        result.append({"key": item["Key"], "etag": item["ETag"]})
        return sorted(result, key=lambda item: item["key"], reverse=True)

    def snapshots(self, inventory):
        for entry in inventory:
            raw, etag = self._read(entry["key"])
            if etag != entry["etag"]:
                raise ValueError("An archived odds snapshot changed during evaluation.")
            record = json.loads(raw)
            if (record.get("provider") != "The Odds API" or record.get("season") != self.season
                    or record.get("sport_key") != "basketball_nba" or not isinstance(record.get("events"), list)
                    or instant(record["collected_at"]).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ.json") != entry["key"].rsplit("/", 1)[-1]):
                raise ValueError("Invalid immutable Odds API snapshot.")
            yield record, {"uri": f"s3://{self.bucket}/{entry['key']}", "sha256": sha256(raw).hexdigest()}

    def selection(self, day):
        prefix = self._day_prefix(day)
        try:
            raw, etag = self._read(prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
                return None, None
            raise
        pointer = json.loads(raw)
        digest = pointer.get("sha256", "")
        if (not re.fullmatch(r"[a-f0-9]{64}", digest) or pointer.get("uri") != f"s3://{self.bucket}/{prefix}{digest}.json"
                or pointer.get("season") != self.season or pointer.get("model_version") != self.model_version
                or pointer.get("game_date_pacific") != day):
            raise ValueError("Invalid accepted market-performance selection.")
        return pointer, etag

    def _validate(self, record):
        games = record["games"]
        if (record.get("schema_version") != 1 or record.get("artifact_kind") != "market_performance"
                or record.get("season") != self.season or record.get("model_version") != self.model_version
                or record.get("calculation_contract") != CONTRACT
                or len(games) != record["coverage"]["scheduled_game_count"]
                or len({game["game_id"] for game in games}) != len(games)
                or record["daily_totals"] != daily_totals(games)):
            raise ValueError("Invalid daily market-performance artifact.")

    def load(self, pointer):
        day, digest = pointer["game_date_pacific"], pointer["sha256"]
        key = self._day_prefix(day)+digest+".json"
        if pointer["uri"] != f"s3://{self.bucket}/{key}":
            raise ValueError("Market performance belongs to a different date or model.")
        raw, _ = self._read(key)
        if sha256(raw).hexdigest() != digest:
            raise ValueError("Market-performance checksum differs from S3.")
        record = json.loads(raw)
        self._validate(record)
        return record

    def _immutable(self, key, raw):
        try:
            boto3.client("s3").put_object(Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
        if self._read(key)[0] != raw:
            raise ValueError("Immutable market-performance read-back differs.")

    def save(self, record, *, expected_etag):
        self._validate(record)
        day = record["game_date_pacific"]
        selected, etag = self.selection(day)
        if etag != expected_etag:
            raise ValueError("Another run changed the market-performance selection.")
        if selected and self.load(selected)["input_fingerprint"] == record["input_fingerprint"]:
            return {"uri": selected["uri"], "sha256": selected["sha256"], "reused": True}
        raw = json_bytes(record)
        digest = sha256(raw).hexdigest()
        prefix = self._day_prefix(day)
        key = prefix+digest+".json"
        self._immutable(key, raw)
        pointer = {"uri": f"s3://{self.bucket}/{key}", "sha256": digest,
                   **{k: record[k] for k in ("schema_version", "season", "model_version", "game_date_pacific", "input_fingerprint")}}
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        boto3.client("s3").put_object(Bucket=self.bucket, Key=prefix+"latest.json", Body=json_bytes(pointer), ContentType="application/json", **condition)
        if self._read(prefix+"latest.json")[0] != json_bytes(pointer):
            raise ValueError("Market-performance selection read-back differs.")
        return {"uri": pointer["uri"], "sha256": digest, "reused": False}

    def save_event(self, *, day, run_id, message, inputs, status="failed", original=None):
        now = datetime.now(UTC)
        identity = sha256(json_bytes([run_id, day, status, original])).hexdigest()
        suffix = "failure" if status == "failed" else "recovery"
        event = {"schema_version": 1, "event_id": identity, "pipeline": "market_performance",
                 "season": self.season, "model_version": self.model_version, "game_date_pacific": day,
                 "dagster_run_id": run_id, "timestamp": now.isoformat(), "status": status,
                 "message": message, "inputs": inputs, "original_failure": original}
        key = (f"{ERRORS_PREFIX}/pipeline=market_performance/season={self.season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{identity}.{suffix}.json")
        try:
            raw, _ = self._read(key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"NoSuchKey", "404", "NotFound"}:
                raise
            raw = json_bytes(event)
            self._immutable(key, raw)
        saved = json.loads(raw)
        if {k: v for k, v in saved.items() if k != "timestamp"} != {k: v for k, v in event.items() if k != "timestamp"}:
            raise ValueError("Saved market-performance event differs from this run.")
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

    def parent_failure(self, context, day):
        parent = context.instance.get_run_by_id(context.run.parent_run_id) if context.run.parent_run_id else None
        if (parent is None or parent.status.value != "FAILURE"
                or (parent.job_name != "evaluate_market_performance" and AssetKey("market_performance_by_date") not in (parent.asset_selection or set()))):
            return None
        selected = parent.tags.get("dagster/partition")
        if selected is None and parent.tags.get("dagster/asset_partition_range_start") == parent.tags.get("dagster/asset_partition_range_end"):
            selected = parent.tags.get("dagster/asset_partition_range_start")
        if selected != day:
            return None
        for entry in reversed(context.instance.get_records_for_run(parent.run_id, of_type=DagsterEventType.STEP_FAILURE).records):
            event = entry.event_log_entry.dagster_event
            if event.step_key != "market_performance_by_date":
                continue
            failure = event.event_specific_data.user_failure_data
            if failure and "market_performance_failure" in failure.metadata:
                ref = failure.metadata["market_performance_failure"].value
                prefix = f"s3://{self.bucket}/{ERRORS_PREFIX}/pipeline=market_performance/season={self.season}/"
                if not ref["uri"].startswith(prefix) or not ref["uri"].endswith(".failure.json"):
                    raise ValueError("Recovery belongs to a different pipeline.")
                raw, _ = self._read(ref["uri"].removeprefix(f"s3://{self.bucket}/"))
                record = json.loads(raw)
                if (sha256(raw).hexdigest() != ref["sha256"] or record["dagster_run_id"] != parent.run_id
                        or record["game_date_pacific"] != day or record["model_version"] != self.model_version or record["status"] != "failed"):
                    raise ValueError("Recovery does not match the failed parent.")
                return ref
        return None
