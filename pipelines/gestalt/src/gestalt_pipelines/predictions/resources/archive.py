"""Immutable game predictions, accepted selections, and failure/recovery records."""

import json
import re
from contextlib import closing
from datetime import UTC, date, datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import AssetKey, ConfigurableResource, DagsterEventType

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC
from ..utils import MODEL_VERSION, PredictionError, json_bytes, validate_prediction


class GamePredictionArchive(ConfigurableResource):
    season: str
    model_version: str = MODEL_VERSION
    bucket: str = DATA_BUCKET

    @property
    def prefix(self) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}", self.season) or not re.fullmatch(r"[a-zA-Z0-9_-]+", self.model_version):
            raise PredictionError("Invalid prediction season or version.")
        return f"lake/artifacts/game_predictions/season={self.season}/model_version={self.model_version}/"

    def _game_prefix(self, day: str, game_id: str) -> str:
        if date.fromisoformat(day).isoformat() != day or not re.fullmatch(r"\d{10}", game_id):
            raise PredictionError("Invalid prediction game/date identity.")
        return self.prefix+f"game_date_pacific={day}/game_id={game_id}/"

    def _read(self, key: str) -> tuple[bytes, str]:
        response = boto3.client("s3").get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as body:
            return body.read(), response["ETag"]

    def _immutable(self, key: str, raw: bytes) -> None:
        try:
            boto3.client("s3").put_object(Bucket=self.bucket, Key=key, Body=raw,
                                          ContentType="application/json", IfNoneMatch="*")
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
        if self._read(key)[0] != raw:
            raise PredictionError("Prediction S3 read-back differs from the immutable bytes.")

    def selection(self, day: str, game_id: str) -> tuple[dict | None, str | None]:
        prefix = self._game_prefix(day, game_id)
        try:
            raw, etag = self._read(prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None
            raise
        pointer = json.loads(raw)
        digest = pointer.get("sha256", "")
        if (not re.fullmatch(r"[a-f0-9]{64}", digest)
                or pointer.get("uri") != f"s3://{self.bucket}/{prefix}{digest}.json"
                or pointer.get("season") != self.season or pointer.get("model_version") != self.model_version
                or pointer.get("game_date_pacific") != day or pointer.get("game_id") != game_id):
            raise PredictionError("Invalid accepted prediction pointer.")
        return pointer, etag

    def load(self, pointer: dict) -> dict:
        raw, _ = self._read(pointer["uri"].removeprefix(f"s3://{self.bucket}/"))
        if sha256(raw).hexdigest() != pointer["sha256"]:
            raise PredictionError("Accepted prediction checksum differs from S3.")
        record = json.loads(raw)
        validate_prediction(record, self.season, self.model_version)
        prefix = self._game_prefix(record["game_date_pacific"], record["game_id"])
        if pointer["uri"] != f"s3://{self.bucket}/{prefix}{pointer['sha256']}.json":
            raise PredictionError("Selected prediction has a different game identity.")
        return record

    def save(self, record: dict, *, expected_etag: str | None) -> dict:
        validate_prediction(record, self.season, self.model_version)
        day, gid = record["game_date_pacific"], record["game_id"]
        selected, etag = self.selection(day, gid)
        accepted = self.load(selected) if selected else None
        if accepted and accepted["purpose"] == "pregame_projection" and (
                record["purpose"] == "historical_replay"
                or datetime.now(UTC) >= datetime.fromisoformat(accepted["scheduled_start_utc"])):
            return {"frozen": True}
        if record["purpose"] == "pregame_projection" and datetime.now(UTC) >= datetime.fromisoformat(record["scheduled_start_utc"]):
            return {"frozen": True}
        if etag != expected_etag:
            raise PredictionError("Another run changed the accepted prediction.")
        if accepted and accepted["input_fingerprint"] == record["input_fingerprint"]:
            return {"uri": selected["uri"], "sha256": selected["sha256"], "reused": True, "frozen": False}
        raw = json_bytes(record)
        digest = sha256(raw).hexdigest()
        prefix = self._game_prefix(day, gid)
        key = prefix+digest+".json"
        self._immutable(key, raw)
        # The calculation may cross tipoff while S3 is writing.
        if record["purpose"] == "pregame_projection" and datetime.now(UTC) >= datetime.fromisoformat(record["scheduled_start_utc"]):
            return {"frozen": True}
        pointer = {"uri": f"s3://{self.bucket}/{key}", "sha256": digest,
                   **{k: record[k] for k in ("schema_version", "season", "model_version", "game_date_pacific",
                                              "game_id", "scheduled_start_utc", "purpose", "input_fingerprint")}}
        condition = {"IfMatch": expected_etag} if expected_etag else {"IfNoneMatch": "*"}
        boto3.client("s3").put_object(Bucket=self.bucket, Key=prefix+"latest.json", Body=json_bytes(pointer),
                                      ContentType="application/json", **condition)
        if self._read(prefix+"latest.json")[0] != json_bytes(pointer):
            raise PredictionError("Prediction selection read-back differs from this publication.")
        return {"uri": pointer["uri"], "sha256": digest, "reused": False, "frozen": False}

    def save_event(self, *, day: str, run_id: str, message: str, inputs: dict,
                   status="failed", game_id: str | None = None, details: dict | None = None,
                   original: dict | None = None) -> dict:
        suffix = {"failed": "failure", "recovered": "recovery", "warning": "warning"}[status]
        identity = [run_id, day, game_id, status, details or {}, original]
        now = datetime.now(UTC)
        event = {"schema_version": 1, "event_id": sha256(json_bytes(identity)).hexdigest(),
                 "pipeline": "predictions", "stage": "game_predictions", "season": self.season,
                 "model_version": self.model_version, "status": status, "timestamp": now.isoformat(),
                 "game_date_pacific": day, "game_id": game_id, "dagster_run_id": run_id,
                 "message": message, "input": inputs, "details": details or {},
                 "artifact_path": f"s3://{self.bucket}/{self.prefix}game_date_pacific={day}/"}
        if original:
            event["original_failure"] = original
        key = (f"{ERRORS_PREFIX}/pipeline=predictions/season={self.season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{event['event_id']}.{suffix}.json")
        try:
            raw, _ = self._read(key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
            raw = json_bytes(event)
            self._immutable(key, raw)
        saved = json.loads(raw)
        if {k: v for k, v in saved.items() if k != "timestamp"} != {k: v for k, v in event.items() if k != "timestamp"}:
            raise PredictionError("Saved prediction event differs from this run.")
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

    def parent_failure(self, context, day: str) -> dict | None:
        run = context.instance.get_run_by_id(context.run.parent_run_id) if context.run.parent_run_id else None
        if (run is None or run.status.value != "FAILURE"
                or (run.job_name != "predict_game_margins" and AssetKey("game_predictions") not in (run.asset_selection or set()))):
            return None
        selected_day = run.tags.get("dagster/partition")
        if selected_day is None:
            first = run.tags.get("dagster/asset_partition_range_start")
            last = run.tags.get("dagster/asset_partition_range_end")
            selected_day = first if first == last else None
        if selected_day != day:
            return None
        events = context.instance.get_records_for_run(run.run_id, of_type=DagsterEventType.STEP_FAILURE).records
        for entry in reversed(events):
            event = entry.event_log_entry.dagster_event
            if event.step_key != "game_predictions":
                continue
            data = event.event_specific_data.user_failure_data
            if data and "prediction_failure" in data.metadata:
                ref = data.metadata["prediction_failure"].value
                prefix = f"s3://{self.bucket}/{ERRORS_PREFIX}/pipeline=predictions/season={self.season}/"
                if not ref["uri"].startswith(prefix) or not ref["uri"].endswith(".failure.json"):
                    raise PredictionError("Recovery refers to a different pipeline.")
                raw, _ = self._read(ref["uri"].removeprefix(f"s3://{self.bucket}/"))
                failure = json.loads(raw)
                if (sha256(raw).hexdigest() != ref["sha256"] or failure["dagster_run_id"] != run.run_id
                        or failure["game_date_pacific"] != day or failure["status"] != "failed"
                        or failure["model_version"] != self.model_version):
                    raise PredictionError("Recovery failure identity differs from its parent.")
                return ref
        return None
