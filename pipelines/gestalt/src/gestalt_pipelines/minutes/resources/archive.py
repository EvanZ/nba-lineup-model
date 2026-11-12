"""Durable versioned minute states and exact canonical NBA input selections."""

import base64
import re
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import AssetKey, ConfigurableResource, DagsterEventType

from ..utils import (
    MODEL_VERSION, PRIOR_REFERENCES, MinutesError, json_bytes, official_json,
    validate_priors, validate_state,
)

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC

ROOT = "lake/artifacts/player_minutes"
ERROR_ROOT = f"{ERRORS_PREFIX}/pipeline=minutes"


class MinutesArchive(ConfigurableResource):
    season: str
    model_version: str = MODEL_VERSION
    bucket: str = DATA_BUCKET
    prior_uri: str = ""

    @property
    def prefix(self) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}", self.season) or not re.fullmatch(r"[a-zA-Z0-9_-]+", self.model_version):
            raise MinutesError("Invalid minutes season or model version.")
        return f"{ROOT}/season={self.season}/model_version={self.model_version}/"

    def _read(self, key: str) -> bytes:
        with closing(boto3.client("s3").get_object(Bucket=self.bucket, Key=key)["Body"]) as stream:
            return stream.read()

    def _optional(self, key: str) -> bytes | None:
        try:
            return self._read(key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
            return None

    def _ref(self, pointer: dict, prefix: str) -> dict:
        digest = pointer.get("sha256")
        uri = pointer.get("uri", "")
        if (
            not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not uri.startswith(f"s3://{self.bucket}/{prefix}")
            or not uri.endswith(".json")
        ):
            raise MinutesError("Invalid S3 source reference.")
        return {"uri": uri, "sha256": digest}

    def read_reference(self, reference: dict, prefix: str) -> dict:
        ref = self._ref(reference, prefix)
        raw = self._read(ref["uri"].removeprefix(f"s3://{self.bucket}/"))
        if sha256(raw).hexdigest() != ref["sha256"]:
            raise MinutesError("S3 input bytes do not match the recorded checksum.", {"input": ref})
        return official_json(raw)

    def _put(self, key: str, raw: bytes) -> bool:
        digest = sha256(raw)
        try:
            boto3.client("s3").put_object(
                Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json",
                IfNoneMatch="*", ChecksumAlgorithm="SHA256",
                ChecksumSHA256=base64.b64encode(digest.digest()).decode("ascii"),
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
            if self._read(key) != raw:
                raise MinutesError("An existing immutable minutes file differs from its contents.")
            return False
        return True

    def save_priors(self, record: dict) -> dict:
        validate_priors(record, self.season, self.model_version)
        raw = json_bytes(record)
        digest = sha256(raw).hexdigest()
        key = self.prefix+f"priors/{digest}.json"
        self._put(key, raw)
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": digest}

    def load_priors(self) -> tuple[dict, dict]:
        uri = self.prior_uri
        if not uri:
            ref = PRIOR_REFERENCES.get((self.season, self.model_version))
            if not ref:
                raise MinutesError("No approved historical minutes input is registered for this season/version.")
            uri = ref["uri"]
        digest = uri.rsplit("/", 1)[-1].removesuffix(".json")
        reference = self._ref({"uri": uri, "sha256": digest}, self.prefix+"priors/")
        if uri != f"s3://{self.bucket}/{self.prefix}priors/{digest}.json":
            raise MinutesError("Starting-prior file is outside the selected model's canonical layout.")
        record = self.read_reference(reference, self.prefix+"priors/")
        validate_priors(record, self.season, self.model_version)
        return record, reference

    def state(self, day: str) -> tuple[dict | None, dict | None, str | None]:
        prefix = self.prefix+f"game_date_pacific={day}/"
        try:
            response = boto3.client("s3").get_object(Bucket=self.bucket, Key=prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None, None
            raise
        with closing(response["Body"]) as stream:
            pointer = official_json(stream.read())
        ref = self._ref(pointer, prefix)
        if (
            pointer.get("model_version") != self.model_version or pointer.get("season") != self.season
            or pointer.get("game_date_pacific") != day
            or ref["uri"] != f"s3://{self.bucket}/{prefix}{ref['sha256']}.json"
        ):
            raise MinutesError("Invalid previous/current minutes selection.")
        record = self.read_reference(ref, prefix)
        validate_state(record, self.season, self.model_version)
        if record["game_date_pacific"] != day:
            raise MinutesError("Selected minutes state has a different date.")
        return record, ref, response["ETag"]

    def _pointer_ref(self, prefix: str) -> dict:
        raw = self._optional(prefix+"latest.json")
        if raw is None:
            raise MinutesError("Missing accepted upstream selection.", {"uri": f"s3://{self.bucket}/{prefix}latest.json"})
        return self._ref(official_json(raw), prefix)

    def _roster_ref(self, day: str) -> dict:
        prefix = f"lake/artifacts/rosters/season={self.season}/game_date_pacific={day}/"
        ref = self._pointer_ref(prefix)
        if ref["uri"] != f"s3://{self.bucket}/{prefix}{ref['sha256']}.json":
            raise MinutesError("Invalid canonical roster reference.")
        return ref

    def load_day(self, day: str) -> tuple[dict, dict]:
        stints_prefix = f"lake/artifacts/stints/season={self.season}/game_date_pacific={day}/"
        injuries_prefix = f"lake/artifacts/injury_status_by_game_date/season={self.season}/game_date_pacific={day}/"
        roster_prefix = f"lake/artifacts/rosters/season={self.season}/game_date_pacific={day}/"
        refs = {
            "stints": self._pointer_ref(stints_prefix),
            "injuries": self._pointer_ref(injuries_prefix), "rosters": self._roster_ref(day),
        }
        if refs["stints"]["uri"] != f"s3://{self.bucket}/{stints_prefix}{refs['stints']['sha256']}.json":
            raise MinutesError("Invalid canonical stints reference.")
        if not re.fullmatch(
            re.escape(f"s3://{self.bucket}/{injuries_prefix}")+r"selection_sha256=[a-f0-9]{64}/statuses\.json",
            refs["injuries"]["uri"],
        ):
            raise MinutesError("Invalid canonical injury status reference.")
        stints = self.read_reference(refs["stints"], stints_prefix)
        injuries = self.read_reference(refs["injuries"], injuries_prefix)
        rosters = self.read_reference(refs["rosters"], roster_prefix)
        if injuries.get("cutoff_source") != "nba_scheduled_start":
            raise MinutesError("Injury inputs must use the approved scheduled-tipoff cutoff.")
        boxes, refs["boxes"] = {}, {}
        for gid, sources in stints["inputs"]["games"].items():
            prefix = f"lake/raw/cdn/boxscore/season={self.season}/game_id={gid}/"
            ref = self._ref(sources["box_score"], prefix)
            if ref["uri"] != f"s3://{self.bucket}/{prefix}{ref['sha256']}.json":
                raise MinutesError("Invalid canonical box-score reference.")
            boxes[gid] = self.read_reference(ref, prefix)
            metadata_ref = {
                "uri": sources["box_score"]["metadata_uri"],
                "sha256": sources["box_score"]["metadata_sha256"],
            }
            metadata = self.read_reference(metadata_ref, prefix)
            if (
                metadata.get("box_score_sha256") != ref["sha256"]
                or metadata.get("game_id") != gid or metadata.get("season") != self.season
                or metadata.get("home_team_id") != boxes[gid]["game"]["homeTeam"]["teamId"]
                or metadata.get("away_team_id") != boxes[gid]["game"]["awayTeam"]["teamId"]
            ):
                raise MinutesError("Box-score sidecar differs from the exact selected raw source.")
            refs["boxes"][gid] = {**ref, **{
                k: sources["box_score"][k] for k in ("metadata_uri", "metadata_sha256")
            }}
        return {"stints": stints, "injuries": injuries, "rosters": rosters, "boxes": boxes}, refs

    def save_state(self, record: dict, *, prior: dict | None, expected_etag: str | None) -> dict:
        validate_state(record, self.season, self.model_version)
        day = record["game_date_pacific"]
        inputs = record["inputs"]
        if inputs["parent"] is not None:
            _, current_parent, _ = self.state(str(date.fromisoformat(day)-timedelta(days=1)))
            if current_parent != inputs["parent"]:
                raise MinutesError("Previous minutes selection changed during this run.")
        for kind, prefix in (
            ("stints", f"lake/artifacts/stints/season={self.season}/game_date_pacific={day}/"),
            ("injuries", f"lake/artifacts/injury_status_by_game_date/season={self.season}/game_date_pacific={day}/"),
        ):
            if self._pointer_ref(prefix) != inputs[kind]:
                raise MinutesError(f"The selected {kind} changed during this run.")
        if self._roster_ref(day) != inputs["rosters"]:
            raise MinutesError("The selected roster changed during this run.")
        raw = json_bytes(record)
        digest = sha256(raw).hexdigest()
        prefix = self.prefix+f"game_date_pacific={day}/"
        key = prefix+digest+".json"
        created = self._put(key, raw)
        reference = {"uri": f"s3://{self.bucket}/{key}", "sha256": digest}
        if reference != prior:
            pointer = {
                **reference, "schema_version": 1, "model_version": self.model_version,
                "season": self.season, "game_date_pacific": day,
            }
            condition = {"IfMatch": expected_etag} if expected_etag else {"IfNoneMatch": "*"}
            boto3.client("s3").put_object(
                Bucket=self.bucket, Key=prefix+"latest.json",
                Body=json_bytes(pointer), ContentType="application/json", **condition,
            )
        return {**reference, "reused": not created}

    def _error_key(self, event: dict) -> str:
        occurred = datetime.fromisoformat(event["timestamp"]).astimezone(PACIFIC).date()
        suffix = "recovery" if event["status"] == "recovered" else "failure"
        return f"{ERROR_ROOT}/season={self.season}/occurred_date_pacific={occurred}/{event['event_id']}.{suffix}.json"

    def save_error(
        self, *, day: str, run_id: str, message: str, inputs: dict,
        details: dict | None = None, original: dict | None = None,
    ) -> dict:
        status = "recovered" if original else "failed"
        identity = [run_id, "player_minutes", self.model_version, status, original]
        event = {
            "schema_version": 1, "event_id": sha256(json_bytes(identity)).hexdigest(),
            "pipeline": "minutes", "stage": "player_minutes", "season": self.season,
            "model_version": self.model_version, "status": status,
            "timestamp": datetime.now(UTC).isoformat(), "game_date_pacific": day,
            "dagster_run_id": run_id, "message": message, "input": inputs,
            "details": details or {}, "artifact_path": f"s3://{self.bucket}/{self.prefix}game_date_pacific={day}/",
        }
        if original:
            event["original_failure"] = original
        key = self._error_key(event)
        raw = self._optional(key)
        if raw is None:
            raw = json_bytes(event)
            self._put(key, raw)
        elif {k: v for k, v in official_json(raw).items() if k != "timestamp"} != {
            k: v for k, v in event.items() if k != "timestamp"
        }:
            raise MinutesError("Existing minutes error event differs from this run.")
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

    def read_failure(self, reference: dict, day: str) -> dict:
        prefix = f"{ERROR_ROOT}/season={self.season}/"
        event = self.read_reference(reference, prefix)
        if (
            event.get("pipeline") != "minutes" or event.get("stage") != "player_minutes"
            or event.get("season") != self.season or event.get("model_version") != self.model_version
            or event.get("game_date_pacific") != day or event.get("status") != "failed"
            or reference["uri"] != f"s3://{self.bucket}/{self._error_key(event)}"
            or event.get("event_id") != sha256(json_bytes([
                event["dagster_run_id"], "player_minutes", self.model_version, "failed", None,
            ])).hexdigest()
        ):
            raise MinutesError("Failure reference belongs to another model, date, or pipeline.")
        return {**reference, "event_id": event["event_id"]}

    def parent_failure(self, context, day: str) -> dict | None:
        if not context.run.parent_run_id:
            return None
        run = context.instance.get_run_by_id(context.run.parent_run_id)
        if (
            run is None or (run.job_name != "update_player_minutes"
                            and AssetKey("player_minutes") not in (run.asset_selection or set()))
            or run.status.value != "FAILURE"
        ):
            return None
        selected_day = run.tags.get("dagster/partition")
        if selected_day is None:
            start = run.tags.get("dagster/asset_partition_range_start")
            end = run.tags.get("dagster/asset_partition_range_end")
            selected_day = start if start == end else None
        if selected_day != day:
            return None
        records = context.instance.get_records_for_run(run.run_id, of_type=DagsterEventType.STEP_FAILURE).records
        for record in reversed(records):
            event = record.event_log_entry.dagster_event
            if event.step_key != "player_minutes":
                continue
            data = event.event_specific_data.user_failure_data
            if data and "minutes_failure" in data.metadata:
                return self.read_failure(data.metadata["minutes_failure"].value, day)
        return None
