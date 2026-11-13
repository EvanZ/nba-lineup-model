"""Durable versioned minute states and exact canonical NBA input selections."""

import base64
import os
import re
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import AssetKey, ConfigurableResource, DagsterEventType

from ..utils import (
    MODEL_VERSION, PRIOR_REFERENCES, MinutesError, json_bytes, official_json,
    validate_priors, validate_state, PROJECTION_VERSION, validate_game_projection,
)

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC

ROOT = "lake/artifacts/player_minutes"
ERROR_ROOT = f"{ERRORS_PREFIX}/pipeline=minutes"


class MinutesArchive(ConfigurableResource):
    season: str
    model_version: str = MODEL_VERSION
    bucket: str = DATA_BUCKET
    prior_uri: str = ""
    projection_version: str = PROJECTION_VERSION

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

    def prior_reference(self) -> dict:
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
        return reference

    def load_priors(self) -> tuple[dict, dict]:
        reference = self.prior_reference()
        record = self.read_reference(reference, self.prefix+"priors/")
        validate_priors(record, self.season, self.model_version)
        return record, reference

    def _state_selection(self, day: str) -> tuple[dict | None, str | None]:
        prefix = self.prefix+f"game_date_pacific={day}/"
        try:
            response = boto3.client("s3").get_object(Bucket=self.bucket, Key=prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None
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
        return ref, response["ETag"]

    def state(self, day: str) -> tuple[dict | None, dict | None, str | None]:
        ref, etag = self._state_selection(day)
        if ref is None:
            return None, None, None
        prefix = self.prefix+f"game_date_pacific={day}/"
        record = self.read_reference(ref, prefix)
        validate_state(record, self.season, self.model_version)
        if record["game_date_pacific"] != day:
            raise MinutesError("Selected minutes state has a different date.")
        return record, ref, etag

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
        return {"stints": stints, "injuries": injuries, "rosters": rosters}, refs

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
        stage: str = "player_minutes", game_id: str | None = None,
    ) -> dict:
        if stage not in {"player_minutes", "game_minutes"}:
            raise MinutesError("Unknown minutes error stage.")
        version = self.model_version if stage == "player_minutes" else self.projection_version
        status = "recovered" if original else "failed"
        identity = [run_id, stage, version, status, original]
        if game_id is not None:
            identity.append(game_id)
        event = {
            "schema_version": 1, "event_id": sha256(json_bytes(identity)).hexdigest(),
            "pipeline": "minutes", "stage": stage, "season": self.season,
            "model_version": version, "status": status,
            "timestamp": datetime.now(UTC).isoformat(), "game_date_pacific": day,
            "dagster_run_id": run_id, "message": message, "input": inputs,
            "details": details or {}, "artifact_path": f"s3://{self.bucket}/{self.prefix if stage == 'player_minutes' else self.game_prefix}game_date_pacific={day}/",
        }
        if game_id is not None:
            event["game_id"] = game_id
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

    def read_failure(self, reference: dict, day: str, *, stage: str = "player_minutes") -> dict:
        version = self.model_version if stage == "player_minutes" else self.projection_version
        prefix = f"{ERROR_ROOT}/season={self.season}/"
        event = self.read_reference(reference, prefix)
        identity = [event["dagster_run_id"],stage,version,"failed",None]
        if event.get("game_id") is not None:
            identity.append(event["game_id"])
        if (
            event.get("pipeline") != "minutes" or event.get("stage") != stage
            or event.get("season") != self.season or event.get("model_version") != version
            or event.get("game_date_pacific") != day or event.get("status") != "failed"
            or reference["uri"] != f"s3://{self.bucket}/{self._error_key(event)}"
            or event.get("event_id") != sha256(json_bytes(identity)).hexdigest()
        ):
            raise MinutesError("Failure reference belongs to another model, date, or pipeline.")
        return {**reference, "event_id": event["event_id"]}

    def parent_failure(self, context, day: str, *, stage: str = "player_minutes") -> dict | None:
        job = "update_player_minutes" if stage == "player_minutes" else "project_game_minutes"
        if not context.run.parent_run_id:
            return None
        run = context.instance.get_run_by_id(context.run.parent_run_id)
        if (
            run is None or (run.job_name != job
                            and AssetKey(stage) not in (run.asset_selection or set()))
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
            if event.step_key != stage:
                continue
            data = event.event_specific_data.user_failure_data
            if data and "minutes_failure" in data.metadata:
                return self.read_failure(data.metadata["minutes_failure"].value, day, stage=stage)
        return None

    @property
    def game_prefix(self) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.projection_version):
            raise MinutesError("Invalid game-minutes version.")
        return f"lake/artifacts/game_minutes/season={self.season}/model_version={self.projection_version}/"

    def projection_selections(self, day: str, *, today: date, roster_archive) -> dict:
        target = date.fromisoformat(day)
        historical = target < today
        baseline_day = str((target if historical else today)-timedelta(days=1))
        baseline, _ = self._state_selection(baseline_day)
        first_date = date.fromisoformat(os.environ["GESTALT_NBA_SEASON_START_DATE"])
        if baseline is None and target != first_date:
            raise FileNotFoundError(f"No accepted daily minutes through {baseline_day}.")
        if historical:
            roster_day = day
            roster, _ = roster_archive.selection(self.season, roster_day)
            if roster is None:
                raise FileNotFoundError(f"No accepted roster for historical game date {day}.")
        else:
            roster_day, roster = roster_archive.latest_selection(self.season, today)
        prefix = f"lake/artifacts/injury_status_by_game_date/season={self.season}/game_date_pacific={day}/"
        raw = self._optional(prefix+"latest.json")
        if raw is None:
            raise FileNotFoundError(f"No accepted injury selection for {day}.")
        injury = self._ref(official_json(raw),prefix)
        if not re.fullmatch(
            re.escape(f"s3://{self.bucket}/{prefix}")+r"selection_sha256=[a-f0-9]{64}/statuses\.json",
            injury["uri"],
        ):
            raise MinutesError("Invalid canonical game injury selection.")
        refs = {
            "daily_minutes": baseline, "priors": self.prior_reference(),
            "rosters": {**roster, "game_date_pacific": roster_day}, "injuries": injury,
        }
        if historical:
            refs["historical_replay"] = True
        return refs

    def projection_data(self, day: str, inputs: dict) -> dict:
        baseline = None
        if inputs["daily_minutes"] is not None:
            baseline = self.read_reference(inputs["daily_minutes"],self.prefix)
            validate_state(baseline,self.season,self.model_version)
            expected_uri = f"s3://{self.bucket}/{self.prefix}game_date_pacific={baseline['game_date_pacific']}/{inputs['daily_minutes']['sha256']}.json"
            if inputs["daily_minutes"]["uri"] != expected_uri:
                raise MinutesError("Projection baseline date differs from its accepted selection.")
        elif day != os.environ["GESTALT_NBA_SEASON_START_DATE"]:
            raise MinutesError("Only opening-day projections may initialize without previous daily state.")
        priors, ref = self.load_priors()
        if inputs["priors"] != ref or (baseline is not None and baseline["inputs"]["priors"] != ref):
            raise MinutesError("Projection initialization differs from the accepted daily prior.")
        roster_day = inputs["rosters"]["game_date_pacific"]
        rosters = self.read_reference(inputs["rosters"],f"lake/artifacts/rosters/season={self.season}/game_date_pacific={roster_day}/")
        if rosters["game_date_pacific"] != roster_day:
            raise MinutesError("Projection roster date differs from its accepted selection.")
        injuries = self.read_reference(inputs["injuries"],f"lake/artifacts/injury_status_by_game_date/season={self.season}/game_date_pacific={day}/")
        if injuries.get("cutoff_source") != "nba_scheduled_start":
            raise MinutesError("Projection injuries must use scheduled-tipoff selection.")
        return {"baseline":baseline,"priors":priors,"rosters":rosters,"injuries":injuries}

    def game_selection(self, day: str, game_id: str) -> tuple[dict | None, str | None]:
        if not re.fullmatch(r"[0-9]{10}",game_id) or date.fromisoformat(day).isoformat() != day:
            raise MinutesError("Invalid game-minutes date/game identity.")
        prefix = self.game_prefix+f"game_date_pacific={day}/game_id={game_id}/"
        try:
            response = boto3.client("s3").get_object(Bucket=self.bucket,Key=prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404","NoSuchKey","NotFound"}:
                return None,None
            raise
        with closing(response["Body"]) as body:
            pointer = official_json(body.read())
        ref = self._ref(pointer,prefix)
        if (
            pointer.get("schema_version") != 1 or pointer.get("season") != self.season
            or pointer.get("game_date_pacific") != day or pointer.get("game_id") != game_id
            or pointer.get("model_version") != self.projection_version
            or ref["uri"] != f"s3://{self.bucket}/{prefix}{ref['sha256']}.json"
            or not re.fullmatch(r"[0-9a-f]{64}",pointer.get("input_fingerprint",""))
        ):
            raise MinutesError("Invalid accepted game-minutes pointer.")
        return pointer,response["ETag"]

    def save_game(self, record: dict, *, expected_etag: str | None, roster_archive) -> dict:
        validate_game_projection(record,self.season,self.projection_version)
        day,gid = record["game_date_pacific"],record["game_id"]
        start = datetime.fromisoformat(record["scheduled_start_utc"])
        if start.tzinfo is None:
            raise MinutesError("Game projection tipoff must be timezone-aware.")
        historical = record["purpose"] == "historical_replay"
        if not historical and datetime.now(UTC) >= start:
            return {"frozen":True}
        selected,etag = self.game_selection(day,gid)
        if historical and selected is not None:
            accepted = self.read_reference(selected,self.game_prefix)
            validate_game_projection(accepted,self.season,self.projection_version)
            if accepted["purpose"] == "pregame_projection":
                return {"frozen":True}
        if etag != expected_etag:
            raise MinutesError("Game projection selection changed during this run.")
        refs = self.projection_selections(day,today=datetime.now(UTC).astimezone(PACIFIC).date(),roster_archive=roster_archive)
        if any(record["inputs"].get(k) != v for k,v in refs.items()):
            raise MinutesError("Accepted projection inputs changed during this run.")
        raw = json_bytes(record)
        digest = sha256(raw).hexdigest()
        prefix = self.game_prefix+f"game_date_pacific={day}/game_id={gid}/"
        key = prefix+digest+".json"
        created = self._put(key,raw)
        ref = {"uri":f"s3://{self.bucket}/{key}","sha256":digest}
        if selected is None or any(selected.get(k)!=v for k,v in ref.items()):
            # A calculation started before tipoff may finish after it. Never
            # advance its accepted selection once the game has started.
            if not historical and datetime.now(UTC) >= start:
                return {"frozen":True}
            pointer = {
                **ref,"schema_version":1,"season":self.season,
                "model_version":self.projection_version,"game_date_pacific":day,
                "game_id":gid,"scheduled_start_utc":record["scheduled_start_utc"],
                "purpose":record["purpose"], "input_fingerprint":record["input_fingerprint"],
            }
            condition = {"IfMatch":expected_etag} if expected_etag else {"IfNoneMatch":"*"}
            boto3.client("s3").put_object(
                Bucket=self.bucket,Key=prefix+"latest.json",Body=json_bytes(pointer),
                ContentType="application/json",**condition,
            )
        return {**ref,"frozen":False,"reused":not created}
