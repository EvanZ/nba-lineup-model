"""S3 inputs, immutable initialization JSON, and ratings failure/recovery records."""

import json
import re
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import AssetKey, ConfigurableResource, DagsterEventType

from ...utils import DATA_BUCKET, ERRORS_PREFIX, PACIFIC
from ..utils import MODEL_VERSION, json_bytes, source_references, validate_initialization


class RatingsArchive(ConfigurableResource):
    season: str
    model_version: str = MODEL_VERSION
    bucket: str = DATA_BUCKET

    @property
    def prefix(self) -> str:
        source_references(self.season, self.model_version)
        return f"lake/artifacts/player_ratings/season={self.season}/model_version={self.model_version}/priors/"

    def _read(self, key: str) -> tuple[bytes, str]:
        response = boto3.client("s3").get_object(Bucket=self.bucket, Key=key)
        with closing(response["Body"]) as stream:
            return stream.read(), response["ETag"]

    def read_sources(self) -> dict[str, bytes]:
        blobs = {}
        client = boto3.client("s3")
        for name, reference in source_references(self.season, self.model_version).items():
            bucket, key = reference["uri"].removeprefix("s3://").split("/", 1)
            with closing(client.get_object(Bucket=bucket, Key=key)["Body"]) as stream:
                raw = stream.read()
            if sha256(raw).hexdigest() != reference["sha256"]:
                raise ValueError(f"The {name} checksum differs from the approved input.")
            blobs[name] = raw
        return blobs

    def selection(self) -> tuple[dict | None, str | None]:
        try:
            raw, etag = self._read(self.prefix+"latest.json")
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None
            raise
        pointer = json.loads(raw)
        digest = pointer["sha256"]
        if (pointer.get("schema_version") != 1 or pointer.get("season") != self.season
                or pointer.get("model_version") != self.model_version
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or pointer["uri"] != f"s3://{self.bucket}/{self.prefix}{digest}.json"):
            raise ValueError("Invalid accepted ratings initialization selection.")
        return {"uri": pointer["uri"], "sha256": digest}, etag

    def _immutable(self, key: str, raw: bytes) -> None:
        try:
            boto3.client("s3").put_object(
                Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"412", "PreconditionFailed"}:
                raise
        saved, _ = self._read(key)
        if saved != raw:
            raise ValueError("S3 read-back differs from the immutable ratings artifact.")

    def save_priors(self, record: dict, *, prior: dict | None, expected_etag: str | None) -> dict:
        validate_initialization(record, self.season, self.model_version)
        raw = json_bytes(record)
        digest = sha256(raw).hexdigest()
        reference = {"uri": f"s3://{self.bucket}/{self.prefix}{digest}.json", "sha256": digest}
        if prior is not None and prior != reference:
            raise ValueError(
                "Different ratings initialization is already selected. "
                "Replacing it requires an approved model-version/replay decision."
            )
        self._immutable(self.prefix+digest+".json", raw)
        if prior == reference:
            current, etag = self.selection()
            if current != prior or etag != expected_etag:
                raise ValueError("The initialization selection changed during this run.")
            return {**reference, "reused": True}
        if prior is not None or expected_etag is not None:
            raise ValueError("Invalid initialization publication state.")
        pointer = {"schema_version": 1, "season": self.season, "model_version": self.model_version, **reference}
        try:
            boto3.client("s3").put_object(
                Bucket=self.bucket, Key=self.prefix+"latest.json", Body=json_bytes(pointer),
                ContentType="application/json", IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] in {"412", "PreconditionFailed"}:
                raise ValueError("Another run selected initialization during this run; reload and retry.") from error
            raise
        selected, _ = self.selection()
        if selected != reference:
            raise ValueError("Ratings initialization selection failed read-back verification.")
        return {**reference, "reused": False}

    def save_event(self, *, run_id: str, message: str, inputs: dict, original: dict | None = None,
                   stage: str = "player_rating_priors", day: str | None = None) -> dict:
        if stage not in {"player_rating_priors", "player_ratings"}:
            raise ValueError("Unknown ratings stage.")
        now = datetime.now(UTC)
        status = "recovered" if original else "failed"
        event_id = sha256(json_bytes([run_id, stage, status, original])).hexdigest()
        event = {
            "schema_version": 1, "event_id": event_id, "pipeline": "ratings",
            "stage": stage, "season": self.season, "model_version": self.model_version,
            "status": status, "timestamp": now.isoformat(), "dagster_run_id": run_id,
            "message": message, "input": inputs,
        }
        if day is not None:
            event["game_date_pacific"] = day
            if stage == "player_ratings":
                event["observed_through_game_date_pacific"] = str(date.fromisoformat(day)-timedelta(days=1))
        if original:
            event["original_failure"] = original
        suffix = "recovery" if original else "failure"
        key = (f"{ERRORS_PREFIX}/pipeline=ratings/season={self.season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{event_id}.{suffix}.json")
        raw = json_bytes(event)
        self._immutable(key, raw)
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

    def parent_failure(self, context, *, stage: str = "player_rating_priors", day: str | None = None) -> dict | None:
        parent_id = context.run.parent_run_id
        if not parent_id:
            return None
        job = {"player_rating_priors": "initialize_player_ratings", "player_ratings": "update_player_ratings"}[stage]
        parent = context.instance.get_run_by_id(parent_id)
        if (parent is None or (parent.job_name != job and AssetKey(stage) not in (parent.asset_selection or set()))
                or parent.status.value != "FAILURE"):
            return None
        records = context.instance.get_records_for_run(
            parent_id, of_type=DagsterEventType.STEP_FAILURE,
        ).records
        for record in reversed(records):
            event = record.event_log_entry.dagster_event
            if event.step_key != stage:
                continue
            failure = event.event_specific_data.user_failure_data
            if not failure or "ratings_failure" not in failure.metadata:
                continue
            reference = failure.metadata["ratings_failure"].value
            prefix = f"s3://{self.bucket}/{ERRORS_PREFIX}/pipeline=ratings/season={self.season}/"
            if not reference["uri"].startswith(prefix) or not reference["uri"].endswith(".failure.json"):
                raise ValueError("Failure reference is outside this ratings pipeline/season.")
            raw, _ = self._read(reference["uri"].removeprefix(f"s3://{self.bucket}/"))
            source = json.loads(raw)
            if (sha256(raw).hexdigest() != reference["sha256"] or source.get("status") != "failed"
                    or source.get("dagster_run_id") != parent_id or source.get("pipeline") != "ratings"
                    or source.get("stage") != stage or source.get("season") != self.season
                    or source.get("game_date_pacific") != day
                    or source.get("model_version") != self.model_version):
                raise ValueError("Failure record does not match this failed parent run.")
            return reference
        return None

    @property
    def state_prefix(self) -> str:
        return self.prefix.removesuffix("priors/")

    def _reference_json(self, reference: dict, prefix: str) -> dict:
        digest = reference["sha256"]
        expected = f"s3://{self.bucket}/{prefix}{digest}.json"
        if (reference["uri"] != expected or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid canonical ratings input reference.")
        raw, _ = self._read(prefix+digest+".json")
        if sha256(raw).hexdigest() != digest:
            raise ValueError("Saved ratings input differs from its recorded checksum.")
        return json.loads(raw)

    def load_priors(self) -> tuple[dict, dict]:
        reference, _ = self.selection()
        if reference is None:
            raise FileNotFoundError("Run initialize_player_ratings before the daily update.")
        record = self._reference_json(reference, self.prefix)
        validate_initialization(record, self.season, self.model_version)
        return record, reference

    def _date_selection(self, prefix: str, day: str, *, optional: bool = False) -> tuple[dict | None, str | None]:
        from datetime import date
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError("Invalid ratings partition date.")
        try:
            raw, etag = self._read(prefix+"latest.json")
        except ClientError as error:
            if optional and error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None, None
            raise
        pointer = json.loads(raw)
        if pointer.get("season") != self.season or pointer.get("game_date_pacific") != day:
            raise ValueError("Selected ratings input has a different season/date.")
        return {"uri": pointer["uri"], "sha256": pointer["sha256"]}, etag

    def state_selection(self, day: str) -> tuple[dict | None, str | None]:
        return self._date_selection(self.state_prefix+f"game_date_pacific={day}/", day, optional=True)

    def state(self, day: str) -> tuple[dict | None, dict | None, str | None]:
        from ..utils import validate_state
        reference, etag = self.state_selection(day)
        if reference is None:
            return None, None, None
        record = self._reference_json(reference, self.state_prefix+f"game_date_pacific={day}/")
        # Old effective-date pointers remain archived, but are not completed states.
        from datetime import date, timedelta
        if (record.get("date_contract") is None
                and record.get("game_date_pacific") == day
                and record.get("observed_through_game_date_pacific") == str(date.fromisoformat(day)-timedelta(days=1))):
            return None, None, None
        validate_state(record, self.season)
        if record["game_date_pacific"] != day:
            raise ValueError("Selected ratings state belongs to a different partition.")
        return record, reference, etag

    def read_state(self, reference: dict, *, effective_date: str, observed_through: str) -> dict:
        """Read a forecast's immutable source, preserving its recorded old or new date meaning."""
        from datetime import date, timedelta
        from ..utils import validate_state
        match = re.fullmatch(re.escape(f"s3://{self.bucket}/{self.state_prefix}")+
                             r"game_date_pacific=(\d{4}-\d{2}-\d{2})/[a-f0-9]{64}\.json", reference["uri"])
        if match is None:
            raise ValueError("Invalid pinned ratings path.")
        day = match.group(1)
        record = self._reference_json(reference, self.state_prefix+f"game_date_pacific={day}/")
        validate_state(record, self.season, allow_legacy_effective_date=True)
        if (record["game_date_pacific"] != day
                or record["observed_through_game_date_pacific"] != observed_through
                or str(date.fromisoformat(observed_through)+timedelta(days=1)) != effective_date):
            raise ValueError("Pinned ratings dates differ from the saved forecast.")
        return record

    def read_priors(self, reference: dict) -> dict:
        """Read the exact initialization already recorded by a saved rating state."""
        record = self._reference_json(reference, self.prefix)
        validate_initialization(record, self.season, self.model_version)
        return record

    def day_inputs(self, day: str) -> tuple[dict, dict]:
        records, references = {}, {}
        for name, root in (("stints", "stints"), ("rosters", "rosters")):
            prefix = f"lake/artifacts/{root}/season={self.season}/game_date_pacific={day}/"
            reference, _ = self._date_selection(prefix, day)
            record = self._reference_json(reference, prefix)
            if record.get("season") != self.season or record.get("game_date_pacific") != day:
                raise ValueError(f"The accepted {name} artifact has the wrong season/date.")
            records[name], references[name] = record, reference
        return records, references

    def verify_selections(self, day: str, inputs: dict) -> None:
        from datetime import date, timedelta
        _, reference = self.load_priors()
        if reference != inputs["priors"]:
            raise ValueError("Ratings initialization changed during this run.")
        observed_day = day
        for name in ("stints", "rosters"):
            if inputs[name] is None:
                continue
            prefix = f"lake/artifacts/{name}/season={self.season}/game_date_pacific={observed_day}/"
            reference, _ = self._date_selection(prefix, observed_day)
            if reference != inputs[name]:
                raise ValueError(f"Accepted {name} changed during this run; rerun the date.")
        if inputs["parent"] is not None:
            reference, _ = self.state_selection(str(date.fromisoformat(day)-timedelta(days=1)))
            if reference != inputs["parent"]:
                raise ValueError("The accepted ratings predecessor changed during this run.")

    def save_state(self, record: dict, *, prior: dict | None, expected_etag: str | None) -> dict:
        from ..utils import validate_state
        validate_state(record, self.season)
        prefix = self.state_prefix+f"game_date_pacific={record['game_date_pacific']}/"
        raw = json_bytes(record); digest = sha256(raw).hexdigest()
        reference = {"uri": f"s3://{self.bucket}/{prefix}{digest}.json", "sha256": digest}
        self._immutable(prefix+digest+".json", raw)
        current, etag = self.state_selection(record["game_date_pacific"])
        if current != prior or etag != expected_etag:
            raise ValueError("The accepted ratings date changed during this run.")
        if reference == prior:
            return {**reference, "reused": True}
        pointer = {"schema_version": 1, "season": self.season, "model_version": self.model_version,
                   "game_date_pacific": record["game_date_pacific"], **reference}
        condition = {"IfMatch": expected_etag} if expected_etag else {"IfNoneMatch": "*"}
        try:
            boto3.client("s3").put_object(Bucket=self.bucket, Key=prefix+"latest.json",
                                         Body=json_bytes(pointer), ContentType="application/json", **condition)
        except ClientError as error:
            if error.response["Error"]["Code"] in {"412", "PreconditionFailed"}:
                raise ValueError("A competing ratings run advanced this date; reload and retry.") from error
            raise
        selected, _ = self.state_selection(record["game_date_pacific"])
        if selected != reference:
            raise ValueError("Ratings date selection failed read-back verification.")
        return {**reference, "reused": False}

    def save_replacement_warning(self, *, day: str, run_id: str, player: dict, inputs: dict) -> dict:
        now = datetime.now(UTC)
        event_id = sha256(json_bytes([run_id, "player_ratings", "replacement_initialization", player["player_id"]])).hexdigest()
        label = player.get("player_name") or f"NBA player {player['player_id']}"
        message = (f"{label} (NBA ID {player['player_id']}) initialized for ratings effective {day}; "
                   f"assigned replacement starting NAIL {player['starting_nail']}. Review this player.")
        event = {
            "schema_version": 1, "event_id": event_id, "pipeline": "ratings", "stage": "player_ratings",
            "season": self.season, "model_version": self.model_version, "status": "warning",
            "warning_type": "replacement_initialization", "timestamp": now.isoformat(),
            "dagster_run_id": run_id, "game_date_pacific": day,
            "observed_through_game_date_pacific": str(date.fromisoformat(day)-timedelta(days=1)),
            "player_id": player["player_id"], "player_name": player.get("player_name"),
            "starting_nail": player["starting_nail"], "forecast_source": player["forecast_source"],
            "initialization": player["initialization"],
            "message": message, "input": inputs,
        }
        key = (f"{ERRORS_PREFIX}/pipeline=ratings/season={self.season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{event_id}.warning.json")
        raw = json_bytes(event); self._immutable(key, raw)
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}
