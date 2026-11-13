"""S3 inputs, immutable initialization JSON, and ratings failure/recovery records."""

import json
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256

import boto3
from botocore.exceptions import ClientError
from dagster import ConfigurableResource, DagsterEventType

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

    def save_event(self, *, run_id: str, message: str, inputs: dict, original: dict | None = None) -> dict:
        now = datetime.now(UTC)
        status = "recovered" if original else "failed"
        event_id = sha256(json_bytes([run_id, "player_rating_priors", status, original])).hexdigest()
        event = {
            "schema_version": 1, "event_id": event_id, "pipeline": "ratings",
            "stage": "player_rating_priors", "season": self.season, "model_version": self.model_version,
            "status": status, "timestamp": now.isoformat(), "dagster_run_id": run_id,
            "message": message, "input": inputs,
        }
        if original:
            event["original_failure"] = original
        suffix = "recovery" if original else "failure"
        key = (f"{ERRORS_PREFIX}/pipeline=ratings/season={self.season}/"
               f"occurred_date_pacific={now.astimezone(PACIFIC).date()}/{event_id}.{suffix}.json")
        raw = json_bytes(event)
        self._immutable(key, raw)
        return {"uri": f"s3://{self.bucket}/{key}", "sha256": sha256(raw).hexdigest()}

    def parent_failure(self, context) -> dict | None:
        parent_id = context.run.parent_run_id
        if not parent_id:
            return None
        parent = context.instance.get_run_by_id(parent_id)
        if (parent is None or parent.job_name != "initialize_player_ratings"
                or parent.status.value != "FAILURE"):
            return None
        records = context.instance.get_records_for_run(
            parent_id, of_type=DagsterEventType.STEP_FAILURE,
        ).records
        for record in reversed(records):
            event = record.event_log_entry.dagster_event
            if event.step_key != "player_rating_priors":
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
                    or source.get("stage") != "player_rating_priors" or source.get("season") != self.season
                    or source.get("model_version") != self.model_version):
                raise ValueError("Failure record does not match this failed parent run.")
            return reference
        return None
