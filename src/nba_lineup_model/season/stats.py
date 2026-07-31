from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.ingest.nba_cdn import validate_game_id
from nba_lineup_model.ingest.nba_stats import (
    DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS,
    NbaStatsClient,
    NbaStatsEndpoint,
    NbaStatsError,
    NbaStatsRawCache,
    StatsCachedResponse,
)
from nba_lineup_model.season.schema import (
    PARTITION_VALUE_PATTERN,
    SEASON_PATTERN,
    SHA256_PATTERN,
    CatalogGame,
    validate_season,
)

StatsFetchStatus = Literal["succeeded", "failed", "skipped"]

STATS_FETCH_COLUMNS = (
    "schema_version",
    "run_id",
    "prefect_flow_run_id",
    "prefect_task_run_id",
    "attempt_number",
    "game_id",
    "season",
    "season_type",
    "endpoint",
    "started_at",
    "finished_at",
    "duration_seconds",
    "status",
    "refresh",
    "cache_hit",
    "source_url",
    "sha256",
    "byte_count",
    "error_type",
    "error_message",
    "skip_reason",
)
_STRING_COLUMNS = (
    "run_id",
    "prefect_flow_run_id",
    "prefect_task_run_id",
    "game_id",
    "season",
    "season_type",
    "endpoint",
    "status",
    "source_url",
    "sha256",
    "error_type",
    "error_message",
    "skip_reason",
)
_INTEGER_COLUMNS = (
    "schema_version",
    "attempt_number",
    "byte_count",
)


@dataclass(frozen=True)
class StatsArtifactEvidence:
    """Validated exact-byte evidence for one NBA Stats response."""

    response: StatsCachedResponse
    sha256: str
    byte_count: int


class StatsFetchRecord(BaseModel):
    """Terminal outcome for one game and NBA Stats endpoint."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    prefect_flow_run_id: str | None = None
    prefect_task_run_id: str | None = None
    attempt_number: int = Field(ge=1)
    game_id: str
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)
    endpoint: NbaStatsEndpoint
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    status: StatsFetchStatus
    refresh: bool
    cache_hit: bool
    source_url: str | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    byte_count: int | None = Field(default=None, gt=0)
    error_type: str | None = None
    error_message: str | None = None
    skip_reason: str | None = None

    @field_validator("game_id")
    @classmethod
    def validate_id(cls, game_id: str) -> str:
        return validate_game_id(game_id)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, season: str) -> str:
        return validate_season(season)

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Datetime values must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_outcome(self) -> StatsFetchRecord:
        elapsed = (self.finished_at - self.started_at).total_seconds()
        if elapsed < 0:
            raise ValueError("Stats fetch finish time precedes its start time")
        if abs(elapsed - self.duration_seconds) > 0.01:
            raise ValueError("Stats fetch duration does not match its timestamps")
        if (self.sha256 is None) != (self.byte_count is None):
            raise ValueError("Stats fetch hash and byte count must be recorded together")

        has_artifact = (
            self.sha256 is not None
            and self.byte_count is not None
            and self.source_url is not None
        )
        if self.status == "succeeded":
            if not has_artifact or self.cache_hit:
                raise ValueError("Successful Stats fetches require a new raw artifact")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful Stats fetches cannot contain errors")
            if self.skip_reason is not None:
                raise ValueError("Successful Stats fetches cannot contain a skip reason")
        elif self.status == "skipped":
            if not has_artifact or not self.cache_hit or not self.skip_reason:
                raise ValueError("Skipped Stats fetches require a validated cache artifact")
            if self.refresh:
                raise ValueError("Refresh Stats fetches cannot be skipped")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Skipped Stats fetches cannot contain errors")
        else:
            if not self.error_type or not self.error_message:
                raise ValueError("Failed Stats fetches require error details")
            if self.skip_reason is not None:
                raise ValueError("Failed Stats fetches cannot contain a skip reason")
        return self


class StatsFetchManifest(BaseModel):
    """Append-oriented terminal history for NBA Stats endpoint work."""

    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1] = 1
    records: list[StatsFetchRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_work_per_run(self) -> StatsFetchManifest:
        keys = [
            (record.run_id, record.game_id, record.endpoint)
            for record in self.records
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Stats manifest contains duplicate endpoint work within a run")
        return self


def fetch_stats_endpoint_raw(
    game: CatalogGame,
    endpoint: NbaStatsEndpoint,
    *,
    run_id: str,
    raw_dir: Path | str = Path("data/raw/stats"),
    refresh: bool = False,
    attempt_number: int = 1,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
    started_at: datetime | None = None,
    client: NbaStatsClient | None = None,
    min_request_interval_seconds: float = 0.0,
    request_interval_jitter_seconds: float = 0.0,
    access_denial_cooldown_seconds: float = (
        DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS
    ),
) -> StatsFetchRecord:
    """Fetch one endpoint without requiring Prefect."""

    started_at = started_at or datetime.now(UTC)
    cache = client.cache if client is not None else NbaStatsRawCache(raw_dir)
    initial = None if refresh else _safe_artifact_evidence(cache, endpoint, game.game_id)
    if initial is not None:
        return _terminal_record(
            game,
            endpoint,
            run_id=run_id,
            started_at=started_at,
            attempt_number=attempt_number,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            status="skipped",
            refresh=False,
            cache_hit=True,
            evidence=initial,
            skip_reason="already_cached",
        )

    owns_client = client is None
    client = client or NbaStatsClient(
        cache=cache,
        min_request_interval_seconds=min_request_interval_seconds,
        request_interval_jitter_seconds=request_interval_jitter_seconds,
        access_denial_cooldown_seconds=access_denial_cooldown_seconds,
    )
    try:
        client.fetch(endpoint, game.game_id, use_cache=False)
    finally:
        if owns_client:
            client.close()

    evidence = require_stats_artifact_evidence(cache, endpoint, game.game_id)
    return _terminal_record(
        game,
        endpoint,
        run_id=run_id,
        started_at=started_at,
        attempt_number=attempt_number,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
        status="succeeded",
        refresh=refresh,
        cache_hit=False,
        evidence=evidence,
    )


def failed_stats_fetch_record(
    game: CatalogGame,
    endpoint: NbaStatsEndpoint,
    *,
    run_id: str,
    started_at: datetime,
    error: Exception,
    raw_dir: Path | str = Path("data/raw/stats"),
    refresh: bool = False,
    attempt_number: int = 1,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
) -> StatsFetchRecord:
    """Create a durable endpoint failure while retaining prior valid evidence."""

    evidence = _safe_artifact_evidence(
        NbaStatsRawCache(raw_dir),
        endpoint,
        game.game_id,
    )
    finished_at = datetime.now(UTC)
    return StatsFetchRecord(
        run_id=run_id,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
        attempt_number=attempt_number,
        game_id=game.game_id,
        season=game.season,
        season_type=game.season_type,
        endpoint=endpoint,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        status="failed",
        refresh=refresh,
        cache_hit=False,
        source_url=evidence.response.url if evidence is not None else None,
        sha256=evidence.sha256 if evidence is not None else None,
        byte_count=evidence.byte_count if evidence is not None else None,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def stats_artifact_evidence(
    cache: NbaStatsRawCache,
    endpoint: NbaStatsEndpoint,
    game_id: str,
) -> StatsArtifactEvidence | None:
    """Read, validate, and hash one retained Stats response."""

    response = cache.read(endpoint, game_id)
    if response is None:
        return None
    raw_body = cache.path_for(endpoint, game_id).read_bytes()
    return StatsArtifactEvidence(
        response=response,
        sha256=hashlib.sha256(raw_body).hexdigest(),
        byte_count=len(raw_body),
    )


def require_stats_artifact_evidence(
    cache: NbaStatsRawCache,
    endpoint: NbaStatsEndpoint,
    game_id: str,
) -> StatsArtifactEvidence:
    evidence = stats_artifact_evidence(cache, endpoint, game_id)
    if evidence is None:
        raise NbaStatsError(f"NBA {endpoint.value} response was not cached for {game_id}")
    return evidence


def is_transient_stats_fetch_error(error: BaseException) -> bool:
    """Return whether Prefect should retry a Stats endpoint failure."""

    current: BaseException | None = error
    while current is not None:
        if isinstance(current, NbaStatsError):
            return current.transient
        current = current.__cause__
    return False


def stats_fetch_manifest_frame(
    manifest: StatsFetchManifest | Sequence[StatsFetchRecord],
) -> pd.DataFrame:
    records = manifest.records if isinstance(manifest, StatsFetchManifest) else list(manifest)
    validated = StatsFetchManifest(records=records)
    rows = [record.model_dump(mode="json") for record in validated.records]
    frame = pd.DataFrame(rows, columns=STATS_FETCH_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["season", "game_id", "endpoint", "started_at"],
            kind="stable",
        ).reset_index(drop=True)
    for column in _STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    frame["refresh"] = frame["refresh"].astype("boolean")
    frame["cache_hit"] = frame["cache_hit"].astype("boolean")
    frame["duration_seconds"] = frame["duration_seconds"].astype("Float64")
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
    frame["finished_at"] = pd.to_datetime(frame["finished_at"], utc=True)
    return frame


def stats_fetch_manifest_from_frame(frame: pd.DataFrame) -> StatsFetchManifest:
    records = []
    for row in frame.to_dict(orient="records"):
        record = _python_record(row)
        record["endpoint"] = NbaStatsEndpoint(record["endpoint"])
        records.append(StatsFetchRecord.model_validate(record))
    return StatsFetchManifest(records=records)


def write_stats_fetch_manifest(
    manifest: StatsFetchManifest | Sequence[StatsFetchRecord],
    path: Path | str,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    stats_fetch_manifest_frame(manifest).to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)
    return output_path


def read_stats_fetch_manifest(path: Path | str) -> StatsFetchManifest:
    input_path = Path(path)
    if not input_path.exists():
        return StatsFetchManifest()
    return stats_fetch_manifest_from_frame(pd.read_parquet(input_path))


def append_stats_fetch_records(
    records: Sequence[StatsFetchRecord],
    path: Path | str,
) -> StatsFetchManifest:
    manifest = read_stats_fetch_manifest(path)
    updated = StatsFetchManifest(records=[*manifest.records, *records])
    write_stats_fetch_manifest(updated, path)
    return updated


def _terminal_record(
    game: CatalogGame,
    endpoint: NbaStatsEndpoint,
    *,
    run_id: str,
    started_at: datetime,
    attempt_number: int,
    prefect_flow_run_id: str | None,
    prefect_task_run_id: str | None,
    status: StatsFetchStatus,
    refresh: bool,
    cache_hit: bool,
    evidence: StatsArtifactEvidence,
    skip_reason: str | None = None,
) -> StatsFetchRecord:
    finished_at = datetime.now(UTC)
    return StatsFetchRecord(
        run_id=run_id,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
        attempt_number=attempt_number,
        game_id=game.game_id,
        season=game.season,
        season_type=game.season_type,
        endpoint=endpoint,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        status=status,
        refresh=refresh,
        cache_hit=cache_hit,
        source_url=evidence.response.url,
        sha256=evidence.sha256,
        byte_count=evidence.byte_count,
        skip_reason=skip_reason,
    )


def _safe_artifact_evidence(
    cache: NbaStatsRawCache,
    endpoint: NbaStatsEndpoint,
    game_id: str,
) -> StatsArtifactEvidence | None:
    try:
        return stats_artifact_evidence(cache, endpoint, game_id)
    except (OSError, ValueError, NbaStatsError):
        return None


def _python_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _python_value(value) for key, value in record.items()}


def _python_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if pd.api.types.is_scalar(value) and bool(pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    item = getattr(value, "item", None)
    return item() if callable(item) else value
