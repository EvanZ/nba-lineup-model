from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.ingest.nba_cdn import validate_game_id

SEASON_PATTERN = r"^\d{4}-\d{2}$"
PARTITION_VALUE_PATTERN = r"^[a-z][a-z0-9_]*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

GameStatus = Literal[
    "scheduled",
    "live",
    "final",
    "postponed",
    "cancelled",
    "unknown",
]
BuildStatus = Literal["succeeded", "failed", "skipped"]
BuildStage = Literal[
    "preflight",
    "fetch",
    "reconstruct",
    "persist",
    "validate",
    "complete",
]


def validate_season(value: str) -> str:
    if len(value) != 7 or value[4] != "-":
        raise ValueError("Season must use YYYY-YY format")
    try:
        start_year = int(value[:4])
        end_suffix = int(value[5:])
    except ValueError as exc:
        raise ValueError("Season must use YYYY-YY format") from exc
    if end_suffix != (start_year + 1) % 100:
        raise ValueError("Season end year must immediately follow its start year")
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime values must include a timezone")
    return value.astimezone(UTC)


class CatalogGame(BaseModel):
    """One canonical NBA game catalog row."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    game_id: str
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)
    game_date: date
    game_time_utc: datetime | None = None
    game_status: GameStatus = "unknown"
    home_team_id: int = Field(gt=0)
    home_team_tricode: str = Field(pattern=r"^[A-Z0-9]{3}$")
    away_team_id: int = Field(gt=0)
    away_team_tricode: str = Field(pattern=r"^[A-Z0-9]{3}$")
    period_count: int | None = Field(default=None, ge=0)
    is_overtime: bool | None = None
    source_status_code: int | None = None
    source_status_text: str | None = None
    source_url: str = Field(pattern=r"^https?://", min_length=1)
    source_fetched_at: datetime

    @field_validator("game_id")
    @classmethod
    def validate_id(cls, game_id: str) -> str:
        return validate_game_id(game_id)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, season: str) -> str:
        return validate_season(season)

    @field_validator("game_time_utc", "source_fetched_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value) if value is not None else None

    @model_validator(mode="after")
    def validate_game(self) -> CatalogGame:
        start_year = int(self.season[:4])
        if self.game_date.year not in {start_year, start_year + 1}:
            raise ValueError("Game date does not fall within its NBA season")
        if self.home_team_id == self.away_team_id:
            raise ValueError("Home and away team IDs must differ")
        if self.home_team_tricode == self.away_team_tricode:
            raise ValueError("Home and away team tricodes must differ")
        if self.period_count is not None and self.is_overtime is not None:
            if self.is_overtime != (self.period_count > 4):
                raise ValueError("Overtime flag does not match period count")
        return self


class GameCatalog(BaseModel):
    """Versioned, unique collection of canonical NBA games."""

    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1] = 1
    games: list[CatalogGame] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_games(self) -> GameCatalog:
        game_ids = [game.game_id for game in self.games]
        if len(game_ids) != len(set(game_ids)):
            raise ValueError("Game catalog contains duplicate game IDs")
        return self


class GameBuildRecord(BaseModel):
    """Terminal outcome of one attempt to process a catalog game."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    game_id: str
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    status: BuildStatus
    terminal_stage: BuildStage
    use_cache: bool
    code_version: str | None = None
    play_by_play_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    boxscore_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    event_count: int | None = Field(default=None, ge=0)
    lineup_stint_count: int | None = Field(default=None, ge=0)
    possession_count: int | None = Field(default=None, ge=0)
    possession_segment_count: int | None = Field(default=None, ge=0)
    validation_issue_count: int | None = Field(default=None, ge=0)
    output_table_count: int = Field(default=0, ge=0)
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
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_outcome(self) -> GameBuildRecord:
        elapsed = (self.finished_at - self.started_at).total_seconds()
        if elapsed < 0:
            raise ValueError("Build finish time precedes its start time")
        if abs(elapsed - self.duration_seconds) > 0.01:
            raise ValueError("Build duration does not match its timestamps")

        count_fields = (
            self.event_count,
            self.lineup_stint_count,
            self.possession_count,
            self.possession_segment_count,
            self.validation_issue_count,
        )
        if self.status == "succeeded":
            if self.terminal_stage != "complete":
                raise ValueError("Successful builds must terminate at the complete stage")
            if any(value is None for value in count_fields):
                raise ValueError("Successful builds require all processing counts")
            if self.output_table_count < 1:
                raise ValueError("Successful builds must write at least one table")
            if self.play_by_play_sha256 is None or self.boxscore_sha256 is None:
                raise ValueError("Successful builds require both source hashes")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful builds cannot contain error details")
            if self.skip_reason is not None:
                raise ValueError("Successful builds cannot contain a skip reason")
        elif self.status == "failed":
            if self.terminal_stage == "complete":
                raise ValueError("Failed builds cannot terminate at the complete stage")
            if not self.error_type or not self.error_message:
                raise ValueError("Failed builds require error type and message")
            if self.skip_reason is not None:
                raise ValueError("Failed builds cannot contain a skip reason")
        else:
            if self.terminal_stage != "preflight":
                raise ValueError("Skipped builds must terminate during preflight")
            if not self.skip_reason:
                raise ValueError("Skipped builds require a reason")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Skipped builds cannot contain error details")
        return self


class BuildLedger(BaseModel):
    """Versioned collection of immutable game build outcomes."""

    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1] = 1
    records: list[GameBuildRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_attempts(self) -> BuildLedger:
        attempt_ids = [record.attempt_id for record in self.records]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("Build ledger contains duplicate attempt IDs")
        attempt_keys = [
            (record.run_id, record.game_id, record.attempt_number)
            for record in self.records
        ]
        if len(attempt_keys) != len(set(attempt_keys)):
            raise ValueError("Build ledger contains duplicate game attempt numbers")
        return self
