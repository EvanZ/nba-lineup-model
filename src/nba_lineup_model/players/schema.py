from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.season.schema import (
    PARTITION_VALUE_PATTERN,
    SEASON_PATTERN,
    SHA256_PATTERN,
    validate_season,
)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime values must include a timezone")
    return value.astimezone(UTC)


class PlayerIdentity(BaseModel):
    """One historical NBA player identity and latest listed attributes."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    player_id: int = Field(gt=0)
    first_name: str | None = None
    last_name: str | None = None
    display_name: str = Field(min_length=1)
    player_slug: str | None = None
    listed_position: str | None = None
    height_raw: str | None = None
    height_inches: int | None = Field(default=None, ge=48, le=100)
    weight_pounds: int | None = Field(default=None, ge=80, le=500)
    college: str | None = None
    country: str | None = None
    draft_year: int | None = Field(default=None, ge=1946, le=2100)
    draft_round: int | None = Field(default=None, ge=1, le=20)
    draft_number: int | None = Field(default=None, ge=1, le=500)
    is_undrafted: bool | None = None
    roster_status: bool | None = None
    from_year: int | None = Field(default=None, ge=1946, le=2100)
    to_year: int | None = Field(default=None, ge=1946, le=2100)
    latest_team_id: int | None = Field(default=None, gt=0)
    latest_team_abbreviation: str | None = None
    latest_team_slug: str | None = None
    latest_jersey_number: str | None = None
    is_defunct_team: bool | None = None
    supplemental_status: str | None = None
    source_season: str = Field(pattern=SEASON_PATTERN)
    source_url: str = Field(pattern=r"^https?://", min_length=1)
    source_fetched_at: datetime
    source_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("source_season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("source_fetched_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_draft_and_career(self) -> PlayerIdentity:
        if self.is_undrafted and any(
            value is not None for value in (self.draft_round, self.draft_number)
        ):
            raise ValueError("Undrafted players cannot have a draft round or number")
        if (
            self.from_year is not None
            and self.to_year is not None
            and self.to_year < self.from_year
        ):
            raise ValueError("Player career end year precedes start year")
        return self


class PlayerCatalog(BaseModel):
    """Versioned historical NBA player identity catalog."""

    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1] = 1
    players: list[PlayerIdentity] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_players(self) -> PlayerCatalog:
        player_ids = [player.player_id for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("Player catalog contains duplicate player IDs")
        return self


class PlayerSeasonBio(BaseModel):
    """Leakage-safe physical and background fields for one player-season."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    player_id: int = Field(gt=0)
    player_name: str = Field(min_length=1)
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)
    team_id: int = Field(gt=0)
    team_abbreviation: str = Field(pattern=r"^[A-Z0-9]{2,3}$")
    age: float = Field(ge=15, le=80)
    listed_position: str | None = None
    height_raw: str | None = None
    height_inches: int | None = Field(default=None, ge=48, le=100)
    weight_pounds: int | None = Field(default=None, ge=80, le=500)
    college: str | None = None
    country: str | None = None
    draft_year: int | None = Field(default=None, ge=1946, le=2100)
    draft_round: int | None = Field(default=None, ge=1, le=20)
    draft_number: int | None = Field(default=None, ge=1, le=500)
    is_undrafted: bool
    player_index_source_sha256: str = Field(pattern=SHA256_PATTERN)
    bio_source_url: str = Field(pattern=r"^https?://", min_length=1)
    bio_source_fetched_at: datetime
    bio_source_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("bio_source_fetched_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_draft(self) -> PlayerSeasonBio:
        if self.is_undrafted and any(
            value is not None for value in (self.draft_round, self.draft_number)
        ):
            raise ValueError("Undrafted players cannot have a draft round or number")
        return self


class PlayerSeasonBioDataset(BaseModel):
    """Versioned player-season bio rows for one season and competition type."""

    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1] = 1
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)
    players: list[PlayerSeasonBio] = Field(min_length=1)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @model_validator(mode="after")
    def validate_players(self) -> PlayerSeasonBioDataset:
        player_ids = [player.player_id for player in self.players]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("Player-season bio dataset contains duplicate player IDs")
        if any(
            player.season != self.season or player.season_type != self.season_type
            for player in self.players
        ):
            raise ValueError("Player-season bio rows do not match their dataset")
        return self


class PlayerSeasonBioManifest(BaseModel):
    """Integrity and source evidence for one player-season bio partition."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)
    created_at: datetime
    player_count: int = Field(ge=1)
    player_ids: tuple[int, ...] = Field(min_length=1)
    player_index_source_sha256: str = Field(pattern=SHA256_PATTERN)
    bio_source_sha256: str = Field(pattern=SHA256_PATTERN)
    part_filename: Literal["part-00000.parquet"] = "part-00000.parquet"
    row_count: int = Field(ge=1)
    byte_count: int = Field(gt=0)
    part_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def validate_counts(self) -> PlayerSeasonBioManifest:
        if self.player_count != len(self.player_ids):
            raise ValueError("Player manifest count does not match player IDs")
        if len(set(self.player_ids)) != self.player_count:
            raise ValueError("Player manifest IDs must be unique")
        if self.row_count != self.player_count:
            raise ValueError("Player manifest rows must match player count")
        return self
