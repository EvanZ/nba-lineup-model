from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nba_lineup_model.season.schema import (
    PARTITION_VALUE_PATTERN,
    SEASON_PATTERN,
    validate_season,
)

CURATED_TABLES = (
    "events",
    "players",
    "event_lineups",
    "lineup_stints",
    "possessions",
    "possession_segments",
)
CuratedTable = Literal[
    "events",
    "players",
    "event_lineups",
    "lineup_stints",
    "possessions",
    "possession_segments",
]


class CuratedPartition(BaseModel):
    """One season and season-type partition of a curated table."""

    model_config = ConfigDict(strict=True, extra="forbid")

    table: CuratedTable
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: str = Field(pattern=PARTITION_VALUE_PATTERN)

    @field_validator("season")
    @classmethod
    def validate_season_value(cls, season: str) -> str:
        return validate_season(season)


@dataclass(frozen=True)
class CuratedDatasetLayout:
    """Construct deterministic Hive-style paths for curated Parquet datasets."""

    root: Path = Path("data/curated")

    def partition_dir(self, partition: CuratedPartition) -> Path:
        return (
            self.root
            / partition.table
            / f"season={partition.season}"
            / f"season_type={partition.season_type}"
        )

    def part_path(self, partition: CuratedPartition, part_number: int = 0) -> Path:
        if part_number < 0:
            raise ValueError("part_number must be non-negative")
        return self.partition_dir(partition) / f"part-{part_number:05d}.parquet"
