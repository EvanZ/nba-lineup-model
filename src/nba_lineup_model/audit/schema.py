from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nba_lineup_model.ingest.nba_cdn import validate_game_id


class AuditGameSpec(BaseModel):
    """One game selected for a cross-season reconstruction audit."""

    model_config = ConfigDict(strict=True, extra="forbid")

    game_id: str
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    season_type: str = Field(min_length=1)
    sample_group: str = Field(default="default", min_length=1)
    expected_overtime: bool | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("game_id")
    @classmethod
    def validate_id(cls, game_id: str) -> str:
        return validate_game_id(game_id)


class AuditManifest(BaseModel):
    """Versioned, reproducible list of games to audit."""

    model_config = ConfigDict(strict=True, extra="forbid")

    version: Literal[1] = 1
    games: list[AuditGameSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_games(self) -> AuditManifest:
        game_ids = [game.game_id for game in self.games]
        if len(game_ids) != len(set(game_ids)):
            raise ValueError("Audit manifest contains duplicate game IDs")
        return self

    @classmethod
    def read(cls, path: Path | str) -> AuditManifest:
        return cls.model_validate_json(Path(path).read_text())

    def write(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n")
        return output_path


class AuditGameResult(BaseModel):
    """Compact audit outcome for one game."""

    model_config = ConfigDict(strict=True)

    game_id: str
    season: str
    season_type: str
    sample_group: str
    status: Literal["pass", "warning", "fail", "error"]
    game_time_utc: str | None = None
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_tricode: str | None = None
    away_tricode: str | None = None
    score_home: int | None = None
    score_away: int | None = None
    period_count: int | None = None
    is_overtime: bool | None = None
    event_count: int | None = None
    lineup_stint_count: int | None = None
    possession_count: int | None = None
    home_possession_count: int | None = None
    away_possession_count: int | None = None
    possession_count_difference: int | None = None
    possession_segment_count: int | None = None
    multi_segment_possession_count: int | None = None
    source_possession_override_count: int | None = None
    source_change_terminal_count: int | None = None
    opponent_technical_free_throw_possession_count: int | None = None
    estimated_home_possessions: float | None = None
    estimated_away_possessions: float | None = None
    home_possession_estimate_difference: float | None = None
    away_possession_estimate_difference: float | None = None
    score_matches_boxscore: bool | None = None
    possession_score_conserved: bool | None = None
    segment_score_conserved: bool | None = None
    segment_duration_conserved: bool | None = None
    balanced_possession_counts: bool | None = None
    lineup_warning_count: int = Field(default=0, ge=0)
    lineup_error_count: int = Field(default=0, ge=0)
    possession_warning_count: int = Field(default=0, ge=0)
    possession_error_count: int = Field(default=0, ge=0)
    segment_warning_count: int = Field(default=0, ge=0)
    segment_error_count: int = Field(default=0, ge=0)
    issue_codes: tuple[str, ...] = ()
    error_stage: Literal["fetch", "reconstruct", "audit"] | None = None
    error_type: str | None = None
    error_message: str | None = None
