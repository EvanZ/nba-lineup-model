from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PossessionStartReason(StrEnum):
    SOURCE_SIGNAL = "source_signal"
    MADE_SCORE = "made_score"
    TURNOVER = "turnover"
    DEFENSIVE_REBOUND = "defensive_rebound"
    JUMP_BALL = "jump_ball"


class PossessionTerminalReason(StrEnum):
    MADE_FIELD_GOAL = "made_field_goal"
    MADE_FINAL_FREE_THROW = "made_final_free_throw"
    TURNOVER = "turnover"
    DEFENSIVE_REBOUND = "defensive_rebound"
    HELD_BALL = "held_ball"
    PERIOD_END = "period_end"
    SOURCE_CHANGE = "source_change"
    FEED_END = "feed_end"


class Possession(BaseModel):
    """One team possession, independent of lineup changes within it."""

    model_config = ConfigDict(strict=True)

    game_id: str
    possession_id: str
    possession_index: int = Field(ge=0)
    period_possession_index: int = Field(ge=0)
    period: int = Field(ge=1)
    period_type: str
    offense_team_id: int
    defense_team_id: int
    start_event_id: str
    end_event_id: str
    start_event_index: int = Field(ge=0)
    end_event_index: int = Field(ge=0)
    start_order_number: int
    end_order_number: int
    start_clock: str
    end_clock: str
    start_elapsed_game_seconds: float = Field(ge=0)
    end_elapsed_game_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    points_home: int
    points_away: int
    offense_points: int
    event_count: int = Field(ge=1)
    start_reason: PossessionStartReason
    terminal_reason: PossessionTerminalReason
    source_possession_mismatch_count: int = Field(ge=0)
    validation_flags: tuple[str, ...] = ()


class EventPossession(BaseModel):
    """Assignment of one substantive event to one possession."""

    model_config = ConfigDict(strict=True)

    game_id: str
    event_id: str
    event_index: int = Field(ge=0)
    possession_id: str
    possession_index: int = Field(ge=0)
    offense_team_id: int


class PossessionSegment(BaseModel):
    """A possession interval played by one fixed five-on-five lineup."""

    model_config = ConfigDict(strict=True)

    game_id: str
    segment_id: str
    segment_index: int = Field(ge=0)
    possession_id: str
    possession_index: int = Field(ge=0)
    possession_segment_index: int = Field(ge=0)
    period: int = Field(ge=1)
    period_type: str
    offense_team_id: int
    defense_team_id: int
    home_player_ids: tuple[int, int, int, int, int]
    away_player_ids: tuple[int, int, int, int, int]
    start_event_id: str
    end_event_id: str
    start_event_index: int = Field(ge=0)
    end_event_index: int = Field(ge=0)
    start_order_number: int
    end_order_number: int
    start_clock: str
    end_clock: str
    start_elapsed_game_seconds: float = Field(ge=0)
    end_elapsed_game_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    points_home: int
    points_away: int
    offense_points: int
    event_count: int = Field(ge=0)
    start_reason: Literal["possession_start", "substitution"]
    end_reason: Literal["substitution", "possession_end"]
    starts_possession: bool
    ends_possession: bool


class PossessionIssue(BaseModel):
    code: str
    severity: Literal["warning", "error"]
    detail: str
    event_id: str | None = None
    possession_id: str | None = None


class PossessionReconstruction(BaseModel):
    game_id: str
    possessions: list[Possession]
    event_possessions: list[EventPossession]
    issues: list[PossessionIssue]


class PossessionSegmentation(BaseModel):
    game_id: str
    segments: list[PossessionSegment]
    issues: list[PossessionIssue]
