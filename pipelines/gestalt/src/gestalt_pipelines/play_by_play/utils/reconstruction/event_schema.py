from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """A typed event derived from one NBA play-by-play action."""

    model_config = ConfigDict(strict=True)

    game_id: str
    event_id: str
    event_index: int = Field(ge=0)
    source_action_number: int
    source_order_number: int
    period: int = Field(ge=1)
    period_type: str
    source_clock: str
    clock: str
    seconds_remaining_period: float = Field(ge=0)
    elapsed_game_seconds: float = Field(ge=0)
    event_type: str
    event_subtype: str | None = None
    descriptor: str | None = None
    team_id: int | None = None
    team_tricode: str | None = None
    player_id: int | None = None
    related_player_ids: tuple[int, ...] = ()
    qualifiers: tuple[str, ...] = ()
    source_possession_team_id: int | None = None
    score_home: int = Field(ge=0)
    score_away: int = Field(ge=0)
    score_home_delta: int
    score_away_delta: int
    is_field_goal: bool = False
    shot_result: str | None = None
    description: str | None = None
    validation_flags: tuple[str, ...] = ()
