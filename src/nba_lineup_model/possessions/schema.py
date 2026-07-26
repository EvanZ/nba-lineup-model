from __future__ import annotations

from pydantic import BaseModel, Field


class PossessionRecord(BaseModel):
    game_id: str
    period: int
    possession_index: int
    offense_team_id: int
    defense_team_id: int
    points: int = Field(ge=0)
    start_clock: str
    end_clock: str
    home_lineup: tuple[int, int, int, int, int]
    away_lineup: tuple[int, int, int, int, int]
