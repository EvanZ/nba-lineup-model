"""Daily Pacific states, including explicit no-game dates."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

minutes_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone=PACIFIC.key,
)

game_minutes_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone=PACIFIC.key, end_offset=2,
)

__all__ = ["minutes_dates", "game_minutes_dates"]
