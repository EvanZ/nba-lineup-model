"""Daily effective dates; opening-day projections use frozen priors."""

import os
from datetime import date, timedelta

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

minutes_dates = DailyPartitionsDefinition(
    start_date=(date.fromisoformat(os.environ["GESTALT_NBA_SEASON_START_DATE"])+timedelta(days=1)).isoformat(),
    timezone=PACIFIC.key, end_offset=1,
)

game_minutes_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone=PACIFIC.key, end_offset=2,
)

__all__ = ["minutes_dates", "game_minutes_dates"]
