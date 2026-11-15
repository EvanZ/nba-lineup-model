"""Daily Pacific dates supported by the saved source archive."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

scoreboard_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_SCOREBOARD_START_DATE"],
    timezone=PACIFIC.key,
    end_offset=1,
)

__all__ = ["scoreboard_dates"]
