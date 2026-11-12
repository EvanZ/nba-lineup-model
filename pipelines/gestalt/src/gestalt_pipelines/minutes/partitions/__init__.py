"""Daily Pacific states, including explicit no-game dates."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

minutes_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone=PACIFIC.key,
)

__all__ = ["minutes_dates"]
