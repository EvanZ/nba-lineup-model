"""Independent Pacific roster-date partitions."""

import os
from dagster import DailyPartitionsDefinition

roster_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone="America/Los_Angeles",
    end_offset=1,
)

__all__ = ["roster_dates"]
