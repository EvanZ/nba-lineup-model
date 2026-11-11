"""Daily transaction partitions for the configured NBA season."""

import os

from dagster import DailyPartitionsDefinition

transaction_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone="America/Los_Angeles",
    end_offset=1,
)

__all__ = ["transaction_dates"]
