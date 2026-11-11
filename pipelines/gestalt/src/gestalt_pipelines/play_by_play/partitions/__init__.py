"""Completed Pacific game dates for the configured NBA season."""

import os

from dagster import DailyPartitionsDefinition

play_by_play_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone="America/Los_Angeles",
)

__all__ = ["play_by_play_dates"]
