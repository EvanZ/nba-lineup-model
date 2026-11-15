"""Completed game dates; each state includes games through its own date."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

ratings_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"], timezone=PACIFIC.key,
)

__all__ = ["ratings_dates"]
