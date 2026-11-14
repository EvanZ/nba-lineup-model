"""Ratings effective dates, including today using observations through yesterday."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

ratings_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"], timezone=PACIFIC.key, end_offset=1,
)

__all__ = ["ratings_dates"]
