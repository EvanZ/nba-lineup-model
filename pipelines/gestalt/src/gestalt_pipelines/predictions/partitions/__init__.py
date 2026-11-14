"""Forecast dates, including tomorrow's injury-report window."""
import os
from dagster import DailyPartitionsDefinition
from ...utils import PACIFIC

game_prediction_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"], timezone=PACIFIC.key, end_offset=2,
)

actual_minutes_prediction_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"], timezone=PACIFIC.key,
)

__all__ = ["game_prediction_dates", "actual_minutes_prediction_dates"]
