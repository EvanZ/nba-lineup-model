"""Evidence through each completed Pacific game date."""
import os
from dagster import DailyPartitionsDefinition
from ...utils import PACIFIC

team_rotation_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"], timezone=PACIFIC.key,
)
__all__ = ["team_rotation_dates"]
