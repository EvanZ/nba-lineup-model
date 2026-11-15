import os
from dagster import DailyPartitionsDefinition
from ...utils import PACIFIC

market_performance_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"], timezone=PACIFIC.key,
)

__all__ = ["market_performance_dates"]
