"""Pacific game dates for the configured NBA season."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

box_score_game_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone=PACIFIC.key,
    end_offset=1,
)

__all__ = ["box_score_game_dates"]
