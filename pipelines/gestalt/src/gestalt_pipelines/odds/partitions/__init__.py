"""Daily Pacific dates supported by the saved source archive."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

odds_capture_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_ODDS_CAPTURE_START_DATE"],
    timezone=PACIFIC.key,
    end_offset=1,
)

game_odds_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_NBA_SEASON_START_DATE"],
    timezone=PACIFIC.key,
    end_offset=8,
)

__all__ = ["odds_capture_dates", "game_odds_dates"]
