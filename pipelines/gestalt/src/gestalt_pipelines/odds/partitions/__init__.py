"""Daily Pacific dates supported by the saved source archive."""

import os

from dagster import DailyPartitionsDefinition

from ...utils import PACIFIC

odds_capture_dates = DailyPartitionsDefinition(
    start_date=os.environ["GESTALT_ODDS_CAPTURE_START_DATE"],
    timezone=PACIFIC.key,
    end_offset=1,
)

__all__ = ["odds_capture_dates"]
