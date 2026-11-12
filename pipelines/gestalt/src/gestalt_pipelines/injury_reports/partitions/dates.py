"""NBA publication dates and Pacific game dates for injury reports."""

import os
import re
from datetime import date

from dagster import DailyPartitionsDefinition

from ...utils import EASTERN, PACIFIC

INJURY_REPORT_SEASON = os.environ["GESTALT_NBA_SEASON"]
start_date = date.fromisoformat(os.environ["GESTALT_NBA_SEASON_START_DATE"])
season_match = re.fullmatch(r"([0-9]{4})-([0-9]{2})", INJURY_REPORT_SEASON)
if season_match is None:
    raise ValueError("GESTALT_NBA_SEASON must be in YYYY-YY format.")
season_start_year = int(season_match[1])
if int(season_match[2]) != (season_start_year + 1) % 100:
    raise ValueError("GESTALT_NBA_SEASON must use consecutive years.")
if start_date.year not in {season_start_year, season_start_year + 1}:
    raise ValueError("GESTALT_NBA_SEASON_START_DATE must be within GESTALT_NBA_SEASON.")

injury_game_dates = DailyPartitionsDefinition(
    start_date=start_date.isoformat(),
    timezone=PACIFIC.key,
    end_offset=2,  # Today and tomorrow.
)


injury_publication_dates = DailyPartitionsDefinition(
    start_date=start_date.isoformat(),
    timezone=EASTERN.key,
    end_offset=1,  # Include the current Eastern publication day.
)
