"""Load definitions with the season used by the test fixtures."""

import os

os.environ["GESTALT_NBA_SEASON"] = "2026-27"
os.environ["GESTALT_NBA_SEASON_START_DATE"] = "2026-10-20"

os.environ["GESTALT_ODDS_CAPTURE_START_DATE"] = "2026-10-26"
os.environ["GESTALT_SCOREBOARD_START_DATE"] = "2026-11-10"
