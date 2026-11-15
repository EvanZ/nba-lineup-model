"""Shared league, timezone, and storage definitions."""

from datetime import date, timedelta
from zoneinfo import ZoneInfo

NBA_TEAM_IDS = tuple(range(1610612737, 1610612767))
PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")
DATA_BUCKET = "nba-gestalt-data"
ERRORS_PREFIX = "lake/artifacts/errors"

__all__ = ["NBA_TEAM_IDS", "PACIFIC", "EASTERN", "DATA_BUCKET", "ERRORS_PREFIX"]


COMPLETED_GAME_DATE_CONTRACT = "completed_game_date_v1"


def forecast_observed_date(game_day: date, today: date) -> date:
    """Historical games use D−1; tomorrow's provisional forecast uses today's available state."""
    return min(game_day, today)-timedelta(days=1)


__all__ += ["COMPLETED_GAME_DATE_CONTRACT", "forecast_observed_date"]
