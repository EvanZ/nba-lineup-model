"""Shared league, timezone, and storage definitions."""

from zoneinfo import ZoneInfo

NBA_TEAM_IDS = tuple(range(1610612737, 1610612767))
PACIFIC = ZoneInfo("America/Los_Angeles")
EASTERN = ZoneInfo("America/New_York")
DATA_BUCKET = "nba-gestalt-data"
ERRORS_PREFIX = "lake/artifacts/errors"

__all__ = ["NBA_TEAM_IDS", "PACIFIC", "EASTERN", "DATA_BUCKET", "ERRORS_PREFIX"]
