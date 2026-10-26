"""Calendar-date helpers for NBA slates shown in Pacific time."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


def pacific_date(value: object) -> date:
    """Convert an absolute timestamp to its Pacific calendar date."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(PACIFIC).date()


def summary_slate_date(summary: Mapping[str, Any]) -> date:
    """Return a final-summary's Pacific slate date, with legacy fallback."""
    pregame = summary.get("pregame")
    pregame_start = pregame.get("start_time") if isinstance(pregame, Mapping) else None
    start_time = summary.get("start_time") or pregame_start
    if start_time:
        try:
            return pacific_date(start_time)
        except ValueError:
            pass
    return date.fromisoformat(str(summary["game_date"]))
