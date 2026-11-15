"""Pacific capture-date polling for NBA market odds."""

from datetime import UTC, datetime, timedelta

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ..jobs import archive_nba_market_odds
from ...utils import PACIFIC


@schedule(
    job=archive_nba_market_odds,
    name="nba_market_odds_game_day",
    cron_schedule="20 8,12,15-19 * * *",
    execution_timezone=PACIFIC.key,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Capture NBA odds at 08:20, 12:20 and 15:20–19:20 Pacific, under their capture date.",
)
def nba_market_odds_game_day(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC)-tick.astimezone(UTC) >= timedelta(hours=1):
        return SkipReason("This live odds poll was missed; the next poll captures current prices.")
    return RunRequest(run_key=tick.astimezone(UTC).isoformat(), partition_key=tick.date().isoformat())


__all__ = ["nba_market_odds_game_day"]
