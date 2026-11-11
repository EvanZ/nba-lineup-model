"""Refresh yesterday's final play-by-play each Pacific morning."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ..jobs import capture_nba_play_by_play
from ..partitions import play_by_play_dates

PACIFIC = ZoneInfo(play_by_play_dates.timezone)


@schedule(
    job=capture_nba_play_by_play,
    cron_schedule="0 3 * * *",
    execution_timezone=play_by_play_dates.timezone,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Refresh the preceding Pacific game date at 3 a.m. Pacific each morning.",
)
def play_by_play_morning(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(days=1):
        return SkipReason("This morning poll was missed. Use a manual backfill for older dates.")
    day = tick.date() - timedelta(days=1)
    if day < play_by_play_dates.start.date():
        return SkipReason("The preceding date is before the configured NBA season start.")
    return RunRequest(run_key=tick.astimezone(UTC).isoformat(), partition_key=day.isoformat())


__all__ = ["play_by_play_morning"]
