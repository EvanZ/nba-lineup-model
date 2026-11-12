"""Ensure every completed Pacific date has a roster, even without transactions."""

from datetime import UTC, datetime, timedelta

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ..jobs import build_nba_rosters
from ..partitions import roster_dates

from ...utils import PACIFIC



@schedule(
    job=build_nba_rosters,
    cron_schedule="30 3 * * *",
    execution_timezone=roster_dates.timezone,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Build yesterday's Pacific rosters at 3:30 a.m., including days without transactions.",
)
def rosters_morning(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(days=1):
        return SkipReason("This morning run was missed. Use a manual backfill for older dates.")
    day = tick.date() - timedelta(days=1)
    if day < roster_dates.start.date():
        return SkipReason("The preceding date is before the configured NBA season start.")
    return RunRequest(run_key=tick.astimezone(UTC).isoformat(), partition_key=day.isoformat())


__all__ = ["rosters_morning"]
