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
    description="At 03:30 Pacific, finalize yesterday and initialize today, including days without new transactions.",
)
def rosters_morning(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(days=1):
        return SkipReason("This morning run was missed. Use a manual backfill for older dates.")
    days = [tick.date() - timedelta(days=1), tick.date()]
    requests = [
        RunRequest(run_key=f"{tick.astimezone(UTC).isoformat()}:{day}", partition_key=day.isoformat())
        for day in days if day >= roster_dates.start.date()
    ]
    return requests or SkipReason("The requested dates precede the configured NBA season start.")


__all__ = ["rosters_morning"]
