"""Publish today's ratings from completed games through yesterday."""

from datetime import UTC, datetime, timedelta

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ...utils import PACIFIC
from ..jobs import update_player_ratings
from ..partitions import ratings_dates


@schedule(
    job=update_player_ratings,
    cron_schedule="15 4 * * *",
    execution_timezone=PACIFIC.key,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Update today's player ratings/HCA at 4:15 a.m. Pacific using games through yesterday.",
)
def update_player_ratings_schedule(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(days=1):
        return SkipReason("This morning run was missed. Use a manual backfill for older dates.")
    day = tick.date()
    if day < ratings_dates.start.date():
        return SkipReason("The ratings date is before the configured NBA season start.")
    return RunRequest(
        run_key=tick.astimezone(UTC).isoformat(), partition_key=day.isoformat(),
        tags={"dagster/priority": "-1", "ratings/morning_date": day.isoformat()},
    )


__all__ = ["update_player_ratings_schedule"]
