"""Update yesterday's player minutes once each morning."""

from datetime import UTC, datetime, timedelta

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ...utils import PACIFIC
from ..jobs import update_player_minutes
from ..partitions import minutes_dates


@schedule(
    job=update_player_minutes,
    cron_schedule="0 4 * * *",
    execution_timezone=PACIFIC.key,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Update yesterday's player minutes at 4 a.m. Pacific; failures require manual repair.",
)
def update_player_minutes_schedule(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(days=1):
        return SkipReason("This morning run was missed. Use a manual backfill for older dates.")
    day = tick.date() - timedelta(days=1)
    if day < minutes_dates.start.date():
        return SkipReason("The preceding date is before the configured NBA season start.")
    return RunRequest(
        run_key=tick.astimezone(UTC).isoformat(), partition_key=day.isoformat(),
        tags={"dagster/priority": "-1", "minutes/morning_date": tick.date().isoformat()},
    )


__all__ = ["update_player_minutes_schedule"]
