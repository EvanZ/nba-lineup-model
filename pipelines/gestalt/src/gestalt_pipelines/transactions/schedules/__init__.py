"""Hourly transaction polling during the Pacific daytime."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ..jobs import capture_nba_transactions
from ..partitions import transaction_dates

PACIFIC = ZoneInfo(transaction_dates.timezone)


@schedule(
    job=capture_nba_transactions,
    cron_schedule="0 8-20 * * *",
    execution_timezone=transaction_dates.timezone,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Refresh today's transactions hourly from 8 a.m. through 8 p.m. Pacific.",
)
def transactions_hourly(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(hours=1):
        return SkipReason("This transaction poll was missed. The next poll refreshes the current date.")
    return RunRequest(run_key=tick.astimezone(UTC).isoformat(), partition_key=tick.date().isoformat())


__all__ = ["transactions_hourly"]
