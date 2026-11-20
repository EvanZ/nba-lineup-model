"""Daytime and end-of-day transaction polling in Pacific time."""

from datetime import UTC, datetime, timedelta

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ..jobs import capture_nba_transactions
from ..partitions import transaction_dates

from ...utils import PACIFIC



@schedule(
    job=capture_nba_transactions,
    cron_schedule=["0 8-20 * * *", "55 23 * * *", "15 3 * * *"],
    execution_timezone=transaction_dates.timezone,
    default_status=DefaultScheduleStatus.STOPPED,
    description="Poll transactions from 8 a.m. through 8 p.m., at 23:55, and check yesterday at 03:15 Pacific.",
)
def transactions_hourly(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC) - tick.astimezone(UTC) >= timedelta(hours=1):
        return SkipReason("This transaction poll was missed. The next poll checks today and yesterday.")
    day = tick.date() - timedelta(days=1) if tick.hour == 3 else tick.date()
    if day < transaction_dates.start.date():
        return SkipReason("The requested date precedes the configured NBA season start.")
    return RunRequest(run_key=tick.astimezone(UTC).isoformat(), partition_key=day.isoformat())


__all__ = ["transactions_hourly"]
