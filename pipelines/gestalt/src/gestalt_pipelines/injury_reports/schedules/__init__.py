"""Schedules for the official injury report archive."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ..jobs import capture_injury_pdfs
from ..partitions import injury_publication_dates

EASTERN = ZoneInfo(injury_publication_dates.timezone)


@schedule(
    job=capture_injury_pdfs,
    cron_schedule="2,17,32,47 * * * *",
    execution_timezone=injury_publication_dates.timezone,
    default_status=DefaultScheduleStatus.STOPPED,
)
def injury_pdf_poll(context: ScheduleEvaluationContext):
    """Check the preceding hour for official reports at each quarter-hour release."""
    end = context.scheduled_execution_time.astimezone(UTC)
    if datetime.now(UTC) - end >= timedelta(minutes=15):
        yield SkipReason("This poll was missed. Use a manual publication-date backfill for older gaps.")
        return
    start_day = (end - timedelta(hours=1)).astimezone(EASTERN).date()
    end_day = end.astimezone(EASTERN).date()
    for day in sorted({start_day, end_day}):
        if day >= injury_publication_dates.start.date():
            yield RunRequest(
                run_key=f"{end.isoformat()}:{day}",
                partition_key=day.isoformat(),
                run_config={"ops": {"injury_pdf_archive": {"config": {"window_end_utc": end.isoformat()}}}},
                tags={"dagster/priority": "10"},
            )


__all__ = ["injury_pdf_poll"]
