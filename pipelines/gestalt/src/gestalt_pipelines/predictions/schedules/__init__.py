"""Daily postgame actual-minute counterfactuals."""
from datetime import UTC, datetime, timedelta

from dagster import DefaultScheduleStatus, RunRequest, ScheduleEvaluationContext, SkipReason, schedule

from ...utils import PACIFIC
from ..jobs import predict_game_margins_actual_minutes
from ..partitions import actual_minutes_prediction_dates


@schedule(
    job=predict_game_margins_actual_minutes,
    cron_schedule="30 4 * * *", execution_timezone=PACIFIC.key,
    default_status=DefaultScheduleStatus.STOPPED,
    description="At 04:30 Pacific, compare yesterday's saved predictions with final actual minutes.",
)
def predict_game_margins_actual_minutes_schedule(context: ScheduleEvaluationContext):
    tick = context.scheduled_execution_time.astimezone(PACIFIC)
    if datetime.now(UTC)-tick.astimezone(UTC) >= timedelta(days=1):
        return SkipReason("This morning run was missed. Use a manual backfill for older game dates.")
    day = tick.date()-timedelta(days=1)
    if day < actual_minutes_prediction_dates.start.date():
        return SkipReason("The game date precedes the configured NBA season start.")
    return RunRequest(
        run_key=tick.astimezone(UTC).isoformat(), partition_key=str(day),
        tags={"dagster/priority": "-1"},
    )


__all__ = ["predict_game_margins_actual_minutes_schedule"]
