"""Follow successful completed-date box refreshes with postgame attribution."""
from datetime import UTC, date, datetime

from dagster import (
    DagsterEventType, DagsterRunStatus, DefaultSensorStatus, RunRequest,
    RunStatusSensorContext, SkipReason, run_status_sensor,
)

from ...box_scores.jobs import capture_nba_box_scores
from ...utils import PACIFIC
from ..jobs import predict_game_margins_actual_minutes


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS, monitored_jobs=[capture_nba_box_scores],
    request_job=predict_game_margins_actual_minutes, minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.STOPPED,
    description="Rebuild postgame attribution for the same completed date after a successful final-box refresh.",
)
def predict_game_margins_actual_minutes_sensor(context: RunStatusSensorContext):
    run = context.dagster_run
    if run.status != DagsterRunStatus.SUCCESS:
        return SkipReason("The final-box refresh did not succeed.")
    day = run.tags.get("dagster/partition")
    if not day:
        return SkipReason("The final-box refresh has no single date partition.")
    if date.fromisoformat(day) >= datetime.now(UTC).astimezone(PACIFIC).date():
        return SkipReason("Postgame attribution requires a completed Pacific date.")
    records = context.instance.get_records_for_run(
        run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION,
    ).records
    for entry in records:
        materialization = entry.event_log_entry.dagster_event.event_specific_data.materialization
        if materialization.asset_key.to_user_string() != "nba_box_scores" or materialization.partition != day:
            continue
        metadata = materialization.metadata
        if not metadata.get("refresh_existing") or not metadata["refresh_existing"].value:
            return SkipReason("Live collection and ordinary reuse do not trigger postgame attribution.")
        boxes = metadata["box_scores"].value
        if metadata["pending_count"].value or len(boxes) != metadata["expected_game_count"].value:
            return SkipReason("The final-box refresh did not cover the complete slate.")
        return RunRequest(
            run_key=f"box-score-refresh:{run.run_id}", partition_key=day,
            tags={"dagster/priority": "-1", "box_scores/refresh_run_id": run.run_id},
        )
    return SkipReason("No complete same-run final-box materialization was found.")
