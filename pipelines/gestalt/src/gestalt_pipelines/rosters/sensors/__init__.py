"""Rebuild a date's rosters when its transaction collection adds rows."""

from dagster import (
    AssetKey, DagsterEventType, DagsterRunStatus, DefaultSensorStatus,
    RunRequest, RunStatusSensorContext, SkipReason, run_status_sensor,
)

from ...transactions.jobs import capture_nba_transactions
from ..jobs import build_nba_rosters


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS,
    monitored_jobs=[capture_nba_transactions],
    request_job=build_nba_rosters,
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.STOPPED,
    description="Build the same Pacific roster date after transaction collection adds new rows.",
)
def roster_transactions_ready(context: RunStatusSensorContext):
    records = context.instance.get_records_for_run(
        context.dagster_run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION,
    ).records
    for record in records:
        materialization = record.event_log_entry.dagster_event.event_specific_data.materialization
        if materialization.asset_key != AssetKey("transactions_by_date"):
            continue
        if materialization.metadata["new_transaction_count"].value == 0:
            return SkipReason("Transaction collection added no rows; the daily roster run still applies.")
        return RunRequest(
            run_key=context.dagster_run.run_id,
            partition_key=materialization.partition,
        )
    return SkipReason("The successful run did not materialize transactions_by_date.")


__all__ = ["roster_transactions_ready"]
