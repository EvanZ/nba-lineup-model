"""Rebuild a date's rosters when its transaction collection adds rows."""

from datetime import date, timedelta

from dagster import (
    AssetKey, DagsterEventType, DagsterRunStatus, DefaultSensorStatus,
    RunRequest, RunStatusSensorContext, SensorResult, SkipReason, run_status_sensor,
)

from ...transactions.jobs import capture_nba_transactions
from ..jobs import build_nba_rosters


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS,
    monitored_jobs=[capture_nba_transactions],
    request_job=build_nba_rosters,
    minimum_interval_seconds=30,
    default_status=DefaultSensorStatus.STOPPED,
    description="Rebuild changed transaction dates through the poll date, including late reports.",
)
def roster_transactions_ready(context: RunStatusSensorContext):
    records = context.instance.get_records_for_run(
        context.dagster_run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION,
    ).records
    changed = set()
    partitions = set()
    for record in records:
        materialization = record.event_log_entry.dagster_event.event_specific_data.materialization
        if materialization.asset_key != AssetKey("transactions_by_date"):
            continue
        day = date.fromisoformat(materialization.partition)
        partitions.add(day)
        if materialization.metadata["new_transaction_count"].value:
            changed.add(day)
    if not changed:
        return SkipReason("Transaction collection added no rows; the daily roster run still applies.")
    first, last = min(changed), max(partitions)
    days = [first + timedelta(days=i) for i in range((last-first).days+1)]
    return SensorResult(run_requests=[
        RunRequest(
            run_key=f"{context.dagster_run.run_id}:{day}",
            partition_key=day.isoformat(),
        )
        for day in days
    ])


__all__ = ["roster_transactions_ready"]
