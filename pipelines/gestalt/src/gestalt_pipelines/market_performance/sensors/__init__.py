"""Evaluate the same date after its actual-minute counterfactual succeeds."""
from hashlib import sha256

from dagster import (
    DagsterEventType, DagsterRunStatus, DefaultSensorStatus, RunRequest,
    RunStatusSensorContext, SkipReason, run_status_sensor,
)

from ...predictions.jobs import predict_game_margins_actual_minutes
from ..jobs import evaluate_market_performance
from ..utils import implementation_identity, json_bytes


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS, monitored_jobs=[predict_game_margins_actual_minutes],
    request_job=evaluate_market_performance, minimum_interval_seconds=60,
    default_status=DefaultSensorStatus.STOPPED,
    description="Evaluate the same completed date after actual-minute counterfactuals succeed.",
)
def evaluate_market_performance_sensor(context: RunStatusSensorContext):
    day = context.dagster_run.tags.get("dagster/partition")
    if not day:
        return SkipReason("The successful counterfactual run has no single game-date partition.")
    events = context.instance.get_records_for_run(context.dagster_run.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION).records
    outputs = None
    for entry in events:
        materialization = entry.event_log_entry.dagster_event.event_specific_data.materialization
        if materialization.asset_key.to_user_string() == "game_predictions_actual_minutes":
            outputs = {gid: {key: ref[key] for key in ("uri", "sha256")}
                       for gid, ref in materialization.metadata["counterfactuals"].value.items()}
    if outputs is None:
        return SkipReason("No same-run actual-minute materialization was found.")
    key = sha256(json_bytes({"day": day, "counterfactuals": outputs, "implementation": implementation_identity()})).hexdigest()
    return RunRequest(run_key=key, partition_key=day, tags={"dagster/priority": "-1"})


__all__ = ["evaluate_market_performance_sensor"]
