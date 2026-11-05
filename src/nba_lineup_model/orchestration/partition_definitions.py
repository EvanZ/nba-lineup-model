"""Isolated #56 Dagster entrypoint; no legacy job imports or collection jobs.

The sensor is stopped by default. It executes explicitly pinned requests only;
neither a successful run nor a request file authorizes environment activation.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from dagster import (
    DefaultSensorStatus, Definitions, Field, RunRequest, RunsFilter, SkipReason,
    job, op, sensor, in_process_executor, DagsterRunStatus,
)

from nba_lineup_model.orchestration.daily_work import read_object, verify_spec
from nba_lineup_model.orchestration.daily_instance import run_guard, binding_for_instance


@op(config_schema={"config_path": str, "spec_path": str,
                   "approval_path": Field(str, is_required=False)})
def prepare_daily_partition_op(context):
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
    from nba_lineup_model.orchestration.resumable_pipeline import prepare_partition
    cfg = context.op_config
    spec = read_object(Path(cfg["spec_path"]))
    approval = read_object(Path(cfg["approval_path"])) if cfg.get("approval_path") else None
    with run_guard(context, "prepare_daily_partition", cfg):
        result = prepare_partition(DailyPipelineConfig.read(cfg["config_path"]), spec,
                                   approval=approval)
    context.add_output_metadata({"work_id": result["work_id"],
        "status": result["status"], "runtime_release": result["runtime_release"]})
    return result


@job(name="prepare_daily_partition", executor_def=in_process_executor,
     tags={"gestalt/workload": "daily_site_release"})
def prepare_daily_partition_job():
    prepare_daily_partition_op()


@sensor(name="resumable_daily_partition", job=prepare_daily_partition_job,
        minimum_interval_seconds=300, default_status=DefaultSensorStatus.STOPPED)
def resumable_daily_partition_sensor(context):
    binding = binding_for_instance(context.instance)
    queue = binding.requests
    configured = os.environ.get("GESTALT_DAILY_REQUEST_ROOT")
    if configured and Path(configured).absolute() != queue:
        raise ValueError("Daily request queue differs from its environment binding")
    pending = []
    for path in sorted(Path(queue).glob("*.json")):
        request = read_object(path)
        binding.tags("prepare_daily_partition", request)
        spec = read_object(Path(request["spec_path"]))
        verify_spec(spec)
        pending.append((spec["identity"]["game_date_pacific"], spec["work_id"], request))
    active = context.instance.get_run_records(
        filters=RunsFilter(job_name="prepare_daily_partition", statuses=[
            DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING, DagsterRunStatus.STARTED,
            DagsterRunStatus.CANCELING]), limit=1)
    if active:
        return SkipReason("One daily partition is already queued or running")
    for day, work_id, request in sorted(pending):
        runs = context.instance.get_run_records(filters=RunsFilter(
            job_name="prepare_daily_partition", tags={"gestalt/work_id": work_id}), limit=3)
        if any(run.dagster_run.status == DagsterRunStatus.SUCCESS for run in runs):
            continue
        if len(runs) >= 3:
            return SkipReason(f"{day}: three failed attempts; review checkpoint failure before continuing")
        if runs and (datetime.now(UTC) - runs[0].update_timestamp).total_seconds() < 900:
            return SkipReason(f"{day}: retry cooling down for 15 minutes")
        return RunRequest(run_key=f"daily:{work_id}:attempt-{len(runs)+1}",
            run_config={"ops":{"prepare_daily_partition_op":{"config":request}}},
            tags={**binding.tags("prepare_daily_partition", request),
                  "gestalt/trigger": "sensor"})
    return SkipReason("All explicitly requested partitions are validated; activation remains separate")


@op(config_schema={"config_path": str, "forecast_date": str,
    "prepared_schedule": Field(str, is_required=False),
    "approval_reference": Field(str, is_required=False)})
def prepare_forecast_date_op(context):
    from datetime import date
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
    from nba_lineup_model.orchestration.forecast_partition import prepare_forecast_date
    with run_guard(context, "prepare_forecast_date", context.op_config):
        kwargs = {}
        if context.op_config.get("prepared_schedule"):
            kwargs = {"prepared_schedule": Path(context.op_config["prepared_schedule"]),
                      "approval_reference": context.op_config.get("approval_reference")}
        result = prepare_forecast_date(DailyPipelineConfig.read(context.op_config["config_path"]),
                                      date.fromisoformat(context.op_config["forecast_date"]), **kwargs)
    context.add_output_metadata({key: result[key] for key in (
        "forecast_date", "model_date", "model_sha256", "work_id", "schedule_artifact")})
    return result


@job(name="prepare_forecast_date", executor_def=in_process_executor,
     tags={"gestalt/workload": "daily_site_release"})
def prepare_forecast_date_job():
    prepare_forecast_date_op()


defs = Definitions(jobs=[prepare_daily_partition_job, prepare_forecast_date_job],
                   sensors=[resumable_daily_partition_sensor])
