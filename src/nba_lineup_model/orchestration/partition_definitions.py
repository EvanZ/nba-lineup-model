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
    from nba_lineup_model.orchestration.shared_daily_control import automation_enabled
    if not automation_enabled(binding):
        return SkipReason("Dev automation is disabled outside an explicit validation window")
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
            job_name="prepare_daily_partition", tags={"gestalt/work_id": work_id, "gestalt/environment": binding.environment}), limit=3)
        if any(run.dagster_run.status == DagsterRunStatus.SUCCESS for run in runs):
            continue
        if len(runs) >= 3:
            return SkipReason(f"{day}: three failed attempts; review checkpoint failure before continuing")
        if runs and (datetime.now(UTC) - runs[0].update_timestamp).total_seconds() < 900:
            return SkipReason(f"{day}: retry cooling down for 15 minutes")
        return RunRequest(run_key=f"{binding.environment}:daily:{work_id}:attempt-{len(runs)+1}",
            run_config={"ops":{"prepare_daily_partition_op":{"config":request}}},
            tags={**binding.tags("prepare_daily_partition", request),
                  "gestalt/trigger": "sensor"})
    return SkipReason("All explicitly requested partitions are validated; activation remains separate")


@op(config_schema={"config_path": str, "forecast_date": str,
    "automation_key": Field(str, is_required=False),
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
        if context.op_config.get("automation_key"):
            from nba_lineup_model.orchestration.daily_work import atomic_json
            config = DailyPipelineConfig.read(context.op_config["config_path"])
            if result["publication"]["status"] != "activated":
                raise ValueError("Automatic forecast work did not publish")
            atomic_json(config.pipeline_root / "automation/forecast-inputs" / (context.op_config["automation_key"] + ".json"), result, immutable=True)
    context.add_output_metadata({key: result[key] for key in (
        "forecast_date", "model_date", "model_sha256", "work_id", "schedule_artifact")})
    return result


@job(name="prepare_forecast_date", executor_def=in_process_executor,
     tags={"gestalt/workload": "daily_site_release"})
def prepare_forecast_date_job():
    prepare_forecast_date_op()


@op(config_schema={"config_path": str, "game_date": str})
def advance_daily_environment_op(context):
    from datetime import date
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
    from nba_lineup_model.orchestration.forward_daily import advance_daily_environment
    with run_guard(context, "advance_daily_environment", context.op_config):
        return advance_daily_environment(DailyPipelineConfig.read(context.op_config["config_path"]),
                                         date.fromisoformat(context.op_config["game_date"]))


@job(name="advance_daily_environment", executor_def=in_process_executor,
     tags={"gestalt/workload": "daily_site_release"})
def advance_daily_environment_job():
    advance_daily_environment_op()


@op(config_schema={"config_path": str, "game_date": str,
    "capture_slot": Field(str, is_required=False), "request_path": Field(str, is_required=False),
    "request_sha256": Field(str, is_required=False)})
def reconcile_in_season_rosters_op(context):
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
    from nba_lineup_model.orchestration.roster_reconciliation import reconcile_environment
    cfg=context.op_config
    with run_guard(context, 'reconcile_in_season_rosters', cfg):
        result=reconcile_environment(DailyPipelineConfig.read(cfg['config_path']),
            capture_slot=cfg.get('capture_slot'),request_path=cfg.get('request_path'),
            request_sha256=cfg.get('request_sha256'))
    context.add_output_metadata({'status':result['status'],
        'reconciliation_id':result['reference']['content_id'], 'generation':result['generation']})
    return result


@job(name='reconcile_in_season_rosters',executor_def=in_process_executor,
     tags={'gestalt/workload':'daily_site_release'})
def reconcile_in_season_rosters_job():
    reconcile_in_season_rosters_op()



@sensor(name="automatic_daily_lifecycle", jobs=[reconcile_in_season_rosters_job, advance_daily_environment_job, prepare_forecast_date_job],
        minimum_interval_seconds=300, default_status=DefaultSensorStatus.STOPPED)
def automatic_daily_lifecycle_sensor(context):
    return evaluate_automatic_lifecycle(context)


def evaluate_automatic_lifecycle(context):
    from nba_lineup_model.orchestration.lifecycle import automation_request
    binding = binding_for_instance(context.instance)
    from nba_lineup_model.orchestration.shared_daily_control import automation_enabled
    if not automation_enabled(binding):
        return SkipReason("Dev automation is disabled outside an explicit validation window")
    configured = os.environ.get("GESTALT_AUTOMATION_CONFIG")
    if not configured:
        return SkipReason("Automatic lifecycle is not enabled for this environment")
    if context.instance.get_runs(filters=RunsFilter(statuses=[DagsterRunStatus.QUEUED,
        DagsterRunStatus.STARTING, DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING]), limit=1):
        return SkipReason("Daily work is already queued or running in this instance")
    request = reconciliation_due(context, binding, configured) or automation_request(Path(configured))
    if request is None:
        return SkipReason("No completed slate or changed pregame inputs need publication")
    tags = binding.tags(request["job"], request["config"])
    runs = context.instance.get_run_records(filters=RunsFilter(job_name=request["job"],
        tags={"gestalt/automation_key": request["key"], "gestalt/environment": binding.environment}), limit=3)
    if any(r.dagster_run.status == DagsterRunStatus.SUCCESS for r in runs):
        return SkipReason("This input selection was already processed successfully")
    if len(runs) >= 3:
        raise RuntimeError("Three failed attempts for the same lifecycle work; inspect Data Quality and rerun after repair")
    if runs and (datetime.now(UTC) - runs[0].update_timestamp).total_seconds() < 900:
        return SkipReason("Retry cooling down for 15 minutes")
    return RunRequest(job_name=request["job"],
        run_key=f"{binding.environment}:automatic:{request['key']}:attempt-{len(runs)+1}",
        run_config={"ops":{request["job"] + "_op":{"config":request["config"]}}},
        tags={**tags, "gestalt/trigger":"sensor", "gestalt/automation_key":request["key"]})


def reconciliation_due(context, binding, configured):
    """A verified scheduled reconciliation is a dependency, not a competing sensor."""
    from nba_lineup_model.orchestration.roster_reconciliation import scheduled_slot
    day, slot = scheduled_slot(datetime.now(UTC))
    request = {'config_path': configured, 'game_date': day, 'capture_slot': slot}
    tags = binding.tags('reconcile_in_season_rosters', request)
    successful = context.instance.get_run_records(filters=RunsFilter(
        job_name='reconcile_in_season_rosters', statuses=[DagsterRunStatus.SUCCESS],
        tags={'gestalt/environment': binding.environment, 'gestalt/work_id': tags['gestalt/work_id']}), limit=1)
    if successful:
        return None
    return {'job': 'reconcile_in_season_rosters', 'key': tags['gestalt/work_id'], 'config': request}


defs = Definitions(jobs=[reconcile_in_season_rosters_job, prepare_daily_partition_job, prepare_forecast_date_job, advance_daily_environment_job],
                   sensors=[resumable_daily_partition_sensor, automatic_daily_lifecycle_sensor])
