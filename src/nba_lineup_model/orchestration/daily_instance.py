"""Durable daily history with explicit environment boundaries for each code location."""
from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import fcntl

from nba_lineup_model.orchestration.daily_work import atomic_json, read_object, digest


def request_identity(job_name: str, request: dict) -> tuple[str, str, str]:
    """Keep attempt identity, but match recovery by the output being repaired."""
    if job_name == "prepare_daily_partition":
        spec = read_object(Path(request["spec_path"]))
        identity = spec["identity"]
        if spec["work_id"] != digest(identity) or identity.get("schema_version") != 1:
            raise ValueError("Work identity checksum/schema mismatch")
        day = identity["game_date_pacific"]
        key = f"{job_name}:{identity['season']}:{day}:{identity['edition_id']}"
        return spec["work_id"], day, key
    from datetime import date
    if job_name == "reconcile_in_season_rosters":
        day = date.fromisoformat(request['game_date']).isoformat()
        key = request.get('request_sha256') or request.get('capture_slot')
        if not key: raise ValueError('Reconciliation requires pinned manual input or a capture slot')
        return f'reconcile:{key}', day, f'{job_name}:{day}:{key}'
    if job_name == "prepare_forecast_date":
        day = date.fromisoformat(request["forecast_date"]).isoformat()
        track = read_object(Path(request["config_path"])).get("model_track", "legacy")
        return f"forecast:{day}", day, f"{job_name}:{day}:{track}"
    if job_name == "advance_daily_environment":
        day = date.fromisoformat(request["game_date"]).isoformat()
        track = read_object(Path(request["config_path"]))["model_track"]
        return f"nightly:{day}", day, f"{job_name}:{day}:{track}"
    if job_name == "prepare_daily_site_release":
        day = date.fromisoformat(request["game_date"]).isoformat()
        return f"legacy:{day}", day, f"{job_name}:legacy:{day}"
    raise ValueError(f"Unsupported daily job: {job_name}")


@dataclass(frozen=True)
class DailyInstance:
    environment: str
    root: Path
    instance_home: Path | None = None

    @property
    def home(self) -> Path:
        return self.instance_home or self.root / "daily-dagster"

    @property
    def requests(self) -> Path:
        return self.root / "daily-requests"

    @classmethod
    def read(cls, home: Path, environment: str | None = None) -> "DailyInstance":
        home = home.absolute()
        if home != home.resolve():
            raise ValueError("Dagster history cannot be a link to another instance")
        value = read_object(home / "environment.json")
        if value.get("schema_version") == 2:
            selected = environment or os.environ.get("GESTALT_DAILY_ENVIRONMENT")
            roots = value.get("environments", {})
            if not roots or set(roots) - {"dev", "prod"} or selected not in roots:
                raise ValueError("Shared daily history requires an explicit registered environment")
            resolved = {name: Path(root).resolve(strict=True) for name, root in roots.items()}
            if any(str(resolved[name]) != root for name, root in roots.items()):
                raise ValueError("Shared environment roots must be canonical")
            if len(set(resolved.values())) != len(resolved) or any(
                left != right and (left.is_relative_to(right) or right.is_relative_to(left))
                for left in resolved.values() for right in resolved.values()
            ):
                raise ValueError("Shared environment roots must be distinct and disjoint")
            return cls(selected, resolved[selected], home)
        if value.get("schema_version") != 1 or value.get("environment") not in {"dev", "prod"}:
            raise ValueError("Invalid daily Dagster environment binding")
        binding = cls(value["environment"], Path(value["environment_root"]))
        if environment and environment != binding.environment:
            raise ValueError("Requested worker and history belong to different environments")
        if binding.root != binding.root.resolve() or home != binding.home:
            raise ValueError("Daily Dagster history must be inside its own environment")
        return binding

    def tags(self, job_name: str, request: dict) -> dict[str, str]:
        config = read_object(Path(request["config_path"]))
        if Path(config["code_root"]).resolve() != Path(__file__).resolve().parents[3]:
            raise ValueError("Requested code differs from the executing daily worker")
        if Path(config["environment_root"]).resolve() != self.root:
            raise ValueError("Pipeline configuration belongs to a different environment")
        if Path(config["live_pointer"]).absolute().parent.resolve() != self.root:
            raise ValueError("Pipeline live pointer belongs to a different environment")
        if job_name not in {"prepare_daily_partition", "prepare_forecast_date", "advance_daily_environment", "reconcile_in_season_rosters"}:
            raise ValueError(f"Unsupported daily job: {job_name}")
        work, day, recovery_key = request_identity(job_name, request)
        return {"gestalt/environment": self.environment,
                "gestalt/environment_root": str(self.root),
                "gestalt/work_id": work, "gestalt/game_date": day,
                "gestalt/recovery_key": recovery_key}


def initialize(environment: str, root: Path, template: Path) -> DailyInstance:
    if environment not in {"dev", "prod"}:
        raise ValueError("Expected dev or prod")
    binding = DailyInstance(environment, root.resolve(strict=True))
    binding.home.mkdir(exist_ok=True)
    if binding.home.is_symlink():
        raise ValueError("Dagster history cannot be a link to another instance")
    atomic_json(binding.home / "environment.json", {
        "schema_version": 1, "environment": environment,
        "environment_root": str(binding.root),
    }, immutable=True)
    configuration = binding.home / "dagster.yaml"
    content = template.read_bytes()
    if configuration.exists():
        if configuration.read_bytes() != content:
            raise ValueError("Existing Dagster configuration differs; do not overwrite it")
    else:
        with configuration.open("xb") as output:
            output.write(content)
    binding.requests.mkdir(exist_ok=True)
    return DailyInstance.read(binding.home)


def binding_for_instance(instance) -> DailyInstance:
    binding = DailyInstance.read(Path(instance.root_directory))
    selected_environment = os.environ.get("GESTALT_DAILY_ENVIRONMENT")
    if selected_environment and selected_environment != binding.environment:
        raise ValueError("Code location belongs to a different environment")
    selected = os.environ.get("DAGSTER_HOME")
    if selected and Path(selected).absolute() != binding.home:
        raise ValueError("Worker DAGSTER_HOME differs from its actual Dagster instance")
    return binding


def bind_run(context, job_name: str, request: dict) -> None:
    binding = binding_for_instance(context.instance)
    tags = binding.tags(job_name, request)
    recorded = context.instance.get_run_by_id(context.run_id).tags
    if any(key in recorded and recorded[key] != value for key, value in tags.items()):
        raise ValueError("Run identity differs from the requested environment or work")
    context.instance.add_run_tags(context.run_id, tags)


@contextmanager
def run_guard(context, job_name: str, request: dict):
    binding = binding_for_instance(context.instance)
    bind_run(context, job_name, request)
    from nba_lineup_model.orchestration.shared_daily_control import automation_enabled
    recorded = context.instance.get_run_by_id(context.run_id).tags
    if recorded.get("gestalt/trigger") == "sensor" and not automation_enabled(binding):
        raise RuntimeError("Dev automation window closed before this run started")
    with (binding.home / ".daily-execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another daily job is executing in this Dagster instance") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def execute_manual(home: Path, job_name: str, request: dict):
    """Manual runs use exactly the same jobs, checks and storage as the sensor."""
    from dagster import DagsterInstance
    from nba_lineup_model.orchestration.partition_definitions import (
        prepare_daily_partition_job, prepare_forecast_date_job, advance_daily_environment_job, reconcile_in_season_rosters_job,
    )
    binding = DailyInstance.read(home)
    tags = binding.tags(job_name, request)
    jobs = {"reconcile_in_season_rosters": reconcile_in_season_rosters_job, "prepare_daily_partition": prepare_daily_partition_job,
            "prepare_forecast_date": prepare_forecast_date_job,
            "advance_daily_environment": advance_daily_environment_job}
    with DagsterInstance.from_config(str(binding.home)) as instance:
        from dagster import RunsFilter, DagsterRunStatus
        active = instance.get_runs(filters=RunsFilter(statuses=[
            DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING,
            DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING]), limit=1)
        if active:
            raise RuntimeError("Daily work is already queued or running; retry after it finishes")
        binding_for_instance(instance)
        return jobs[job_name].execute_in_process(instance=instance,
            tags={**tags, "gestalt/trigger": "manual"},
            run_config={"ops": {job_name + "_op": {"config": request}}},
            raise_on_error=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--environment", choices=("dev", "prod"), required=True)
    init.add_argument("--environment-root", type=Path, required=True)
    init.add_argument("--template", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--home", type=Path, required=True)
    run.add_argument("--job", choices=("prepare_daily_partition", "prepare_forecast_date", "advance_daily_environment", "reconcile_in_season_rosters"), required=True)
    run.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        binding = initialize(args.environment, args.environment_root, args.template)
        print(json.dumps({"environment": binding.environment, "dagster_home": str(binding.home)}))
    else:
        result = execute_manual(args.home, args.job, read_object(args.request))
        print(json.dumps({"run_id": result.run_id, "success": result.success}))
        raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
