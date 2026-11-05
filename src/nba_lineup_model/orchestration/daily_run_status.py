"""Read only the bound instance; recovery requires a success for the same work."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from nba_lineup_model.orchestration.daily_instance import DailyInstance

JOBS = ("prepare_daily_partition", "prepare_forecast_date", "prepare_daily_site_release")


def stamp(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(value, UTC).isoformat()
    return datetime.fromisoformat(value).replace(tzinfo=UTC).isoformat()


def read_daily_runs(home: Path, environment: str, environment_root: Path) -> dict:
    binding = DailyInstance.read(home)
    if binding.environment != environment or binding.root != environment_root.resolve():
        raise ValueError("Watchdog and daily Dagster instance belong to different environments")
    database = binding.home / "history" / "runs.db"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        records = [dict(row) for row in connection.execute(
            "select * from runs where pipeline_name in (?, ?, ?) order by create_timestamp, id", JOBS)]
    finally:
        connection.close()
    from nba_lineup_model.orchestration.daily_history_migration import migration_bindings, record_digest
    migrated = migration_bindings(binding.home)
    groups = defaultdict(list)
    daily = []
    for record in records:
        run_id, job, status, created, started, ended, updated, serialized = (
            record[key] for key in ("run_id", "pipeline_name", "status", "create_timestamp",
                                   "start_time", "end_time", "update_timestamp", "run_body"))
        tags = (json.loads(serialized).get("tags") or {})
        if run_id in migrated:
            original = migrated[run_id]
            if record_digest(record) != original["record_sha256"]:
                raise ValueError(f"Migrated run record changed: {run_id}")
            tags = {**tags, **original["tags"]}
        if (tags.get("gestalt/environment") != environment or
                tags.get("gestalt/environment_root") != str(binding.root)):
            raise ValueError(f"Run {run_id} has a missing or conflicting environment binding")
        key = tags.get("gestalt/recovery_key")
        if not key or not key.startswith(job + ":"):
            raise ValueError(f"Run {run_id} has no valid work identity")
        if tags.get("gestalt/history_only") == "true":
            # Retained and integrity-checked above; not a run against the live site.
            continue
        row = {"run_id": run_id, "job_name": job, "status": status,
               "recovery_key": key, "created_at": stamp(created),
               "started_at": stamp(started), "finished_at": stamp(ended),
               "updated_at": stamp(updated)}
        groups[key].append(row)
        if job in {"prepare_daily_partition", "prepare_daily_site_release"}:
            daily.append(row)
    failures = []
    unfinished = []
    for key, runs in groups.items():
        successes = [r for r in runs if r["status"] == "SUCCESS"]
        failed = [r for r in runs if r["status"] == "FAILURE"]
        latest_success = max((r["finished_at"] or r["updated_at"] for r in successes), default="")
        if failed:
            failure = max(failed, key=lambda r: r["finished_at"] or r["updated_at"])
            if not latest_success or latest_success <= (failure["finished_at"] or failure["updated_at"]):
                failures.append(failure)
        unfinished.extend(r for r in runs if r["status"] in {"QUEUED", "STARTING", "STARTED", "CANCELING"})
    latest = daily[-1] if daily else {}
    successes = [r for r in daily if r["status"] == "SUCCESS"]
    last_success = max(successes, key=lambda r: r["finished_at"] or r["updated_at"], default={})
    return {**latest, "last_success_at": last_success.get("finished_at", "") or last_success.get("updated_at", ""),
            "last_success_run_id": last_success.get("run_id", ""),
            "unresolved_failures": failures, "unfinished_runs": unfinished,
            "run_count": len(daily), "total_run_count": len(records),
            "history_path": str(database),
            "environment": environment, "environment_root": str(binding.root)}
