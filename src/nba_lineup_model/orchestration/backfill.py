"""Coverage-driven, resumable official-NBA catch-up (EC2 only).

CLI and Dagster call the same workflow. Publication remains an explicit,
environment-scoped atomic activation; individual downloads never publish.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nba_lineup_model.orchestration.daily_pipeline import (
    DailyPipelineConfig, _report, prepare_daily_release, require_complete_slate, schedule_lock,
)
from nba_lineup_model.season.backfill_recovery import recover_official_slate
from nba_lineup_model.season.dev_site_release import _tree_sha256
from nba_lineup_model.season.source_policy import require_clean_artifact_sources
from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.web_api.win_projections import _write_json


def evidence_date(config):
    return date.fromisoformat(json.loads((config.live_pointer / "site-releases/latest.json").read_text())["evidence_through_date"])


def run_backfill(config: DailyPipelineConfig, *, end_date: date, start_date: date | None = None,
                 game_ids=None, recover=None, prepare=None, activate=None):
    config.validate()
    if end_date > datetime.now(UTC).astimezone(PACIFIC).date():
        raise ValueError("Catch-up cannot target future Pacific dates")
    start_date = start_date or config.season_start
    if not config.season_start <= start_date <= end_date:
        raise ValueError("Invalid backfill date range")
    # Explicit IDs select dates, never allow a partial date to advance the model.
    if game_ids:
        import pandas as pd
        catalog = pd.read_parquet(config.catalog_path)
        rows = catalog.loc[catalog.game_id.astype(str).isin(game_ids) & catalog.season.astype(str).eq(config.season) & catalog.season_type.eq("regular")]
        if set(rows.game_id.astype(str)) != set(game_ids):
            raise ValueError("Requested NBA game IDs are absent from the official season catalog")
        dates = pd.to_datetime(rows.game_time_utc, utc=True).dt.tz_convert(PACIFIC).dt.date
        start_date, end_date = min(dates), max(dates)
        if end_date > datetime.now(UTC).astimezone(PACIFIC).date():
            raise ValueError("Requested games are on future Pacific dates")
    config.pipeline_root.mkdir(parents=True, exist_ok=True)
    with (config.pipeline_root / ".backfill.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another backfill is already running") from error
        tip = evidence_date(config)
        revision = subprocess.check_output(["git", "-C", str(config.code_root), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(config.code_root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise ValueError("Commit tested code before backfill materialization")
        with schedule_lock(config.schedule_root):
            require_clean_artifact_sources(config.schedule_root)
            source_sha = _tree_sha256(config.schedule_root)
            identity = hashlib.sha256(f"{source_sha}:{revision}".encode()).hexdigest()
            workspace = config.pipeline_root / "backfills" / identity
            workspace.mkdir(parents=True, exist_ok=True)
            schedule = workspace / "schedule"
            if not schedule.exists():
                temporary = workspace / "schedule.copying"
                if temporary.exists():
                    # A killed copy is never accepted as a checkpoint.
                    shutil.rmtree(temporary)
                shutil.copytree(config.schedule_root.resolve(), temporary)
                temporary.rename(schedule)
        journal_path = workspace / "checkpoint.json"
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"completed_dates": {}}
        status = {"game_date": end_date.isoformat(), "status": "running", "stage": "backfill", "source_schedule_sha256": source_sha, "requested_start_date": start_date.isoformat(), "requested_end_date": end_date.isoformat(), "days": []}
        report_root = config.pipeline_root / "reports" / "backfill"
        recover = recover or recover_official_slate
        prepare = prepare or prepare_daily_release
        last = None
        changed_history = False
        try:
            # Published dates are cheap local coverage checks. Any older hole is
            # recovered too, without replacing its existing model/data history.
            day = config.season_start
            while day <= max(tip, end_date):
                work = replace(config, schedule_root=schedule, activate_after_validation=False)
                if day <= tip:
                    try:
                        require_complete_slate(work, day)
                    except (ValueError, OSError):
                        if day < start_date or day > end_date:
                            raise ValueError(f"Published date {day} has a coverage gap; include it in the requested backfill")
                        recover(game_date=day, state_root=schedule, catalog_path=config.catalog_path,
                            coverage_path=config.coverage_path, checkpoint_root=workspace / "inputs", season=config.season)
                        require_complete_slate(work, day)
                        changed_history = True
                    if day != tip or not changed_history:
                        day += timedelta(days=1)
                        continue
                # Missing predecessor days are dependencies even when the user
                # selected only a later game. Never skip across the model chain.
                key = day.isoformat()
                if key in journal["completed_dates"]:
                    last = journal["completed_dates"][key]
                    from nba_lineup_model.season.production_site_release import verify_production_site_release
                    verify_production_site_release(Path(last["runtime_release"]))
                    schedule = Path(last["runtime_release"]) / "schedule"
                    status["days"].append({"game_date": key, "status": "reused_validated_candidate"})
                else:
                    day_schedule = workspace / "working" / key
                    if not day_schedule.exists():
                        day_schedule.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(schedule.resolve(), day_schedule)
                    schedule = day_schedule
                    work = replace(config, schedule_root=schedule, activate_after_validation=False)
                    status.update(stage="recover_inputs", processing_date=key)
                    _report(report_root, status)
                    recovery = recover(game_date=day, state_root=schedule, catalog_path=config.catalog_path,
                        coverage_path=config.coverage_path, checkpoint_root=workspace / "inputs", season=config.season)
                    require_complete_slate(work, day)
                    status.update(stage="daily_artifacts")
                    _report(report_root, status)
                    last = prepare(work, day)
                    # Activation compares the ORIGINAL served schedule, not our
                    # private recovered copy. A concurrent live write invalidates it.
                    last["source_schedule_sha256"] = source_sha
                    last["backfill"] = recovery
                    _report(config.pipeline_root / "reports" / key, last)
                    journal["completed_dates"][key] = last
                    _write_json(journal_path, journal)
                    schedule = Path(last["runtime_release"]) / "schedule"
                    status["days"].append({"game_date": key, "status": "validated_candidate", "recovery": recovery})
                day += timedelta(days=1)
            status.update(status="validated_candidate" if last else "already_complete", stage="awaiting_activation" if last else "complete")
            if last:
                status["runtime_release"] = last["runtime_release"]
            _report(report_root, status)
            if last and config.activate_after_validation:
                if activate is None:
                    from nba_lineup_model.orchestration.daily_activation import activate_daily_release
                    activate = activate_daily_release
                activate(config, last)
                status.update(status="activated", stage="complete")
                _report(report_root, status)
            return status
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            if hasattr(error, "report"):
                status["recovery"] = error.report
            _report(report_root, status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat, default=datetime.now(UTC).astimezone(PACIFIC).date() - timedelta(days=1))
    parser.add_argument("--game-id", action="append")
    args = parser.parse_args()
    print(json.dumps(run_backfill(DailyPipelineConfig.read(args.config), end_date=args.end_date, start_date=args.start_date, game_ids=args.game_id), indent=2))


if __name__ == "__main__":
    main()
