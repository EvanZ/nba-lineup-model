from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from dagster._core.remote_origin import RemoteJobOrigin, RemoteRepositoryOrigin, RegisteredCodeLocationOrigin
from dagster import DagsterInstance, DagsterRunStatus, Definitions, build_sensor_context

from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig, prepare_daily_release, require_complete_slate
from nba_lineup_model.orchestration.definitions import complete_slate_daily_release_sensor, defs, prepare_daily_site_release_job


def config_fixture(tmp_path):
    root = tmp_path / "nba-gestalt-dev"
    root.mkdir()
    code = root / "code"
    code.mkdir()
    subprocess.run(["git", "init", "-q", str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "test"], check=True)
    config = DailyPipelineConfig(environment_root=root, pipeline_root=root / "pipeline", data_root=root / "data", model_root=root / "models", schedule_root=root / "schedule", coverage_path=root / "coverage.json", catalog_path=root / "catalog.parquet", preseason_rankings_path=root / "rankings.parquet", preseason_minutes_path=root / "minutes.json", preseason_profiles_path=root / "profiles.parquet", release_root=code, code_root=code, frontend_root=root / "frontend", live_pointer=root / "live-release", matchup_url="http://127.0.0.1:8011/api/matchups")
    pd.DataFrame([{"season":"2026-27", "season_type":"regular", "game_id": game_id, "game_time_utc":"2026-10-20T23:00:00Z"} for game_id in ("0022600001", "0022600002")]).to_parquet(config.catalog_path)
    for game_id in ("0022600001", "0022600002"):
        for relative, payload in [(f"final/{game_id}.summary.json", {"game_id":game_id, "final":{"home_score":100,"away_score":90}}), (f"final/{game_id}.nba.json", {"game":{"gameId":game_id}}), (f"frozen/{game_id}.json", {"game_id":game_id,"projection":{"home_margin":2.0}})]:
            path=config.schedule_root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
    config.coverage_path.write_text(json.dumps({"season":"2026-27", "games":{"0022600001":["a"*64], "0022600002":["b"*64]}}))
    config.live_pointer.mkdir()
    (config.live_pointer / "unchanged").write_text("original")
    return config


def test_complete_slate_uses_catalog_not_only_reported_finals(tmp_path):
    config = config_fixture(tmp_path)
    assert len(require_complete_slate(config, date(2026,10,20))) == 2
    (config.schedule_root / "final/0022600002.nba.json").unlink()
    with pytest.raises(ValueError, match="0022600002"):
        require_complete_slate(config, date(2026,10,20))
    assert not (config.pipeline_root / "candidates").exists()


@pytest.mark.parametrize("failed_stage", ["data", "model", "serving", "lineups", "validate_and_stage", None])
def test_pipeline_failure_at_every_stage_never_publishes_partial_state(tmp_path, failed_stage):
    config = config_fixture(tmp_path)
    seen=[]
    class Steps:
        def __init__(self, config, candidate, revision):
            self.candidate=candidate
        def __getattr__(self, name):
            def run(*args):
                seen.append(name)
                if name == failed_stage:
                    raise RuntimeError("injected failure")
                return self.candidate / "runtime"
            return run
    if failed_stage:
        with pytest.raises(RuntimeError, match="injected failure"):
            prepare_daily_release(config, date(2026,10,20), steps_class=Steps)
        assert seen[-1] == failed_stage
    else:
        result=prepare_daily_release(config, date(2026,10,20), steps_class=Steps)
        assert result["status"] == "validated_candidate"
        assert seen == ["data","model","serving","lineups","validate_and_stage"]
        assert (Path(result["candidate"])/"schedule/frozen/0022600001.json").read_bytes() == (config.schedule_root/"frozen/0022600001.json").read_bytes()
    assert (config.live_pointer / "unchanged").read_text() == "original"
    report=json.loads((config.pipeline_root/"reports/2026-10-20/report.json").read_text())
    assert report["status"] == ("failed" if failed_stage else "validated_candidate")
    assert (config.pipeline_root/"reports/2026-10-20/report.html").is_file()


def test_daily_sensor_suppresses_queued_failed_and_successful_date_attempts(tmp_path, monkeypatch):
    config=config_fixture(tmp_path)
    path=config.environment_root/"config.json"
    path.write_text(json.dumps(asdict(config),default=str))
    pointer=config.live_pointer/"site-releases/latest.json"
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({"evidence_through_date":"2026-10-19"}))
    monkeypatch.setenv("GESTALT_DAILY_PIPELINE_CONFIG", str(path))
    for status in (DagsterRunStatus.QUEUED, DagsterRunStatus.STARTED, DagsterRunStatus.FAILURE, DagsterRunStatus.SUCCESS):
        with DagsterInstance.ephemeral() as instance:
            context=build_sensor_context(instance=instance)
            request=complete_slate_daily_release_sensor(context)
            assert request.run_key == "daily-site:2026-10-20"
            instance.create_run_for_job(prepare_daily_site_release_job,remote_job_origin=RemoteJobOrigin(RemoteRepositoryOrigin(RegisteredCodeLocationOrigin("test"), "test_repository"), "prepare_daily_site_release"),run_config=request.run_config,status=status,tags={"gestalt/game_date":"2026-10-20"})
            result=complete_slate_daily_release_sensor(context)
            assert not hasattr(result,"run_key")


def test_actual_dagster_daily_job_records_incomplete_slate_without_partial_outputs(tmp_path):
    config=config_fixture(tmp_path)
    path=config.environment_root/"config.json"
    path.write_text(json.dumps(asdict(config),default=str))
    (config.schedule_root/"final/0022600002.nba.json").unlink()
    result=prepare_daily_site_release_job.execute_in_process(run_config={"ops":{"build_daily_site_candidate":{"config":{"game_date":"2026-10-20","config_path":str(path)}}}},raise_on_error=False)
    assert not result.success
    assert (config.live_pointer / "unchanged").read_text() == "original"
    assert not config.model_root.exists()
    assert json.loads((config.pipeline_root/"reports/2026-10-20/report.json").read_text())["stage"] == "preflight"


def test_only_coupled_daily_model_job_is_registered():
    Definitions.validate_loadable(defs)
    names={job.name for job in defs.jobs}
    assert "prepare_daily_site_release" in names
    assert not names & {"update_in_season_nail_production", "update_in_season_minutes_production", "promote_current_season_serving", "materialize_current_season_lineups"}
