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


def test_schedule_lock_remains_same_across_release_switches(tmp_path, monkeypatch):
    from nba_lineup_model.orchestration.daily_pipeline import schedule_lock_path
    monkeypatch.delenv("GESTALT_SCHEDULE_LOCK_PATH", raising=False)
    root = tmp_path / "nba-gestalt-dev"
    assert schedule_lock_path(root / "live-release/schedule") == root / ".schedule-writer.lock"
    assert schedule_lock_path(root / "official-schedule/current") == root / ".schedule-writer.lock"
    assert schedule_lock_path(root / "pipeline/candidates/x/schedule") == root / ".schedule-writer.lock"


@pytest.mark.parametrize("scenario", ["success", "stale", "reload_failure", "health_failure", "rollback_failure"])
def test_activation_covers_stale_state_and_process_rollback(tmp_path, monkeypatch, scenario):
    from nba_lineup_model.orchestration import daily_activation as activation
    from nba_lineup_model.season.dev_site_release import _tree_sha256
    config = config_fixture(tmp_path)
    # Replace the test marker directory with a normal release pointer.
    old = config.pipeline_root / "releases/old"
    new = config.pipeline_root / "releases/new"
    old.mkdir(parents=True)
    new.mkdir()
    (config.live_pointer / "unchanged").unlink()
    config.live_pointer.rmdir()
    config.live_pointer.symlink_to(old)
    (new / "schedule").mkdir()
    report = {"status":"validated_candidate", "game_date":"2026-10-20", "runtime_release":str(new), "source_schedule_sha256":_tree_sha256(config.schedule_root)}
    manifest = {"identity":{"live_pointer":str(config.live_pointer), "evidence_through_date":"2026-10-20"}}
    monkeypatch.setattr(activation, "verify_production_site_release", lambda path: manifest)
    monkeypatch.setattr("nba_lineup_model.season.production_site_release.verify_production_site_release", lambda path: manifest)
    events = []
    def reload_api(c):
        events.append(("reload", c.live_pointer.resolve().name))
        if scenario == "reload_failure" and c.live_pointer.resolve() == new:
            raise RuntimeError("restart failed")
    def verify_api(c, release):
        events.append(("verify", release.name))
        if scenario in {"health_failure", "rollback_failure"}:
            if release == new or scenario == "rollback_failure":
                raise RuntimeError("API mismatch")
    if scenario == "stale":
        (config.schedule_root / "changed.json").write_text('{}')
        with pytest.raises(ValueError, match="Schedule changed"):
            activation.activate_daily_release(config, report, reload_api=reload_api, verify_api=verify_api)
        assert events == []
    elif scenario == "success":
        activation.activate_daily_release(config, report, reload_api=reload_api, verify_api=verify_api)
        assert config.live_pointer.resolve() == new
        assert report["status"] == "activated"
        assert events == [("reload","new"),("verify","new")]
        return
    else:
        with pytest.raises(Exception):
            activation.activate_daily_release(config, report, reload_api=reload_api, verify_api=verify_api)
        assert ("reload","old") in events
        assert ("verify","old") in events
        assert report["status"] == ("rollback_failed" if scenario == "rollback_failure" else "activation_failed")
    assert config.live_pointer.resolve() == old


def test_pipeline_report_is_accessible_without_publishing_a_failed_candidate(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from nba_lineup_model.web_api.app import create_app
    from nba_lineup_model.orchestration.daily_pipeline import _report
    monkeypatch.setenv("GESTALT_DAILY_PIPELINE_ROOT", str(tmp_path))
    client = TestClient(create_app(evaluator=object()))
    assert client.get("/api/reports/daily-pipeline").status_code == 404
    _report(tmp_path / "reports/2026-10-31", {"game_date":"2026-10-31", "status":"failed", "error":"Missing official box score"})
    response = client.get("/api/reports/daily-pipeline")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Missing official box score" in response.text
    assert "failed" in response.text


def test_automatic_activation_is_opt_in(tmp_path):
    assert config_fixture(tmp_path).activate_after_validation is False


@pytest.mark.parametrize("missing_context", [False, True])
def test_model_uses_pinned_context_api_and_rejects_neutral_fallback(tmp_path, monkeypatch, missing_context):
    from contextlib import contextmanager
    from nba_lineup_model.orchestration.daily_pipeline import CanonicalPipelineSteps
    from nba_lineup_model.season import daily_model_replay as replay
    config = config_fixture(tmp_path)
    config.preseason_profiles_path.write_bytes(b"fixed profile artifact")
    candidate = config.pipeline_root / "candidate"
    candidate.mkdir(parents=True)
    steps = CanonicalPipelineSteps(config, candidate, "revision")
    @contextmanager
    def api(**kwargs):
        assert kwargs["log_name"] == "context-api.log"
        yield "http://127.0.0.1:12345"
    monkeypatch.setattr(steps, "api", api)
    def provider(url):
        assert url == "http://127.0.0.1:12345/api/matchups"
        assert url != config.matchup_url
        return lambda *args: ([1.0], ["missing player"] if missing_context else [])
    monkeypatch.setattr(replay, "_http_context_provider", provider)
    def run(**kwargs):
        cfg = kwargs["config"]
        assert cfg.matchup_url == "isolated://frozen-context"
        identity = json.loads(cfg.context_model_version)
        assert identity["model_release"] == str(config.release_root.resolve())
        assert len(identity["profiles_sha256"]) == 64
        assert kwargs["context_provider"](None, False, False) == ([1.0], [])
    monkeypatch.setattr(replay, "replay_daily_model_chain", run)
    if missing_context:
        with pytest.raises(ValueError, match="Pinned context inputs incomplete"):
            steps.model(date(2026,10,20))
    else:
        steps.model(date(2026,10,20))
