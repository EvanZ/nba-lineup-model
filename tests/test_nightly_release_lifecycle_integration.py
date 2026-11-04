"""Nightly Dagster -> coupled artifacts -> release -> API integration.

Opt-in EC2 test. It consumes the exact approved #56 work selection, creates a
new private environment, and reuses immutable verified source/data/model
artifacts. It does not activate dev/prod or retrain the accepted model.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
from nba_lineup_model.orchestration.daily_work import atomic_json, digest, read_object
from nba_lineup_model.orchestration.partition_definitions import prepare_daily_partition_job
from nba_lineup_model.orchestration.resumable_pipeline import bind_execution
from nba_lineup_model.season.production_site_release import verify_production_site_release
from nba_lineup_model.season.release_publisher import (
    activate_release, read_active_activation,
)
from nba_lineup_model.web_api.release_serving import ReleaseServing

SOURCE = os.environ.get("GESTALT_LIFECYCLE_SOURCE_WORK_ROOT")
pytestmark = pytest.mark.skipif(
    not SOURCE, reason="Requires explicitly selected approved EC2 work root"
)


def _hardlink_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, copy_function=os.link)


def _current_revision(repo: Path) -> str:
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    if status:
        pytest.fail("Lifecycle verification requires a clean EC2 checkout")
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _prepare_private_work(tmp_path: Path):
    source = Path(SOURCE).resolve(strict=True)
    assert source.is_relative_to(Path("/var/lib/nba-gestalt-dev"))
    source_config = DailyPipelineConfig.read(source / "config.json")
    source_spec = read_object(source / "spec.json")
    source_approval = read_object(source / "approval.json")
    env = tmp_path / "nba-gestalt-dev"
    env.mkdir()
    schedule = env / "schedule"
    _hardlink_tree(source_config.schedule_root, schedule)
    model_configuration = env / "model-configuration.json"
    shutil.copyfile(source_config.model_configuration_path, model_configuration)
    base_manifest_ref = source_spec["identity"]["execution"]["base_release_manifest"]
    base_release = Path(base_manifest_ref["path"]).resolve().parent
    live = env / "live-release"
    live.symlink_to(base_release, target_is_directory=True)
    config = replace(
        source_config,
        environment_root=env,
        pipeline_root=env / "pipeline",
        data_root=env / "data",
        model_root=env / "model",
        schedule_root=schedule,
        live_pointer=live,
        model_configuration_path=model_configuration,
        code_root=Path("/home/ubuntu/nba-lineup-model"),
        frontend_root=Path("/var/lib/nba-gestalt-dev/live-release/web").resolve(),
        activate_after_validation=False,
    )
    config.validate()
    revision = _current_revision(config.code_root)
    identity = copy.deepcopy(source_spec["identity"])
    execution = identity.pop("execution")
    identity.pop("configuration_digest", None)
    identity["code_identity"] = revision
    identity["inputs"][str(model_configuration)] = {
        "path": str(model_configuration),
        "sha256": __import__("hashlib").sha256(model_configuration.read_bytes()).hexdigest(),
        "size": model_configuration.stat().st_size,
    }
    spec = bind_execution({"identity": identity}, config, execution)
    config_path = env / "config.json"
    spec_path = env / "spec.json"
    approval_path = env / "approval.json"
    atomic_json(config_path, asdict(config))
    atomic_json(spec_path, spec)
    atomic_json(approval_path, source_approval)
    return source, config, spec, source_approval, config_path, spec_path, approval_path


def _api_checks(runtime: Path, pointer: Path, observation_root: Path, monkeypatch):
    pointer.symlink_to(runtime, target_is_directory=True)
    serving = ReleaseServing.load(pointer)
    assert serving is not None
    monkeypatch.setenv("GESTALT_ACTIVE_RELEASE_ROOT", str(pointer))
    monkeypatch.setenv("GESTALT_LIVE_SCORE_STATE_DIR", str(observation_root))
    monkeypatch.chdir(serving.bindings["historical_runtime"])
    from nba_lineup_model.web_api.app import create_app

    with TestClient(create_app()) as client:
        endpoints = {
            "health": "/api/health",
            "serving": "/api/serving-view",
            "rankings": "/api/rankings?season=2026-27",
            "lineups": "/api/lineups?season=2026-27&minimum_possessions=0",
            "wins": "/api/win-projections",
            "records": "/api/current-season-records",
            "performance": "/api/schedule-performance",
            "schedule": "/api/schedule?date=2026-11-02",
            "player": "/api/players/201939",
            "team": "/api/teams/GSW?season=2026-27",
        }
        responses = {name: client.get(path) for name, path in endpoints.items()}
        assert all(value.status_code == 200 for value in responses.values()), {
            name: value.text[:300] for name, value in responses.items()
            if value.status_code != 200
        }
        release_ids = {value.headers.get("X-Gestalt-Release") for value in responses.values()}
        assert release_ids == {serving.release_id}
        schedule = responses["schedule"].json()
        assert len(schedule["games"]) == 15
        assert all(g.get("final_audit") for g in schedule["games"])
        assert all(g.get("retrospective", {}).get("ledger") for g in schedule["games"])
        player = responses["player"].json()
        curry = [g for g in player["current_season"]["games"] if g["game_date"] == "2026-11-02"]
        assert len(curry) == 1 and curry[0]["opponent"] == "PHX"
        performance = responses["performance"].json()
        season = next(row for row in performance["windows"] if row["key"] == "season")
        assert season["games"] == 107
        records = responses["records"].json()
        assert len(records["completed_game_ids"]) == 107
        wins = responses["wins"].json()
        assert wins["win_projection"]["as_of"] == "2026-11-02"
        assert len(responses["rankings"].json()["players"]) == 528
        assert responses["lineups"].json()["lineups"]
        return {
            "release_id": serving.release_id,
            "edition_id": serving.bindings["edition_id"],
            "schedule_games": len(schedule["games"]),
            "performance_games": season["games"],
            "completed_games": len(records["completed_game_ids"]),
            "ranking_players": 528,
            "curry_nov2": True,
            "endpoint_release_ids": sorted(release_ids),
        }


def test_isolated_dagster_job_builds_all_surfaces_and_actual_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, config, spec, approval, config_path, spec_path, approval_path = (
        _prepare_private_work(tmp_path)
    )
    result = prepare_daily_partition_job.execute_in_process(
        run_config={
            "ops": {
                "prepare_daily_partition_op": {
                    "config": {
                        "config_path": str(config_path),
                        "spec_path": str(spec_path),
                        "approval_path": str(approval_path),
                    }
                }
            }
        },
        raise_on_error=False,
    )
    assert result.success
    output = result.output_for_node("prepare_daily_partition_op")
    assert output["status"] == "serving_validated"
    assert output["reused_stages"] == []
    runtime = Path(output["runtime_release"]).resolve()
    manifest = verify_production_site_release(runtime)
    identity = manifest["identity"]
    assert identity["evidence_through_date"] == "2026-11-02"
    assert identity["game_count"] == 107
    assert identity["live_pointer"] == str(config.live_pointer)
    bindings = ReleaseServing.load(runtime).bindings
    assert set(bindings["surface_bindings"]) == {
        "player_pages", "rankings", "teams", "lineups", "wins", "schedule"
    }
    api = _api_checks(
        runtime, config.environment_root / "test-api-release",
        config.observation_root, monkeypatch,
    )
    evidence = {
        "dagster_run_id": result.run_id,
        "source_work_id": read_object(source / "spec.json")["work_id"],
        "work_id": spec["work_id"],
        "runtime_release": str(runtime),
        "runtime_sha256": manifest["runtime_sha256"],
        "artifact_ids": {
            key: manifest["approved_manifest"]["identity"][key]["bundle_sha256"]
            for key in ("data", "model", "serving", "lineups")
        },
        "api": api,
        "model_retrained": False,
        "production_changed": False,
    }
    atomic_json(tmp_path / "nightly-lifecycle-evidence.json", evidence)


def test_assembled_runtime_activation_rollback_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, config, _, _, config_path, spec_path, approval_path = _prepare_private_work(tmp_path)
    built = prepare_daily_partition_job.execute_in_process(
        run_config={"ops":{"prepare_daily_partition_op":{"config":{
            "config_path":str(config_path), "spec_path":str(spec_path),
            "approval_path":str(approval_path),
        }}}},
        raise_on_error=True,
    ).output_for_node("prepare_daily_partition_op")
    runtime = Path(built["runtime_release"]).resolve()
    publisher = config.environment_root / "test-publisher"
    pointer = config.environment_root / "activation-live"
    pointer.symlink_to(config.live_pointer.resolve(), target_is_directory=True)
    validated = []

    def verify(record):
        release = Path(record["release_path"]).resolve()
        verified = verify_production_site_release(release)
        serving = ReleaseServing.load(release)
        assert serving is not None
        assert serving.release_id == verified["runtime_sha256"]
        validated.append((record["generation"], serving.release_id))

    success = activate_release(
        release=runtime, live_pointer=pointer, publisher_root=publisher,
        environment="dev", operation_id="integrated-success",
        expected_previous_generation=0, reason="issue 57 isolated integration",
        reload_api=lambda record: verify(record), verify_api=verify,
    )
    assert success.status == "activated"
    assert read_active_activation(pointer)["generation"] == 1

    calls = []
    def fail_second(record):
        calls.append(record["generation"])
        if record["generation"] == 2:
            raise RuntimeError("injected assembled API verification failure")
        verify(record)

    with pytest.raises(RuntimeError, match="injected assembled"):
        activate_release(
            release=runtime, live_pointer=pointer, publisher_root=publisher,
            environment="dev", operation_id="integrated-failure",
            expected_previous_generation=1, reason="issue 57 rollback injection",
            reload_api=lambda record: None, verify_api=fail_second,
        )
    active = read_active_activation(pointer)
    assert active["generation"] == 3
    assert active["operation_id"].startswith("integrated-failure:rollback")
    assert calls == [2, 3]

    recovery_publisher = config.environment_root / "test-recovery-publisher"
    recovery_pointer = config.environment_root / "recovery-live"
    recovery_pointer.symlink_to(config.live_pointer.resolve(), target_is_directory=True)
    with pytest.raises(SystemExit):
        activate_release(
            release=runtime, live_pointer=recovery_pointer,
            publisher_root=recovery_publisher, environment="dev",
            operation_id="integrated-interruption", expected_previous_generation=0,
            reason="issue 57 interruption injection",
            reload_api=lambda record: None, verify_api=lambda record: None,
            after_pointer_switch=lambda record: (_ for _ in ()).throw(SystemExit()),
        )
    recovered = activate_release(
        release=runtime, live_pointer=recovery_pointer,
        publisher_root=recovery_publisher, environment="dev",
        operation_id="integrated-interruption", expected_previous_generation=0,
        reason="issue 57 interruption injection",
        reload_api=lambda record: verify(record), verify_api=verify,
    )
    assert recovered.status == "recovered"
    assert read_active_activation(recovery_pointer)["generation"] == 1
