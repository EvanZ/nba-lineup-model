from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from dagster import DagsterInstance, DagsterRunStatus, Definitions, build_sensor_context
from dagster._core.remote_origin import (
    RegisteredCodeLocationOrigin,
    RemoteJobOrigin,
    RemoteRepositoryOrigin,
)

from nba_lineup_model.orchestration.daily_pipeline import (
    DailyPipelineConfig,
    causal_schedule_sha256,
    prepare_daily_release,
    require_complete_slate,
    require_clean_code_revision,
)
from nba_lineup_model.orchestration.definitions import (
    complete_slate_daily_release_sensor,
    defs,
    prepare_daily_site_release_job,
)


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
    bootstrap_release = root / "bootstrap-release"
    bootstrap_release.mkdir()
    (bootstrap_release / "unchanged").write_text("original")
    config.live_pointer.symlink_to(bootstrap_release, target_is_directory=True)
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


def test_historical_data_replay_dates_are_explicit_and_reported(tmp_path):
    config = config_fixture(tmp_path)
    seen = []

    class Steps:
        def __init__(self, _config, candidate, _revision):
            self.candidate = candidate

        def __getattr__(self, name):
            def run(*_args):
                if name == "data":
                    seen.extend(self.data_replay_dates)
                return self.candidate / "runtime"

            return run

    result = prepare_daily_release(
        config,
        date(2026, 10, 20),
        steps_class=Steps,
        data_replay_dates=(date(2026, 10, 20),),
    )

    assert seen == [date(2026, 10, 20)]
    assert result["data_replay_dates"] == ["2026-10-20"]
    report = json.loads(
        (config.pipeline_root / "reports/2026-10-20/report.json").read_text()
    )
    assert report["data_replay_dates"] == ["2026-10-20"]


def test_historical_model_replay_approval_is_exact_and_reported(tmp_path):
    config = config_fixture(tmp_path)
    approved = date(2026, 10, 20)
    seen = []

    class Steps:
        def __init__(self, _config, candidate, _revision):
            self.candidate = candidate

        def __getattr__(self, name):
            def run(*_args):
                if name == "model":
                    seen.append(self.approved_historical_model_replay_from)
                return self.candidate / "runtime"

            return run

    result = prepare_daily_release(
        config,
        approved,
        steps_class=Steps,
        approved_historical_model_replay_from=approved,
    )

    assert seen == [approved]
    assert result["approved_historical_model_replay_from"] == "2026-10-20"
    report = json.loads(
        (config.pipeline_root / "reports/2026-10-20/report.json").read_text()
    )
    assert report["approved_historical_model_replay_from"] == "2026-10-20"


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


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "mutable_live",
        "causal_stale",
        "reload_failure",
        "health_failure",
        "rollback_failure",
    ],
)
def test_activation_covers_stale_state_and_process_rollback(tmp_path, monkeypatch, scenario):
    from nba_lineup_model.orchestration import daily_activation as activation
    config = config_fixture(tmp_path)
    # Replace the test marker directory with a normal release pointer.
    old = config.pipeline_root / "releases/old"
    new = config.pipeline_root / "releases/new"
    old.mkdir(parents=True)
    new.mkdir()
    config.live_pointer.unlink()
    config.live_pointer.symlink_to(old, target_is_directory=True)
    (new / "schedule").mkdir()
    report = {
        "status": "validated_candidate",
        "game_date": "2026-10-20",
        "runtime_release": str(new),
        "activation_operation_id": "daily-test-operation",
        "expected_previous_generation": 0,
        "source_causal_schedule_sha256": causal_schedule_sha256(
            config.schedule_root
        ),
    }
    manifest = {"identity":{"live_pointer":str(config.live_pointer), "evidence_through_date":"2026-10-20"}}
    monkeypatch.setattr(activation, "verify_production_site_release", lambda path: manifest)
    monkeypatch.setattr(
        "nba_lineup_model.season.release_publisher.verify_production_site_release",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        "nba_lineup_model.season.release_publisher._validated_release_identity",
        lambda release: {
            "release_path": Path(release).resolve().as_posix(),
            "release_id": f"release-{Path(release).resolve().name}",
            "runtime_sha256": "runtime-sha",
            "binding_map_sha256": "bindings-sha",
            "edition_id": "original",
            "code": {
                "git_revision": "a" * 40,
                "package_sha256": "b" * 64,
                "dependency_lock_sha256": "c" * 64,
            },
        },
    )
    events = []
    def reload_api(c, record):
        release_name = Path(record["release_path"]).name
        events.append(("reload", release_name))
        if scenario == "reload_failure" and release_name == "new":
            raise RuntimeError("restart failed")
    def verify_api(c, release, _record):
        events.append(("verify", release.name))
        if scenario in {"health_failure", "rollback_failure"}:
            if release == new or scenario == "rollback_failure":
                raise RuntimeError("API mismatch")
    if scenario == "mutable_live":
        latest = config.schedule_root / "latest/2026-10-20.json"
        latest.parent.mkdir()
        latest.write_text('{"market": "new odds"}')
    if scenario == "causal_stale":
        (config.schedule_root / "frozen/0022600001.json").write_text('{}')
        with pytest.raises(ValueError, match="Causal schedule inputs changed"):
            activation.activate_daily_release(
                config, report, reload_api=reload_api, verify_api=verify_api
            )
        assert events == []
    elif scenario in {"success", "mutable_live"}:
        activation.activate_daily_release(
            config, report, reload_api=reload_api, verify_api=verify_api
        )
        active = activation.read_active_activation(config.live_pointer)
        assert active["release_path"] == new.as_posix()
        assert active["generation"] == 1
        assert report["status"] == "activated"
        assert events == [("reload", "new"), ("verify", "new")]
        return
    else:
        with pytest.raises(Exception):
            activation.activate_daily_release(config, report, reload_api=reload_api, verify_api=verify_api)
        assert ("reload","old") in events
        assert ("verify","old") in events
        assert report["status"] == ("rollback_failed" if scenario == "rollback_failure" else "activation_failed")
    active = activation.read_active_activation(config.live_pointer)
    assert active["release_path"] == old.as_posix()


def test_pipeline_report_is_accessible_without_publishing_a_failed_candidate(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nba_lineup_model.orchestration.daily_pipeline import _report
    from nba_lineup_model.web_api.app import create_app
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


def test_isolated_daily_dagster_boundary_exposes_only_atomic_jobs():
    from nba_lineup_model.orchestration.daily_definitions import defs as daily_defs

    Definitions.validate_loadable(daily_defs)
    assert {job.name for job in daily_defs.jobs} == {
        "backfill_official_history",
        "prepare_daily_site_release",
    }
    assert {sensor.name for sensor in daily_defs.sensors} == {
        "complete_slate_daily_release"
    }
    assert not daily_defs.schedules


def test_daily_systemd_boundary_is_separate_from_collectors_and_legacy():
    root = Path(__file__).parents[1]
    for name in (
        "gestalt-daily-code-server.service",
        "gestalt-daily-webserver.service",
        "gestalt-daily-daemon.service",
    ):
        unit = (root / "deploy" / name).read_text()
        assert "DAGSTER_HOME=/var/lib/nba-gestalt/daily-dagster" in unit
        assert "GESTALT_DAILY_PIPELINE_CONFIG=" in unit
        assert "gestalt-dagster-" not in unit
        assert "collection_definitions" not in unit
    code_server = (root / "deploy/gestalt-daily-code-server.service").read_text()
    assert "orchestration.daily_definitions" in code_server
    assert "MemoryHigh=750M" in code_server
    assert "MemoryMax=1G" in code_server

    production_config = json.loads(
        (root / "deploy/gestalt-daily-production-config.json").read_text()
    )
    assert production_config["release_root"] == "/opt/nba-gestalt/current"

def test_git_preflight_timeout_is_visible_and_bounded(tmp_path, monkeypatch):
    from nba_lineup_model.orchestration import daily_pipeline

    config = config_fixture(tmp_path)

    def blocked(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=30)

    monkeypatch.setattr(daily_pipeline.subprocess, "check_output", blocked)
    with pytest.raises(RuntimeError, match="Git preflight exceeded 30 seconds"):
        prepare_daily_release(config, date(2026, 10, 20))

    report = json.loads(
        (
            config.pipeline_root
            / "reports/2026-10-20/report.json"
        ).read_text()
    )
    assert report["status"] == "failed"
    assert report["stage"] == "preflight"
    assert "Git preflight exceeded 30 seconds" in report["error"]


def test_post_activation_production_gate_checks_full_player_history(
    tmp_path, monkeypatch
):
    import io

    from nba_lineup_model.orchestration import daily_activation
    from nba_lineup_model.season import dev_site_release

    config = config_fixture(tmp_path)
    release = config.pipeline_root / "release"
    release.mkdir(parents=True)
    manifest = {
        "identity": {
            "approved_release_sha256": "release-sha",
            "evidence_through_date": "2026-10-20",
            "game_count": 2,
            "serving_bundle_sha256": "serving-sha",
            "lineup_bundle_sha256": "lineup-sha",
        },
        "approved_manifest": {
            "identity": {"schedule": {"final_game_count": 2}}
        },
    }
    monkeypatch.setattr(
        daily_activation,
        "verify_production_site_release",
        lambda _release: manifest,
    )
    health = {"site_release": {"release_sha256": "release-sha"}}
    monkeypatch.setattr(
        daily_activation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(health).encode()),
    )
    seen = []
    monkeypatch.setattr(
        daily_activation,
        "collect_api_validation",
        lambda _api, **kwargs: seen.append(kwargs) or {},
    )
    monkeypatch.setattr(
        dev_site_release,
        "_normalize_api_validation",
        lambda *_args, **_kwargs: {},
    )

    monkeypatch.setattr(
        daily_activation,
        "verify_loaded_identity",
        lambda *_args, **_kwargs: None,
    )

    daily_activation.verify_environment_api(config, release, {})
    assert seen == [
        {
            "evidence_date": "2026-10-20",
            "daily_model_root": config.model_root,
        }
    ]


def test_daily_deploy_stops_daemon_and_rejects_active_runs():
    script = (
        Path(__file__).parents[1]
        / "scripts/deploy_gestalt_daily_orchestration.sh"
    ).read_text()

    stop = script.index("systemctl stop gestalt-daily-daemon")
    switch = script.index('sudo mv -Tf "$temporary" "$APP_ROOT/current"')
    assert stop < switch
    assert '"QUEUED", "STARTING", "STARTED", "CANCELING"' in script
    assert "Refusing daily deployment while" in script
    assert script.index('git -C "$SOURCE_REPO" worktree add') < script.index(
        'model_configuration_source="$release/deploy/'
    )
    assert "Refusing to overwrite a different accepted HCA configuration" in script


def test_interrupted_daily_stage_writes_terminal_report(tmp_path):
    config = config_fixture(tmp_path)

    class Steps:
        def __init__(self, _config, candidate, _revision):
            self.candidate = candidate

        def data(self, _target):
            raise KeyboardInterrupt("injected deployment interrupt")

    with pytest.raises(KeyboardInterrupt, match="deployment interrupt"):
        prepare_daily_release(
            config,
            date(2026, 10, 20),
            steps_class=Steps,
        )
    report = json.loads(
        (
            config.pipeline_root
            / "reports/2026-10-20/report.json"
        ).read_text()
    )
    assert report["status"] == "failed"
    assert report["stage"] == "data"
    assert report["error"].startswith("KeyboardInterrupt:")


def test_canonical_steps_pin_the_resolved_model_runtime(tmp_path):
    from nba_lineup_model.orchestration.daily_pipeline import CanonicalPipelineSteps

    config = config_fixture(tmp_path)
    runtime_link = config.environment_root / "runtime-link"
    runtime_link.symlink_to(config.release_root)
    config = replace(config, release_root=runtime_link)
    candidate = config.pipeline_root / "candidate"
    candidate.mkdir(parents=True)

    steps = CanonicalPipelineSteps(config, candidate, "revision")
    assert steps.runtime_root == config.release_root.resolve(strict=True)
    assert steps.runtime_root != runtime_link

def test_accepted_hca_model_is_extended_with_exact_provenance(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from nba_lineup_model.orchestration import daily_pipeline
    from nba_lineup_model.orchestration.daily_pipeline import CanonicalPipelineSteps
    from nba_lineup_model.season import daily_model_replay as replay
    from nba_lineup_model.season.daily_artifact_common import file_sha256

    config = config_fixture(tmp_path)
    selection = config.environment_root / "accepted-hca.json"
    selection.write_text(
        json.dumps(
            {
                "candidate_id": daily_pipeline.CAUSAL_HCA_MODEL_TRACK,
                "target_season": "2026-27",
                "prior_mean": 1.7,
                "prior_equivalent_possessions": 42000.0,
                "observation_variance_per_100": 14000.0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    provenance_url = "http://127.0.0.1:18013/api/matchups"
    accepted = replay.CausalHcaReplayConfig(
        season="2026-27",
        home_court_prior_mean=1.7,
        home_court_prior_equivalent_possessions=42000.0,
        home_court_observation_variance_per_100=14000.0,
        home_court_prior_source=(
            "artifact://causal-hca-prior/" + file_sha256(selection)
        ),
        matchup_url=provenance_url,
    )
    config = replace(
        config,
        model_track=daily_pipeline.CAUSAL_HCA_MODEL_TRACK,
        model_code_revision="accepted-revision",
        model_configuration_path=selection,
        model_configuration_sha256=accepted.sha256,
        model_provenance_matchup_url=provenance_url,
    )
    config.preseason_profiles_path.write_bytes(b"fixed profile artifact")
    candidate = config.pipeline_root / "candidate"
    candidate.mkdir(parents=True)
    steps = CanonicalPipelineSteps(config, candidate, "orchestration-revision")
    checked = []
    monkeypatch.setattr(
        daily_pipeline,
        "require_model_implementation_compatible",
        lambda root, revision: checked.append((root, revision)),
    )

    @contextmanager
    def api(**kwargs):
        assert kwargs["log_name"] == "context-api.log"
        yield "http://127.0.0.1:12345"

    monkeypatch.setattr(steps, "api", api)

    def provider(url, *, court):
        assert url == "http://127.0.0.1:12345/api/matchups"
        assert court == "neutral"
        return lambda *_args: ([1.0], [])

    monkeypatch.setattr(replay, "_http_context_provider", provider)

    def run(**kwargs):
        assert kwargs["config"].sha256 == accepted.sha256
        assert kwargs["code_revision"] == "accepted-revision"
        assert kwargs["context_provider"](None, False, False) == ([1.0], [])

    monkeypatch.setattr(replay, "replay_daily_model_chain", run)
    steps.model(date(2026, 11, 2))
    assert checked == [(config.code_root, "accepted-revision")]


def test_model_implementation_compatibility_rejects_model_code_change(tmp_path):
    from nba_lineup_model.orchestration.daily_pipeline import (
        require_model_implementation_compatible,
    )

    repo = tmp_path / "repo"
    model_file = repo / "src/nba_lineup_model/season/daily_model_replay.py"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("VALUE = 1\n")
    inference = repo / "src/nba_lineup_model/web_api/inference.py"
    inference.parent.mkdir(parents=True)
    inference.write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "accepted",
        ],
        check=True,
    )
    accepted = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    from nba_lineup_model.orchestration.model_compatibility import calculation_inventory
    policy = repo / "src/nba_lineup_model/orchestration/model_compatibility" / (accepted + ".json")
    policy.parent.mkdir(parents=True)
    for relative in ["pyproject.toml", "uv.lock", "deploy/orchestration-requirements.txt",
                     "deploy/web-requirements.txt"]:
        dependency = repo / relative
        dependency.parent.mkdir(parents=True, exist_ok=True)
        dependency.write_text("fixed dependencies\n")
    policy.write_text(json.dumps({"schema_version":1, "accepted_revision":accepted,
        "compatible_files":calculation_inventory(repo)}))
    (repo / "README").write_text("orchestration-only\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "unrelated",
        ],
        check=True,
    )
    require_model_implementation_compatible(repo, accepted)
    model_file.write_text("VALUE = 2\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "model change",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="explicitly approved causal replay"):
        require_model_implementation_compatible(repo, accepted)


def test_production_config_names_accepted_hca_lineage():
    root = Path(__file__).parents[1]
    config = json.loads(
        (root / "deploy/gestalt-daily-production-config.json").read_text()
    )
    assert config["activate_after_validation"] is False
    assert config["model_track"] == "causal_hca_replacement_candidate_v2"
    assert config["model_code_revision"] == (
        "1459c83f82b1fa50d901048e65bb39c9fd678204"
    )
    assert config["model_configuration_sha256"] == (
        "558385560bbb2b2d518bceb847468e2f87900a33964191f54a4ecc2bd07534c3"
    )
    assert config["model_root"].endswith("/daily-model-causal-hca-v2")


@pytest.mark.parametrize("damage", [None, "unfinished", "input", "checkpoint", "artifact", "active"])
def test_resumable_work_handoff_verifies_every_receipt(tmp_path, monkeypatch, damage):
    import shutil

    from nba_lineup_model.orchestration import daily_activation as activation
    from nba_lineup_model.orchestration.daily_work import (
        STAGES,
        DailyWork,
        atomic_json,
        digest,
        file_reference,
        work_spec,
    )
    from nba_lineup_model.orchestration.resumable_pipeline import bind_execution

    config = config_fixture(tmp_path)
    base = config.live_pointer.resolve() / "runtime-manifest.json"
    atomic_json(base, {"runtime_sha256": "old"})
    runtime = config.pipeline_root / "releases" / "tested"
    runtime.mkdir(parents=True)
    artifact = runtime / "payload"
    artifact.write_text("verified")
    shutil.copytree(config.schedule_root, runtime / "schedule-snapshot")
    spec = work_spec(
        season=config.season, game_date=date(2026, 10, 20), source_revisions={},
        model_configuration={"version": "test"}, parent_model=None, edition_id="test",
        code_identity="c" * 40, inputs={str(base): file_reference(base)}, expected_game_ids=[],
    )
    spec = bind_execution(spec, config, {"base_release_manifest": file_reference(base)})
    work = DailyWork(config.pipeline_root / "partitions", spec)
    work.run({stage: lambda: {
        "artifacts": [file_reference(artifact)], "result": {"runtime_release": str(runtime)},
    } for stage in STAGES})
    monkeypatch.setattr(activation, "verify_production_site_release", lambda _: {
        "identity": {"code_revision": "c" * 40},
    })
    monkeypatch.setattr(activation, "read_active_activation", lambda _: {
        "runtime_sha256": "different" if damage == "active" else "old", "generation": 7,
    })
    if damage == "unfinished":
        status = json.loads((work.root / "status.json").read_text())
        atomic_json(work.root / "status.json", {**status, "status": "running"})
    elif damage == "input":
        atomic_json(base, {"runtime_sha256": "changed"})
    elif damage == "artifact":
        artifact.write_text("changed")
    elif damage == "checkpoint":
        path = work.root / "checkpoints/model.json"
        receipt = json.loads(path.read_text())
        receipt["stage_key"] = "f" * 64
        receipt["receipt_id"] = digest({k:v for k,v in receipt.items() if k != "receipt_id"})
        atomic_json(path, receipt)
    if damage:
        with pytest.raises(ValueError):
            activation.activation_report_from_work(config, work.root)
        assert not (config.pipeline_root / "reports/2026-10-20/report.json").exists()
    else:
        report = activation.activation_report_from_work(config, work.root)
        assert report["status"] == "validated_candidate"
        assert report["expected_previous_generation"] == 7
        assert report["work_id"] == spec["work_id"]
        assert len(report["checkpoint_ids"]) == 5
        assert activation.activation_report_from_work(config, work.root) == report
    assert (config.live_pointer / "unchanged").read_text() == "original"


def test_activation_cli_accepts_approved_migration_and_validated_work():
    from nba_lineup_model.orchestration.daily_activation import _parser

    args = _parser().parse_args([
        "--config", "/dev/config.json", "--work", "/dev/work",
        "--legacy-migration-approval", "explicit-user-approval",
    ])
    assert args.legacy_migration_approval == "explicit-user-approval"
    assert args.work == Path("/dev/work")
    assert args.report is None


def test_clean_code_revision_accepts_verified_sealed_package(tmp_path, monkeypatch):
    package = tmp_path / "revision=test"
    code = package / "code"
    code.mkdir(parents=True)
    (package / "code-package.json").write_text("{}")
    revision = "d" * 40
    from nba_lineup_model.season import sealed_code_package

    monkeypatch.setattr(
        sealed_code_package,
        "verify_sealed_code_package",
        lambda root: {"identity": {"git_revision": revision}},
    )
    assert require_clean_code_revision(code) == revision
