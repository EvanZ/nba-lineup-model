"""Real Dagster storage/execution tests; basketball calculations are test doubles."""
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from dagster import DagsterInstance, build_sensor_context

from nba_lineup_model.orchestration import data_quality_probe as probe
from nba_lineup_model.orchestration.daily_instance import initialize, execute_manual, DailyInstance
from nba_lineup_model.orchestration.daily_run_status import read_daily_runs
from nba_lineup_model.orchestration.daily_work import atomic_json, work_spec
from nba_lineup_model.orchestration.partition_definitions import (
    prepare_daily_partition_job, resumable_daily_partition_sensor,
)
from nba_lineup_model.season.data_quality_report import reconcile_alert_state, publish_data_quality_report
from nba_lineup_model.web_api.data_quality_reports import read_data_quality_status
from test_data_quality_report import report, finding_codes


@pytest.fixture
def pipeline(monkeypatch):
    state = {"fail": True, "calls": 0}
    def prepare(*args, **kwargs):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("Deliberate test failure")
        return {"work_id": "tested", "status": "validated", "runtime_release": "test-only"}
    monkeypatch.setattr("nba_lineup_model.orchestration.daily_pipeline.DailyPipelineConfig.read",
                        lambda *a: SimpleNamespace())
    monkeypatch.setattr("nba_lineup_model.orchestration.resumable_pipeline.prepare_partition", prepare)
    monkeypatch.delenv("DAGSTER_HOME", raising=False)
    return state


def environment(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    template = Path(__file__).resolve().parents[1] / "deploy/dagster-daily.yaml"
    binding = initialize(name, root, template)
    with DagsterInstance.from_config(str(binding.home)):
        pass
    return binding


def request(binding, day="2026-10-30"):
    folder = binding.root / day
    folder.mkdir(exist_ok=True)
    atomic_json(folder / "config.json", {"environment_root": str(binding.root),
                "code_root": str(Path(__file__).resolve().parents[1]),
                "live_pointer": str(binding.root / "live-release")})
    spec = work_spec(season="2026-27", game_date=date.fromisoformat(day),
        source_revisions={}, model_configuration={"version": "test"}, parent_model=None,
        edition_id="test", code_identity="test", inputs={}, expected_game_ids=[])
    atomic_json(folder / "spec.json", spec)
    return {"config_path": str(folder / "config.json"), "spec_path": str(folder / "spec.json")}


def quality(binding, monkeypatch):
    monkeypatch.setattr(probe, "effective_automation_states", lambda *a: {
        "resumable_daily_partition": {"automation_status": "RUNNING",
                                     "automation_status_source": "operator_override"}})
    row = probe.daily_publication_row(dagster_home=binding.home, environment=binding.environment,
        environment_root=binding.root, activation={}, thresholds={}, expected={
            "latest_eligible_completed_date": "2026-10-30",
            "active_evidence_through_date": "2026-10-30"})
    return report(collectors=[row])


def test_sensor_failure_manual_retry_recovers_without_editing_alerts(tmp_path, monkeypatch, pipeline):
    binding = environment(tmp_path, "dev")
    req = request(binding)
    atomic_json(binding.requests / "request.json", req)
    with DagsterInstance.from_config(str(binding.home)) as instance:
        with build_sensor_context(instance=instance) as context:
            selected = resumable_daily_partition_sensor(context)
        failed = prepare_daily_partition_job.execute_in_process(instance=instance,
            run_config=selected.run_config, tags=selected.tags, raise_on_error=False)
    assert not failed.success
    payload = quality(binding, monkeypatch)
    assert "collector_run_failed" in finding_codes(payload)
    state, events = reconcile_alert_state(payload, None)
    assert len(events) == 1
    retry = execute_manual(binding.home, "prepare_daily_partition", req)
    assert not retry.success
    payload = quality(binding, monkeypatch)
    state, events = reconcile_alert_state(payload, state)
    assert state["active"] and not events
    pipeline["fail"] = False
    repaired = execute_manual(binding.home, "prepare_daily_partition", req)
    assert repaired.success
    payload = quality(binding, monkeypatch)
    assert "collector_run_failed" not in finding_codes(payload)
    state, events = reconcile_alert_state(payload, state)
    assert not state["active"] and [e["event"] for e in events] == ["recovered"]
    assert not reconcile_alert_state(payload, state)[1]
    output = tmp_path / "quality"
    publish_data_quality_report(payload, output)
    atomic_json(output / "alert-state.json", state)
    assert not read_data_quality_status(output)["urgent"]
    with DagsterInstance.from_config(str(binding.home)) as instance:
        assert len(instance.get_runs()) == 3
        assert instance.get_run_by_id(repaired.run_id).tags["gestalt/trigger"] == "manual"
        with build_sensor_context(instance=instance) as context:
            assert "validated" in resumable_daily_partition_sensor(context).skip_message


def test_other_environment_other_work_and_forecast_do_not_clear_failure(tmp_path, monkeypatch, pipeline):
    dev = environment(tmp_path, "dev")
    prod = environment(tmp_path, "prod")
    failed = execute_manual(dev.home, "prepare_daily_partition", request(dev))
    pipeline["fail"] = False
    assert execute_manual(prod.home, "prepare_daily_partition", request(prod)).success
    assert execute_manual(dev.home, "prepare_daily_partition", request(dev, "2026-10-31")).success
    monkeypatch.setattr("nba_lineup_model.orchestration.forecast_partition.prepare_forecast_date",
        lambda *a: {k: "test" for k in ["forecast_date", "model_date", "model_sha256", "work_id", "runtime_release"]})
    assert execute_manual(dev.home, "prepare_forecast_date", {
        "config_path": request(dev)["config_path"], "forecast_date": "2026-11-01"}).success
    status = read_daily_runs(dev.home, "dev", dev.root)
    assert [r["run_id"] for r in status["unresolved_failures"]] == [failed.run_id]
    assert "collector_run_failed" in finding_codes(quality(dev, monkeypatch))


def test_wrong_environment_or_history_is_rejected_before_calculations(tmp_path, pipeline):
    dev = environment(tmp_path, "dev")
    prod = environment(tmp_path, "prod")
    with pytest.raises(ValueError, match="different environment"):
        execute_manual(dev.home, "prepare_daily_partition", request(prod))
    with pytest.raises(ValueError, match="different environments"):
        read_daily_runs(prod.home, "dev", dev.root)
    assert pipeline["calls"] == 0


def test_instance_binding_survives_restarts_and_rejects_redirection(tmp_path):
    binding = environment(tmp_path, "dev")
    assert DailyInstance.read(binding.home) == binding
    template = Path(__file__).resolve().parents[1] / "deploy/dagster-daily.yaml"
    assert initialize("dev", binding.root, template) == binding
    with pytest.raises(ValueError):
        initialize("prod", binding.root, template)


def test_wrong_code_request_cannot_execute_under_a_new_worker(tmp_path, pipeline):
    binding = environment(tmp_path, "dev")
    req = request(binding)
    import json
    config = Path(req["config_path"])
    value = json.loads(config.read_text())
    value["code_root"] = str(tmp_path / "different-release")
    atomic_json(config, value)
    with pytest.raises(ValueError, match="Requested code differs"):
        execute_manual(binding.home, "prepare_daily_partition", req)
    assert pipeline["calls"] == 0


def test_empty_permanent_history_is_not_reported_as_recovered(tmp_path, monkeypatch):
    binding = environment(tmp_path, "dev")
    payload = quality(binding, monkeypatch)
    assert "daily_history_unverified" in finding_codes(payload)
    assert payload["status"] != "passed"
