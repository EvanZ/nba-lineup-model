from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dagster import DagsterRunStatus

from nba_lineup_model.orchestration.lifecycle import next_nightly_date
from nba_lineup_model.orchestration import partition_definitions as jobs


@pytest.mark.parametrize("finals,expected", [
    (set(), None), ({"a"}, None), ({"a", "b"}, date(2025, 1, 4)),
    ({"b", "later"}, None),
])
def test_complete_official_slate_not_partial_scoreboard(finals, expected):
    assert next_nightly_date(evidence=date(2025, 1, 3),
        slates={"2025-01-04":["a", "b"], "2025-01-05":["later"]},
        final_ids=finals, now=datetime(2025, 1, 6, tzinfo=UTC)) == expected


def test_zero_game_day_alone_does_not_trigger_model():
    assert next_nightly_date(evidence=date(2025, 1, 3),
        slates={"2025-01-04":[], "2025-01-05":["a"]},
        final_ids=set(), now=datetime(2025, 1, 5, 12, tzinfo=UTC)) is None


def test_empty_dates_carried_only_toward_next_completed_slate():
    assert next_nightly_date(evidence=date(2025, 1, 3),
        slates={"2025-01-05":["a"]}, final_ids={"a"},
        now=datetime(2025, 1, 6, 8, tzinfo=UTC)) == date(2025, 1, 4)


def test_future_final_cannot_trigger_nightly():
    assert next_nightly_date(evidence=date(2025, 1, 3),
        slates={"2025-01-05":["a"]}, final_ids={"a"},
        now=datetime(2025, 1, 5, 1, tzinfo=UTC)) is None


def sensor_context(monkeypatch, *, running=False, records=()):
    monkeypatch.setenv("GESTALT_AUTOMATION_CONFIG", "/dev/worker/config.json")
    binding = SimpleNamespace(tags=lambda job, config: {"gestalt/environment":"dev"})
    monkeypatch.setattr(jobs, "binding_for_instance", lambda instance: binding)
    import nba_lineup_model.orchestration.lifecycle as lifecycle
    monkeypatch.setattr(lifecycle, "automation_request", lambda path: {
        "job":"advance_daily_environment", "key":"same-inputs",
        "config":{"config_path":str(path), "game_date":"2025-01-04"}})
    return SimpleNamespace(instance=SimpleNamespace(
        get_runs=Mock(return_value=[object()] if running else []),
        get_run_records=Mock(return_value=list(records))))


def test_sensor_submits_existing_environment_job(monkeypatch):
    context = sensor_context(monkeypatch)
    result = jobs.evaluate_automatic_lifecycle(context)
    assert result.job_name == "advance_daily_environment"
    assert result.tags["gestalt/environment"] == "dev"
    assert result.run_key == "automatic:same-inputs:attempt-1"


def test_sensor_never_queues_behind_active_environment_work(monkeypatch):
    context = sensor_context(monkeypatch, running=True)
    result = jobs.evaluate_automatic_lifecycle(context)
    assert "queued or running" in result.skip_message
    context.instance.get_run_records.assert_not_called()


def test_duplicate_success_does_not_submit_again(monkeypatch):
    record = SimpleNamespace(dagster_run=SimpleNamespace(status=DagsterRunStatus.SUCCESS))
    result = jobs.evaluate_automatic_lifecycle(
        sensor_context(monkeypatch, records=[record]))
    assert "already processed" in result.skip_message


def test_repeated_failure_surfaces_instead_of_creating_endless_queue(monkeypatch):
    record = SimpleNamespace(dagster_run=SimpleNamespace(status=DagsterRunStatus.FAILURE))
    with pytest.raises(RuntimeError, match="Three failed attempts"):
        jobs.evaluate_automatic_lifecycle(
            sensor_context(monkeypatch, records=[record] * 3))


def test_empty_partition_reuses_every_model_component_without_fitting(tmp_path, monkeypatch):
    from test_daily_artifacts import _publish_data, _publish_model
    from nba_lineup_model.orchestration.forward_daily import ForwardPartitionSteps
    from nba_lineup_model.orchestration.daily_work import file_reference, read_object
    from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
    first = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    parent = _publish_model(tmp_path, data_bundle=first.path)
    empty = _publish_data(tmp_path, game_date=date(2026, 10, 21), game_ids=[], suffix="empty")
    steps = ForwardPartitionSteps.__new__(ForwardPartitionSteps)
    steps.spec = {"identity":{"expected_game_ids":[]}}
    steps.model_inputs = [file_reference(parent.path / "manifest.json")]
    steps.data_inputs = [file_reference(empty.path / "manifest.json")]
    steps.config = SimpleNamespace(data_root=tmp_path / "data-root", model_root=tmp_path / "model-root", season="2026-27")
    steps.build_target_model(date(2026, 10, 21))
    pointer = read_object(steps.config.model_root / "latest.json")
    result = verify_daily_model_artifact(steps.config.model_root / pointer["bundle_path"])
    assert result["identity"]["components"] == parent.manifest["identity"]["components"]
    assert result["identity"]["parent_model_sha256"] == parent.manifest["bundle_sha256"]
    assert result["identity"]["daily_game_ids"] == []
    assert result["identity"]["cumulative_game_ids"] == ["0022600001"]


def test_dev_reload_has_no_sudo_and_never_targets_prod(monkeypatch):
    from pathlib import Path
    from nba_lineup_model.orchestration import daily_activation
    run = Mock()
    monkeypatch.setattr(daily_activation.subprocess, "run", run)
    daily_activation.reload_environment_api(SimpleNamespace(environment_root=Path("/var/lib/nba-gestalt-dev")), {})
    assert run.call_args.args[0] == ["systemctl", "--no-ask-password", "restart", "gestalt-web-dev"]
