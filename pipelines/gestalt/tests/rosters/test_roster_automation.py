from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from dagster import (
    DagsterEventType, DagsterInstance, DagsterRunStatus, SensorResult, SkipReason,
    build_run_status_sensor_context, build_schedule_context,
)

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.rosters import schedules
from gestalt_pipelines.rosters.sensors import roster_transactions_ready
from gestalt_pipelines.transactions.resources import NBATransactions


@pytest.mark.parametrize("day", ["2026-10-20", "2026-11-08"])
def test_new_transactions_trigger_same_date_and_stable_run_key(monkeypatch, day):
    # Execute the actual producer in a test instance; never write to S3.
    monkeypatch.setattr(NBATransactions, "fetch", lambda _: b"test")
    monkeypatch.setattr(NBATransactions, "publish", lambda *args, **kwargs: {
        "uri": "s3://test/transactions.json", "sha256": "a"*64,
        "transaction_count": 2, "new_transaction_count": 1, "source_date_row_count": 2,
    })
    job = defs.resolve_job_def("capture_nba_transactions")
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance, partition_key=day)
        success_event = next(e for e in result.all_events if e.event_type == DagsterEventType.RUN_SUCCESS)
        with build_run_status_sensor_context(
            sensor_name="roster_transactions_ready", dagster_event=success_event,
            dagster_instance=instance, dagster_run=instance.get_run_by_id(result.run_id),
        ) as context:
            request = roster_transactions_ready(context)
            repeated = roster_transactions_ready(context)
        assert isinstance(request, SensorResult)
        [run] = request.run_requests
        [again] = repeated.run_requests
        assert run.partition_key == day
        assert run.run_key == f"{result.run_id}:{day}" == again.run_key
        assert run.run_config == {}


def test_zero_new_rows_does_not_launch_rosters(monkeypatch):
    monkeypatch.setattr(NBATransactions, "fetch", lambda _: b"test")
    monkeypatch.setattr(NBATransactions, "publish", lambda *args, **kwargs: {
        "uri": "s3://test/transactions.json", "sha256": "a"*64,
        "transaction_count": 2, "new_transaction_count": 0, "source_date_row_count": 2,
    })
    with DagsterInstance.ephemeral() as instance:
        result = defs.resolve_job_def("capture_nba_transactions").execute_in_process(
            instance=instance, partition_key="2026-11-08",
        )
        event = next(e for e in result.all_events if e.event_type == DagsterEventType.RUN_SUCCESS)
        with build_run_status_sensor_context(
            sensor_name="roster_transactions_ready", dagster_event=event,
            dagster_instance=instance, dagster_run=instance.get_run_by_id(result.run_id),
        ) as context:
            assert isinstance(roster_transactions_ready(context), SkipReason)


def test_sensor_only_monitors_successful_transaction_runs():
    assert roster_transactions_ready._run_status == DagsterRunStatus.SUCCESS
    assert [job.name for job in roster_transactions_ready._monitored_jobs] == ["capture_nba_transactions"]


@pytest.mark.parametrize("tick", [
    datetime(2026, 11, 12, 3, 30, tzinfo=ZoneInfo("America/Los_Angeles")),
    datetime(2027, 3, 14, 3, 30, tzinfo=ZoneInfo("America/Los_Angeles")),
    datetime(2027, 11, 7, 3, 30, tzinfo=ZoneInfo("America/Los_Angeles")),
])
def test_daily_rosters_use_yesterday_in_pacific_even_across_dst(monkeypatch, tick):
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _: tick.astimezone(UTC)))
    schedule = defs.get_schedule_def("rosters_morning")
    assert schedule.cron_schedule == "30 3 * * *"
    assert schedule.execution_timezone == "America/Los_Angeles"
    with build_schedule_context(
        scheduled_execution_time=tick, repository_def=defs.get_repository_def(),
    ) as context:
        [request] = schedule.evaluate_tick(context).run_requests
    assert request.partition_key == (tick.date() - timedelta(days=1)).isoformat()
    assert request.run_config == {}


def test_missed_daily_run_requires_explicit_backfill(monkeypatch):
    tick = datetime(2026, 11, 12, 3, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(
        now=lambda _: tick.astimezone(UTC)+timedelta(days=2),
    ))
    with build_schedule_context(
        scheduled_execution_time=tick, repository_def=defs.get_repository_def(),
    ) as context:
        evaluated = schedules.rosters_morning.evaluate_tick(context)
    assert not evaluated.run_requests
    assert "manual backfill" in evaluated.skip_message
