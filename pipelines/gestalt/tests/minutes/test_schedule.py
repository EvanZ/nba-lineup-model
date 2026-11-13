from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from dagster import DagsterRunStatus, build_schedule_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.minutes import schedules
from gestalt_pipelines.minutes.assets.update import _require_morning_inputs
from gestalt_pipelines.minutes.utils import MinutesError
from gestalt_pipelines.utils import PACIFIC


@pytest.mark.parametrize("tick", [
    datetime(2026, 11, 13, 4, tzinfo=PACIFIC),
    datetime(2027, 3, 14, 4, tzinfo=PACIFIC),
    datetime(2027, 11, 7, 4, tzinfo=PACIFIC),
])
def test_four_am_schedule_uses_yesterday_across_dst(monkeypatch, tick):
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _: tick.astimezone(UTC)))
    schedule = defs.get_schedule_def("update_player_minutes_schedule")
    assert schedule.cron_schedule == "0 4 * * *"
    assert schedule.execution_timezone == "America/Los_Angeles"
    with build_schedule_context(scheduled_execution_time=tick, repository_def=defs.get_repository_def()) as context:
        first, = schedule.evaluate_tick(context).run_requests
        again, = schedule.evaluate_tick(context).run_requests
    assert first.partition_key == str(tick.date()-timedelta(days=1))
    assert first.run_key == again.run_key == tick.astimezone(UTC).isoformat()
    assert first.tags["minutes/morning_date"] == str(tick.date())
    assert first.run_config == {}


def test_missed_schedule_does_not_automatically_replay_old_dates(monkeypatch):
    tick = datetime(2026, 11, 13, 4, tzinfo=PACIFIC)
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _: tick.astimezone(UTC)+timedelta(days=2)))
    with build_schedule_context(scheduled_execution_time=tick, repository_def=defs.get_repository_def()) as context:
        result = schedules.update_player_minutes_schedule.evaluate_tick(context)
    assert not result.run_requests and "manual backfill" in result.skip_message


@pytest.mark.parametrize("bad", ["missing", "failed", "running", "old"])
def test_scheduled_update_fails_when_either_morning_input_is_not_ready(bad):
    morning = datetime(2026, 11, 13, tzinfo=PACIFIC).timestamp()
    for bad_job in ("parse_nba_games", "build_nba_rosters"):
        def records(*, filters, limit):
            assert filters.tags == {"dagster/partition": "2026-11-12"}
            status, end = DagsterRunStatus.SUCCESS, morning+60
            if filters.job_name == bad_job:
                if bad == "missing":
                    return []
                if bad == "failed":
                    status = DagsterRunStatus.FAILURE
                if bad == "running":
                    status, end = DagsterRunStatus.STARTED, None
                if bad == "old":
                    end = morning-1
            return [SimpleNamespace(dagster_run=SimpleNamespace(status=status), end_time=end)]
        context = SimpleNamespace(
            run=SimpleNamespace(tags={"minutes/morning_date": "2026-11-13"}),
            instance=SimpleNamespace(get_run_records=records),
        )
        with pytest.raises(MinutesError, match=bad_job):
            _require_morning_inputs(context, "2026-11-12")


def test_successful_morning_inputs_allow_update_and_manual_backfills_need_no_run_history():
    morning = datetime(2026, 11, 13, tzinfo=PACIFIC).timestamp()
    calls = []
    def records(*, filters, limit):
        calls.append(filters.job_name)
        return [SimpleNamespace(dagster_run=SimpleNamespace(status=DagsterRunStatus.SUCCESS), end_time=morning+60)]
    context = SimpleNamespace(run=SimpleNamespace(tags={"minutes/morning_date": "2026-11-13"}), instance=SimpleNamespace(get_run_records=records))
    _require_morning_inputs(context, "2026-11-12")
    assert calls == ["parse_nba_games", "build_nba_rosters"]
    _require_morning_inputs(SimpleNamespace(run=SimpleNamespace(tags={})), "2026-10-20")
