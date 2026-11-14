from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from dagster import build_schedule_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.ratings import schedules
from gestalt_pipelines.utils import PACIFIC


@pytest.mark.parametrize("tick", [
    datetime(2026, 11, 15, 4, 15, tzinfo=PACIFIC),
    datetime(2027, 3, 14, 4, 15, tzinfo=PACIFIC),
    datetime(2027, 11, 7, 4, 15, tzinfo=PACIFIC),
    datetime(2026, 10, 20, 4, 15, tzinfo=PACIFIC),
])
def test_schedule_targets_today_with_previous_day_observations(monkeypatch, tick):
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _:tick.astimezone(UTC)))
    schedule=defs.get_schedule_def("update_player_ratings_schedule")
    assert schedule.cron_schedule=="15 4 * * *"
    assert schedule.execution_timezone=="America/Los_Angeles"
    with build_schedule_context(scheduled_execution_time=tick,repository_def=defs.get_repository_def()) as context:
        request,=schedule.evaluate_tick(context).run_requests
        again,=schedule.evaluate_tick(context).run_requests
    assert request.partition_key==str(tick.date())
    assert request.run_key==again.run_key==tick.astimezone(UTC).isoformat()
    assert request.tags['ratings/morning_date']==str(tick.date())
    assert request.run_config=={}


@pytest.mark.parametrize('before_start',[False,True])
def test_missed_or_preseason_tick_does_not_launch_replay(monkeypatch,before_start):
    tick=datetime(2026,10,19 if before_start else 20,4,15,tzinfo=PACIFIC)
    now=tick.astimezone(UTC)+(timedelta() if before_start else timedelta(days=2))
    monkeypatch.setattr(schedules,'datetime',SimpleNamespace(now=lambda _:now))
    with build_schedule_context(scheduled_execution_time=tick,repository_def=defs.get_repository_def()) as context:
        result=schedules.update_player_ratings_schedule.evaluate_tick(context)
    assert not result.run_requests and result.skip_message
