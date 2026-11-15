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
    datetime(2026, 10, 21, 4, 15, tzinfo=PACIFIC),
])
def test_schedule_targets_previous_completed_date(monkeypatch, tick):
    monkeypatch.setattr(schedules, "datetime", SimpleNamespace(now=lambda _:tick.astimezone(UTC)))
    schedule=defs.get_schedule_def("update_player_ratings_schedule")
    assert schedule.cron_schedule=="15 4 * * *"
    assert schedule.execution_timezone=="America/Los_Angeles"
    with build_schedule_context(scheduled_execution_time=tick,repository_def=defs.get_repository_def()) as context:
        request,=schedule.evaluate_tick(context).run_requests
        again,=schedule.evaluate_tick(context).run_requests
    assert request.partition_key==str(tick.date()-timedelta(days=1))
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


def test_morning_ratings_waits_for_fresh_same_date_sources_and_manual_replay_is_independent():
    from types import SimpleNamespace
    from dagster import DagsterRunStatus
    from gestalt_pipelines.ratings.assets.update import _require_morning_inputs
    midnight = datetime(2026, 11, 14, tzinfo=PACIFIC).timestamp()
    rows = []
    def records(*, filters, limit):
        assert filters.tags == {'dagster/partition': '2026-11-13'}
        assert filters.job_name in {'parse_nba_games', 'build_nba_rosters'}
        return rows
    context = SimpleNamespace(run=SimpleNamespace(tags={'ratings/morning_date': '2026-11-14'}),
                              instance=SimpleNamespace(get_run_records=records))
    with pytest.raises(ValueError, match='has not completed successfully'):
        _require_morning_inputs(context, '2026-11-13')
    rows.append(SimpleNamespace(dagster_run=SimpleNamespace(status=DagsterRunStatus.SUCCESS), end_time=midnight-1))
    with pytest.raises(ValueError, match='has not completed successfully'):
        _require_morning_inputs(context, '2026-11-13')
    rows[0].end_time = midnight+3600
    _require_morning_inputs(context, '2026-11-13')
    rows[0].dagster_run.status = DagsterRunStatus.FAILURE
    with pytest.raises(ValueError, match='has not completed successfully'):
        _require_morning_inputs(context, '2026-11-13')
    _require_morning_inputs(SimpleNamespace(run=SimpleNamespace(tags={})), '2026-10-20')
