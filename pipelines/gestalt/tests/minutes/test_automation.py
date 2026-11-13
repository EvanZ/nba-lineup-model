from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from dagster import (
    AssetMaterialization, AssetSelection, Definitions, DagsterInstance, Failure,
    asset, define_asset_job, evaluate_automation_conditions, in_process_executor,
)

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.minutes.assets.update import _require_morning_inputs
from gestalt_pipelines.minutes.partitions import minutes_dates
from gestalt_pipelines.minutes.utils import MinutesError
from gestalt_pipelines.utils import PACIFIC


def test_upstream_events_do_not_launch_the_replaced_minutes_sensor():
    yesterday = datetime.now(UTC).astimezone(PACIFIC).date()-timedelta(days=1)
    with DagsterInstance.ephemeral() as instance:
        for name in ("nba_stints", "rosters_by_date", "injury_status_by_game_date"):
            instance.report_runless_asset_event(AssetMaterialization(asset_key=name, partition=str(yesterday)))
        instance.report_runless_asset_event(AssetMaterialization(
            asset_key="player_minutes", partition=str(yesterday-timedelta(days=1)),
        ))
        result = evaluate_automation_conditions(
            defs, instance=instance, asset_selection=AssetSelection.assets("player_minutes"),
            evaluation_time=datetime.now(UTC)+timedelta(seconds=1),
        )
        assert result.total_requested == 0
    sensor_names = {s.name for s in defs.sensors}
    assert "update_player_minutes_sensor" not in sensor_names
    assert "minutes_ready" not in sensor_names
    assert defs.get_schedule_def("update_player_minutes_schedule").cron_schedule == "0 4 * * *"


def test_latest_failed_morning_input_blocks_even_after_an_earlier_success():
    today = datetime.now(UTC).astimezone(PACIFIC).date()
    day = str(today-timedelta(days=1))
    should_fail = [False]

    @asset(name="nba_stints", partitions_def=minutes_dates)
    def morning_stints():
        if should_fail[0]:
            raise Failure("Intentional upstream failure in an isolated schedule test.")

    @asset(name="rosters_by_date", partitions_def=minutes_dates)
    def morning_rosters():
        pass

    definitions = Definitions(
        assets=[morning_stints, morning_rosters],
        jobs=[
            define_asset_job("parse_nba_games", selection=AssetSelection.assets(morning_stints), executor_def=in_process_executor),
            define_asset_job("build_nba_rosters", selection=AssetSelection.assets(morning_rosters), executor_def=in_process_executor),
        ],
    )
    with DagsterInstance.ephemeral() as instance:
        context = SimpleNamespace(
            instance=instance, run=SimpleNamespace(tags={"minutes/morning_date": str(today)}),
        )
        parser = definitions.resolve_job_def("parse_nba_games")
        rosters = definitions.resolve_job_def("build_nba_rosters")
        assert parser.execute_in_process(instance=instance, partition_key=day).success
        assert rosters.execute_in_process(instance=instance, partition_key=day).success
        _require_morning_inputs(context, day)
        should_fail[0] = True
        assert not parser.execute_in_process(instance=instance, partition_key=day, raise_on_error=False).success
        with pytest.raises(MinutesError, match="parse_nba_games"):
            _require_morning_inputs(context, day)
        should_fail[0] = False
        assert parser.execute_in_process(instance=instance, partition_key=day).success
        _require_morning_inputs(context, day)
