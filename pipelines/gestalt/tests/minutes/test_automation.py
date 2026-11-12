from datetime import UTC, datetime, timedelta

from dagster import (
    AssetKey, AssetMaterialization, AssetSelection, AssetSpec, DailyPartitionsDefinition,
    Definitions, DagsterInstance, Failure, asset, define_asset_job,
    evaluate_automation_conditions, in_process_executor,
)

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.minutes.assets import player_minutes
from gestalt_pipelines.minutes.partitions import minutes_dates
from gestalt_pipelines.minutes.resources import MinutesArchive
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.utils import PACIFIC


def day_keys():
    yesterday = datetime.now(UTC).astimezone(PACIFIC).date()-timedelta(days=1)
    return str(yesterday), str(yesterday-timedelta(days=1))


def evaluate(definitions, instance, cursor=None):
    return evaluate_automation_conditions(
        definitions, instance=instance, cursor=cursor,
        asset_selection=AssetSelection.assets("player_minutes"),
        evaluation_time=datetime.now(UTC)+timedelta(seconds=1),
    )


def report_inputs(instance, day, parent, *, rosters=True):
    for name in ("nba_stints", "nba_box_scores", "injury_status_by_game_date"):
        instance.report_runless_asset_event(AssetMaterialization(asset_key=name, partition=day))
    instance.report_runless_asset_event(AssetMaterialization(asset_key="player_minutes", partition=parent))
    if rosters:
        instance.report_runless_asset_event(AssetMaterialization(asset_key="rosters_by_date", partition=day))


def test_automation_waits_for_rosters_then_requests_once_and_holds_after_success():
    day, parent = day_keys()
    with DagsterInstance.ephemeral() as instance:
        initial = evaluate(defs, instance)
        report_inputs(instance, day, parent, rosters=False)
        waiting = evaluate(defs, instance, initial.cursor)
        assert waiting.total_requested == 0
        instance.report_runless_asset_event(AssetMaterialization(asset_key="rosters_by_date", partition=day))
        ready = evaluate(defs, instance, waiting.cursor)
        assert ready.total_requested == 1
        assert ready.get_requested_partitions(AssetKey("player_minutes")) == {day}
        instance.report_runless_asset_event(AssetMaterialization(asset_key="player_minutes", partition=day))
        done = evaluate(defs, instance, ready.cursor)
        assert done.total_requested == 0


def test_latest_failed_upstream_blocks_even_with_previous_successful_materialization():
    day, parent = day_keys()

    should_fail = [True]

    @asset(name="nba_stints", partitions_def=minutes_dates)
    def failed_stints():
        if should_fail[0]:
            raise Failure("Intentional upstream failure in an isolated automation test.")

    fail_job = define_asset_job(
        "fail_stints_test", selection=AssetSelection.assets(failed_stints),
        executor_def=in_process_executor,
    )
    definitions = Definitions(
        assets=[player_minutes, failed_stints]+[
            AssetSpec(name, partitions_def=minutes_dates)
            for name in ("nba_box_scores", "rosters_by_date", "injury_status_by_game_date")
        ],
        jobs=[fail_job],
        resources={"minutes_archive": MinutesArchive(season="2026-27"),
                   "nba_schedule": NBAScheduleResource(season="2026-27")},
    )
    with DagsterInstance.ephemeral() as instance:
        initial = evaluate(definitions, instance)
        report_inputs(instance, day, parent)
        failed = definitions.resolve_job_def("fail_stints_test").execute_in_process(
            instance=instance, partition_key=day, raise_on_error=False,
        )
        assert not failed.success
        blocked = evaluate(definitions, instance, initial.cursor)
        assert blocked.total_requested == 0

        should_fail[0] = False
        recovered = definitions.resolve_job_def("fail_stints_test").execute_in_process(
            instance=instance, partition_key=day, raise_on_error=False,
        )
        assert recovered.success
        resumed = evaluate(definitions, instance, blocked.cursor)
        assert resumed.get_requested_partitions(AssetKey("player_minutes")) == {day}
