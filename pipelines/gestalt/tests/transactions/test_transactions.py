import json
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance, build_schedule_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.transactions import schedules
from gestalt_pipelines.transactions.assets import capture
from gestalt_pipelines.transactions.resources import NBATransactions, nba_transactions
from gestalt_pipelines.transactions.utils import transactions_for_date

DAY = date(2026, 11, 8)
OBSERVED = datetime(2026, 11, 11, 3, tzinfo=UTC)
ROW = {
    "Transaction_Type": "Trade", "TRANSACTION_DATE": "2026-11-08T00:00:00",
    "TRANSACTION_DESCRIPTION": "Oklahoma City received Nique Clifford.",
    "TEAM_ID": 1610612760.0, "TEAM_SLUG": "thunder",
    "PLAYER_ID": 1642363.0, "PLAYER_SLUG": "nique-clifford",
    "Additional_Sort": 1610612758.0, "GroupSort": "Trade example",
}


def feed(rows):
    return json.dumps({"NBA_Player_Movement": {
        "columns": [{"Name": key, "DataType": "String"} for key in ROW],
        "rows": rows,
    }}).encode()


@pytest.fixture
def transaction_job(monkeypatch):
    state = {"source": feed([dict(ROW)]), "when": OBSERVED, "conflict": False}
    objects = {}

    def get_object(**kwargs):
        key = kwargs["Key"]
        if key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = objects[key]
        return {"Body": BytesIO(raw), "ETag": sha256(raw).hexdigest()}

    def put_object(**kwargs):
        assert kwargs["Bucket"] == "nba-gestalt-data"
        key = kwargs["Key"]
        if "IfMatch" in kwargs:
            assert kwargs["IfMatch"] == sha256(objects[key]).hexdigest()
        else:
            assert kwargs["IfNoneMatch"] == "*" and key not in objects
        if state["conflict"]:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        objects[key] = kwargs["Body"]

    monkeypatch.setattr(nba_transactions.boto3, "client", lambda _: SimpleNamespace(
        get_object=get_object, put_object=put_object,
    ))
    monkeypatch.setattr(NBATransactions, "fetch", lambda _: state["source"])
    monkeypatch.setattr(capture, "datetime", SimpleNamespace(now=lambda tz: state["when"].astimezone(tz)))
    return defs.resolve_job_def("capture_nba_transactions"), state, objects


def test_date_runs_append_new_rows_and_preserve_first_observation(transaction_job):
    job, state, objects = transaction_job
    other_day = {**ROW, "TRANSACTION_DATE": "2026-11-07T00:00:00"}
    state["source"] = feed([ROW, other_day])
    first = job.execute_in_process(partition_key=DAY.isoformat())
    assert first.success
    key = "lake/raw/nba/transactions/season=2026-27/game_date_pacific=2026-11-08/latest.json"
    assert list(objects) == [key]
    saved = json.loads(objects[key])
    assert saved["transactions"] == [{**ROW, "first_observed_at": OBSERVED.isoformat()}]
    assert saved["last_polled_at"] == OBSERVED.isoformat()

    state["when"] += timedelta(hours=1)
    new_row = {**ROW, "PLAYER_ID": 99.0, "PLAYER_SLUG": "new-player"}
    # Preserve observed history even when an earlier row disappears from the live feed.
    state["source"] = feed([new_row, new_row, other_day])
    second = job.execute_in_process(partition_key=DAY.isoformat())
    saved = json.loads(objects[key])
    assert saved["transactions"] == [
        {**ROW, "first_observed_at": OBSERVED.isoformat()},
        {**new_row, "first_observed_at": state["when"].isoformat()},
    ]
    metadata = second.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["transaction_count"].value == 2
    assert metadata["new_transaction_count"].value == 1

    again = job.execute_in_process(partition_key=DAY.isoformat())
    metadata = again.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["new_transaction_count"].value == 0
    assert json.loads(objects[key])["transactions"] == saved["transactions"]


def test_empty_date_is_valid_but_invalid_source_does_not_replace_history(transaction_job):
    job, state, objects = transaction_job
    assert job.execute_in_process(partition_key="2026-11-09").success
    [raw] = objects.values()
    assert json.loads(raw)["transactions"] == []
    original = dict(objects)
    state["source"] = b'{"NBA_Player_Movement":{"rows":[],"columns":[]}}'
    assert not job.execute_in_process(partition_key="2026-11-09", raise_on_error=False).success
    assert objects == original


def test_intervening_s3_update_fails_without_overwriting_history(transaction_job):
    job, state, objects = transaction_job
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    original = dict(objects)
    state["conflict"] = True
    assert not job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False).success
    assert objects == original


def test_source_date_is_not_shifted_to_previous_pacific_day():
    assert transactions_for_date(feed([ROW]), DAY) == [ROW]
    assert transactions_for_date(feed([ROW]), DAY - timedelta(days=1)) == []


def test_hourly_schedule_selects_pacific_date_and_skips_missed_ticks(monkeypatch):
    schedule = defs.get_schedule_def("transactions_hourly")
    assert schedule.cron_schedule == ["0 8-20 * * *", "59 23 * * *"]
    assert schedule.execution_timezone == "America/Los_Angeles"
    clock = SimpleNamespace(now=lambda _: None)
    monkeypatch.setattr(schedules, "datetime", clock)
    with DagsterInstance.local_temp() as instance:
        for hour, minute in [(8, 0), (20, 0), (23, 59)]:
            tick = datetime(2026, 11, 10, hour, minute, tzinfo=schedules.PACIFIC)
            clock.now = lambda _: tick.astimezone(UTC)
            with build_schedule_context(
                instance=instance, scheduled_execution_time=tick,
                repository_def=defs.get_repository_def(),
            ) as context:
                result = schedule.evaluate_tick(context)
                [request] = result.run_requests
                assert request.partition_key == "2026-11-10"
                assert request.run_config == {}

        clock.now = lambda _: tick.astimezone(UTC) + timedelta(hours=2)
        with build_schedule_context(
            instance=instance, scheduled_execution_time=tick,
            repository_def=defs.get_repository_def(),
        ) as context:
            assert not schedule.evaluate_tick(context).run_requests


def test_current_poll_recovers_yesterday_once_and_triggers_both_roster_dates(transaction_job):
    from dagster import DagsterEventType, SensorResult, build_run_status_sensor_context
    from gestalt_pipelines.rosters.sensors import roster_transactions_ready

    job, state, objects = transaction_job
    yesterday, today = "2026-11-09", "2026-11-10"
    old = {**ROW, "TRANSACTION_DATE": yesterday+"T00:00:00", "Transaction_Type": "Waive"}
    late = {
        **old, "Transaction_Type": "Signing", "PLAYER_ID": 1643721.0,
        "TRANSACTION_DESCRIPTION": "Los Angeles Lakers signed Carson Cooper.",
    }
    state["source"] = feed([old])
    assert job.execute_in_process(partition_key=yesterday).success
    key = f"lake/raw/nba/transactions/season=2026-27/game_date_pacific={yesterday}/latest.json"
    first_stamp = json.loads(objects[key])["transactions"][0]["first_observed_at"]

    state["when"] += timedelta(minutes=5)
    # Include the late row twice and an older row: only yesterday/today are checked.
    state["source"] = feed([old, late, late, ROW])
    with DagsterInstance.ephemeral() as instance:
        recovered = job.execute_in_process(instance=instance, partition_key=today)
        assert recovered.success
        saved = json.loads(objects[key])
        assert saved["transactions"] == [
            {**old, "first_observed_at": first_stamp},
            {**late, "first_observed_at": state["when"].isoformat()},
        ]
        assert set(objects) == {
            key,
            f"lake/raw/nba/transactions/season=2026-27/game_date_pacific={today}/latest.json",
        }
        materials = {
            event.event_specific_data.materialization.partition:
            event.event_specific_data.materialization.metadata
            for event in recovered.get_asset_materialization_events()
        }
        assert materials[yesterday]["new_transaction_count"].value == 1
        assert materials[today]["new_transaction_count"].value == 0

        success = next(e for e in recovered.all_events if e.event_type == DagsterEventType.RUN_SUCCESS)
        with build_run_status_sensor_context(
            sensor_name="roster_transactions_ready", dagster_event=success,
            dagster_instance=instance, dagster_run=instance.get_run_by_id(recovered.run_id),
        ) as context:
            result = roster_transactions_ready(context)
            repeated = roster_transactions_ready(context)
        assert isinstance(result, SensorResult)
        assert [r.partition_key for r in result.run_requests] == [yesterday, today]
        assert [r.run_key for r in result.run_requests] == [r.run_key for r in repeated.run_requests]
        assert len({r.run_key for r in result.run_requests}) == 2

    yesterday_raw = objects[key]
    state["when"] += timedelta(minutes=5)
    # Existing history remains even if the NBA removes the old row from its feed.
    state["source"] = feed([late, late, ROW])
    again = job.execute_in_process(partition_key=today)
    assert again.success
    assert objects[key] == yesterday_raw
    [event] = again.get_asset_materialization_events()
    assert event.event_specific_data.materialization.metadata["new_transaction_count"].value == 0


def test_current_poll_stops_at_configured_season_start(transaction_job):
    job, state, objects = transaction_job
    state["when"] = datetime(2026, 10, 20, 16, tzinfo=UTC)
    state["source"] = feed([{**ROW, "TRANSACTION_DATE": "2026-10-19T00:00:00"}])
    assert job.execute_in_process(partition_key="2026-10-20").success
    assert list(objects) == [
        "lake/raw/nba/transactions/season=2026-27/game_date_pacific=2026-10-20/latest.json",
    ]
