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
    monkeypatch.setattr(capture, "datetime", SimpleNamespace(now=lambda _: state["when"]))
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
    assert schedule.cron_schedule == "0 8-20 * * *"
    assert schedule.execution_timezone == "America/Los_Angeles"
    clock = SimpleNamespace(now=lambda _: None)
    monkeypatch.setattr(schedules, "datetime", clock)
    with DagsterInstance.local_temp() as instance:
        for hour in [8, 20]:
            tick = datetime(2026, 11, 10, hour, tzinfo=schedules.PACIFIC)
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
