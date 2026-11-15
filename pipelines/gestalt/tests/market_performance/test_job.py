import json
from copy import deepcopy
from hashlib import sha256

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance
from gestalt_pipelines.market_performance.utils import json_bytes
from .conftest import DAY


def test_manual_job_saves_full_date_and_reuses_unchanged_output(setup):
    job, archive, s3, data = setup
    sources = dict(s3.objects)
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance, partition_key=DAY).success
        before = dict(s3.objects)
        assert job.execute_in_process(instance=instance, partition_key=DAY).success
    assert before == s3.objects
    pointer, _ = archive.selection(DAY)
    record = archive.load(pointer)
    row = record["games"][0]
    assert record["coverage"]["scheduled_game_count"] == record["coverage"]["shared_comparison_game_count"] == 1
    assert row["comparisons"]["model"]["kind"] == "historical_replay"
    assert row["comparisons"]["polymarket"]["home_win_probability"] == .625
    assert row["comparisons"]["bookmaker_consensus"]["predicted_home_margin"] == 5
    assert record["summary"]["model"]["margin_rmse"] == 2
    assert row["inputs"]["prediction"] == data["refs"]["prediction"]
    assert all(s3.objects[key] == raw for key, raw in sources.items())


def test_missing_quotes_preserve_game_and_empty_denominators(setup):
    job, archive, s3, data = setup
    s3.objects.clear()
    assert job.execute_in_process(partition_key=DAY).success
    pointer, _ = archive.selection(DAY)
    record = archive.load(pointer)
    assert len(record["games"]) == 1
    assert record["coverage"]["shared_comparison_game_count"] == 0
    assert record["coverage"]["missing_market_fields"]
    assert record["summary"]["model"]["brier"] is None
    assert record["games"][0]["comparisons"]["model"]["brier_loss"] is not None


@pytest.mark.parametrize("failure", ["prediction", "counter", "non_final", "stale_counter"])
def test_required_inputs_fail_and_real_parent_recovery_is_scoped(setup, failure):
    from dagster._core.execution.api import execute_run
    from dagster._core.definitions.job_base import InMemoryJob
    job, archive, s3, data = setup
    if failure in {"prediction", "counter"}: data["missing"] = failure
    elif failure == "non_final": data["box"]["gameStatus"] = 2
    else: data["counter"]["inputs"]["prediction"] = {"uri": "different", "sha256": "e"*64}
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(instance=instance, partition_key=DAY, raise_on_error=False)
        assert not failed.success and archive.selection(DAY)[0] is None
        errors = [json.loads(raw) for key, raw in s3.objects.items() if key.endswith(".failure.json")]
        assert len(errors) == 1 and errors[0]["pipeline"] == "market_performance"
        data["missing"] = None
        data["box"]["gameStatus"] = 3
        data["counter"]["inputs"]["prediction"] = data["refs"]["prediction"]
        assert job.execute_in_process(instance=instance, partition_key=DAY).success
        assert not any(key.endswith(".recovery.json") for key in s3.objects)
        child = instance.create_run_for_job(job, parent_run_id=failed.run_id, root_run_id=failed.run_id, tags={"dagster/partition": DAY})
        assert execute_run(InMemoryJob(job), child, instance).success
    recoveries = [json.loads(raw) for key, raw in s3.objects.items() if key.endswith(".recovery.json")]
    assert len(recoveries) == 1 and recoveries[0]["original_failure"]["sha256"]


def test_new_accepted_inputs_version_report_without_touching_originals(setup):
    job, archive, s3, data = setup
    assert job.execute_in_process(partition_key=DAY).success
    first, _ = archive.selection(DAY)
    original = dict(s3.objects)
    data["refs"]["counter"] = {"uri": "s3://nba-gestalt-data/test/corrected-counter.json", "sha256": "e"*64}
    data["counter"]["counterfactual_home_margin"] = 7.25
    assert job.execute_in_process(partition_key=DAY).success
    second, _ = archive.selection(DAY)
    assert first["sha256"] != second["sha256"]
    assert all(s3.objects[key] == raw for key, raw in original.items() if not key.endswith("latest.json"))


def test_pointer_race_fails_without_publishing_partial_selection(setup):
    job, archive, s3, _ = setup
    s3.race = True
    result = job.execute_in_process(partition_key=DAY, raise_on_error=False)
    assert not result.success and archive.selection(DAY)[0] is None


def test_saved_checksum_failure_is_not_hidden(setup):
    job, archive, s3, _ = setup
    assert job.execute_in_process(partition_key=DAY).success
    pointer, _ = archive.selection(DAY)
    key = pointer["uri"].removeprefix("s3://nba-gestalt-data/")
    s3.objects[key] += b" "
    with pytest.raises(ValueError, match="checksum"):
        archive.load(pointer)


def test_empty_date_publishes_zero_game_record(setup):
    job, archive, s3, data = setup
    data["games"].clear()
    assert job.execute_in_process(partition_key=DAY).success
    pointer, _ = archive.selection(DAY)
    record = archive.load(pointer)
    assert record["games"] == [] and record["daily_totals"]["game_count"] == 0


def test_current_date_is_not_treated_as_completed(setup, monkeypatch):
    from datetime import UTC, datetime
    from gestalt_pipelines.market_performance.assets import evaluate
    class Today(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 21, 18, tzinfo=UTC)
    monkeypatch.setattr(evaluate, "datetime", Today)
    job, archive, _, _ = setup
    result = job.execute_in_process(partition_key=DAY, raise_on_error=False)
    assert not result.success and archive.selection(DAY)[0] is None


def test_input_selection_changes_fail_before_performance_publication(setup, monkeypatch):
    from gestalt_pipelines.predictions.resources import GamePredictionArchive
    job, archive, _, _ = setup
    original = GamePredictionArchive.selection
    calls = []
    def selection(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not kwargs.get("actual_minutes"):
            calls.append(True)
            if len(calls) > 1:
                return ({**result[0], "sha256": "e"*64}, result[1])
        return result
    monkeypatch.setattr(GamePredictionArchive, "selection", selection)
    assert not job.execute_in_process(partition_key=DAY, raise_on_error=False).success
    assert archive.selection(DAY)[0] is None


@pytest.mark.parametrize("field,value", [("provider", "unapproved source"), ("season", "2025-26"), ("collected_at", "2026-10-20T23:20:00Z")])
def test_raw_odds_identity_is_checked_before_use(setup, field, value):
    job, archive, s3, _ = setup
    key = next(iter(s3.objects))
    record = json.loads(s3.objects[key])
    record[field] = value
    s3.objects[key] = json_bytes(record)
    assert not job.execute_in_process(partition_key=DAY, raise_on_error=False).success
    assert archive.selection(DAY)[0] is None
