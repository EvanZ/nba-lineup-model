"""Manual jobs with accepted source fixtures and real isolated Dagster history."""
import json
from copy import deepcopy
from datetime import timedelta

import pytest
from dagster import DagsterInstance, Definitions

from gestalt_pipelines.minutes.resources import MinutesArchive
from gestalt_pipelines.predictions.assets import game_predictions
from gestalt_pipelines.predictions.jobs import predict_game_margins
from gestalt_pipelines.ratings.resources import RatingsArchive
from gestalt_pipelines.resources import NBAScheduleResource

from .conftest import DAY, NOW


@pytest.fixture
def job(archive, data, monkeypatch, clock):
    store, s3 = archive
    minutes = MinutesArchive(season=store.season)
    ratings = RatingsArchive(season=store.season)
    schedule = NBAScheduleResource(season=store.season)
    allocation = data["minutes"]
    injuries_ref = allocation["inputs"]["injuries"]
    extraction_ref = {"uri": "s3://nba-gestalt-data/test/extraction.json", "sha256": "f"*64}
    injuries = {"games": {data["game"]["game_id"]: {"report_used": True}},
                "inputs": {"extractions": {data["game"]["game_id"]: extraction_ref}}}
    source_documents = [(data["inputs"]["game_minutes"], allocation), (injuries_ref, injuries),
                        (extraction_ref, data["report"])]
    seen = []
    def read(self, ref, prefix):
        seen.append(ref)
        for reference, record in source_documents:
            if ref == reference:
                return deepcopy(record)
        raise AssertionError("Attempt to read an unselected source")
    def state(self, day):
        assert day == data["ratings"]["game_date_pacific"]
        return deepcopy(data["ratings"]), data["inputs"]["player_ratings"], "ratings-etag"
    previous = {**data["game"], "game_id": "0022600001", "game_time_utc": data["game"]["game_time_utc"]-timedelta(days=1),
                "away_team_id": 1610612750}
    monkeypatch.setattr(MinutesArchive, "read_reference", read)
    monkeypatch.setattr(MinutesArchive, "game_selection", lambda self, day, gid: (data["inputs"]["game_minutes"], "minute-etag"))
    monkeypatch.setattr(RatingsArchive, "state", state)
    monkeypatch.setattr(RatingsArchive, "state_selection", lambda self, day: (data["inputs"]["player_ratings"], "ratings-etag"))
    monkeypatch.setattr(RatingsArchive, "load_priors", lambda self: (data["priors"], data["ratings"]["inputs"]["priors"]))
    monkeypatch.setattr(RatingsArchive, "selection", lambda self: (data["ratings"]["inputs"]["priors"], "prior-etag"))
    monkeypatch.setattr(NBAScheduleResource, "games_between", lambda self, *args, **kw: ([previous, data["game"]], data["inputs"]["schedule"]))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda self: (b"{}", data["inputs"]["schedule"]))
    defs = Definitions(assets=[game_predictions], jobs=[predict_game_margins], resources={
        "minutes_archive": minutes, "ratings_archive": ratings,
        "nba_schedule": schedule, "game_prediction_archive": store})
    return defs.resolve_job_def("predict_game_margins"), seen


def test_manual_job_selects_exact_sources_and_rerun_does_not_write(job, archive, data, clock):
    executable, seen = job
    store, s3 = archive
    with DagsterInstance.ephemeral() as instance:
        assert executable.execute_in_process(partition_key=DAY, instance=instance).success
        before = dict(s3.objects)
        clock.value += timedelta(minutes=1)
        assert executable.execute_in_process(partition_key=DAY, instance=instance).success
    assert s3.objects == before
    selected, _ = store.selection(DAY, data["game"]["game_id"])
    result = store.load(selected)
    assert result["ratings_effective_date_pacific"] == DAY
    assert result["ratings_observed_through_game_date_pacific"] == "2026-10-20"
    assert result["injury_report"] == data["report"]
    assert result["components"]["home_is_b2b"] is True
    assert result["components"]["away_is_b2b"] is False
    assert seen.count(data["minutes"]["inputs"]["injuries"]) == 1


def test_missing_source_fails_then_actual_parent_reexecution_records_recovery(job, archive, data, monkeypatch):
    from dagster._core.execution.api import execute_run
    from dagster._core.definitions.job_base import InMemoryJob
    executable, _ = job
    store, s3 = archive
    original_state = RatingsArchive.state
    monkeypatch.setattr(RatingsArchive, "state", lambda *args: (None, None, None))
    with DagsterInstance.ephemeral() as instance:
        failed = executable.execute_in_process(partition_key=DAY, instance=instance, raise_on_error=False)
        assert not failed.success
        failures = [json.loads(v) for k, v in s3.objects.items() if k.endswith(".failure.json")]
        assert len(failures) == 1 and failures[0]["pipeline"] == "predictions"
        assert not any(k.endswith("latest.json") for k in s3.objects)
        monkeypatch.setattr(RatingsArchive, "state", original_state)
        assert executable.execute_in_process(partition_key=DAY, instance=instance).success
        assert not any(k.endswith(".recovery.json") for k in s3.objects)
        child = instance.create_run_for_job(executable, parent_run_id=failed.run_id,
                    root_run_id=failed.run_id, tags={"dagster/partition": DAY})
        assert execute_run(InMemoryJob(executable), child, instance).success
    recovery = [json.loads(v) for k, v in s3.objects.items() if k.endswith(".recovery.json")]
    assert len(recovery) == 1 and recovery[0]["original_failure"]["sha256"]


def test_replacement_warning_is_written_once_for_the_same_player_and_game(job, archive, data):
    executable, _ = job
    store, s3 = archive
    game_id = data["game"]["game_id"]
    removed = data["ratings"]["players"].pop(0)
    data["priors"] = {"players": []}
    with DagsterInstance.ephemeral() as instance:
        first = executable.execute_in_process(partition_key=DAY, instance=instance)
        assert first.success
        initial, _ = store.selection(DAY, game_id)
        repeat = executable.execute_in_process(partition_key=DAY, instance=instance)
        assert repeat.success
        # A new input creates a new prediction, but this player's fallback persists.
        data["inputs"]["player_ratings"] = {**data["inputs"]["player_ratings"], "sha256": "c" * 64}
        refreshed = executable.execute_in_process(partition_key=DAY, instance=instance)
        assert refreshed.success
        updated, _ = store.selection(DAY, game_id)
        assert updated["sha256"] != initial["sha256"]
        for run in (repeat, refreshed):
            metadata = run.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
            assert metadata["warnings"].value == []
    warnings = [(k, json.loads(v)) for k, v in s3.objects.items() if k.endswith(".warning.json")]
    assert len(warnings) == 1
    key, warning = warnings[0]
    assert key.startswith("lake/artifacts/errors/pipeline=predictions/season=2026-27/occurred_date_pacific=")
    assert warning["details"]["player_id"] == removed["player_id"]
    assert warning["status"] == "warning" and warning["dagster_run_id"] == first.run_id


def test_historical_date_uses_that_days_ratings_and_labels_replay(job, archive, data, clock):
    executable, _ = job
    clock.value += timedelta(days=2)
    with DagsterInstance.ephemeral() as instance:
        assert executable.execute_in_process(partition_key=DAY, instance=instance).success
    store, _ = archive
    selected, _ = store.selection(DAY, data["game"]["game_id"])
    output = store.load(selected)
    assert output["purpose"] == "historical_replay"
    assert output["ratings_effective_date_pacific"] == DAY


def test_missing_one_game_leaves_entire_new_slate_unpublished(job, archive, data, monkeypatch):
    executable, _ = job
    second = {**data["game"], "game_id": "0022600003"}
    monkeypatch.setattr(NBAScheduleResource, "games_between", lambda *args, **kw:
                       ([data["game"], second], data["inputs"]["schedule"]))
    monkeypatch.setattr(MinutesArchive, "game_selection", lambda self, day, gid:
                       (data["inputs"]["game_minutes"], "etag") if gid == data["game"]["game_id"] else (None, None))
    with DagsterInstance.ephemeral() as instance:
        result = executable.execute_in_process(partition_key=DAY, instance=instance, raise_on_error=False)
        assert not result.success
    assert not any(k.endswith("latest.json") for k in archive[1].objects)


def test_changed_input_selection_prevents_publication(job, archive, monkeypatch):
    executable, _ = job
    monkeypatch.setattr(RatingsArchive, "state_selection", lambda *args: (None, None))
    with DagsterInstance.ephemeral() as instance:
        assert not executable.execute_in_process(partition_key=DAY, instance=instance, raise_on_error=False).success
    assert not any(k.endswith("latest.json") for k in archive[1].objects)


def test_tomorrow_uses_latest_completed_state_available_today(job, archive, data, clock, monkeypatch):
    from gestalt_pipelines.minutes.utils import projection_fingerprint
    from datetime import date
    executable, _ = job
    target = str(date.fromisoformat(DAY)+timedelta(days=1))
    data["game"]["game_time_utc"] += timedelta(days=1)
    allocation = data["minutes"]
    allocation["game_date_pacific"] = target
    allocation["scheduled_start_utc"] = data["game"]["game_time_utc"].isoformat()
    allocation["input_fingerprint"] = projection_fingerprint(data["game"], allocation["inputs"], implementation=allocation["implementation"])
    calls = []
    def state(self, day):
        calls.append(day)
        return data["ratings"], data["inputs"]["player_ratings"], "etag"
    monkeypatch.setattr(RatingsArchive, "state", state)
    with DagsterInstance.ephemeral() as instance:
        assert executable.execute_in_process(partition_key=target, instance=instance).success
    assert calls == ["2026-10-20"]
    store, _ = archive
    selected, _ = store.selection(target, data["game"]["game_id"])
    assert store.load(selected)["ratings_effective_date_pacific"] == "2026-10-21"


def test_explicit_historical_prediction_correction_preserves_original_and_cutoffs(job,archive,data,clock):
    executable,_ = job
    store,s3 = archive
    with DagsterInstance.ephemeral() as instance:
        assert executable.execute_in_process(partition_key=DAY,instance=instance).success
        original,_ = store.selection(DAY,data["game"]["game_id"])
        original_key = original["uri"].removeprefix("s3://nba-gestalt-data/")
        original_bytes = s3.objects[original_key]
        clock.value += timedelta(days=2)
        data["inputs"]["game_minutes"]["sha256"] = "e"*64
        assert executable.execute_in_process(partition_key=DAY,instance=instance).success
        assert store.selection(DAY,data["game"]["game_id"])[0] == original
        config = {"ops":{"game_predictions":{"config":{"allow_historical_correction":True}}}}
        assert executable.execute_in_process(partition_key=DAY,instance=instance,run_config=config).success
    corrected,_ = store.selection(DAY,data["game"]["game_id"])
    assert corrected["uri"] != original["uri"]
    assert s3.objects[original_key] == original_bytes
    output = store.load(corrected)
    assert output["purpose"] == "historical_replay"
    assert output["ratings_effective_date_pacific"] == DAY
    assert output["inputs"]["player_ratings"] == data["inputs"]["player_ratings"]
    assert output["inputs"]["game_minutes"] == data["inputs"]["game_minutes"]


def test_historical_correction_cannot_override_live_predictions(job,archive,data,clock):
    executable,_ = job
    config = {"ops":{"game_predictions":{"config":{"allow_historical_correction":True}}}}
    with DagsterInstance.ephemeral() as instance:
        assert not executable.execute_in_process(partition_key=DAY,instance=instance,run_config=config,raise_on_error=False).success
    assert not any(k.endswith("latest.json") for k in archive[1].objects)
