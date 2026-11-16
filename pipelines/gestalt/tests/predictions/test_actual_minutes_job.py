"""Isolated manual job, pinned inputs, warnings and actual failed-parent recovery."""
import json
from copy import deepcopy
from datetime import timedelta

import pytest
from dagster import DagsterInstance, Definitions
from gestalt_pipelines.box_scores.resources import NBABoxScoreArchive
from gestalt_pipelines.predictions.assets import game_predictions_actual_minutes
from gestalt_pipelines.predictions.jobs import predict_game_margins_actual_minutes
from gestalt_pipelines.ratings.resources import RatingsArchive
from gestalt_pipelines.resources import NBAScheduleResource
from .conftest import DAY, NOW
from .test_actual_minutes import box


@pytest.fixture
def counter_job(archive, record, data, box, clock, monkeypatch):
    store, s3 = archive
    store.save(record, expected_etag=None)
    pointer, _ = store.selection(DAY, record["game_id"])
    baseline_files = dict(s3.objects)
    clock.value = NOW+timedelta(days=1)
    monkeypatch.setattr("gestalt_pipelines.predictions.assets.actual_minutes.datetime", clock)
    refs = {"box": {"uri": "s3://nba-gestalt-data/test/final-box.json", "sha256": "d"*64},
            "schedule": data["inputs"]["schedule"]}
    reads = []
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *args, **kwargs: ([data["game"]], refs["schedule"]))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda *args: (b"{}", refs["schedule"]))
    monkeypatch.setattr(NBABoxScoreArchive, "read_saved", lambda *args: (json.dumps({"game":box}).encode(), refs["box"]))
    def state(self, reference, *, effective_date, observed_through):
        reads.append((reference, effective_date, observed_through))
        assert reference == record["inputs"]["player_ratings"]
        assert effective_date == record["ratings_effective_date_pacific"]
        assert observed_through == record["ratings_observed_through_game_date_pacific"]
        return deepcopy(data["ratings"])
    monkeypatch.setattr(RatingsArchive, "read_state", state)
    monkeypatch.setattr(RatingsArchive, "read_priors", lambda self, reference: {"players": []})
    monkeypatch.setattr(RatingsArchive, "state", lambda *args: pytest.fail("Must not read current daily selection"))
    monkeypatch.setattr(RatingsArchive, "load_priors", lambda *args: pytest.fail("Must not read current prior selection"))
    defs = Definitions(assets=[game_predictions_actual_minutes], jobs=[predict_game_margins_actual_minutes], resources={
        "nba_schedule": NBAScheduleResource(season="2026-27"), "nba_box_score_archive": NBABoxScoreArchive(),
        "ratings_archive": RatingsArchive(season="2026-27"), "game_prediction_archive": store})
    return defs.resolve_job_def("predict_game_margins_actual_minutes"), baseline_files, refs, reads


def test_manual_counterfactual_and_identical_rerun_preserve_baseline(counter_job, archive):
    job, baseline, _, reads = counter_job
    store, s3 = archive
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(partition_key=DAY, instance=instance).success
        before = dict(s3.objects)
        assert job.execute_in_process(partition_key=DAY, instance=instance).success
    assert s3.objects == before
    assert all(s3.objects[k] == value for k, value in baseline.items())
    assert reads == []


def test_extra_actual_player_uses_exact_pinned_state_and_warns(counter_job, box, archive):
    job, _, _, reads = counter_job
    box["homeTeam"]["players"][0]["personId"] = 99999
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(partition_key=DAY, instance=instance).success
    assert len(reads) == 1
    warnings = [(k, json.loads(v)) for k, v in archive[1].objects.items() if k.endswith(".warning.json")]
    assert len(warnings) == 1
    key, event = warnings[0]
    assert key.startswith("lake/artifacts/errors/pipeline=predictions/season=2026-27/")
    assert event["stage"] == "game_predictions_actual_minutes" and event["details"]["player_id"] == 99999


def test_actual_failure_then_real_child_reexecution_records_stage_scoped_recovery(counter_job, box, archive):
    from dagster._core.execution.api import execute_run
    from dagster._core.definitions.job_base import InMemoryJob
    job, baseline, _, _ = counter_job
    store, s3 = archive
    box["gameStatus"] = 2
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(partition_key=DAY, instance=instance, raise_on_error=False)
        assert not failed.success
        failures = [json.loads(v) for k, v in s3.objects.items() if k.endswith(".failure.json")]
        assert len(failures) == 1 and failures[0]["stage"] == "game_predictions_actual_minutes"
        assert not any("/game_predictions_actual_minutes/" in k and k.endswith("latest.json") for k in s3.objects)
        box["gameStatus"] = 3
        assert job.execute_in_process(partition_key=DAY, instance=instance).success
        assert not any(k.endswith(".recovery.json") for k in s3.objects)
        child = instance.create_run_for_job(job, parent_run_id=failed.run_id, root_run_id=failed.run_id,
                                            tags={"dagster/partition": DAY})
        assert execute_run(InMemoryJob(job), child, instance).success
    recovered = [json.loads(v) for k, v in s3.objects.items() if k.endswith(".recovery.json")]
    assert len(recovered) == 1 and recovered[0]["stage"] == "game_predictions_actual_minutes"
    assert recovered[0]["original_failure"]["sha256"]
    assert all(s3.objects[k] == value for k, value in baseline.items())


def test_missing_game_input_leaves_full_slate_unpublished(counter_job, archive, data, monkeypatch):
    job, _, refs, _ = counter_job
    other = {**data["game"], "game_id": "0022609999"}
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *args, **kw: ([data["game"], other], refs["schedule"]))
    with DagsterInstance.ephemeral() as instance:
        assert not job.execute_in_process(partition_key=DAY, instance=instance, raise_on_error=False).success
    assert not any("/game_predictions_actual_minutes/" in k and k.endswith("latest.json") for k in archive[1].objects)


def test_changed_box_selection_cannot_advance_counterfactual(counter_job, archive, box, monkeypatch):
    job, _, refs, _ = counter_job
    calls = []
    def read(*args):
        calls.append(True)
        ref = dict(refs["box"])
        if len(calls) > 1: ref["sha256"] = "e"*64
        return json.dumps({"game": box}).encode(), ref
    monkeypatch.setattr(NBABoxScoreArchive, "read_saved", read)
    with DagsterInstance.ephemeral() as instance:
        assert not job.execute_in_process(partition_key=DAY, instance=instance, raise_on_error=False).success
    assert not any("/game_predictions_actual_minutes/" in k and k.endswith("latest.json") for k in archive[1].objects)


def test_no_game_date_succeeds_without_writing_counterfactuals(counter_job, archive, monkeypatch):
    job, baseline, refs, _ = counter_job
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *args, **kw: ([], refs["schedule"]))
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(partition_key=DAY, instance=instance).success
    assert archive[1].objects == baseline


@pytest.mark.parametrize("change",["unrelated_stats","score","minutes"])
def test_later_box_refresh_uses_pregame_ratings_only(change,counter_job,archive,box,clock,record,monkeypatch):
    job,baseline,refs,reads = counter_job
    store,s3 = archive
    for name in ("state_selection","selection"):
        monkeypatch.setattr(RatingsArchive,name,lambda *a,**k:pytest.fail("Must not select newer ratings"))
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance,partition_key=DAY).success
        pointer,_ = store.selection(DAY,record["game_id"],actual_minutes=True)
        before = store.load(pointer,actual_minutes=True)
        clock.value += timedelta(days=30)
        refs["box"]["sha256"] = "e"*64
        if change == "unrelated_stats":box["homeTeam"]["players"][0]["statistics"]["reboundsTotal"] = 99
        elif change == "score":box["homeTeam"]["score"] += 2
        else:box["homeTeam"]["players"][0]["statistics"]["minutes"] = "PT47M00.00S"
        assert job.execute_in_process(instance=instance,partition_key=DAY).success
    pointer,_ = store.selection(DAY,record["game_id"],actual_minutes=True)
    after = store.load(pointer,actual_minutes=True)
    assert all(s3.objects[k] == value for k,value in baseline.items()) and reads == []
    assert after["inputs"]["prediction"] == before["inputs"]["prediction"]
    assert after["inputs"]["box_score"] == refs["box"]
    for side in ("home","away"):
        ratings = {p["player_id"]:p["nail"] for p in record[side]["players"]}
        assert all(p["nail"] == ratings[p["player_id"]] for p in after[side]["players"])
    if change == "minutes":assert after["counterfactual_home_margin"] != before["counterfactual_home_margin"]
    else:assert after["counterfactual_home_margin"] == before["counterfactual_home_margin"]
    if change == "score":assert after["actual_home_score"] == before["actual_home_score"]+2
    else:assert after["actual_home_score"] == before["actual_home_score"]
