"""Real completed-date captures drive only postgame jobs, not learned ratings."""
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from dagster import DagsterEventType, DagsterInstance, RunRequest, SkipReason, build_run_status_sensor_context

from box_scores.test_box_score_capture import box_score_job, DAY, GAMES, REFRESH_CONFIG, _box
from gestalt_pipelines.box_scores.resources import archive
from gestalt_pipelines.definitions import defs
from gestalt_pipelines.predictions.sensors import predict_game_margins_actual_minutes_sensor as sensor
from gestalt_pipelines.predictions.sensors import actual_minutes as automation


def request(instance, result):
    event = next(e for e in result.all_events if e.event_type in (DagsterEventType.RUN_SUCCESS,DagsterEventType.RUN_FAILURE))
    with build_run_status_sensor_context(sensor_name=sensor.name, dagster_event=event,
            dagster_instance=instance, dagster_run=instance.get_run_by_id(result.run_id)) as context:
        return sensor(context)


def test_complete_refresh_triggers_same_date_and_allows_operator_retries(box_score_job):
    job,_,final_ids,_ = box_score_job
    final_ids.update(g["game_id"] for g in GAMES)
    with DagsterInstance.ephemeral() as instance:
        first = job.execute_in_process(instance=instance,partition_key=str(DAY),run_config=REFRESH_CONFIG)
        a = request(instance,first)
        again = request(instance,first)
        second = job.execute_in_process(instance=instance,partition_key=str(DAY),run_config=REFRESH_CONFIG)
        b = request(instance,second)
    assert isinstance(a,RunRequest) and a.partition_key == str(DAY) and a.run_config == {}
    assert a.run_key == again.run_key and b.run_key != a.run_key
    assert a.tags["box_scores/refresh_run_id"] == first.run_id and a.tags["dagster/priority"] == "-1"
    assert "predict_game_margins_actual_minutes_schedule" not in {s.name for s in defs.schedules}
    assert defs.get_sensor_def(sensor.name).name == "predict_game_margins_actual_minutes_sensor"


@pytest.mark.parametrize("change",["metadata","unrelated_stats","score","minutes"])
def test_each_changed_box_version_requests_postgame_refresh(change,monkeypatch,box_score_job):
    job,objects,final_ids,_ = box_score_job
    final_ids.update(g["game_id"] for g in GAMES)
    original_get = archive.requests.get
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance,partition_key=str(DAY)).success
        originals = dict(objects)
        document = json.loads(_box(GAMES[0],3))
        if change == "metadata":document["meta"] = {"time":"new NBA publication"}
        elif change == "unrelated_stats":document["game"]["homeTeam"]["statistics"] = {"reboundsTotal":99}
        elif change == "score":document["game"]["homeTeam"]["score"] += 2
        else:document["game"]["homeTeam"]["players"][0]["statistics"] = {"minutes":"PT40M00.00S"}
        def get(url,**kwargs):
            if GAMES[0]["game_id"] in url:
                return nullcontext(SimpleNamespace(status_code=200,raise_for_status=lambda:None,content=json.dumps(document).encode()))
            return original_get(url,**kwargs)
        monkeypatch.setattr(archive.requests,"get",get)
        result = job.execute_in_process(instance=instance,partition_key=str(DAY),run_config=REFRESH_CONFIG)
        r = request(instance,result)
    assert result.success and isinstance(r,RunRequest) and r.partition_key == str(DAY)
    assert all(objects[k] == v for k,v in originals.items())


@pytest.fixture(autouse=True)
def fixed_postgame_clock(monkeypatch):
    monkeypatch.setattr(automation,"datetime",SimpleNamespace(now=lambda _:datetime(2026,11,5,12,tzinfo=UTC)))


def test_failed_refresh_and_ordinary_capture_do_not_trigger(box_score_job):
    job,_,final_ids,_ = box_score_job
    with DagsterInstance.ephemeral() as instance:
        failed = job.execute_in_process(instance=instance,partition_key=str(DAY),run_config=REFRESH_CONFIG,raise_on_error=False)
        assert not failed.success and isinstance(request(instance,failed),SkipReason)
        final_ids.update(g["game_id"] for g in GAMES)
        ordinary = job.execute_in_process(instance=instance,partition_key=str(DAY))
        assert ordinary.success and isinstance(request(instance,ordinary),SkipReason)


@pytest.mark.parametrize("clock_day",[3,4])
def test_today_and_future_slates_do_not_trigger_postgame(clock_day,monkeypatch,box_score_job):
    job,_,final_ids,_ = box_score_job
    final_ids.update(g["game_id"] for g in GAMES)
    monkeypatch.setattr(automation,"datetime",SimpleNamespace(now=lambda _:datetime(2026,11,clock_day,12,tzinfo=UTC)))
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance,partition_key=str(DAY),run_config=REFRESH_CONFIG)
        assert isinstance(request(instance,result),SkipReason)


def test_pending_live_capture_has_no_final_materialization(monkeypatch,box_score_job):
    from gestalt_pipelines.box_scores.assets import capture
    job,_,_,_ = box_score_job
    monkeypatch.setattr(capture,"datetime",SimpleNamespace(now=lambda _:datetime(2026,11,4,20,tzinfo=UTC)))
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance,partition_key=str(DAY))
        assert result.success and not result.get_asset_materialization_events()
        assert isinstance(request(instance,result),SkipReason)
