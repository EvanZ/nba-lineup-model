from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json

import pytest
from dagster import DagsterInstance, build_sensor_context
from dagster._core.definitions.job_base import InMemoryJob
from dagster._core.execution.api import execute_run

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.minutes.assets import project
from gestalt_pipelines.minutes.resources import archive
from gestalt_pipelines.minutes.sensors import projections
from gestalt_pipelines.minutes.utils import (
    MinutesError, build_game_projection, json_bytes, update_state, validate_game_projection,
)
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.rosters.resources import NBARosterArchive

from .helpers import FIRST, HOME, SEASON, priors, slate
from .test_asset import minutes_job

DAY = FIRST+timedelta(days=1)
NOW = datetime(DAY.year,DAY.month,DAY.day,18,tzinfo=UTC)


def example():
    prior = priors()
    baseline = update_state(priors=prior,parent=None,**slate())
    data = slate(DAY,"0022600002")
    game = data["games"][0]
    game["game_time_utc"] = NOW+timedelta(hours=2)
    data["injuries"]["games"][game["game_id"]]["scheduled_start_utc"] = game["game_time_utc"].isoformat()
    return dict(
        game=game,day=str(DAY),season=SEASON,baseline=baseline,
        rosters=data["rosters"],injuries=data["injuries"],priors=prior,inputs=data["inputs"],
    )


def test_projection_weights_active15_and_240_with_no_individual_cap():
    data = example()
    home = data["rosters"]["teams"][str(HOME)]
    for pid in range(300,320):
        home.append(pid)
        data["baseline"]["players"][str(pid)] = {
            "estimate":20,"name":None,"initialization":{"branch":"test"},
        }
    record = build_game_projection(**data)
    team = record["teams"][str(HOME)]
    assert sum(p["selected_rotation"] for p in team["players"]) == 15
    assert team["total_projected_minutes"] == pytest.approx(240)
    by_id = {p["player_id"]:p for p in team["players"]}
    assert by_id[17]["projected_minutes"] == 0
    assert by_id[18]["allocation_multiplier"] == .5
    assert by_id[18]["injury_adjusted_minutes"] == data["baseline"]["players"]["18"]["estimate"]*.5
    assert any(p["projected_minutes"] == 0 and p["conditional_minutes"] > 0 for p in team["players"])
    # Preserve the existing proportional formula even for extreme rotations.
    data = example()
    for pid in data["rosters"]["teams"][str(HOME)]:
        data["baseline"]["players"][str(pid)]["estimate"] = 48 if pid == 1 else 0
    record = build_game_projection(**data)
    assert next(p for p in record["teams"][str(HOME)]["players"] if p["player_id"]==1)["projected_minutes"] == 240


def test_pending_report_is_labelled_unknown_without_seasonal_probability():
    data = example()
    report = data["injuries"]["games"][data["game"]["game_id"]]["teams"][str(HOME)]
    report["players"] = {}
    report["submission_status"] = "not_yet_submitted"
    record = build_game_projection(**data)
    for p in record["teams"][str(HOME)]["players"]:
        assert p["injury_status"] == "Report pending" and p["play_probability"] is None
        assert p["allocation_multiplier"] == 1


def test_new_roster_player_uses_approved_initializer_and_output_validation():
    data = example()
    data["rosters"]["teams"][str(HOME)].append(999999)
    record = build_game_projection(**data)
    p = next(p for p in record["teams"][str(HOME)]["players"] if p["player_id"]==999999)
    assert p["initialized_for_projection"] and p["conditional_minutes"] == data["priors"]["cold_start_minutes"]
    record["teams"][str(HOME)]["players"][0]["projected_minutes"] += 1
    with pytest.raises(MinutesError,match="allocation"):
        validate_game_projection(record,SEASON)


def test_reconciled_injury_rows_outside_roster_do_not_block_allocation():
    data = example()
    game = data["injuries"]["games"][data["game"]["game_id"]]
    game["teams"][str(HOME)]["players"]["999999"] = {"status":"Out"}
    record=build_game_projection(**data)
    assert all(p["player_id"]!=999999 for t in record["teams"].values() for p in t["players"])
    assert record["teams"][str(HOME)]["total_projected_minutes"] == pytest.approx(240)
    data["injuries"]["games"] = {}
    with pytest.raises(MinutesError,match="Missing matching"):
        build_game_projection(**data)


class Clock:
    value = NOW

    @classmethod
    def now(cls,*args):
        return cls.value

    fromisoformat = staticmethod(datetime.fromisoformat)


@pytest.fixture
def projection_job(minutes_job,monkeypatch):
    daily,a,objects,writes,_ = minutes_job
    assert daily.execute_in_process(partition_key=str(FIRST)).success
    data = example()
    game = data["game"]
    monkeypatch.setattr(Clock,"value",NOW)
    for module in (project,archive,projections):
        monkeypatch.setattr(module,"datetime",Clock)
    ref = data["inputs"]["schedule"]
    monkeypatch.setattr(NBAScheduleResource,"games_for_date",lambda *a,**k:([game],ref))
    monkeypatch.setattr(NBAScheduleResource,"games_between",lambda *a,**k:([game],ref))
    monkeypatch.setattr(NBAScheduleResource,"load",lambda *a,**k:(b"schedule",ref))

    def save_injuries(document):
        raw = json_bytes(document)
        digest = sha256(raw).hexdigest()
        prefix = f"lake/artifacts/injury_status_by_game_date/season={SEASON}/game_date_pacific={DAY}/"
        key = prefix+f"selection_sha256={digest}/statuses.json"
        objects[key] = raw
        objects[prefix+"latest.json"] = json_bytes({"uri":"s3://nba-gestalt-data/"+key,"sha256":digest})
        return prefix+"latest.json"
    pointer = save_injuries(data["injuries"])
    return defs.resolve_job_def("project_game_minutes"),a,objects,writes,data,pointer,save_injuries


def test_real_job_publishes_reuses_and_preserves_started_games(projection_job):
    job,a,objects,writes,data,_,_ = projection_job
    assert job.execute_in_process(partition_key=str(DAY)).success
    gid = data["game"]["game_id"]
    selected,_ = a.game_selection(str(DAY),gid)
    key = a.game_prefix+f"game_date_pacific={DAY}/game_id={gid}/latest.json"
    before = objects[key]
    count = len(writes)
    assert job.execute_in_process(partition_key=str(DAY)).success
    assert objects[key] == before and len(writes) == count
    record = a.read_reference(selected,a.game_prefix)
    assert record["baseline_through_date"] == str(FIRST)
    assert record["roster_date_pacific"] == str(FIRST)
    assert len(record["teams"]) == 2
    Clock.value = data["game"]["game_time_utc"]+timedelta(seconds=1)
    assert job.execute_in_process(partition_key=str(DAY)).success
    assert objects[key] == before and len(writes) == count


def test_projection_failure_and_real_parent_reexecution_write_linked_recovery(projection_job):
    job,a,objects,_,data,pointer,_ = projection_job
    gid = data["game"]["game_id"]
    with DagsterInstance.ephemeral() as instance:
        assert job.execute_in_process(instance=instance,partition_key=str(DAY)).success
        accepted,_ = a.game_selection(str(DAY),gid)
        source = objects.pop(pointer)
        failed = job.execute_in_process(instance=instance,partition_key=str(DAY),raise_on_error=False)
        assert not failed.success
        assert a.game_selection(str(DAY),gid)[0] == accepted
        failure_key = next(k for k in objects if k.endswith(".failure.json"))
        failure = json.loads(objects[failure_key])
        assert failure["stage"] == "game_minutes"
        objects[pointer] = source
        parent = instance.get_run_by_id(failed.run_id)
        child = instance.create_run_for_job(
            job,run_config=parent.run_config,tags=parent.tags,
            parent_run_id=parent.run_id,root_run_id=parent.run_id,
        )
        assert execute_run(InMemoryJob(job),child,instance).success
        recovery_key = next(k for k in objects if k.endswith(".recovery.json"))
        recovery = json.loads(objects[recovery_key])
        assert recovery["original_failure"]["uri"].endswith(failure_key)
        assert recovery["stage"] == "game_minutes"


def test_projection_cannot_advance_pointer_if_tipoff_passes_during_publication(projection_job,monkeypatch):
    job,a,objects,_,data,_,save = projection_job
    assert job.execute_in_process(partition_key=str(DAY)).success
    gid = data["game"]["game_id"]
    before,etag = a.game_selection(str(DAY),gid)
    injuries = deepcopy(data["injuries"])
    injuries["games"][gid]["teams"][str(HOME)]["players"]["17"]["status"] = "Questionable"
    save(injuries)
    refs = a.projection_selections(str(DAY),today=DAY,roster_archive=NBARosterArchive())
    refs["schedule"] = data["inputs"]["schedule"]
    record = build_game_projection(
        game=data["game"],day=str(DAY),season=SEASON,inputs=refs,
        **a.projection_data(str(DAY),refs),
    )
    original_put = archive.MinutesArchive._put
    def cross_tipoff(self,key,raw):
        result = original_put(self,key,raw)
        Clock.value = data["game"]["game_time_utc"]+timedelta(seconds=1)
        return result
    monkeypatch.setattr(archive.MinutesArchive,"_put",cross_tipoff)
    assert a.save_game(record,expected_etag=etag,roster_archive=NBARosterArchive())["frozen"]
    assert a.game_selection(str(DAY),gid)[0] == before


def test_sensor_seeds_then_skips_unchanged_then_requests_changed_injuries(projection_job):
    job,a,_,_,data,_,save = projection_job
    sensor = defs.get_sensor_def("game_minutes_ready")
    with DagsterInstance.ephemeral() as instance:
        with build_sensor_context(
            instance=instance,repository_def=defs.get_repository_def(),
            resources={"minutes_archive":a,"nba_roster_archive":NBARosterArchive(),
                       "nba_schedule":NBAScheduleResource(season=SEASON)},
        ) as context:
            first = sensor.evaluate_tick(context)
            assert len(first.run_requests)==1 and first.run_requests[0].partition_key==str(DAY)
            assert job.execute_in_process(partition_key=str(DAY),instance=instance).success
            assert not sensor.evaluate_tick(context).run_requests
            injuries=deepcopy(data["injuries"])
            injuries["games"][data["game"]["game_id"]]["teams"][str(HOME)]["players"]["17"]["status"]="Questionable"
            save(injuries)
            changed = sensor.evaluate_tick(context)
            assert len(changed.run_requests)==1
            assert changed.run_requests[0].run_key != first.run_requests[0].run_key
            assert sensor.evaluate_tick(context).run_requests[0].run_key == changed.run_requests[0].run_key
            Clock.value = data["game"]["game_time_utc"]+timedelta(seconds=1)
            assert not sensor.evaluate_tick(context).run_requests


def test_later_game_updates_do_not_touch_earlier_frozen_selection(projection_job,monkeypatch):
    job,a,objects,_,data,_,save = projection_job
    first=data["game"]
    second={**first,"game_id":"0022600003","game_time_utc":first["game_time_utc"]+timedelta(hours=3)}
    injuries=deepcopy(data["injuries"])
    injuries["games"][second["game_id"]]=deepcopy(injuries["games"][first["game_id"]])
    injuries["games"][second["game_id"]]["scheduled_start_utc"]=second["game_time_utc"].isoformat()
    save(injuries)
    monkeypatch.setattr(NBAScheduleResource,"games_for_date",lambda *a,**k:([first,second],data["inputs"]["schedule"]))
    assert job.execute_in_process(partition_key=str(DAY)).success
    old_first,_=a.game_selection(str(DAY),first["game_id"])
    old_second,_=a.game_selection(str(DAY),second["game_id"])
    Clock.value=first["game_time_utc"]+timedelta(seconds=1)
    for g in injuries["games"].values():
        g["teams"][str(HOME)]["players"]["17"]["status"]="Questionable"
    save(injuries)
    assert job.execute_in_process(partition_key=str(DAY)).success
    assert a.game_selection(str(DAY),first["game_id"])[0]==old_first
    assert a.game_selection(str(DAY),second["game_id"])[0]!=old_second
    assert old_first["uri"].removeprefix("s3://nba-gestalt-data/") in objects


def test_sensor_waits_for_missing_baseline_and_skips_active_date(projection_job):
    from dagster import DagsterRunStatus
    job,a,objects,_,_,_,_=projection_job
    sensor=defs.get_sensor_def("game_minutes_ready")
    baseline_key=a.prefix+f"game_date_pacific={FIRST}/latest.json"
    source=objects.pop(baseline_key)
    with DagsterInstance.ephemeral() as instance:
        with build_sensor_context(
            instance=instance,repository_def=defs.get_repository_def(),
            resources={"minutes_archive":a,"nba_roster_archive":NBARosterArchive(),
                       "nba_schedule":NBAScheduleResource(season=SEASON)},
        ) as context:
            assert not sensor.evaluate_tick(context).run_requests
            objects[baseline_key]=source
            assert sensor.evaluate_tick(context).run_requests
            instance.create_run_for_job(job,status=DagsterRunStatus.STARTED,tags={"dagster/partition":str(DAY)})
            assert not sensor.evaluate_tick(context).run_requests


def test_manual_historical_projection_cannot_rewrite_accepted_forecast(projection_job):
    job,a,objects,_,data,_,_=projection_job
    assert job.execute_in_process(partition_key=str(DAY)).success
    old,_=a.game_selection(str(DAY),data["game"]["game_id"])
    Clock.value=NOW+timedelta(days=1)
    assert job.execute_in_process(partition_key=str(DAY)).success
    assert a.game_selection(str(DAY),data["game"]["game_id"])[0]==old


def test_historical_partition_uses_previous_day_and_preserves_replay_versions(projection_job,monkeypatch):
    job,a,objects,writes,data,_,save = projection_job
    Clock.value=NOW+timedelta(days=5)
    # Historical membership is the accepted target-date roster, not today's snapshot.
    original_ref=a._roster_ref(str(FIRST))
    roster=json.loads(objects[original_ref["uri"].removeprefix("s3://nba-gestalt-data/")])
    roster["game_date_pacific"]=str(DAY)
    NBARosterArchive().publish(roster,expected_etag=None)
    later=deepcopy(roster)
    later["game_date_pacific"]=str(Clock.value.date())
    NBARosterArchive().publish(later,expected_etag=None)
    assert job.execute_in_process(partition_key=str(DAY)).success
    pointer,_=a.game_selection(str(DAY),data["game"]["game_id"])
    record=a.read_reference(pointer,a.game_prefix)
    assert record["purpose"]=="historical_replay"
    assert record["baseline_through_date"]==str(FIRST)
    assert record["roster_date_pacific"]==str(DAY)
    assert record["inputs"]["daily_minutes"]["uri"].startswith(
        "s3://nba-gestalt-data/"+a.prefix+f"game_date_pacific={FIRST}/"
    )
    before=len(writes)
    assert job.execute_in_process(partition_key=str(DAY)).success
    assert a.game_selection(str(DAY),data["game"]["game_id"])[0]==pointer
    assert len(writes)==before


def test_opening_day_historical_projection_uses_frozen_priors(projection_job,monkeypatch):
    job,a,_,_,_,_,_=projection_job
    data=slate()
    monkeypatch.setattr(NBAScheduleResource,"games_for_date",lambda *a,**k:(data["games"],data["inputs"]["schedule"]))
    assert job.execute_in_process(partition_key=str(FIRST)).success
    pointer,_=a.game_selection(str(FIRST),data["games"][0]["game_id"])
    record=a.read_reference(pointer,a.game_prefix)
    assert record["purpose"]=="historical_replay"
    assert record["baseline_kind"]=="frozen_priors"
    assert record["baseline_through_date"] is None
    assert record["inputs"]["daily_minutes"] is None
    p=next(p for t in record["teams"].values() for p in t["players"] if p["player_id"]==1)
    assert p["conditional_minutes"]==priors()["players"]["1"]["starting_minutes"]


def test_historical_missing_previous_day_does_not_fall_back_to_priors(projection_job):
    job,a,objects,_,data,_,_=projection_job
    Clock.value=NOW+timedelta(days=5)
    del objects[a.prefix+f"game_date_pacific={FIRST}/latest.json"]
    assert not job.execute_in_process(partition_key=str(DAY),raise_on_error=False).success
    assert a.game_selection(str(DAY),data["game"]["game_id"])[0] is None
