"""Successful stint evidence triggers one complete cumulative rotation refresh."""
from datetime import UTC, datetime
import pytest
from dagster import (AssetMaterialization, DagsterEventType, DagsterInstance, MetadataValue,
                     RunRequest, SkipReason, build_run_status_sensor_context, job, op)

from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.team_rotation.sensors import build_team_rotation_sensor


@op(config_schema={"day":str,"digest":str,"season":str,"asset":str,"emit":bool,"fail":bool,"coverage":bool})
def rotation_source(context):
    c=context.op_config
    games={"2026-10-20":["0022600001"],"2026-10-22":["0022600002"]}.get(c['day'],[])
    metadata={"season":c['season'],"game_date_pacific":c['day'],"sha256":c['digest'],
              "stints_uri":f"s3://nba-gestalt-data/lake/artifacts/stints/season={c['season']}/game_date_pacific={c['day']}/{c['digest']}.json",
              "game_count":len(games),"inputs":MetadataValue.json({'games':{gid:{} for gid in games} if c['coverage'] else {'missing':{}}})}
    if c['emit']:context.log_event(AssetMaterialization(asset_key=c['asset'],partition=c['day'],metadata=metadata))
    if c['fail']:raise ValueError('Expected upstream failure')


@job
def synthetic_rotation_source():rotation_source()


def request(instance,**changes):
    c={'day':'2026-10-20','digest':'a'*64,'season':'2026-27','asset':'nba_stints','emit':True,'fail':False,'coverage':True}
    c.update(changes)
    result=synthetic_rotation_source.execute_in_process(instance=instance,
        run_config={'ops':{'rotation_source':{'config':c}}},
        tags={'dagster/asset_partition_range_start':c['day'],'dagster/asset_partition_range_end':c['day']},raise_on_error=False)
    kind=DagsterEventType.RUN_FAILURE if c['fail'] else DagsterEventType.RUN_SUCCESS
    event=next(e for e in result.all_events if e.event_type==kind)
    schedule=NBAScheduleResource(season='2026-27')
    with build_run_status_sensor_context(sensor_name='build_team_rotation_sensor',dagster_event=event,
         dagster_instance=instance,dagster_run=instance.get_run_by_id(result.run_id),resources={'nba_schedule':schedule}) as context:
        return build_team_rotation_sensor(context)


@pytest.fixture(autouse=True)
def scheduled_games_and_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return datetime(2026,10,24,18,tzinfo=UTC)
    monkeypatch.setattr('gestalt_pipelines.team_rotation.sensors.datetime',Clock)
    games=[{'game_id':gid,'game_time_utc':datetime(2026,10,day,20,tzinfo=UTC)}
           for day,gid in [(20,'0022600001'),(22,'0022600002')]]
    monkeypatch.setattr(NBAScheduleResource,'games_between',lambda self,first,last,reference=None,allow_empty=False:
        ([g for g in games if first<=g['game_time_utc'].date()<=last],{'uri':'s3://nba-gestalt-data/schedule','sha256':'e'*64}))
    monkeypatch.setattr('boto3.client',lambda *a,**kw:pytest.fail('Tests must not access real S3 or providers'))


def test_new_completed_stints_and_identical_reruns_deduplicate():
    with DagsterInstance.ephemeral() as instance:
        first=request(instance)
        repeat=request(instance)
        assert isinstance(first,RunRequest) and first.partition_key=='2026-10-20'
        assert first.run_key==repeat.run_key
        assert first.tags['dagster/priority']=='-2'
        changed=request(instance,digest='b'*64)
        assert changed.run_key!=first.run_key


def test_older_correction_refreshes_newest_head_and_stable_run_key():
    with DagsterInstance.ephemeral() as instance:
        request(instance)
        head=request(instance,day='2026-10-22')
        correction=request(instance,digest='c'*64)
        repeat=request(instance,digest='c'*64)
        assert head.partition_key==correction.partition_key=='2026-10-22'
        assert correction.run_key!=head.run_key and correction.run_key==repeat.run_key


def test_waits_for_missing_earlier_date_then_gap_repair_launches_current_head():
    with DagsterInstance.ephemeral() as instance:
        missing=request(instance,day='2026-10-22')
        assert isinstance(missing,SkipReason) and '2026-10-20' in missing.skip_message
        fixed=request(instance)
        assert isinstance(fixed,RunRequest) and fixed.partition_key=='2026-10-22'


def test_off_day_advances_without_requiring_fake_missing_empty_partitions():
    with DagsterInstance.ephemeral() as instance:
        request(instance)
        request(instance,day='2026-10-22')
        empty=request(instance,day='2026-10-23')
        assert isinstance(empty,RunRequest) and empty.partition_key=='2026-10-23'


@pytest.mark.parametrize('changes',[{'fail':True},{'emit':False},{'asset':'team_rotation'},
                                  {'season':'2025-26'},{'day':'2026-10-24'},{'day':'2026-10-19'}])
def test_failed_unrelated_foreign_or_uncompleted_events_do_not_launch(changes):
    with DagsterInstance.ephemeral() as instance:
        assert isinstance(request(instance,**changes),SkipReason)


def test_incomplete_game_coverage_waits_until_repaired():
    with DagsterInstance.ephemeral() as instance:
        assert isinstance(request(instance,coverage=False),SkipReason)
        assert isinstance(request(instance),RunRequest)


def test_invalid_hash_is_visible():
    with DagsterInstance.ephemeral() as instance,pytest.raises(ValueError,match='checksum'):
        request(instance,digest='broken')


def test_failed_head_materialization_is_not_reused_on_older_success():
    with DagsterInstance.ephemeral() as instance:
        request(instance)
        assert isinstance(request(instance,day='2026-10-22',fail=True),SkipReason)
        assert isinstance(request(instance,digest='c'*64),SkipReason)
        assert isinstance(request(instance,day='2026-10-22'),RunRequest)
