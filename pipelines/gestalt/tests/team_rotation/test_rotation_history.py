"""History indexes use successful publication evidence and exact immutable inputs."""
import json
from copy import deepcopy
from datetime import date
from hashlib import sha256
from types import SimpleNamespace

import pytest
from dagster import (AssetKey, AssetMaterialization, DagsterEventType, DagsterInstance, MetadataValue,
                     RunRequest, SkipReason, build_run_status_sensor_context, job, op)

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.team_rotation.utils import accepted_publications, json_bytes
from gestalt_pipelines.team_rotation.sensors import publish_team_rotation_history_sensor
from gestalt_pipelines.utils import NBA_TEAM_IDS

SEASON='2026-27'
DAY='2026-10-20'
ROOT='lake/artifacts/team_rotation/season=2026-27/'


@op(config_schema={'document':str,'fail':bool,'asset':str})
def history_source(context):
    c=context.op_config;doc=json.loads(c['document'])
    refs=doc.pop('selections');day=doc['game_date_pacific']
    context.log_event(AssetMaterialization(asset_key=c['asset'],partition=day,
        metadata={**doc,'selections':MetadataValue.json(refs)}))
    if c['fail']:raise ValueError('Expected failed producer after materialization')


@job
def synthetic_rotation_history_source():history_source()


@pytest.fixture
def published(saved,monkeypatch):
    a,store,schedule,stint,accept,save,games=saved
    schedule_doc={'leagueSchedule':{'seasonYear':SEASON,'leagueId':'00','gameDates':[{'games':[]}]}}
    catalog=schedule_doc['leagueSchedule']['gameDates'][0]['games']
    for g in games:
        catalog.append({'gameId':g['game_id'],'gameDateTimeUTC':g['game_time_utc'].isoformat(),
            **{side+'Team':{'teamId':g[side+'_team_id'],'teamTricode':g[side+'_team_tricode']} for side in ('home','away')}})
    for offset,tid in enumerate(NBA_TEAM_IDS):
        catalog.append({'gameId':f'0022699{offset:03}','gameDateTimeUTC':'2027-03-01T03:00:00+00:00',
            'homeTeam':{'teamId':tid,'teamTricode':f'T{offset:02}'},
            'awayTeam':{'teamId':NBA_TEAM_IDS[(offset+1)%30],'teamTricode':f'T{(offset+1)%30:02}'}})
    raw=json_bytes(schedule_doc);digest=sha256(raw).hexdigest()
    key=f'lake/raw/scheduleleaguev2/season={SEASON}/sha256={digest}/schedule.json'
    ref={'uri':'s3://nba-gestalt-data/'+key,'sha256':digest};store.objects[key]=raw
    store.objects[f'lake/raw/scheduleleaguev2/season={SEASON}/latest.json']=json_bytes(ref)
    monkeypatch.setattr(type(schedule),'load',lambda self,reference=None:(raw,ref))
    monkeypatch.setattr(type(schedule),'games_between',lambda self,first,last,reference=None,allow_empty=False:(games,ref))
    stint['inputs']['schedule']=ref;accept()
    records,_,_=a.prepare(season=SEASON,day=date.fromisoformat(DAY),first_day=date.fromisoformat(DAY),nba_schedule=schedule)
    monkeypatch.setattr('gestalt_pipelines.team_rotation.resources.archive.summarize_team',lambda **kw:pytest.fail('No rotation calculation'))
    with DagsterInstance.ephemeral() as instance:
        def stage(day=DAY,correction=False):
            refs=[]
            for original in records:
                doc=deepcopy(original);doc['game_date_pacific']=day
                if correction:doc['test_correction']=True
                prefix=f"{ROOT}team_id={doc['team_id']}/game_date_pacific={day}/"
                refs.append({**save(prefix,doc),'season':SEASON,'team_id':doc['team_id'],
                             'schema_version':1,'processing_version':'v1','game_date_pacific':day})
            return refs
        def emit(refs=None,day=DAY,fail=False,asset='team_rotation',**overrides):
            if refs is None:refs=stage(day)
            doc={'season':SEASON,'processing_version':'v1','game_date_pacific':day,'team_count':30,'selections':refs}
            doc.update(overrides)
            return synthetic_rotation_history_source.execute_in_process(instance=instance,
                run_config={'ops':{'history_source':{'config':{'document':json.dumps(doc),'fail':fail,'asset':asset}}}},raise_on_error=False)
        def publish():return a.publish_history(SEASON,lambda:accepted_publications(instance,SEASON))
        yield SimpleNamespace(a=a,store=store,instance=instance,records=records,stage=stage,emit=emit,publish=publish)


def indexes(p):
    return {tid:json.loads(p.store.objects[f'{ROOT}team_id={tid}/history.json']) for tid in NBA_TEAM_IDS}


def test_initial_publication_and_unchanged_rerun_preserve_all_originals(published):
    p=published;p.emit();before=dict(p.store.objects)
    result=p.publish()
    assert result['team_count']==30 and result['written_count']==30 and result['summary_count']==30
    assert all(k.endswith('/history.json') for k in p.store.writes)
    assert all(p.store.objects[k]==v for k,v in before.items())
    assert all(v['through_game_date_pacific']==DAY and len(v['points'])==1 for v in indexes(p).values())
    for v in indexes(p).values():assert set(v)=={'schema_version','processing_version','artifact_kind','season','team_id','through_game_date_pacific','points'}
    p.store.writes.clear()
    assert p.publish()['written_count']==0 and not p.store.writes


def test_failed_producer_staged_objects_are_never_accepted(published):
    p=published;p.emit();p.publish();before=indexes(p)
    p.emit(p.stage(correction=True),fail=True)
    p.store.writes.clear()
    assert p.publish()['written_count']==0 and indexes(p)==before
    assert len(accepted_publications(p.instance,SEASON))==1


def test_only_failed_producer_cannot_seed_any_index(published):
    p=published;p.emit(fail=True)
    with pytest.raises(FileNotFoundError):p.publish()
    assert not p.store.writes


def test_correction_changes_reference_but_preserves_old_immutable_objects(published):
    p=published;p.emit();p.publish();old=indexes(p);old_objects=dict(p.store.objects)
    p.emit(p.stage(correction=True));p.publish()
    new=indexes(p)
    for tid in NBA_TEAM_IDS:
        assert new[tid]['points'][0]!=old[tid]['points'][0]
        key=old[tid]['points'][0]['uri'].split('nba-gestalt-data/',1)[1]
        assert p.store.objects[key]==old_objects[key]


def test_older_backfill_retains_newer_dates_and_chronological_unique_points(published):
    p=published;p.emit();first=accepted_publications(p.instance,SEASON)
    p.emit(day='2026-10-22');p.publish()
    p.a.publish_history(SEASON,lambda:first)
    for index in indexes(p).values():
        assert index['through_game_date_pacific']=='2026-10-22'
        assert [r['game_date_pacific'] for r in index['points']]==[DAY,'2026-10-22']


def test_newest_upstream_pointers_are_not_required_to_match_pinned_inputs(published):
    p=published;p.emit()
    for key in list(p.store.objects):
        if key.startswith(('lake/raw/','lake/artifacts/stints/')) and key.endswith('/latest.json'):
            p.store.objects[key]=b'newer unrelated selection'
    p.store.reads.clear();p.publish()
    assert not any(k.endswith('/latest.json') for k in p.store.reads)


@pytest.mark.parametrize('kind',['summary','stints','box','metadata','schedule'])
def test_corrupt_original_inputs_fail_before_any_index_write(published,kind):
    p=published;p.emit()
    paths={'summary':ROOT,'stints':'lake/artifacts/stints/','box':'lake/raw/cdn/boxscore/',
           'metadata':'lake/raw/cdn/boxscore/','schedule':'lake/raw/scheduleleaguev2/'}
    key=next(k for k in p.store.objects if k.startswith(paths[kind]) and not k.endswith('/latest.json')
             and (k.endswith('.meta.json') if kind=='metadata' else not k.endswith('.meta.json')))
    if kind=='stints':key=next(iter(p.records[0]['inputs']['stints'].values()))['uri'].split('nba-gestalt-data/',1)[1]
    p.store.objects[key]=b'{}'
    with pytest.raises(ValueError,match='checksum'):p.publish()
    assert not p.store.writes


@pytest.mark.parametrize('new_date',[False,True])
def test_concurrent_publisher_cannot_overwrite_newer_correction_or_date(published,new_date):
    p=published;p.emit();happened=False
    def concurrent(key):
        nonlocal happened
        if happened or not key.endswith('/history.json'):return
        happened=True
        day='2026-10-22' if new_date else DAY
        refs=p.stage(day=day,correction=not new_date);p.emit(refs,day=day)
        tid=int(key.split('team_id=')[1].split('/')[0])
        ref=next(r for r in refs if r['team_id']==tid)
        point={k:ref[k] for k in ('game_date_pacific','uri','sha256')}
        record={'schema_version':1,'processing_version':'v1','artifact_kind':'team_rotation_history_index',
                'season':SEASON,'team_id':tid,'through_game_date_pacific':day,'points':[point]}
        p.store.objects[key]=json_bytes(record)
    p.store.hook=concurrent
    p.publish()
    expected=accepted_publications(p.instance,SEASON)
    for tid,index in indexes(p).items():
        assert index['through_game_date_pacific']==('2026-10-22' if new_date else DAY)
        for point in index['points']:
            ref=next(r for r in expected[point['game_date_pacific']]['selections'] if r['team_id']==tid)
            assert point['sha256']==ref['sha256']


def test_repeated_conflict_fails_visibly_without_accepting_old_points(published):
    from botocore.exceptions import ClientError
    p=published;p.emit()
    def conflict(key):raise ClientError({'Error':{'Code':'PreconditionFailed'}},'PutObject')
    p.store.hook=conflict
    with pytest.raises(ValueError,match='changed repeatedly'):p.publish()
    assert not p.store.writes


def test_partial_index_write_failure_is_recoverable_and_only_indexes_valid_sources(published):
    p=published;p.emit();job=defs.resolve_job_def('publish_team_rotation_history')
    assert job.asset_layer.executable_asset_keys=={AssetKey('team_rotation_history')}
    calls=0
    def fail_second(key):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('Simulated index write failure')
    p.store.hook=fail_second
    failed=job.execute_in_process(instance=p.instance,raise_on_error=False)
    assert not failed.success and len(p.store.writes)==1
    assert not failed.get_asset_materialization_events()
    p.store.hook=None
    recovered=job.execute_in_process(instance=p.instance)
    assert recovered.success and len(indexes(p))==30
    assert all(len(i['points'])==1 for i in indexes(p).values())


def sensor(p,result,fail=False):
    kind=DagsterEventType.RUN_FAILURE if fail else DagsterEventType.RUN_SUCCESS
    event=next(e for e in result.all_events if e.event_type==kind)
    with build_run_status_sensor_context(sensor_name='publish_team_rotation_history_sensor',dagster_event=event,
            dagster_instance=p.instance,dagster_run=p.instance.get_run_by_id(result.run_id)) as context:
        return publish_team_rotation_history_sensor(context)


def test_success_trigger_deduplicates_and_detects_corrections(published):
    p=published
    first=sensor(p,p.emit());repeat=sensor(p,p.emit())
    assert isinstance(first,RunRequest) and first.partition_key is None
    assert first.run_key==repeat.run_key and first.tags['dagster/priority']=='-2'
    correction=sensor(p,p.emit(p.stage(correction=True)))
    assert correction.run_key!=first.run_key


@pytest.mark.parametrize('changes',[{'fail':True},{'asset':'team_rotation_history'},{'season':'2025-26'}])
def test_failed_unrelated_and_foreign_sources_never_trigger(published,changes):
    p=published;result=p.emit(**changes)
    assert isinstance(sensor(p,result,fail=changes.get('fail',False)),SkipReason)


def test_incomplete_publication_cannot_trigger_or_publish(published):
    p=published;result=p.emit(p.stage()[:-1])
    with pytest.raises(ValueError,match='incomplete'):sensor(p,result)
    with pytest.raises(ValueError,match='incomplete'):p.publish()
    assert not p.store.writes


@pytest.mark.parametrize('field,value',[('season','2025-26'),('team_id',1610612766),('processing_version','v2'),('game_date_pacific','2026-10-19')])
def test_summary_identity_verified_even_when_bytes_match_hash(published,field,value):
    p=published;refs=p.stage()
    ref=refs[0];key=ref['uri'].split('nba-gestalt-data/',1)[1]
    doc=json.loads(p.store.objects[key]);doc[field]=value;raw=json_bytes(doc);digest=sha256(raw).hexdigest()
    newkey=key.rsplit('/',1)[0]+'/'+digest+'.json';p.store.objects[newkey]=raw
    ref.update(uri='s3://nba-gestalt-data/'+newkey,sha256=digest)
    p.emit(refs)
    with pytest.raises(ValueError):p.publish()
    assert not p.store.writes


def test_unpublished_dated_objects_are_not_discovered_as_accepted(published):
    p=published;p.emit();p.stage(day='2026-10-22')
    p.publish()
    assert all([point['game_date_pacific'] for point in index['points']]==[DAY] for index in indexes(p).values())


def test_publication_changes_during_verification_are_refreshed_before_writing(published):
    p=published;p.emit();old=accepted_publications(p.instance,SEASON)
    p.emit(p.stage(correction=True));new=accepted_publications(p.instance,SEASON)
    calls=0
    def changing():
        nonlocal calls
        calls+=1
        return old if calls==1 else new
    p.a.publish_history(SEASON,changing)
    for tid,index in indexes(p).items():
        ref=next(r for r in new[DAY]['selections'] if r['team_id']==tid)
        assert index['points'][0]['sha256']==ref['sha256']


def test_acceptance_history_is_paginated_without_losing_earlier_dates(published):
    p=published;p.emit();p.emit(day='2026-10-22')
    events=p.instance.fetch_materializations(AssetKey('team_rotation'),limit=10).records
    calls=[]
    def fetch(filters,limit,cursor):
        calls.append(cursor)
        return SimpleNamespace(records=events[:1] if cursor is None else events[1:],
                               has_more=cursor is None,cursor='next')
    proxy=SimpleNamespace(fetch_materializations=fetch,get_run_by_id=p.instance.get_run_by_id)
    assert sorted(accepted_publications(proxy,SEASON))==[DAY,'2026-10-22']
    assert calls==[None,'next']
