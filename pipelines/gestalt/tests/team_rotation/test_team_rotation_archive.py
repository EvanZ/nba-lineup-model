"""Accepted input coverage, immutable output identity, and manual job boundaries."""
import json
from copy import deepcopy
from datetime import date
from hashlib import sha256
import pytest
from botocore.exceptions import ClientError
from gestalt_pipelines.definitions import defs
from gestalt_pipelines.team_rotation.utils import json_bytes
from gestalt_pipelines.utils import NBA_TEAM_IDS

FIRST=date(2026,10,20)


def prepare(saved,day=FIRST):
    a,store,schedule,*_=saved
    return a.prepare(season="2026-27",day=day,first_day=FIRST,nba_schedule=schedule)


def test_reads_shared_inputs_once_and_all_teams_publish(saved):
    a,store,_,doc,*_=saved
    before=dict(store.objects)
    records,guards,_=prepare(saved)
    assert len(records)==30
    assert sum(r["coverage"]["game_count"] for r in records)==4
    assert store.objects==before and not store.writes
    for gid,inp in doc["inputs"]["games"].items():
        for field in ("uri","metadata_uri"):
            key=inp["box_score"][field].split("nba-gestalt-data/",1)[1]
            assert store.reads.count(key)==1
    result=a.publish(records,guards)
    assert result["created_count"]==result["advanced_count"]==30
    assert len(store.writes)==60
    first=dict(store.objects);store.writes.clear()
    result=a.publish(*prepare(saved)[:2])
    assert result["reused_count"]==30 and result["advanced_count"]==0
    assert not store.writes and store.objects==first
    for pointer in result["selections"]:
        raw=store.objects[pointer["uri"].split("nba-gestalt-data/",1)[1]]
        assert sha256(raw).hexdigest()==pointer["sha256"]
        record=json.loads(raw)
        assert record["processing_version"]=="v1"
        if record["players"]:assert record["players"][0]["on_court_possessions"]==10.123456789
    for key,raw in before.items():assert store.objects[key]==raw


@pytest.mark.parametrize("damage",["missing","checksum","incomplete","coverage","missing_box","metadata","lineup","name"])
def test_bad_inputs_do_not_publish(saved,damage):
    a,store,_,doc,accept,*_=saved
    prefix="lake/artifacts/stints/season=2026-27/game_date_pacific=2026-10-20/"
    if damage=="missing":del store.objects[prefix+"latest.json"]
    elif damage=="checksum":
        pointer=json.loads(store.objects[prefix+"latest.json"])
        store.objects[pointer["uri"].split("nba-gestalt-data/",1)[1]]=b'{}'
    elif damage=="incomplete":doc["validation"]["complete"]=False;accept()
    elif damage=="coverage":doc["rows"].pop();accept()
    elif damage=="lineup":doc["rows"][0]["home_player_ids"]=[1]*5;accept()
    else:
        ref=next(iter(doc["inputs"]["games"].values()))["box_score"]
        if damage=="missing_box":del store.objects[ref["uri"].split("nba-gestalt-data/",1)[1]]
        elif damage=="metadata":store.objects[ref["metadata_uri"].split("nba-gestalt-data/",1)[1]]=b'{}'
        else:
            key=ref["uri"].split("nba-gestalt-data/",1)[1];box=json.loads(store.objects[key]);box["game"]["homeTeam"]["players"][0]["name"]=""
            # Keep the box checksum valid; missing names are a separate contract failure.
            raw=json_bytes(box);digest=sha256(raw).hexdigest();newkey=key.rsplit('/',1)[0]+'/'+digest+'.json'
            store.objects[newkey]=raw;meta=store.objects[ref["metadata_uri"].split("nba-gestalt-data/",1)[1]]
            store.objects[newkey.removesuffix('.json')+'.meta.json']=meta
            ref.update(uri='s3://nba-gestalt-data/'+newkey,sha256=digest,metadata_uri='s3://nba-gestalt-data/'+newkey.removesuffix('.json')+'.meta.json');accept()
    with pytest.raises((ValueError,FileNotFoundError,ClientError)):prepare(saved)
    assert not store.writes


def test_correction_retains_old_bytes_and_older_date_cannot_regress_head(saved):
    a,store,_,doc,accept,*_=saved
    original=a.publish(*prepare(saved)[:2]);old_objects=dict(store.objects)
    doc["rows"][0]["possessions"]+=1;accept()
    corrected=a.publish(*prepare(saved)[:2])
    # The date-level stints checksum changes for all four participating teams,
    # but only the corrected game's two clubs have different numeric exposure.
    assert corrected["created_count"]==4 and corrected["advanced_count"]==4
    previous={ref['team_id']:ref for ref in original['selections']}
    numeric_changes=[]
    for ref in corrected['selections']:
        def read(r):return json.loads(store.objects[r['uri'].split('nba-gestalt-data/',1)[1]])
        if read(ref)['players']!=read(previous[ref['team_id']])['players']:numeric_changes.append(ref['team_id'])
    assert len(numeric_changes)==2
    for ref in original["selections"]:
        key=ref["uri"].split("nba-gestalt-data/",1)[1];assert store.objects[key]==old_objects[key]
    later=a.publish(*prepare(saved,date(2026,10,21))[:2]);head=dict(store.objects)
    result=a.publish(*prepare(saved)[:2])
    assert result["advanced_count"]==0
    for tid in NBA_TEAM_IDS:
        key=f"lake/artifacts/team_rotation/season=2026-27/team_id={tid}/latest.json"
        assert store.objects[key]==head[key]


def test_input_race_or_output_race_cannot_be_silently_accepted(saved):
    a,store,_,doc,accept,*_=saved
    records,guards,_=prepare(saved)
    doc["rows"][0]["possessions"]+=2;accept()
    with pytest.raises(ValueError,match="input changed"):a.publish(records,guards)
    assert not store.writes
    records,guards,_=prepare(saved)
    def race(key):
        if key.endswith("latest.json"):store.objects[key]=b'{}'
    store.hook=race
    with pytest.raises(ClientError):a.publish(records,guards)


def test_manual_job_only_selects_new_asset_and_error_is_recorded(saved):
    a,store,*_=saved
    job=defs.resolve_job_def("build_team_rotation")
    assert job.asset_layer.executable_asset_keys=={__import__('dagster').AssetKey('team_rotation')}
    result=job.execute_in_process(partition_key="2026-10-20")
    assert result.success
    assert all(key.startswith("lake/artifacts/team_rotation/") for key in store.writes)
    # Complete preflight fails before publishing any team when a date is missing.
    del store.objects["lake/artifacts/stints/season=2026-27/game_date_pacific=2026-10-20/latest.json"]
    store.writes.clear()
    result=job.execute_in_process(partition_key="2026-10-20",raise_on_error=False)
    assert not result.success
    assert len(store.writes)==1 and store.writes[0].endswith('.failure.json')
    raw=store.objects[store.writes[0]];ref={'uri':'s3://nba-gestalt-data/'+store.writes[0],'sha256':sha256(raw).hexdigest()}
    assert a.read_failure(ref,season='2026-27',day='2026-10-20')==ref
    recovery=a.save_event(season='2026-27',day='2026-10-20',run_id='recovery-run',message='Passed',original_failure=ref)
    assert recovery['uri'].endswith('.recovery.json')


def test_non_regular_season_game_is_not_read_or_counted(saved):
    a,store,_,doc,accept,_,games=saved
    extra=deepcopy(games[0]);extra['game_id']='0062600001';games.append(extra)
    doc['validation']['expected_game_ids'].append(extra['game_id'])
    doc['validation']['processed_game_ids'].append(extra['game_id'])
    doc['inputs']['games'][extra['game_id']]={'box_score':{'uri':'must-not-read'}}
    row=deepcopy(doc['rows'][0]);row['game_id']=extra['game_id'];doc['rows'].append(row);accept()
    records,_,_=prepare(saved)
    assert sum(r['coverage']['game_count'] for r in records)==4
    assert all(extra['game_id'] not in r['coverage']['game_ids'] for r in records)


def test_input_changes_during_staging_leave_all_pointers_untouched(saved):
    a,store,_,doc,accept,*_=saved
    records,guards,_=prepare(saved)
    changed=False
    def change_input(key):
        nonlocal changed
        if not changed and key.startswith('lake/artifacts/team_rotation/'):
            changed=True;doc['rows'][0]['possessions']+=1;accept()
    store.hook=change_input
    with pytest.raises(ValueError,match='during staging'):a.publish(records,guards)
    assert not any(key.endswith('/latest.json') for key in store.writes)


def test_corrupt_immutable_readback_cannot_advance_team_pointer(saved):
    a,store,*_=saved
    records,guards,_=prepare(saved)
    first=records[0];raw=json_bytes(first);digest=sha256(raw).hexdigest()
    key=f"lake/artifacts/team_rotation/season=2026-27/team_id={first['team_id']}/game_date_pacific=2026-10-20/{digest}.json"
    store.objects[key]=b'{}'
    with pytest.raises(ValueError,match='readback'):a.publish(records,guards)
    assert not store.writes


def test_actual_failed_parent_reexecution_records_recovery(saved):
    from dagster import DagsterInstance
    from dagster._core.definitions.job_base import InMemoryJob
    from dagster._core.execution.api import execute_run
    _,store,*_=saved
    job=defs.resolve_job_def('build_team_rotation')
    source='lake/artifacts/stints/season=2026-27/game_date_pacific=2026-10-20/latest.json'
    selection=store.objects.pop(source)
    with DagsterInstance.ephemeral() as instance:
        failed=job.execute_in_process(instance=instance,partition_key='2026-10-20',raise_on_error=False)
        assert not failed.success
        [failure_key]=[k for k in store.objects if k.endswith('.failure.json')]
        failure_bytes=store.objects[failure_key]
        store.objects[source]=selection
        parent=instance.get_run_by_id(failed.run_id)
        child=instance.create_run_for_job(job,run_config=parent.run_config,tags=parent.tags,
                                          parent_run_id=parent.run_id,root_run_id=parent.run_id)
        recovered=execute_run(InMemoryJob(job),child,instance)
        assert recovered.success
    [recovery_key]=[k for k in store.objects if k.endswith('.recovery.json')]
    event=json.loads(store.objects[recovery_key])
    assert event['original_failure']=={'uri':'s3://nba-gestalt-data/'+failure_key,'sha256':sha256(failure_bytes).hexdigest()}
    assert event['dagster_run_id']==recovered.run_id
    assert store.objects[failure_key]==failure_bytes
