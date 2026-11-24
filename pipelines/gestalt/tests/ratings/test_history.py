"""History copies synthetic saved states; tests never contact real S3 or fit ratings."""
import json
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
from types import SimpleNamespace

import pytest
from gestalt_pipelines.definitions import defs
from gestalt_pipelines.ratings.resources import RatingsArchive
from gestalt_pipelines.ratings.utils import json_bytes
from gestalt_pipelines.ratings.utils.update import CALCULATION_CONTRACT, COMPLETED_GAME_DATE_CONTRACT
from ratings.conftest import S3

FIRST = date(2026, 10, 20)


class Store(S3):
    def __init__(self):
        super().__init__()
        self.writes, self.hook = [], None

    def put_object(self, **kw):
        assert kw['Key'].startswith(('lake/artifacts/player_game_history/', 'lake/artifacts/errors/pipeline=ratings/'))
        if self.hook:
            self.hook(kw)
        result = super().put_object(**kw)
        self.writes.append(kw['Key'])
        return result

    def get_paginator(self, _):
        return SimpleNamespace(paginate=lambda **kw: [{'Contents': [{'Key': key} for key in self.objects if key.startswith(kw['Prefix'])]}])


@pytest.fixture
def saved_history(monkeypatch, record):
    store = Store()
    monkeypatch.setattr('boto3.client', lambda *a, **kw: store)
    monkeypatch.setattr('gestalt_pipelines.ratings.utils.update.solve_joint', lambda *a, **kw: pytest.fail('History must not solve ratings'))
    monkeypatch.setattr('numpy.linalg.solve', lambda *a, **kw: pytest.fail('History must not fit any model'))
    archive = RatingsArchive(season='2026-27')
    def save(prefix, doc):
        raw = json_bytes(doc); digest = sha256(raw).hexdigest()
        ref = {'uri': f's3://nba-gestalt-data/{prefix}{digest}.json', 'sha256': digest}
        store.objects[prefix+digest+'.json'] = raw
        return ref
    priors = save(archive.prefix, record)
    store.objects[archive.prefix+'latest.json'] = json_bytes({**priors, 'schema_version': 1, 'season': '2026-27', 'model_version': 'v1'})
    states, refs = [], []
    for offset in range(3):
        day = str(FIRST+timedelta(days=offset)); players = []
        for pid in (list(range(1, 13)) if offset == 0 else [*range(1, 13), 9999]):
            initial = str(FIRST if pid != 9999 else FIRST+timedelta(days=1))
            starting = record['players'][pid-1]['rapm'] if pid <= 2 else record['settings']['replacement_prior']
            nail = starting+(min(offset, 1)*0.123456789012345 if pid == 1 else 0)
            players.append({'player_id': pid, 'player_name': f'Player {pid}', 'starting_nail': starting,
                'nail': nail, 'residual': nail-starting, 'on_court_possessions': 50.25 if pid == 1 else 0,
                'team_id': (1610612738 if offset == 0 else 1610612755) if pid == 1 else None,
                'initialized_on': initial, 'initialization': {'kind': 'replacement' if pid == 9999 else 'frozen_catalog', 'input': priors},
                'forecast_source': 'in_season_replacement_cold_start_prior' if pid == 9999 else record['players'][min(pid,2)-1]['forecast_source'],
                'standard_error': 1.0, 'information': 0.0})
        n = len(players)
        ref = lambda kind: {'uri': f's3://nba-gestalt-data/fixture/{day}/{kind}.json', 'sha256': sha256((day+kind).encode()).hexdigest()}
        state = {'schema_version': 1, 'artifact_kind': 'player_ratings', 'season': '2026-27', 'model_version': 'v1',
            'model_id': record['model_id'], 'model_run_id': record['model_run_id'], 'settings': deepcopy(record['settings']),
            'game_date_pacific': day, 'observed_through_game_date_pacific': day,
            'effective_from_game_date_pacific': str(FIRST+timedelta(days=offset+1)),
            'date_contract': COMPLETED_GAME_DATE_CONTRACT, 'calculation_contract': CALCULATION_CONTRACT,
            'inputs': {'priors': priors, 'parent': refs[-1] if refs else None, 'schedule': ref('schedule'), 'stints': ref('stints'), 'rosters': ref('rosters')},
            'players': players, 'included_game_ids': [f'002260000{i+1}' for i in range(min(offset+1, 2))],
            'statistics': {'player_ids': [p['player_id'] for p in players], 'hca_column': n,
                'gram': [[0]*(n+1) for _ in range(n+1)], 'score': [0]*(n+1), 'exposure': [0]*(n+1)},
            'hca': {'estimate': record['settings']['hca']['prior_mean']},
            'validation': {'complete': True, 'initialization_only': False, 'expected_game_ids': [f'002260000{offset+1}'] if offset < 2 else [], 'processed_game_ids': [f'002260000{offset+1}'] if offset < 2 else []}}
        prefix = archive.state_prefix+f'game_date_pacific={day}/'
        source = save(prefix, state)
        store.objects[prefix+'latest.json'] = json_bytes({**source, 'season': '2026-27', 'game_date_pacific': day})
        states.append(state); refs.append(source)
    from gestalt_pipelines.resources import NBAScheduleResource
    # Minimal synthetic upstream documents; forecasting arithmetic has its own
    # tests. Real reference/hash checks and game-history validation still run.
    monkeypatch.setattr('gestalt_pipelines.predictions.resources.archive.validate_prediction',lambda *a:None)
    monkeypatch.setattr('gestalt_pipelines.predictions.resources.archive.validate_actual_minutes_prediction',lambda *a:None)
    games_by_day={}
    for offset,state in enumerate(states):
        day=state['game_date_pacific'];games_by_day[day]=[]
        if offset==2:
            state['inputs']['stints']=save(f'lake/artifacts/stints/season=2026-27/game_date_pacific={day}/',{'season':'2026-27','game_date_pacific':day,'rows':[]})
            continue
        gid=state['validation']['expected_game_ids'][0];home=1610612738 if offset==0 else 1610612755;away=1610612747
        game={'game_id':gid,'game_date':day,'home_team_id':home,'away_team_id':away,'home_team_tricode':'BOS' if offset==0 else 'PHI','away_team_tricode':'LAL'}
        games_by_day[day]=[game]
        playing={'home':[1,3,4,5,11],'away':[6,7,8,9,10]}
        pred={'schema_version':1,'artifact_kind':'game_prediction','season':'2026-27','model_version':'v1','game_id':gid,'game_date_pacific':day,'purpose':'historical_replay'}
        box={'game':{'gameId':gid,'gameStatus':3}}
        actual={'schema_version':1,'artifact_kind':'game_prediction_actual_minutes','season':'2026-27','model_version':'v1','game_id':gid,'game_date_pacific':day,'baseline_prediction_purpose':'historical_replay','actual_home_score':110,'actual_away_score':100}
        for side,team,score in [('home',home,110),('away',away,100)]:
            ids=playing[side]+([2]+([9999] if offset else []) if side=='home' else [])
            pred[side]={'team_id':team,'players':[{'player_id':pid,'nail':0.987654321012345,'projected_minutes':30.123456789 if pid in playing[side] else 0,'injury_status':'Available' if pid in playing[side] else 'Out','injury_reason':None if pid in playing[side] else 'ACL repair'} for pid in ids]}
            actual[side]={'team_id':team,'players':[{'player_id':p['player_id'],'nail':p['nail'],'projected_minutes':p['projected_minutes'],'actual_minutes':48.0 if p['player_id'] in playing[side] else 0.0,'in_baseline_prediction':True} for p in pred[side]['players']]}
            box['game'][side+'Team']={'teamId':team,'score':score,'players':[{'personId':pid,'played':'1' if pid in playing[side] else '0','status':'ACTIVE' if pid in playing[side] else 'INACTIVE','notPlayingDescription':None if pid in playing[side] else 'ACL repair','statistics':{'minutes':'PT48M00.00S' if pid in playing[side] else 'PT00M00.00S','points':43,'assists':9,'plusMinusPoints':10}} for pid in ids if not (pid==2 and offset==0)]}
        predref=save(f'lake/artifacts/game_predictions/season=2026-27/model_version=v1/game_date_pacific={day}/game_id={gid}/',pred)
        boxref=save(f'lake/raw/cdn/boxscore/season=2026-27/game_id={gid}/',box)
        metakey=boxref['uri'].split('nba-gestalt-data/',1)[1].removesuffix('.json')+'.meta.json';meta=json_bytes({'source':'official NBA'})
        store.objects[metakey]=meta;boxref.update(metadata_uri='s3://nba-gestalt-data/'+metakey,metadata_sha256=sha256(meta).hexdigest())
        actual['inputs']={'prediction':predref,'box_score':boxref}
        prefix=f'lake/artifacts/game_predictions_actual_minutes/season=2026-27/model_version=v1/game_date_pacific={day}/game_id={gid}/'
        actualref=save(prefix,actual);store.objects[prefix+'latest.json']=json_bytes({**actualref,'schema_version':1,'season':'2026-27','model_version':'v1','game_date_pacific':day,'game_id':gid})
        state['inputs']['stints']=save(f'lake/artifacts/stints/season=2026-27/game_date_pacific={day}/',{'season':'2026-27','game_date_pacific':day,'rows':[{'game_id':gid,'home_team_id':home,'away_team_id':away,'home_player_ids':playing['home'],'away_player_ids':playing['away'],'possessions':50.25}]})
    monkeypatch.setattr(NBAScheduleResource,'games_for_date',lambda self,day,reference=None,allow_empty=False:(games_by_day[str(day)],reference))
    rewrite(archive,store,states,save)
    refs[:]=[archive.state_selection(state['game_date_pacific'])[0] for state in states]
    return defs.resolve_job_def('publish_player_game_history'), archive, store, states, refs, save


def load(store, ref):
    raw = store.objects[ref['uri'].removeprefix('s3://nba-gestalt-data/')]
    assert sha256(raw).hexdigest() == ref['sha256']
    return json.loads(raw)


def rewrite(a, store, states, save):
    references = []
    for state in states:
        state['inputs']['parent'] = references[-1] if references else None
        day = state['game_date_pacific']; prefix = a.state_prefix+f'game_date_pacific={day}/'
        ref = save(prefix, state); references.append(ref)
        store.objects[prefix+'latest.json'] = json_bytes({**ref, 'season': '2026-27', 'game_date_pacific': day})
    return references


def test_manual_history_keeps_games_inactive_players_and_off_day_current(saved_history):
    job,a,store,states,refs,_=saved_history
    sources=dict(store.objects)
    assert job.execute_in_process(partition_key='2026-10-22').success
    for pid,count in [(1,2),(2,2),(9999,1),(12,0)]:
        index,_=a.history_selection(pid)
        assert len(index['points'])==count and index['source_ratings']==refs[-1]
        assert index['current']['game_date_pacific']=='2026-10-22'
        assert index['current']['source_ratings']==refs[-1]
        assert index['current']['nail']==next(p['nail'] for p in states[-1]['players'] if p['player_id']==pid)
        assert index['starting_point']['kind']==('introduced_prior' if pid==9999 else 'opening_prior')
        for point in index['points']:
            row=load(store,point);assert row['games'] and row['game_date_pacific']!='2026-10-22'
            offset=(date.fromisoformat(row['game_date_pacific'])-FIRST).days
            source=states[offset];player=next(p for p in source['players'] if p['player_id']==pid)
            assert row['nail']==player['nail'] and row['cumulative_possessions']==player['on_court_possessions']
            assert row['inputs']=={'ratings_state':refs[offset],**source['inputs']}
            assert 'observed_through_game_date_pacific' not in row and 'slate_game_ids' not in row
    index,_=a.history_selection(2);first,second=[load(store,p)['games'][0] for p in index['points']]
    assert first['participation']=={'status':'not_listed','reason':None,'box_score_row_present':False}
    assert second['participation']['status']=='inactive'
    assert first['pregame']['injury_status']==second['pregame']['injury_status']=='Out'
    assert first['actual_minutes']==second['actual_minutes']==0 and first['stats'] is second['stats'] is None
    index,_=a.history_selection(1);first,second=[load(store,p)['games'][0] for p in index['points']]
    assert first['team_id']!=second['team_id'] and first['stats']['plus_minus']==10
    assert first['stats']['points']==43 and first['stats']['assists']==9
    assert first['pregame']['nail']!=load(store,index['points'][0])['nail']
    assert all(store.objects[k]==v for k,v in sources.items())
    assert not any('game_date_pacific=2026-10-22/' in k for k in store.writes if k.startswith(a.history_root))


def test_first_date_then_later_date_and_identical_reruns_are_idempotent(saved_history):
    job, a, store, _, _, _ = saved_history
    assert job.execute_in_process(partition_key='2026-10-20').success
    assert a.history_selection(9999) == (None, None)
    assert job.execute_in_process(partition_key='2026-10-22').success
    before, written = dict(store.objects), len(store.writes)
    assert job.execute_in_process(partition_key='2026-10-22').success
    assert store.objects == before and len(store.writes) == written


def test_older_backfill_does_not_regress_newer_player_indexes(saved_history):
    job, a, store, _, _, _ = saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    before = dict(store.objects)
    assert job.execute_in_process(partition_key='2026-10-20').success
    assert store.objects == before
    assert a.history_selection(1)[0]['through_game_date_pacific'] == '2026-10-22'


@pytest.mark.parametrize('fault', ['missing_head','missing_parent','bad_hash','wrong_date','wrong_season','wrong_model','wrong_version','wrong_calculation','wrong_priors','wrong_effective','lost_player','changed_initialization','mixed_latest'])
def test_invalid_source_lineage_fails_before_publishing_any_history(saved_history, fault):
    job, a, store, states, refs, save = saved_history
    if fault == 'missing_head': del store.objects[a.state_prefix+'game_date_pacific=2026-10-22/latest.json']
    elif fault == 'missing_parent': del store.objects[refs[1]['uri'].removeprefix('s3://nba-gestalt-data/')]
    elif fault == 'bad_hash': store.objects[refs[1]['uri'].removeprefix('s3://nba-gestalt-data/')] = b'{}'
    elif fault == 'mixed_latest':
        changed = deepcopy(states[0]); changed['players'][0]['nail'] += .1; changed['players'][0]['residual'] += .1
        ref = save(a.state_prefix+'game_date_pacific=2026-10-20/', changed)
        store.objects[a.state_prefix+'game_date_pacific=2026-10-20/latest.json'] = json_bytes({**ref,'season':'2026-27','game_date_pacific':'2026-10-20'})
    else:
        if fault == 'wrong_date': states[1]['game_date_pacific'] = '2026-10-19'
        if fault == 'wrong_season': states[1]['season'] = '2025-26'
        if fault == 'wrong_model': states[1]['model_id'] = 'unapproved-model'
        if fault == 'wrong_version': states[1]['model_version'] = 'v2'
        if fault == 'wrong_calculation': states[1]['calculation_contract'] = 'other-calculation'
        if fault == 'wrong_priors': states[1]['inputs']['priors'] = {'uri':'s3://other/priors','sha256':'f'*64}
        if fault == 'wrong_effective': states[1]['effective_from_game_date_pacific'] = '2026-10-21'
        if fault == 'changed_initialization': states[1]['players'][0]['initialized_on'] = '2026-10-19'
        if fault == 'lost_player': states[1]['players'][0]['player_id'] = 10; states[1]['players'].sort(key=lambda p:p['player_id']); states[1]['statistics']['player_ids'] = [p['player_id'] for p in states[1]['players']]
        rewrite(a, store, states, save)
    assert not job.execute_in_process(partition_key='2026-10-22', raise_on_error=False).success
    assert not any(key.startswith(a.history_root) for key in store.objects)


def test_corrected_suffix_replaces_coherent_indexes_and_preserves_old_versions(saved_history):
    job, a, store, states, refs, save = saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    before = {k:v for k,v in store.objects.items() if k.startswith(a.history_root) and not k.endswith('latest.json')}
    original, _ = a.history_selection(1)
    for state in states[1:]:
        state['players'][0]['nail'] += .25; state['players'][0]['residual'] += .25
    corrected = rewrite(a, store, states, save)
    assert job.execute_in_process(partition_key='2026-10-22').success
    index, _ = a.history_selection(1)
    assert index['source_ratings'] == corrected[-1]
    assert index['points'][0] == original['points'][0]
    assert all(index['points'][i]['source_ratings'] == corrected[i] for i in range(2))
    assert load(store,index['points'][1])['nail'] == states[1]['players'][0]['nail']
    assert all(store.objects[k] == v for k,v in before.items())


def test_changed_source_during_staging_cannot_move_any_index(saved_history):
    job, a, store, states, refs, _ = saved_history
    assert job.execute_in_process(partition_key='2026-10-20').success
    old = store.objects[a.history_prefix(1)+'latest.json']
    def changed(kw):
        if kw['Key'].startswith(a.history_root) and not kw['Key'].endswith('latest.json'):
            store.objects[a.state_prefix+'game_date_pacific=2026-10-22/latest.json'] = json_bytes({
                'uri': refs[2]['uri'].replace(refs[2]['sha256'],'f'*64), 'sha256':'f'*64,'season':'2026-27','game_date_pacific':'2026-10-22'})
            store.hook = None
    store.hook = changed
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert store.objects[a.history_prefix(1)+'latest.json'] == old
    assert a.history_selection(9999) == (None,None)


def test_conditional_index_write_rejects_a_competing_writer(saved_history):
    job, a, store, _, _, _ = saved_history
    assert job.execute_in_process(partition_key='2026-10-20').success
    old = store.objects[a.history_prefix(1)+'latest.json']
    store.race = True
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert store.objects[a.history_prefix(1)+'latest.json'] == old


def test_missing_previously_accepted_record_is_restored_before_index_reuse(saved_history):
    job, a, store, _, _, _ = saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    index,_ = a.history_selection(1)
    key = index['points'][0]['uri'].removeprefix('s3://nba-gestalt-data/')
    raw = store.objects.pop(key)
    assert job.execute_in_process(partition_key='2026-10-22').success
    assert store.objects[key] == raw


@pytest.mark.parametrize('fault',['player','path','chronology','current','model','starting'])
def test_invalid_accepted_index_is_not_silently_replaced(saved_history,fault):
    job,a,store,_,_,_ = saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    index,_=a.history_selection(1)
    if fault=='player':index['player_id']=2
    if fault=='path':index['points'][0]['uri']='s3://other/file.json'
    if fault=='chronology':index['points'].reverse()
    if fault=='current':index['current']=index['points'][0]
    if fault=='model':index['model_id']='other-model'
    if fault=='starting':index['starting_point']['first_state_date']='2026-10-23'
    key=a.history_prefix(1)+'latest.json';store.objects[key]=json_bytes(index)
    original=store.objects[key]
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert store.objects[key]==original


def test_real_failed_parent_reexecution_uses_existing_ratings_error_layout(saved_history):
    from dagster import DagsterInstance
    from dagster._core.execution.api import execute_run
    from dagster._core.definitions.job_base import InMemoryJob
    job,a,store,_,refs,_=saved_history
    key=refs[1]['uri'].removeprefix('s3://nba-gestalt-data/');raw=store.objects.pop(key)
    with DagsterInstance.ephemeral() as instance:
        failed=job.execute_in_process(partition_key='2026-10-22',instance=instance,raise_on_error=False)
        assert not failed.success
        store.objects[key]=raw
        child=instance.create_run_for_job(job,parent_run_id=failed.run_id,root_run_id=failed.run_id,
            tags={'dagster/partition':'2026-10-22'})
        assert execute_run(InMemoryJob(job),child,instance).success
    failures=[json.loads(v) for k,v in store.objects.items() if k.endswith('.failure.json')]
    recoveries=[json.loads(v) for k,v in store.objects.items() if k.endswith('.recovery.json')]
    assert len(failures)==len(recoveries)==1
    assert recoveries[0]['stage']=='player_game_history'
    assert recoveries[0]['original_failure']['sha256']==sha256(json_bytes(failures[0])).hexdigest()


def test_history_job_selects_only_new_asset_and_sensor_defaults_stopped(saved_history):
    job,*_=saved_history
    assert job.name=='publish_player_game_history' and job.op_selection is None
    assert job.tags['dagster/priority']=='-2'
    assert {n.name for n in job.nodes}=={'player_game_history'}
    assert next(s for s in defs.sensors if s.name == 'publish_player_game_history_sensor').default_status.value == 'STOPPED'
    assert all('history' not in s.name for s in defs.schedules)


def test_uncached_existing_record_with_wrong_bytes_is_rejected(saved_history):
    from gestalt_pipelines.ratings.utils import history_records
    job,a,store,_,_,_=saved_history
    chain=a.history_chain('2026-10-22',first_date='2026-10-20')
    assert job.execute_in_process(partition_key='2026-10-22').success
    index,_=a.history_selection(1);row=load(store,index['points'][0]);ref=a.history_record_reference(row)
    for key in list(store.objects):
        if key.startswith(a.history_root):del store.objects[key]
    store.objects[ref['uri'].removeprefix('s3://nba-gestalt-data/')]=b'{}'
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert a.history_selection(1)==(None,None)


def test_index_readback_failure_is_visible(saved_history,monkeypatch):
    from io import BytesIO
    job,a,store,_,_,_=saved_history
    original=store.get_object
    def corrupt(**kw):
        result=original(**kw)
        if kw['Key'].startswith(a.history_root) and kw['Key'].endswith('latest.json'):
            result['Body']=BytesIO(b'{}')
        return result
    monkeypatch.setattr(store,'get_object',corrupt)
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success


def test_per_player_initialization_input_mismatch_is_rejected(saved_history):
    job,a,store,states,_,save=saved_history
    states[1]['players'][0]['initialization']['input']={'uri':'s3://other/prior','sha256':'f'*64}
    rewrite(a,store,states,save)
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert not any(k.startswith(a.history_root) for k in store.objects)


def test_history_reuses_one_ratings_s3_client_per_run(saved_history,monkeypatch):
    job,a,store,_,_,_=saved_history
    clients=[];original=RatingsArchive._s3
    def client(self):
        value=original(self);clients.append(id(value));return value
    monkeypatch.setattr(RatingsArchive,'_s3',client)
    assert job.execute_in_process(partition_key='2026-10-22').success
    assert len(clients)>10 and set(clients)=={id(store)}


def revise_game_box(fixture, *, corrupt=None):
    _,a,store,_,_,save=fixture;day='2026-10-20';gid='0022600001'
    prefix=f'lake/artifacts/game_predictions_actual_minutes/season=2026-27/model_version=v1/game_date_pacific={day}/game_id={gid}/'
    pointer=json.loads(store.objects[prefix+'latest.json']);actual=load(store,pointer)
    oldbox=actual['inputs']['box_score'];box=load(store,oldbox)
    if corrupt=='missing':del store.objects[prefix+'latest.json'];return
    if corrupt=='hash':store.objects[oldbox['uri'].split('nba-gestalt-data/',1)[1]]=b'{}';return
    if corrupt=='sidecar':store.objects[oldbox['metadata_uri'].split('nba-gestalt-data/',1)[1]]=b'{}';return
    if corrupt=='minutes':actual['home']['players'][0]['actual_minutes']=1
    elif corrupt=='identity':box['game']['gameId']='0022609999'
    elif corrupt=='status':box['game']['gameStatus']=2
    elif corrupt=='stat':box['game']['homeTeam']['players'][0]['statistics']['points']='unknown'
    else:box['game']['homeTeam']['players'][0]['statistics']['plusMinusPoints']=-11
    ref=save(f'lake/raw/cdn/boxscore/season=2026-27/game_id={gid}/',box)
    raw=store.objects[oldbox['metadata_uri'].split('nba-gestalt-data/',1)[1]]
    metakey=ref['uri'].split('nba-gestalt-data/',1)[1][:-5]+'.meta.json';store.objects[metakey]=raw
    ref.update(metadata_uri='s3://nba-gestalt-data/'+metakey,metadata_sha256=sha256(raw).hexdigest())
    actual['inputs']['box_score']=ref;new=save(prefix,actual)
    store.objects[prefix+'latest.json']=json_bytes({**pointer,**new})


@pytest.mark.parametrize('fault',['missing','hash','sidecar','minutes','identity','status','stat'])
def test_invalid_postgame_sources_fail_before_history_publication(saved_history,fault):
    job,a,store,_,_,_=saved_history;revise_game_box(saved_history,corrupt=fault)
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert not any(k.startswith(a.history_root) for k in store.objects)


def test_postgame_stat_correction_refreshes_history_without_changing_ratings(saved_history):
    job,a,store,_,refs,_=saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    index,_=a.history_selection(1);old=index['points'][0];oldbytes=store.objects[old['uri'].split('nba-gestalt-data/',1)[1]]
    revise_game_box(saved_history)
    assert job.execute_in_process(partition_key='2026-10-22').success
    corrected,_=a.history_selection(1)
    assert corrected['source_ratings']==refs[-1] and corrected['current']==index['current']
    assert corrected['points'][0]!=old and corrected['points'][1]==index['points'][1]
    row=load(store,corrected['points'][0]);assert row['games'][0]['stats']['plus_minus']==-11
    assert row['nail']==load(store,old)['nail'] and store.objects[old['uri'].split('nba-gestalt-data/',1)[1]]==oldbytes


def test_newer_unselected_box_and_forecast_do_not_invalidate_pinned_history(saved_history):
    job,a,store,_,_,_=saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    original={k:v for k,v in store.objects.items() if k.startswith(a.history_root)}
    for root in ['lake/raw/cdn/boxscore/season=2026-27/game_id=0022600001/',
                 'lake/artifacts/game_predictions/season=2026-27/model_version=v1/game_date_pacific=2026-10-20/game_id=0022600001/']:
        store.objects[root+'latest.json']=b'{"different_unselected_version":true}'
    assert job.execute_in_process(partition_key='2026-10-22').success
    assert {k:v for k,v in store.objects.items() if k.startswith(a.history_root)}==original


def test_current_off_day_change_does_not_create_a_game_point(saved_history):
    job,a,store,states,_,save=saved_history
    assert job.execute_in_process(partition_key='2026-10-22').success
    index,_=a.history_selection(1);points=index['points']
    states[-1]['players'][0]['nail']+=0.25;states[-1]['players'][0]['residual']+=0.25
    rewrite(a,store,states,save)
    assert job.execute_in_process(partition_key='2026-10-22').success
    new,_=a.history_selection(1)
    assert new['points']==points and new['current']['nail']!=index['current']['nail']
    assert not any('game_date_pacific=2026-10-22/' in key for key in store.objects if key.startswith(a.history_root))


def test_changed_postgame_selection_during_staging_does_not_move_indexes(saved_history):
    job,a,store,_,_,_=saved_history
    assert job.execute_in_process(partition_key='2026-10-20').success
    before=store.objects[a.history_prefix(1)+'latest.json']
    def change(kw):
        if kw['Key'].startswith(a.history_root) and not kw['Key'].endswith('latest.json'):
            revise_game_box(saved_history);store.hook=None
    store.hook=change
    assert not job.execute_in_process(partition_key='2026-10-22',raise_on_error=False).success
    assert store.objects[a.history_prefix(1)+'latest.json']==before


def test_one_daily_record_can_retain_multiple_games(saved_history):
    from gestalt_pipelines.ratings.utils import history_records
    from gestalt_pipelines.resources import NBAScheduleResource
    from gestalt_pipelines.predictions.resources import GamePredictionArchive
    _,a,_,_,_,_=saved_history
    chain=a.history_chain('2026-10-20',first_date='2026-10-20');state=chain[0][0]
    games,_=NBAScheduleResource(season='2026-27').games_for_date(FIRST)
    details,_=a.history_games(state,games,GamePredictionArchive(season='2026-27'))
    other=deepcopy(details[1][0]);other['game_id']='0022600099';details[1].append(other)
    records=history_records(chain,{'2026-10-20':details})
    assert len(records[1])==1 and len(records[1][0]['games'])==2
