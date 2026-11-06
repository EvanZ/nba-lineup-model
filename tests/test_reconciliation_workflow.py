from pathlib import Path
import json
import shutil
import pandas as pd
import pytest

from test_daily_model_replay import _prior_files
from test_environment_selection import FakeS3
from nba_lineup_model.players import reconciliation_workflow as workflow
from nba_lineup_model.players import reconciliation_artifact as artifacts
from nba_lineup_model.season.daily_model_replay import _load_frozen_priors


def fixture(tmp_path, monkeypatch, *, omit_state=None, boundary='2026-11-03'):
    import nba_lineup_model.players.in_season_rosters as original
    root=tmp_path/'release';root.mkdir()
    bp,bm=_prior_files(root)
    ranks,minutes,sha=_load_frozen_priors(bp,bm,season='2026-27')
    profiles=root/'profiles.parquet';ranks[['player_id','player_name']].to_parquet(profiles,index=False)
    accepted=tmp_path/'accepted';accepted.mkdir()
    state=ranks.loc[~ranks.player_id.eq(omit_state)].copy()
    state['team_at_cutoff']=state.team
    state.to_parquet(accepted/'ratings.parquet',index=False)
    minutes.loc[~minutes.player_id.eq(omit_state)].to_parquet(accepted/'minutes.parquet',index=False)
    manifest={'bundle_sha256':'accepted', 'identity':{'season':'2026-27','game_date_pacific':boundary,
        'model_provenance':{'prior_sha256':sha},'components':{'player_ratings':{'path':'ratings.parquet'},'minutes_state':{'path':'minutes.parquet'}}}}
    monkeypatch.setattr(workflow,'verify_daily_model_artifact',lambda _:manifest)
    def roster(request,*,season,destination):
        rows=[]
        for item in request['introductions']:
            rows.append({'player_id':item['player_id'],'player_name':'Test Player','team_abbreviation':item['team']})
        (destination/'evidence.json').write_text(json.dumps(rows))
        return pd.DataFrame(rows),[]
    monkeypatch.setattr(workflow,'manual_roster',roster)
    def history(release,destination,season):
        path=destination/f'data/curated/team_rosters/{season}/part-00000.parquet'
        path.parent.mkdir(parents=True);ranks.assign(team_abbreviation=ranks.team).to_parquet(path,index=False)
        return Path('profiles.parquet'),Path('rotations.parquet')
    monkeypatch.setattr(workflow,'pin_history',history)
    def extend(**kw):
        r=pd.read_parquet(kw['base_rankings_path']);p=pd.read_parquet(kw['base_profiles_path'])
        for row in kw['current_roster'].itertuples():
            if int(row.player_id) not in set(r.player_id):
                extra=r.iloc[[0]].copy();extra['player_id']=int(row.player_id);extra['team']=row.team_abbreviation
                r=pd.concat([r,extra],ignore_index=True)
                extra=p.iloc[[0]].copy();extra['player_id']=int(row.player_id)
                p=pd.concat([p,extra],ignore_index=True)
        r.to_parquet(kw['rankings_output_path'],index=False);p.to_parquet(kw['profiles_output_path'],index=False)
    monkeypatch.setattr(original,'_extend_preseason_catalogs',extend)
    def extend_minutes(**kw):
        obj=json.loads(Path(kw['base_cache_path']).read_text());rows=obj['minutes']['players']
        for player in kw['rankings'].player_id:
            if int(player) not in {int(r['player_id']) for r in rows}:
                rows.append({**rows[0],'player_id':int(player)})
        Path(kw['output_path']).write_text(json.dumps(obj))
    monkeypatch.setattr(original,'_extend_runtime_minutes_catalog',extend_minutes)
    return dict(release_root=root,state_root=tmp_path/'state',base_rankings_path=bp,
        base_profiles_path=profiles,base_minutes_path=bm,accepted_model_path=accepted,
        environment='dev',cache_root=tmp_path/'cache',lake_uri='s3://nba-gestalt-data/lake',
        producer_revision='test',client=FakeS3())


def request(*ids, team='HOM', effective='2026-11-04'):
    return {'mode':'manual','introductions':[{'player_id':p,'team':team,'effective_date':effective,
        'discovered_at':'2026-11-06T21:00:00Z','authorization':'test approved introduction'} for p in ids]}


def test_successive_additions_and_no_change_preserve_all_initialization(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    first=workflow.run(**args,request=request(999901))
    second=workflow.run(**args,request=request(999902))
    assert first['generation']==1 and second['generation']==2
    manifest=artifacts.verify(Path(second['path']))
    assert [e['player_id'] for e in manifest['identity']['introductions']]==[999901,999902]
    again=workflow.run(**args,request=request(999901,999902))
    assert again['status']=='unchanged' and again['generation']==2
    workflow.require_same_rows(pd.read_parquet(args['base_rankings_path']),
        pd.read_parquet(Path(second['path'])/'rankings.parquet'),'incumbents')


def test_catalog_known_but_state_missing_gets_recorded_without_changing_its_prior(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch,omit_state=1631104)
    result=workflow.run(**args,request=request(1631104))
    manifest=artifacts.verify(Path(result['path']))
    assert manifest['identity']['introductions'][0]['player_id']==1631104
    assert len(pd.read_parquet(Path(result['path'])/'rankings.parquet'))==len(pd.read_parquet(args['base_rankings_path']))


def test_team_change_is_explicit_and_does_not_move_selection(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    existing=int(pd.read_parquet(args['base_rankings_path']).iloc[0].player_id)
    with pytest.raises(ValueError,match='Unsupported team changes'):
        workflow.run(**args,request=request(existing,team='DIFFERENT'))
    assert artifacts.selected(environment='dev',cache_root=args['cache_root'],lake_uri=args['lake_uri'],client=args['client']) is None


def test_late_historical_introduction_is_archived_without_selection_until_approved(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch,boundary='2026-11-05')
    req=request(999901)
    pending=workflow.run(**args,request=req)
    assert pending['status']=='awaiting_historical_correction_approval'
    assert artifacts.selected(environment='dev',cache_root=args['cache_root'],lake_uri=args['lake_uri'],client=args['client']) is None
    approval={k:pending['historical_impact'][k] for k in ['from_date','through_date','accepted_model_sha256']}
    accepted=workflow.run(**args,request={**req,'historical_correction_approval':{**approval,'decision_reference':'explicit test approval'}})
    assert accepted['generation']==1
    assert pending['reference']!=accepted['reference']
    assert artifacts.verify(Path(pending['path']))['identity']['historical_impact'].get('selection_approval') is None


def test_two_introductions_share_one_canonically_published_official_dependency(tmp_path,monkeypatch):
    from datetime import date
    from test_daily_artifacts import _publish_data
    from nba_lineup_model.season.durable_artifact_storage import daily_data_reference
    args=fixture(tmp_path,monkeypatch)
    data=_publish_data(tmp_path/'official',game_date=date(2026,11,4),game_ids=['0022600180'])
    ref=daily_data_reference(data.path)
    original=workflow.manual_roster
    def source(*a,**kw):
        observed,_=original(*a,**kw)
        return observed,[ref.as_dict(),ref.as_dict()]
    monkeypatch.setattr(workflow,'manual_roster',source)
    req=request(999901,999902)
    for item in req['introductions']:
        item['membership']={'bundle_path':str(data.path),'sha256':ref.content_id.split('=',1)[1]}
    result=workflow.run(**args,request=req)
    manifest=artifacts.verify(Path(result['path']))
    assert manifest['identity']['dependencies']==[ref.as_dict()]
    assert len(manifest['identity']['introductions'])==2
