import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_daily_model_replay import _prior_files, _publish_data, _zero_context
from nba_lineup_model.players.reconciliation_artifact import seal
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.daily_model_replay import (
    _load_frozen_priors, replay_daily_model_chain, HistoricalModelReplayApprovalRequired,
)
from nba_lineup_model.season.reconciliation_inputs import catalog,extend_accumulators,marker


def catalog_fixture(tmp_path, *, effective='2026-10-20', rating_delta=0):
    full=tmp_path/'full';full.mkdir();rp,mp=_prior_files(full)
    base=tmp_path/'base';base.mkdir();bp,bm=_prior_files(base,omit_player=1631104)
    _,_,sha=_load_frozen_priors(bp,bm,season='2026-27')
    root=tmp_path/'reconciliation';root.mkdir()
    ranks=pd.read_parquet(rp);ranks.loc[ranks.player_id.eq(1631104),'rapm']+=rating_delta
    ranks.to_parquet(root/'rankings.parquet',index=False)
    ranks[['player_id','player_name']].to_parquet(root/'profiles.parquet',index=False)
    (root/'minutes.json').write_bytes(mp.read_bytes())
    intro=root/'introductions';intro.mkdir()
    ranks.loc[ranks.player_id.eq(1631104)].to_parquet(intro/'ratings.parquet',index=False)
    rows=[x for x in json.loads(mp.read_text())['minutes']['players'] if x['player_id']==1631104]
    (intro/'minutes.json').write_text(json.dumps({'minutes':{'players':rows}}))
    ranks.loc[ranks.player_id.eq(1631104),['player_id','player_name']].to_parquet(intro/'profiles.parquet',index=False)
    event={'player_id':1631104,'team':'HOM','effective_date':effective,
           'discovered_at':'2026-11-06T21:00:00Z','evidence':{'fixture':'official roster'},
           'authorization':'test late introduction', 'initialization':{
               'rankings':'introductions/ratings.parquet','minutes':'introductions/minutes.json',
               'profiles':'introductions/profiles.parquet'}}
    seal(root,{'season':'2026-27','effective_through':effective,'base_prior_sha256':sha,
        'parent_reference':None,'introductions':[event],'dependencies':[]})
    return root,bp,bm,rp,mp


def test_late_discovery_applies_only_from_effective_date_and_needs_no_live_source(tmp_path,monkeypatch):
    root,bp,bm,_,_=catalog_fixture(tmp_path)
    import nba_lineup_model.players.in_season_rosters as source
    monkeypatch.setattr(source,'fetch_current_nba_rosters',lambda:pytest.fail('Replay attempted live discovery'))
    ranks,minutes,sha=_load_frozen_priors(bp,bm,season='2026-27')
    before,_=catalog(root,date(2026,10,19),ranks,minutes,base_prior_sha256=sha)
    after,_=catalog(root,date(2026,10,20),ranks,minutes,base_prior_sha256=sha)
    assert 1631104 not in set(before.player_id)
    assert 1631104 in set(after.player_id)
    assert marker(root,date(2026,10,19))['player_ids']==[]


def test_daily_processing_consumes_saved_initialization_and_duplicate_rerun_is_noop(tmp_path,monkeypatch):
    root,bp,bm,rp,mp=catalog_fixture(tmp_path)
    _publish_data(tmp_path)
    import nba_lineup_model.players.in_season_rosters as source
    monkeypatch.setattr(source,'fetch_current_nba_rosters',lambda:pytest.fail('Replay attempted live discovery'))
    common=dict(start_date=date(2026,10,20),end_date=date(2026,10,20),
        data_root=tmp_path/'data',code_revision='test-revision',context_provider=_zero_context)
    repaired=replay_daily_model_chain(**common,preseason_rankings_path=bp,preseason_minutes_path=bm,
        output_root=tmp_path/'model',reconciliation_root=root)
    control=replay_daily_model_chain(**common,preseason_rankings_path=rp,preseason_minutes_path=mp,
        output_root=tmp_path/'control')
    for name in ['player_ratings','minutes_state','rating_trajectory','game_projections']:
        def frame(artifact):
            m=verify_daily_model_artifact(artifact.path)
            return pd.read_parquet(artifact.path/m['identity']['components'][name]['path'])
        pd.testing.assert_frame_equal(frame(repaired[0]),frame(control[0]),check_exact=True)
    assert replay_daily_model_chain(**common,preseason_rankings_path=bp,preseason_minutes_path=bm,
        output_root=tmp_path/'model',reconciliation_root=root)==[]
    changed=tmp_path/'changed';changed.mkdir()
    changed_root,*_=catalog_fixture(changed,rating_delta=1)
    with pytest.raises(HistoricalModelReplayApprovalRequired):
        replay_daily_model_chain(**common,preseason_rankings_path=bp,preseason_minutes_path=bm,
            output_root=tmp_path/'model',reconciliation_root=changed_root)


@pytest.mark.parametrize('joint',[False,True])
def test_identity_remap_preserves_all_evidence_and_home_court_cross_terms(joint):
    size=3+int(joint);dense=np.arange(size*size,dtype=float).reshape(size,size)
    score=np.arange(size,dtype=float);exposure=score+100
    gram,new_score,new_exposure=extend_accumulators([20,40,60],[10,20,30,40,60,70],
        sparse.csr_matrix(dense),score,exposure,joint_home_court=joint)
    mapping=[1,3,4]+([6] if joint else [])
    np.testing.assert_array_equal(gram.toarray()[np.ix_(mapping,mapping)],dense)
    np.testing.assert_array_equal(new_score[mapping],score)
    np.testing.assert_array_equal(new_exposure[mapping],exposure)
    for i in [0,2,5]:
        assert gram.getrow(i).nnz==gram.getcol(i).nnz==0
        assert new_score[i]==new_exposure[i]==0


def test_historical_correction_reports_exact_scope_and_requires_matching_decision():
    from nba_lineup_model.players.reconciliation_workflow import historical_impact
    model = {'bundle_sha256': 'accepted-model', 'identity': {'game_date_pacific': '2026-11-05'}}
    events = {1631207: {'effective_date': '2026-11-04'}}
    impact = historical_impact(events, [1631207], accepted_model=model, request={})
    assert impact['from_date'] == '2026-11-04'
    assert impact['through_date'] == '2026-11-05'
    assert 'selection_approval' not in impact
    approval = {k: impact[k] for k in ('from_date', 'through_date', 'accepted_model_sha256')}
    approval['decision_reference'] = 'explicit test approval'
    assert historical_impact(events, [1631207], accepted_model=model,
        request={'historical_correction_approval': approval})['selection_approval'] == approval
    with pytest.raises(ValueError, match='does not match'):
        historical_impact(events, [1631207], accepted_model=model,
            request={'historical_correction_approval': {**approval, 'from_date': '2026-11-05'}})
    assert historical_impact(events, [1631207],
        accepted_model={**model, 'identity': {'game_date_pacific': '2026-11-03'}}, request={}) is None


def test_catalog_known_initialization_is_deferred_until_saved_introduction_date(tmp_path):
    from nba_lineup_model.players.reconciliation_artifact import read_json, seal
    import shutil
    root,bp,bm,rp,mp=catalog_fixture(tmp_path,effective='2026-11-04')
    ranks,minutes,sha=_load_frozen_priors(rp,mp,season='2026-27')
    # Same existing prior, but the selected history says adoption occurs Nov 4.
    corrected=tmp_path/'known-introduction';shutil.copytree(root,corrected)
    identity=read_json(corrected/'manifest.json')['identity']
    (corrected/'manifest.json').unlink();identity.pop('files');identity['base_prior_sha256']=sha
    seal(corrected,identity)
    before,_=catalog(corrected,date(2026,11,3),ranks,minutes,base_prior_sha256=sha)
    after,_=catalog(corrected,date(2026,11,4),ranks,minutes,base_prior_sha256=sha)
    assert 1631104 not in set(before.player_id)
    pd.testing.assert_frame_equal(after,ranks.sort_values('player_id').reset_index(drop=True),check_exact=True)
