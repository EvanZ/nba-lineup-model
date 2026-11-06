from datetime import date, datetime, UTC
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_reconciliation_replay import catalog_fixture
from nba_lineup_model.season.daily_model_replay import _load_frozen_priors, _solve_rating_state, DailyModelReplayConfig
from nba_lineup_model.modeling.in_season_minutes import initialize_minutes_state
from nba_lineup_model.season.forecast_initialization import initialize_forecast_players


def inputs(tmp_path):
    root, bp, bm, _, _ = catalog_fixture(tmp_path)
    ranks, priors, sha = _load_frozen_priors(bp, bm, season='2026-27')
    config = DailyModelReplayConfig()
    ids = tuple(ranks.player_id.astype(int)); n = len(ids)
    cutoff = datetime(2026, 10, 20, tzinfo=UTC)
    ratings = _solve_rating_state(rankings=ranks, gram=sparse.eye(n, format='csr'),
        score=np.ones(n), exposure=np.ones(n), player_ids=ids,
        preseason=ranks.set_index('player_id').rapm.to_dict(), team_by_player=ranks.set_index('player_id').team.to_dict(),
        included_game_ids=['test-game'], contribution_lineage=['test-contribution'],
        cutoff_time=cutoff, cutoff_game_id='test-game', config=config)
    minutes = initialize_minutes_state(priors, season='2026-27', config=config.minutes_config())
    return dict(rankings_state=ratings, minutes_state=minutes, reconciliation_root=root,
        through=date(2026, 11, 6), available_at=datetime(2026, 11, 6, 22, tzinfo=UTC),
        base_rankings=ranks, base_minutes=priors, base_prior_sha256=sha, replay_config=config)


def test_forecast_adds_saved_player_without_changing_incumbent_evidence(tmp_path):
    args = inputs(tmp_path)
    ratings, minutes, selection = initialize_forecast_players(**args)
    for key, result in [('rankings_state', ratings), ('minutes_state', minutes)]:
        old = args[key]
        pd.testing.assert_frame_equal(old.sort_values('player_id').reset_index(drop=True),
            result.loc[result.player_id.isin(old.player_id), old.columns].reset_index(drop=True),
            check_exact=True, check_dtype=False)
    row = ratings.loc[ratings.player_id.eq(1631104)].iloc[0]
    assert row.updated_nail == row.frozen_preseason_nail
    assert row.raw_on_court_possessions == row.effective_information == row.current_season_residual == 0
    assert minutes.loc[minutes.player_id.eq(1631104), 'conditional_observations'].item() == 0
    assert selection['forecast_only_player_ids'] == [1631104]
    again = initialize_forecast_players(**{**args, 'rankings_state': ratings, 'minutes_state': minutes})
    pd.testing.assert_frame_equal(ratings, again[0], check_exact=True)
    pd.testing.assert_frame_equal(minutes, again[1], check_exact=True)
    assert again[2]['forecast_only_player_ids'] == []


def test_late_discovery_cannot_enter_forecast_with_earlier_information_cutoff(tmp_path):
    args = inputs(tmp_path)
    with pytest.raises(ValueError, match='discovered after'):
        initialize_forecast_players(**{**args, 'available_at': datetime(2026, 11, 4, tzinfo=UTC)})


def test_materialized_forecast_serving_keeps_accepted_evidence_and_validates_publication(tmp_path):
    from test_daily_model_replay import _prior_files, _publish_data, _zero_context
    from nba_lineup_model.season.daily_model_replay import replay_daily_model_chain
    from nba_lineup_model.players.reconciliation_artifact import seal
    from nba_lineup_model.modeling.current_season_serving_materialization import materialize_daily_model_serving
    from nba_lineup_model.season.forecast_initialization import validate_publication
    from nba_lineup_model.season.current_serving import resolve_latest_bundle, verify_serving_bundle
    import shutil
    priors=tmp_path/'priors';priors.mkdir();rp,mp=_prior_files(priors)
    ranks,minutes,sha=_load_frozen_priors(rp,mp,season='2026-27')
    _publish_data(tmp_path)
    replay_daily_model_chain(start_date=date(2026,10,20),end_date=date(2026,10,20),
        data_root=tmp_path/'data',output_root=tmp_path/'model',preseason_rankings_path=rp,
        preseason_minutes_path=mp,code_revision='test',context_provider=_zero_context)
    release=tmp_path/'release';release.mkdir();(release/'release.json').write_text(json.dumps({'run_id':'test'}))
    base=tmp_path/'base';(base/'preseason').mkdir(parents=True)
    shutil.copyfile(rp,base/'preseason/rankings.parquet')
    ranks[['player_id','player_name']].to_parquet(base/'preseason/profiles.parquet',index=False)
    common=dict(daily_model_root=tmp_path/'model',daily_data_root=tmp_path/'data',release_root=release)
    materialize_daily_model_serving(**common,output_root=base/'current-season-serving')
    root=tmp_path/'introductions';(root/'new').mkdir(parents=True)
    new=ranks.iloc[[0]].copy();new['player_id']=999901
    combined=pd.concat([ranks,new],ignore_index=True)
    combined.to_parquet(root/'rankings.parquet',index=False)
    combined[['player_id','player_name']].to_parquet(root/'profiles.parquet',index=False)
    new.to_parquet(root/'new/rankings.parquet',index=False)
    new[['player_id','player_name']].to_parquet(root/'new/profiles.parquet',index=False)
    payload=json.loads(mp.read_text());row={**payload['minutes']['players'][0],'player_id':999901}
    payload['minutes']['players'].append(row)
    (root/'minutes.json').write_text(json.dumps(payload))
    (root/'new/minutes.json').write_text(json.dumps({'minutes':{'players':[row]}}))
    event={'player_id':999901,'team':str(new.iloc[0].team),'effective_date':'2026-11-06',
        'discovered_at':'2026-11-06T21:00:00Z','evidence':{'test':'official'},'authorization':'test',
        'initialization':{'rankings':'new/rankings.parquet','minutes':'new/minutes.json','profiles':'new/profiles.parquet'}}
    seal(root,{'season':'2026-27','effective_through':'2026-11-06','base_prior_sha256':sha,
        'parent_reference':None,'introductions':[event],'dependencies':[]})
    initialized=tmp_path/'initialized'
    materialize_daily_model_serving(**common,output_root=initialized,forecast_initialization=dict(
        reconciliation_root=root,through=date(2026,11,6),available_at=datetime(2026,11,6,21,tzinfo=UTC),
        base_rankings=ranks,base_minutes=minutes,base_prior_sha256=sha,replay_config=DailyModelReplayConfig()))
    result=validate_publication(base=base,serving_root=initialized,rankings_path=root/'rankings.parquet',
        profiles_path=root/'profiles.parquet',reconciliation_root=root)
    assert result['forecast_only_player_ids']==[999901]
    manifest=verify_serving_bundle(resolve_latest_bundle(initialized))
    assert manifest['identity']['evidence_through_date']=='2026-10-20'
    assert manifest['identity']['coverage']['serving_player_count']==len(ranks)+1
