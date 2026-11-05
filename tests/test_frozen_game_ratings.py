import copy
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_daily_artifacts import _publish_data, _components, _sha, MODEL_COMPONENTS
from test_official_schedule_runtime import GAME_DATE
from test_release_bound_serving import selection, write

from nba_lineup_model.season.daily_model_artifact import publish_daily_model_artifact
from nba_lineup_model.season.frozen_game_ratings import load_frozen_game_ratings
from nba_lineup_model.season.serving_selection import (
    ObservationSnapshot, read_selected_day, frozen_rating_context,
)


@pytest.fixture
def chain(tmp_path):
    models = []
    for day, rating in ((20, -2.0), (21, 8.0)):
        data = _publish_data(tmp_path, game_date=date(2026, 10, day),
                             game_ids=[f"00226000{day}"], suffix=str(day))
        components = _components(tmp_path / str(day), MODEL_COMPONENTS, 'model')
        path = tmp_path / str(day) / 'ratings.parquet'
        pq.write_table(pa.table({'player_id': [999], 'updated_nail': [rating]}), path)
        components['player_ratings'] = path
        model = publish_daily_model_artifact(root=tmp_path/'models', data_bundle=data.path,
            parent_bundle=models[-1].path if models else None, components=components,
            model_provenance={'code_revision': 'abcdef', 'model_version': 'test',
                'configuration_sha256': _sha('config'), 'prior_sha256': _sha('prior')})
        models.append(model)
    selected = {'start_time': '2026-10-22T00:00:00Z', 'projection': {'input_provenance': {
        'rating_source_uri': f"artifact://daily-model/{models[0].manifest['bundle_sha256']}/player_ratings",
        'rating_as_of': '2026-10-20'}}}
    kwargs = dict(model_bundle=models[-1].path, model_sha256=models[-1].manifest['bundle_sha256'],
                  preseason_rankings=tmp_path/'unused')
    return models, selected, kwargs


def test_older_game_uses_its_exact_model_after_newer_update(chain):
    models, selected, kwargs = chain
    assert load_frozen_game_ratings(selected, **kwargs) == {999: -2.0}


def test_preseason_source_is_exact_and_checksum_verified(chain, tmp_path):
    from nba_lineup_model.season.daily_artifact_common import file_sha256
    _, selected, kwargs = chain
    path = tmp_path/'preseason.parquet'
    pq.write_table(pa.table({'player_id': [999], 'rapm': [-3.0]}), path)
    selected['projection']['input_provenance']['rating_source_uri'] = (
        f'artifact://preseason-rankings/{file_sha256(path)}')
    kwargs['preseason_rankings'] = path
    assert load_frozen_game_ratings(selected, **kwargs) == {999: -3.0}
    pq.write_table(pa.table({'player_id': [999], 'rapm': [4.0]}), path)
    with pytest.raises(ValueError, match='exact preseason'):
        load_frozen_game_ratings(selected, **kwargs)


@pytest.mark.parametrize('case', ['missing', 'future', 'wrong_date', 'tampered'])
def test_no_latest_rating_fallback(chain, case):
    models, selected, kwargs = chain
    p = selected['projection']['input_provenance']
    if case == 'missing':
        p['rating_source_uri'] = f"artifact://daily-model/{'f'*64}/player_ratings"
    elif case == 'future':
        p['rating_as_of'] = '2026-10-21'
    elif case == 'wrong_date':
        p['rating_as_of'] = '2026-10-19'
    else:
        component = models[0].manifest['identity']['components']['player_ratings']
        (models[0].path/component['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        load_frozen_game_ratings(selected, **kwargs)


@pytest.mark.parametrize('has_rating', [True, False])
def test_unprojected_player_is_in_official_actuals_and_never_dropped(tmp_path, monkeypatch, has_rating):
    root, board = selection(tmp_path, monkeypatch)
    game = board['games'][0]; gid = game['game_id']
    live = copy.deepcopy(board)
    live['games'][0].update(home_score=110, away_score=100,
                          status={'state': 'post', 'completed': True})
    observations = tmp_path/'observations'
    write(observations/'latest'/f'{GAME_DATE}.json', live)
    box = {'game': {'gameId': gid}}
    for side, ids in (('home', (4, 5, 999)), ('away', (1, 2, 3))):
        box['game'][side+'Team'] = {'teamTricode': game[side+'_team'], 'players': [
            {'personId': pid, 'firstName': 'Player', 'familyName': str(pid),
             'statistics': {'minutes': minutes}}
            for pid, minutes in zip(ids, ('PT100M', 'PT80M', 'PT60M'))]}
    box_path = observations/'final'/f'{gid}.nba.json'
    write(box_path, box)
    original_box = box_path.read_bytes()
    token = frozen_rating_context.set(lambda selected: {999: -2.0} if has_rating else {})
    try:
        result = read_selected_day(GAME_DATE, root=root,
            observations=ObservationSnapshot.capture(observations, [GAME_DATE]))['games'][0]
    finally:
        frozen_rating_context.reset(token)
    assert box_path.read_bytes() == original_box
    assert result['projection'] == game['projection']
    assert result['final_audit']['final']['home_score'] == 110
    if has_rating:
        actual = result['retrospective']['actual_minutes']['home']
        assert {p['player_id'] for p in actual} == {4, 5, 999}
        assert next(p for p in actual if p['player_id']==999)['actual_minutes'] == 60
        assert result['retrospective']['ledger']['home']['actual_minutes'] == 240
        assert not result.get('data_quality_errors')
    else:
        assert 'retrospective' not in result
        assert any('999' in e and 'rating-before' in e for e in result['data_quality_errors'])
