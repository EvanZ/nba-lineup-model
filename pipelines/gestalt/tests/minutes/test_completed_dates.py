"""Completed-date identity, cutoff checks and corresponding forecast dependencies."""
from copy import deepcopy
from datetime import datetime,timedelta
from hashlib import sha256

import pytest
from dagster import AssetKey
from gestalt_pipelines.minutes.assets import player_minutes
from gestalt_pipelines.minutes.partitions import minutes_dates
from gestalt_pipelines.minutes.utils import MinutesError,build_game_projection,json_bytes,update_state,validate_state
from gestalt_pipelines.utils import PACIFIC
from .helpers import FIRST,FIRST_STATE,SEASON,priors,slate
from .test_asset import minutes_job
from .test_projection import example


def test_completed_range_starts_with_opening_games_and_stops_at_yesterday():
    keys=minutes_dates.get_partition_keys(current_time=datetime(2026,11,14,12,tzinfo=PACIFIC))
    assert keys[0]=='2026-10-20' and keys[-1]=='2026-11-13'


@pytest.mark.parametrize('upstream',['nba_stints','rosters_by_date','injury_status_by_game_date'])
def test_completed_observation_dependencies_use_same_date(upstream):
    assert player_minutes.get_partition_mapping(AssetKey(upstream)) is None


def test_first_updated_state_includes_opening_games_and_preserves_introduction_day():
    record=update_state(priors=priors(),parent=None,**slate())
    assert record['game_date_pacific']==record['observed_through_game_date_pacific']==str(FIRST)
    assert record['effective_from_game_date_pacific']==str(FIRST+timedelta(days=1))
    assert record['included_game_ids']==['0022600001']
    assert record['players']['1']['introduced_on']==str(FIRST)


@pytest.mark.parametrize('source',['rosters','stints','injuries'])
def test_updater_rejects_another_dates_observations(source):
    data=slate();data[source]['game_date_pacific']=str(FIRST+timedelta(days=1))
    with pytest.raises(MinutesError):update_state(priors=priors(),parent=None,**data)


def test_old_effective_date_and_future_history_are_rejected():
    state=update_state(priors=priors(),parent=None,**slate())
    old=deepcopy(state);old.pop('date_contract')
    with pytest.raises(MinutesError,match='completed-game-date'):validate_state(old,SEASON)
    state['players']['1']['history'][0]['game_date_pacific']=str(FIRST+timedelta(days=1))
    with pytest.raises(MinutesError,match='Future or unprocessed'):validate_state(state,SEASON)


def test_replay_can_replace_old_selection_without_deleting_original(minutes_job):
    job,archive,objects,_,_=minutes_job
    assert job.execute_in_process(partition_key=str(FIRST)).success
    state,_,_=archive.state(str(FIRST));state.pop('date_contract')
    raw=json_bytes(state);digest=sha256(raw).hexdigest();prefix=archive.prefix+f'game_date_pacific={FIRST}/'
    objects[prefix+digest+'.json']=raw
    objects[prefix+'latest.json']=json_bytes({'schema_version':1,'season':SEASON,'model_version':'v1','game_date_pacific':str(FIRST),
        'uri':'s3://nba-gestalt-data/'+prefix+digest+'.json','sha256':digest})
    with pytest.raises(MinutesError,match='completed-game-date'):archive.state(str(FIRST))
    assert job.execute_in_process(partition_key=str(FIRST)).success
    assert archive.state(str(FIRST))[0]['observed_through_game_date_pacific']==str(FIRST)
    assert objects[prefix+digest+'.json']==raw


def test_forecast_accepts_previous_completed_state_and_rejects_same_day():
    data=example();record=build_game_projection(**data)
    assert data['baseline']['game_date_pacific']==str(FIRST)
    assert record['baseline_through_date']==str(FIRST)
    data['baseline']['game_date_pacific']=data['day']
    data['baseline']['observed_through_game_date_pacific']=data['day']
    with pytest.raises(MinutesError):build_game_projection(**data)


def test_old_effective_date_pointer_waits_for_completed_state_without_rewriting_it(minutes_job):
    job,archive,objects,_,_=minutes_job
    assert job.execute_in_process(partition_key=str(FIRST)).success
    old=deepcopy(archive.state(str(FIRST))[0]);day=str(FIRST+timedelta(days=1))
    old['game_date_pacific']=day
    del old['date_contract'];del old['effective_from_game_date_pacific']
    raw=json_bytes(old);digest=sha256(raw).hexdigest();prefix=archive.prefix+f'game_date_pacific={day}/'
    objects[prefix+digest+'.json']=raw
    pointer=json_bytes({'schema_version':1,'season':SEASON,'model_version':'v1','game_date_pacific':day,
        'uri':'s3://nba-gestalt-data/'+prefix+digest+'.json','sha256':digest})
    objects[prefix+'latest.json']=pointer
    assert archive.state(day)==(None,None,None)
    tomorrow=FIRST+timedelta(days=2)
    with pytest.raises(FileNotFoundError,match='No accepted completed-date minutes'):
        archive.projection_selections(str(tomorrow),today=tomorrow,roster_archive=object())
    assert objects[prefix+digest+'.json']==raw
    assert objects[prefix+'latest.json']==pointer
