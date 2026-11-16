"""Real sensor/schedule evaluation against isolated Dagster history and saved fixtures."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
from dagster import AssetKey, DagsterInstance, DagsterRunStatus, build_sensor_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.minutes.resources import MinutesArchive
from gestalt_pipelines.predictions.sensors import margins
from gestalt_pipelines.ratings.resources import RatingsArchive
from gestalt_pipelines.resources import NBAScheduleResource
from .conftest import DAY, NOW
from .test_job import job


def tick(instance, store):
    with build_sensor_context(instance=instance, repository_def=defs.get_repository_def(), resources={
        'nba_schedule': NBAScheduleResource(season='2026-27'), 'minutes_archive': MinutesArchive(season='2026-27'),
        'ratings_archive': RatingsArchive(season='2026-27'), 'game_prediction_archive': store,
    }) as context:
        return defs.get_sensor_def('predict_game_margins_sensor').evaluate_tick(context)


def test_seed_then_skip_unchanged_without_reading_large_ratings(job, archive, clock, monkeypatch):
    executable,_=job;store,s3=archive
    monkeypatch.setattr(margins,'datetime',clock)
    with DagsterInstance.ephemeral() as instance:
        first=tick(instance,store)
        assert len(first.run_requests)==1 and first.run_requests[0].partition_key==DAY
        assert tick(instance,store).run_requests[0].run_key==first.run_requests[0].run_key
        assert executable.execute_in_process(instance=instance,partition_key=DAY).success
        before=dict(s3.objects)
        monkeypatch.setattr(RatingsArchive,'state',lambda *a:pytest.fail('Unchanged inputs must not read the large ratings state'))
        assert not tick(instance,store).run_requests
        assert s3.objects==before


@pytest.mark.parametrize('source',['game_minutes','player_ratings','schedule'])
def test_changes_to_each_accepted_source_request_new_margin(job,archive,data,clock,monkeypatch,source):
    executable,_=job;store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    with DagsterInstance.ephemeral() as instance:
        first=tick(instance,store).run_requests[0]
        assert executable.execute_in_process(instance=instance,partition_key=DAY).success
        assert not tick(instance,store).run_requests
        data['inputs'][source]={'uri':f's3://nba-gestalt-data/test/new-{source}.json','sha256':'e'*64}
        changed=tick(instance,store).run_requests
        assert len(changed)==1 and changed[0].partition_key==DAY
        assert changed[0].run_key!=first.run_key
        assert tick(instance,store).run_requests[0].run_key==changed[0].run_key


@pytest.mark.parametrize('missing',['allocation','ratings_selection','completed_ratings'])
def test_missing_inputs_wait_without_requesting_a_failing_run(job,archive,clock,monkeypatch,missing):
    store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    if missing=='allocation':monkeypatch.setattr(MinutesArchive,'game_selection',lambda *a:(None,None))
    elif missing=='ratings_selection':monkeypatch.setattr(RatingsArchive,'state_selection',lambda *a:(None,None))
    else:monkeypatch.setattr(RatingsArchive,'state',lambda *a:(None,None,None))
    with DagsterInstance.ephemeral() as instance:
        result=tick(instance,store)
        assert not result.run_requests and 'Waiting' in result.skip_message


def test_current_and_tomorrow_use_one_completed_rating_check(job,archive,data,clock,monkeypatch):
    store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    tomorrow={**data['game'],'game_id':'0022600003','game_time_utc':data['game']['game_time_utc']+timedelta(days=1)}
    monkeypatch.setattr(NBAScheduleResource,'games_between',lambda *a,**kw:([data['game'],tomorrow],data['inputs']['schedule']))
    original=RatingsArchive.state;seen=[]
    def read(self,day):seen.append(day);return original(self,day)
    monkeypatch.setattr(RatingsArchive,'state',read)
    with DagsterInstance.ephemeral() as instance:
        requests=tick(instance,store).run_requests
        assert [r.partition_key for r in requests]==['2026-10-21','2026-10-22']
        assert seen==['2026-10-20']


def test_scheduled_tipoff_and_historical_games_do_not_trigger(job,archive,data,clock,monkeypatch):
    store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    with DagsterInstance.ephemeral() as instance:
        assert tick(instance,store).run_requests
        clock.value=data['game']['game_time_utc']
        assert not tick(instance,store).run_requests
        clock.value+=timedelta(days=2)
        assert not tick(instance,store).run_requests


@pytest.mark.parametrize('kind',['named_job','asset_backfill'])
def test_queued_current_date_suppresses_duplicate_but_old_dates_do_not(job,archive,clock,monkeypatch,kind):
    store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    with DagsterInstance.ephemeral() as instance:
        if kind=='named_job':
            active=SimpleNamespace(job_name='predict_game_margins',asset_selection=None,tags={'dagster/partition':DAY})
        else:
            active=SimpleNamespace(job_name='__ASSET_JOB',asset_selection={AssetKey('game_predictions')},tags={
                'dagster/asset_partition_range_start':DAY,'dagster/asset_partition_range_end':DAY})
        monkeypatch.setattr(instance,'get_runs',lambda **kw:[active])
        assert not tick(instance,store).run_requests
        active.tags={'dagster/partition':'2026-10-20'}
        assert tick(instance,store).run_requests


def test_fallback_priors_do_not_cause_repeated_requests(job,archive,data,clock,monkeypatch):
    executable,_=job;store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    missing=data['ratings']['players'].pop(0)
    data['priors']={'players':[{'player_id':missing['player_id'],'rapm':missing['nail']}]}
    with DagsterInstance.ephemeral() as instance:
        assert executable.execute_in_process(instance=instance,partition_key=DAY).success
        selected,_=store.selection(DAY,data['game']['game_id'])
        assert 'rating_priors' in store.load(selected)['inputs']
        monkeypatch.setattr(RatingsArchive,'state',lambda *a:pytest.fail('Unchanged fallback must not read learned state'))
        assert not tick(instance,store).run_requests


def test_opening_day_uses_real_priors_without_a_fake_daily_state(job,archive,data,clock,monkeypatch):
    store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    clock.value=NOW-timedelta(days=1)
    data['priors']={'players':[]}
    monkeypatch.setattr(RatingsArchive,'state',lambda *a:pytest.fail('Opening day must use frozen priors'))
    with DagsterInstance.ephemeral() as instance:
        assert [r.partition_key for r in tick(instance,store).run_requests]==['2026-10-20','2026-10-21']


def test_missing_one_allocation_waits_for_the_entire_date(job,archive,data,clock,monkeypatch):
    store,_=archive;monkeypatch.setattr(margins,'datetime',clock)
    second={**data['game'],'game_id':'0022600003'}
    monkeypatch.setattr(NBAScheduleResource,'games_between',lambda *a,**kw:([data['game'],second],data['inputs']['schedule']))
    original=MinutesArchive.game_selection
    monkeypatch.setattr(MinutesArchive,'game_selection',lambda self,day,gid:
        (None,None) if gid==second['game_id'] else original(self,day,gid))
    with DagsterInstance.ephemeral() as instance:
        result=tick(instance,store)
        assert not result.run_requests and second['game_id'] in result.skip_message
