from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from dagster import DagsterInstance, DagsterRunStatus, Definitions, build_sensor_context
from dagster._core.remote_origin import RemoteJobOrigin, RemoteRepositoryOrigin, RegisteredCodeLocationOrigin

from nba_lineup_model.orchestration import backfill
from nba_lineup_model.orchestration import definitions as orchestration
from nba_lineup_model.season import backfill_recovery as recovery
from nba_lineup_model.season.dev_site_release import _tree_sha256
from test_daily_pipeline import config_fixture
from test_official_schedule_runtime import _board, GAME_ID, GAME_DATE


def fixture(tmp_path, count=1):
    root = tmp_path / 'schedule'
    (root / 'latest').mkdir(parents=True)
    (root / 'frozen').mkdir()
    board = _board(collected_at=datetime(2026, 11, 1, 1, 55, tzinfo=UTC))
    template = board['games'][0]
    games, boxes, rows = [], {}, []
    for i in range(count):
        game_id = f'00226001{i:02d}'
        game = json.loads(json.dumps(template).replace(GAME_ID, game_id))
        game['start_time'] = '2026-11-02T00:00:00Z'
        games.append(game)
        (root / 'frozen' / f'{game_id}.json').write_text(json.dumps(game))
        rows.append(dict(game_id=game_id, season='2026-27', season_type='regular', game_time_utc=game['start_time'], home_team_tricode='BBB', away_team_tricode='AAA'))
        boxes[game_id] = {'game': {'gameId': game_id, 'gameStatus': 3, 'gameStatusText': 'Final', 'period': 4,
            'homeTeam': {'teamTricode': 'BBB', 'score': 105, 'players': [{'personId': j, 'statistics': {'minutes':'PT80M'}} for j in (4,5,6)]},
            'awayTeam': {'teamTricode': 'AAA', 'score': 100, 'players': [{'personId': j, 'statistics': {'minutes':'PT80M'}} for j in (1,2,3)]}}}
    board['games'] = games
    (root / 'latest' / f'{GAME_DATE}.json').write_text(json.dumps(board))
    catalog = tmp_path / 'catalog.parquet'
    pd.DataFrame(rows).to_parquet(catalog)
    coverage = tmp_path / 'coverage.json'
    def ingest(**kwargs):
        coverage.write_text(json.dumps({'season':'2026-27','games':{g:['a'*64] for g in kwargs['game_ids']}}))
    return dict(game_date=date(2026,11,1), state_root=root, catalog_path=catalog, coverage_path=coverage, checkpoint_root=tmp_path/'inputs', ingest=ingest), boxes


def test_historical_recovery_never_reads_todays_scoreboard_or_ratings(tmp_path, monkeypatch):
    args, boxes = fixture(tmp_path)
    monkeypatch.setattr('nba_lineup_model.season.official_schedule_runtime._scoreboard_games', lambda *a: pytest.fail('today scoreboard used'))
    monkeypatch.setattr('nba_lineup_model.season.official_schedule_runtime._fetch_json', lambda *a: pytest.fail('current predictions used'))
    before = {p.name:p.read_bytes() for p in (args['state_root']/'frozen').glob('*')}
    fetched = []
    result = recovery.recover_official_slate(**args, fetch_box=lambda g: fetched.append(g) or boxes[g])
    assert result['status'] == 'complete'
    summary = json.loads((args['state_root']/f'final/{GAME_ID}.summary.json').read_text())
    assert summary['grading']['model_favorite_won'] is True
    assert summary['retrospective']['ledger']
    assert len(fetched) == 1
    assert before == {p.name:p.read_bytes() for p in (args['state_root']/'frozen').glob('*')}
    fingerprint = _tree_sha256(args['state_root'])
    recovery.recover_official_slate(**args, fetch_box=lambda g: pytest.fail('refetched validated box'))
    assert _tree_sha256(args['state_root']) == fingerprint


def test_failure_collects_other_games_and_retry_reuses_success(tmp_path):
    args, boxes = fixture(tmp_path, 2)
    ids = sorted(boxes)
    def fetch(g):
        if g == ids[0]:
            raise TimeoutError('injected outage')
        return boxes[g]
    with pytest.raises(recovery.BackfillIncomplete) as caught:
        recovery.recover_official_slate(**args, fetch_box=fetch)
    assert caught.value.report['completed_game_ids'] == [ids[1]]
    assert caught.value.report['failures'][0]['game_id'] == ids[0]
    summary = (args['state_root']/f'final/{ids[1]}.summary.json').read_bytes()
    fetched=[]
    recovery.recover_official_slate(**args, fetch_box=lambda g: fetched.append(g) or boxes[g])
    assert fetched == [ids[0]]
    assert (args['state_root']/f'final/{ids[1]}.summary.json').read_bytes() == summary


@pytest.mark.parametrize('fault', ['not_final','wrong_id','wrong_team','no_frozen','no_rating','wrong_scores','forbidden'])
def test_recovery_fails_closed_without_inventing_inputs(tmp_path, fault):
    args, boxes = fixture(tmp_path)
    box = boxes[GAME_ID]
    if fault == 'not_final': box['game']['gameStatus'] = 2
    if fault == 'wrong_id': box['game']['gameId'] = '0022600999'
    if fault == 'wrong_team': box['game']['homeTeam']['teamTricode'] = 'XXX'
    if fault == 'no_frozen': (args['state_root']/f'frozen/{GAME_ID}.json').unlink()
    if fault == 'no_rating': box['game']['homeTeam']['players'][0]['personId'] = 999
    if fault == 'forbidden': box['source'] = 'espn'
    if fault == 'wrong_scores':
        recovery.recover_official_slate(**args, fetch_box=lambda g: boxes[g])
        path=args['state_root']/f'final/{GAME_ID}.nba.json'
        box['game']['homeTeam']['score']=1
        path.write_text(json.dumps(box))
    with pytest.raises(recovery.BackfillIncomplete):
        recovery.recover_official_slate(**args, fetch_box=lambda g: box)


def test_checkpoint_box_survives_lost_candidate(tmp_path):
    args, boxes = fixture(tmp_path)
    recovery.recover_official_slate(**args, fetch_box=lambda g: boxes[g])
    shutil.rmtree(args['state_root']/'final')
    recovery.recover_official_slate(**args, fetch_box=lambda g: pytest.fail('durable input was lost'))


def setup_pipeline(tmp_path):
    c=config_fixture(tmp_path)
    pointer=c.live_pointer/'site-releases/latest.json'
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({'evidence_through_date':'2026-10-19'}))
    return c


def test_multiday_recovery_stops_then_resumes_without_repeating_validated_day(tmp_path, monkeypatch):
    c=setup_pipeline(tmp_path)
    monkeypatch.setattr('nba_lineup_model.season.production_site_release.verify_production_site_release', lambda path: {})
    events=[]
    failed=[True]
    def recover(**kwargs):
        day=kwargs['game_date']
        events.append(('recover',day))
        if day==date(2026,10,21) and failed[0]: raise RuntimeError('injected day failure')
        return {'status':'complete'}
    def prepare(work, day):
        events.append(('model',day))
        runtime=c.pipeline_root/f'test-releases/{day}'
        runtime.mkdir(parents=True)
        shutil.copytree(work.schedule_root,runtime/'schedule')
        return dict(game_date=str(day), status='validated_candidate', runtime_release=str(runtime))
    before=_tree_sha256(c.schedule_root)
    with pytest.raises(RuntimeError,match='injected'):
        backfill.run_backfill(c,end_date=date(2026,10,22),recover=recover,prepare=prepare)
    assert ('model',date(2026,10,21)) not in events
    assert ('recover',date(2026,10,22)) not in events
    failed[0]=False
    result=backfill.run_backfill(c,end_date=date(2026,10,22),recover=recover,prepare=prepare)
    assert result['status']=='validated_candidate'
    assert events.count(('model',date(2026,10,20)))==1
    assert [day for stage,day in events if stage=='model']==[date(2026,10,20),date(2026,10,21),date(2026,10,22)]
    assert _tree_sha256(c.schedule_root)==before
    assert (c.live_pointer/'unchanged').read_text()=='original'
    previous=list(events)
    backfill.run_backfill(c,end_date=date(2026,10,22),recover=recover,prepare=prepare)
    assert events==previous


def test_historical_repair_names_the_exact_data_replay_date(tmp_path, monkeypatch):
    c = setup_pipeline(tmp_path)
    (c.live_pointer / "site-releases/latest.json").write_text(
        json.dumps({"evidence_through_date": "2026-10-20"})
    )
    missing = c.schedule_root / "final/0022600001.nba.json"
    payload = missing.read_bytes()
    missing.unlink()
    monkeypatch.setattr(
        "nba_lineup_model.season.production_site_release.verify_production_site_release",
        lambda path: {},
    )
    seen = []

    def recover(**kwargs):
        target = kwargs["state_root"] / "final/0022600001.nba.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return {"status": "complete"}

    def prepare(work, day, *, data_replay_dates=()):
        seen.append((day, data_replay_dates))
        runtime = c.pipeline_root / f"test-releases/{day}"
        runtime.mkdir(parents=True)
        shutil.copytree(work.schedule_root, runtime / "schedule")
        return dict(
            game_date=str(day),
            status="validated_candidate",
            runtime_release=str(runtime),
        )

    backfill.run_backfill(
        c, start_date=date(2026, 10, 20), end_date=date(2026, 10, 20),
        recover=recover, prepare=prepare,
    )
    assert seen == [(date(2026, 10, 20), (date(2026, 10, 20),))]


def test_approved_historical_model_replay_resumes_at_the_exact_date(
    tmp_path, monkeypatch
):
    c = setup_pipeline(tmp_path)
    approved = date(2026, 10, 20)
    (c.live_pointer / "site-releases/latest.json").write_text(
        json.dumps({"evidence_through_date": approved.isoformat()})
    )
    monkeypatch.setattr(
        "nba_lineup_model.season.production_site_release.verify_production_site_release",
        lambda path: {},
    )
    seen = []

    def prepare(
        work,
        day,
        *,
        approved_historical_model_replay_from=None,
    ):
        seen.append((day, approved_historical_model_replay_from))
        runtime = c.pipeline_root / f"test-releases/{day}"
        runtime.mkdir(parents=True)
        shutil.copytree(work.schedule_root, runtime / "schedule")
        return dict(
            game_date=str(day),
            status="validated_candidate",
            runtime_release=str(runtime),
        )

    backfill.run_backfill(
        c,
        start_date=approved,
        end_date=approved,
        recover=lambda **_kwargs: {"status": "complete"},
        prepare=prepare,
        approved_historical_model_replay_from=approved,
    )

    assert seen == [(approved, approved)]


def test_already_complete_backfill_is_noop(tmp_path):
    c=setup_pipeline(tmp_path)
    (c.live_pointer/'site-releases/latest.json').write_text(json.dumps({'evidence_through_date':'2026-10-20'}))
    result=backfill.run_backfill(c,end_date=date(2026,10,20),recover=lambda **kw:pytest.fail('recovered complete game'),prepare=lambda *a:pytest.fail('rebuilt complete date'))
    assert result['status']=='already_complete'


def test_explicit_game_ids_include_entire_slate_and_predecessors(tmp_path, monkeypatch):
    c=setup_pipeline(tmp_path)
    seen=[]
    def recover(**kw): seen.append(kw['game_date']); return {}
    def prepare(work,day):
        return dict(game_date=str(day),status='validated_candidate',runtime_release=str(c.pipeline_root/'unused'))
    result=backfill.run_backfill(c,end_date=date(2026,10,25),game_ids=['0022600002'],recover=recover,prepare=prepare)
    assert seen==[date(2026,10,20)]
    assert result['requested_end_date']=='2026-10-20'


def test_sensor_enqueues_missing_inputs_not_skip_and_bounds_retries(tmp_path,monkeypatch):
    c=setup_pipeline(tmp_path)
    path=c.environment_root/'config.json'
    path.write_text(json.dumps(asdict(c),default=str))
    monkeypatch.setenv('GESTALT_DAILY_PIPELINE_CONFIG',str(path))
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None): return datetime.now(UTC)
    monkeypatch.setattr(orchestration,'datetime',Clock)
    for state in (DagsterRunStatus.QUEUED,DagsterRunStatus.STARTED,DagsterRunStatus.FAILURE,DagsterRunStatus.SUCCESS):
        with DagsterInstance.ephemeral() as instance:
            context=build_sensor_context(instance=instance)
            request=orchestration.complete_slate_daily_release_sensor(context)
            assert request.run_key.startswith('backfill:2026-10-19:') and request.run_key.endswith(':attempt-1')
            job=orchestration.backfill_official_history_job
            instance.create_run_for_job(job,remote_job_origin=RemoteJobOrigin(RemoteRepositoryOrigin(RegisteredCodeLocationOrigin('test'),'test_repository'),job.name),run_config=request.run_config,status=state,tags=request.tags)
            result=orchestration.complete_slate_daily_release_sensor(context)
            assert not hasattr(result,'run_key')
    Definitions.validate_loadable(orchestration.defs)


def test_dagster_calls_same_backfill_entrypoint(tmp_path,monkeypatch):
    c=setup_pipeline(tmp_path)
    path=c.environment_root/'config.json'
    path.write_text(json.dumps(asdict(c),default=str))
    calls=[]
    def run(config,**kw): calls.append(kw); return {'status':'already_complete','game_date':'2026-11-01'}
    monkeypatch.setattr(backfill,'run_backfill',run)
    result=orchestration.backfill_official_history_job.execute_in_process(run_config={'ops':{'recover_official_history':{'config':{'config_path':str(path),'end_date':'2026-11-01'}}}})
    assert result.success
    assert calls[0]['end_date']==date(2026,11,1)

def test_sensor_retries_after_cooldown_and_stops_after_three(tmp_path, monkeypatch):
    from types import SimpleNamespace
    c=setup_pipeline(tmp_path)
    path=c.environment_root/'config.json'
    path.write_text(json.dumps(asdict(c),default=str))
    monkeypatch.setenv('GESTALT_DAILY_PIPELINE_CONFIG',str(path))
    with DagsterInstance.ephemeral() as instance:
        context=build_sensor_context(instance=instance)
        records=[SimpleNamespace(dagster_run=SimpleNamespace(status=DagsterRunStatus.FAILURE),update_timestamp=datetime.now(UTC)-timedelta(minutes=16))]
        monkeypatch.setattr(instance,'get_run_records',lambda **kw:records)
        assert orchestration.complete_slate_daily_release_sensor(context).run_key.endswith('attempt-2')
        records*=3
        result=orchestration.complete_slate_daily_release_sensor(context)
        assert 'three times' in result.skip_message


def test_missing_frozen_still_collects_raw_official_inputs(tmp_path):
    args,boxes=fixture(tmp_path)
    (args['state_root']/f'frozen/{GAME_ID}.json').unlink()
    with pytest.raises(recovery.BackfillIncomplete):
        recovery.recover_official_slate(**args,fetch_box=lambda g:boxes[g])
    assert (args['checkpoint_root']/f'boxes/{GAME_ID}.json').exists()
    assert GAME_ID in json.loads(args['coverage_path'].read_text())['games']
    assert not (args['state_root']/f'final/{GAME_ID}.summary.json').exists()


def test_backfill_process_lock_prevents_duplicate_runs(tmp_path):
    import fcntl
    c=setup_pipeline(tmp_path)
    c.pipeline_root.mkdir()
    with (c.pipeline_root/'.backfill.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already running'):
            backfill.run_backfill(c,end_date=date(2026,10,20))


def test_pbp_failure_is_reported_without_losing_final_box(tmp_path):
    from nba_lineup_model.season.cdn_lake import CdnBatchError
    args,boxes=fixture(tmp_path)
    def ingest(**kw):
        raise CdnBatchError({'failed_count':1,'failures':[{'game_id':GAME_ID,'message':'temporary NBA timeout'}]})
    args['ingest']=ingest
    with pytest.raises(recovery.BackfillIncomplete,match='temporary NBA timeout'):
        recovery.recover_official_slate(**args,fetch_box=lambda g:boxes[g])
    assert (args['state_root']/f'final/{GAME_ID}.nba.json').exists()


def test_missing_daily_board_is_recovered_from_catalog_and_frozen_inputs(tmp_path):
    from nba_lineup_model.season.schedule_artifact_contract import validate_schedule_artifacts
    args, boxes = fixture(tmp_path)
    (args['state_root']/f'latest/{GAME_DATE}.json').unlink()
    recovery.recover_official_slate(**args, fetch_box=lambda g:boxes[g])
    board=json.loads((args['state_root']/f'latest/{GAME_DATE}.json').read_text())
    assert board['source'].startswith('NBA official')
    assert board['games'][0]['projection']['ledger']
    assert board['games'][0]['retrospective']['ledger']
    validate_schedule_artifacts(args['state_root'])


def test_backfill_git_preflight_failure_is_reported(tmp_path, monkeypatch):
    c = setup_pipeline(tmp_path)

    def fail(_code_root):
        raise RuntimeError("blocked Git preflight")

    monkeypatch.setattr(backfill, "require_clean_code_revision", fail)
    with pytest.raises(RuntimeError, match="blocked Git preflight"):
        backfill.run_backfill(c, end_date=date(2026, 10, 20))

    report = json.loads(
        (c.pipeline_root / "reports/backfill/report.json").read_text()
    )
    assert report["status"] == "failed"
    assert report["stage"] == "preflight"


def test_stalled_daily_run_is_terminated_for_sensor_retry(tmp_path):
    from types import SimpleNamespace

    c = setup_pipeline(tmp_path)
    now = datetime.now(UTC)
    run = SimpleNamespace(
        run_id="stalled-run",
        status=DagsterRunStatus.STARTED,
        tags={"gestalt/game_date": "2026-11-02"},
    )
    record = SimpleNamespace(
        dagster_run=run,
        update_timestamp=now - timedelta(hours=3),
    )
    terminated = []
    context = SimpleNamespace(
        instance=SimpleNamespace(
            run_launcher=SimpleNamespace(
                terminate=lambda run_id: terminated.append(run_id) or True
            )
        )
    )

    result = orchestration._guard_inflight_daily_run(context, c, record, now)
    assert terminated == ["stalled-run"]
    assert "next sensor tick will retry" in result.skip_message
