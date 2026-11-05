import json
from datetime import UTC, datetime

import pytest

from nba_lineup_model.orchestration import data_quality_probe as probe


@pytest.mark.parametrize('stamp,complete,active,expected', [
    ('2026-11-05T04:00:00+00:00', False, '2026-11-02', 2),
    ('2026-11-05T12:59:59+00:00', True, '2026-11-02', 2),
    ('2026-11-05T13:00:00+00:00', True, '2026-11-02', 3),
    ('2026-11-05T04:00:00+00:00', False, '2026-11-01', 2),
    ('2026-11-05T12:00:00+00:00', True, '2026-11-04', 3),
])
def test_nightly_target_obeys_deadline_not_live_completion(stamp, complete, active, expected):
    official={'completed_dates': ['2026-11-02']+(['2026-11-04'] if complete else []),
        'completed_game_ids_by_date': {'2026-11-02':['a','b'], '2026-11-04':['c']}}
    result=probe.nightly_coverage_target(official,active,datetime.fromisoformat(stamp),5)
    assert result['required_completed_games']==expected
    assert result['required_evidence_through_date']==('2026-11-04' if expected==3 else '2026-11-02')


def test_official_snapshot_dates_finals_and_excludes_partial_slate_from_due_dates(tmp_path):
    latest=tmp_path/'latest';final=tmp_path/'final';latest.mkdir();final.mkdir()
    for day,games in [('2026-11-02',[('a',True),('b',True)]),
                       ('2026-11-03',[]),('2026-11-04',[('c',True),('d',False)])]:
        (latest/f'{day}.json').write_text(json.dumps({'game_date':day,'games':[
            {'game_id':gid,'status':{'completed':done,'state':'post' if done else 'in'}}
            for gid,done in games]}))
        for gid,done in games:
            if done:(final/f'{gid}.summary.json').write_text(json.dumps({'game_id':gid,'game_date':day}))
    result=probe.official_schedule_snapshot(tmp_path,report_date='2026-11-04')
    assert result['completed_games']==3
    assert result['completed_dates']==['2026-11-02']
    assert result['completed_game_ids_by_date']=={'2026-11-02':['a','b'],'2026-11-04':['c']}


@pytest.mark.parametrize('missing',[False,True])
def test_live_ledgers_are_counted_from_served_games_not_release_manifest(monkeypatch,missing):
    calls=[]
    def get(root,path):
        calls.append(path)
        if path.startswith('/api/schedule?'):
            gid='old' if path.endswith('2026-11-02') else 'new'
            row={'game_id':gid,'status':{'completed':True,'state':'post'}}
            if not missing or gid=='old': row['retrospective']={'ledger':{'home_margin':1}}
            return {'games':[row]}
        return {}
    monkeypatch.setattr(probe,'get_json',get)
    result=probe.current_surface_snapshot('test://api','2026-11-02',
        schedule_dates=['2026-11-02','2026-11-04'])
    assert result['actual_minutes_ledger_count']==(1 if missing else 2)
    assert result['schedule_game_count']==2
    assert '/api/schedule?date=2026-11-04' in calls


def test_failed_live_schedule_is_an_explicit_probe_failure(monkeypatch):
    def get(root,path):
        if path.startswith('/api/schedule?'):raise OSError('unavailable')
        return {}
    monkeypatch.setattr(probe,'get_json',get)
    result=probe.current_surface_snapshot('test://api','2026-11-02',schedule_dates=['2026-11-04'])
    assert result['probe_failures'][0]['stage']=='api:schedule'
