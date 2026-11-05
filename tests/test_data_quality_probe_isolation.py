from datetime import UTC, datetime
from urllib.error import HTTPError

import pytest

from nba_lineup_model.orchestration import data_quality_probe as probe
from nba_lineup_model.season.data_quality_report import render_data_quality_html


@pytest.mark.parametrize('failed_path', [
    '/api/health', '/api/serving-view', '/api/rankings?season=2026-27',
    '/api/teams?season=2026-27', '/api/lineups?season=2026-27&minimum_possessions=0',
    '/api/win-projections', '/api/current-season-records', '/api/schedule-performance',
    '/api/schedule?date=2026-11-02',
])
def test_failed_endpoint_does_not_abort_other_checks(monkeypatch, failed_path):
    calls = []
    def get(root, path):
        calls.append(path)
        if path == failed_path:
            raise HTTPError(path, 503, 'Unavailable', {}, None)
        return {}
    monkeypatch.setattr(probe, 'get_json', get)
    snapshot = probe.current_surface_snapshot('http://test', '2026-11-02')
    assert len(calls) == 9
    assert len(snapshot['probe_failures']) == 1
    assert failed_path in snapshot['probe_failures'][0]['message']


def test_report_retains_artifacts_and_job_recovery_when_one_api_fails(tmp_path, monkeypatch):
    approved = {
        'season': '2026-27', 'game_count': 107, 'evidence_through_date': '2026-11-02',
        'data': {'bundle_sha256': 'data'}, 'model': {'bundle_sha256': 'model'},
        'serving': {'bundle_sha256': 'serving'}, 'lineups': {'bundle_sha256': 'lineups'},
        'preseason_catalog': {'player_count': 500}, 'api_validation': {},
        'schedule': {'tree_sha256': 'schedule', 'attribution_audit': {
            'served_games': 107, 'pregame_ledgers': 107, 'actual_minutes_ledgers': 107}},
    }
    monkeypatch.setattr(probe, 'read_active_activation', lambda *_: {'release_id': 'release'})
    monkeypatch.setattr(probe, 'verify_production_site_release', lambda *_: {
        'identity': {}, 'approved_manifest': {'identity': approved}})
    monkeypatch.setattr(probe, 'read_json', lambda *_: {
        'components': {}, 'edition_id': 'edition', 'evidence_through_date': '2026-11-02'})
    monkeypatch.setattr(probe, 'official_schedule_snapshot', lambda *a, **k: {
        'completed_games': 107, 'latest_eligible_completed_date': '2026-11-02'})
    monkeypatch.setattr(probe, 'activation_transition', lambda *a: {
        'changed_components': [], 'forecast_selection_changed': False})
    monkeypatch.setattr(probe, 'collector_rows', lambda **k: [])
    monkeypatch.setattr(probe, 'daily_publication_row', lambda **k: {
        'name': 'daily_publication', 'latest_run_status': 'SUCCESS',
        'automation_status': 'RUNNING', 'required_now': False,
        'unresolved_failures': [], 'last_success_at': '2026-11-04T20:47:23Z'})
    def get(root, path):
        if path == '/api/schedule-performance':
            raise HTTPError(path, 503, 'Unavailable', {}, None)
        return {'release_id': 'release', 'players': [{'id': 1}], 'components': {}}
    monkeypatch.setattr(probe, 'get_json', get)
    report = probe.collect_environment_report(
        environment='dev', live_release=tmp_path / 'live-release', schedule_root=tmp_path,
        dagster_home=tmp_path, daily_dagster_home=tmp_path, api_root='http://test',
        report_date='2026-11-04', generated_at=datetime(2026, 11, 4, 23, tzinfo=UTC))
    assert len(report['artifacts']) == 11
    assert report['collectors'][0]['latest_run_status'] == 'SUCCESS'
    assert report['pipeline_failures'][0]['stage'] == 'api:performance'
    performance = next(a for a in report['artifacts'] if a['name'] == 'aggregate_performance')
    assert performance['states']['served'] is False
    assert performance['coverage']['completed_games']['observed'] is None
    rankings = next(a for a in report['artifacts'] if a['name'] == 'rankings')
    assert rankings['coverage']['players']['observed'] == 1
    assert rankings['states']['served'] is True
    html = render_data_quality_html(report)
    assert 'observed_daily_data' in html and 'daily_publication' in html
    assert 'api:performance' in html
