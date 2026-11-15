"""Historical dates validate captures without polling or moving latest.json."""
import json
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace

import pytest
from dagster import build_schedule_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.odds import schedules
from gestalt_pipelines.odds.partitions import odds_capture_dates
from gestalt_pipelines.odds.resources import OddsApi
from gestalt_pipelines.odds.resources import odds_api as module
from gestalt_pipelines.utils import PACIFIC


@pytest.mark.parametrize('day', [date(2026, 11, 1), date(2027, 3, 14)])
def test_archive_date_uses_pacific_dst_boundaries_and_preserves_bookmakers(monkeypatch, day):
    midnight = datetime.combine(day, datetime.min.time(), PACIFIC).astimezone(UTC)
    end = datetime.combine(day+timedelta(days=1), datetime.min.time(), PACIFIC).astimezone(UTC)
    root = 'lake/raw/odds-api/season=2026-27/snapshots/'
    moments = [midnight-timedelta(seconds=1), midnight, midnight+timedelta(hours=2), end-timedelta(seconds=1), end]
    objects = {}
    for instant in moments:
        objects[root+instant.strftime('%Y%m%dT%H%M%SZ.json')] = json.dumps({
            'provider': 'The Odds API', 'sport_key': 'basketball_nba', 'season': '2026-27',
            'collected_at': instant.isoformat(), 'events': [{'bookmakers': [{'key': 'espnbet'}]}],
        }).encode()
    def paginate(**kwargs):
        return [{'Contents': [{'Key': key} for key in objects if key.startswith(kwargs['Prefix'])]}]
    client = SimpleNamespace(
        get_paginator=lambda _: SimpleNamespace(paginate=paginate),
        get_object=lambda **kw: {'Body': BytesIO(objects[kw['Key']])},
    )
    monkeypatch.setattr(module.boto3, 'client', lambda _: client)
    refs = OddsApi(api_key='unused').archived_snapshots('2026-27', day)
    assert len(refs) == 3
    assert all(datetime.fromisoformat(row['collected_at']).astimezone(PACIFIC).date() == day for row in refs)
    assert all(len(row['sha256']) == 64 for row in refs)


def test_historical_job_uses_archive_only_and_preserves_live_latest(monkeypatch):
    called = []
    def archived(self, season, day):
        called.append((season, day))
        return [{'uri': 's3://nba-gestalt-data/saved.json', 'sha256': 'a'*64}]
    monkeypatch.setattr(OddsApi, 'archived_snapshots', archived)
    monkeypatch.setattr(OddsApi, 'fetch', lambda *a: pytest.fail('Historical dates must not call provider'))
    monkeypatch.setattr(OddsApi, 'publish', lambda *a: pytest.fail('Historical dates must not move latest'))
    result = defs.resolve_job_def('archive_nba_market_odds').execute_in_process(partition_key='2026-10-26')
    assert result.success and called == [('2026-27', date(2026, 10, 26))]
    assert result.get_asset_materialization_events()[0].event_specific_data.materialization.partition == '2026-10-26'


def test_missing_archive_date_is_a_visible_failure(monkeypatch):
    client = SimpleNamespace(get_paginator=lambda _: SimpleNamespace(paginate=lambda **kw: [{}]))
    monkeypatch.setattr(module.boto3, 'client', lambda _: client)
    with pytest.raises(FileNotFoundError, match='No genuine archived'):
        OddsApi(api_key='unused').archived_snapshots('2026-27', date(2026, 10, 26))


def test_capture_schedule_selects_actual_pacific_date_and_skips_old_ticks(monkeypatch):
    tick = datetime(2026, 11, 14, 19, 20, tzinfo=PACIFIC)
    now = [tick.astimezone(UTC)]
    monkeypatch.setattr(schedules, 'datetime', SimpleNamespace(now=lambda _: now[0]))
    schedule = defs.get_schedule_def('nba_market_odds_game_day')
    with build_schedule_context(scheduled_execution_time=tick, repository_def=defs.get_repository_def()) as context:
        [request] = schedule.evaluate_tick(context).run_requests
        assert request.partition_key == '2026-11-14'
        now[0] += timedelta(days=1)
        assert not schedule.evaluate_tick(context).run_requests
    assert odds_capture_dates.get_first_partition_key() == '2026-10-26'
