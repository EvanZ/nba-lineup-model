"""Archived slates cannot be replaced by the NBA's current response."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.scoreboard.assets import capture
from gestalt_pipelines.scoreboard.resources import NBAScoreboardArchive


def test_old_slate_validates_saved_data_without_downloading_or_writing(monkeypatch):
    board = {'gameDate': '2026-11-10', 'games': []}
    monkeypatch.setattr(capture, 'datetime', SimpleNamespace(now=lambda _: datetime(2026, 11, 14, 20, tzinfo=UTC)))
    monkeypatch.setattr(NBAScheduleResource, 'games_for_date', lambda *a, **kw: ([], {}))
    monkeypatch.setattr(NBAScoreboardArchive, 'fetch', lambda *a: pytest.fail('Current endpoint is not historical'))
    monkeypatch.setattr(NBAScoreboardArchive, 'publish', lambda *a: pytest.fail('Do not overwrite saved slate'))
    monkeypatch.setattr(NBAScoreboardArchive, 'load', lambda *a: (board, {'uri': 's3://saved', 'sha256': 'a'*64, 'fetched_at': None}))
    job = defs.resolve_job_def('capture_nba_scoreboard')
    result = job.execute_in_process(partition_key='2026-11-10')
    assert result.success
    assert result.get_asset_materialization_events()[0].event_specific_data.materialization.partition == '2026-11-10'
    monkeypatch.setattr(NBAScoreboardArchive, 'load', lambda *a: None)
    assert not job.execute_in_process(partition_key='2026-11-10', raise_on_error=False).success
