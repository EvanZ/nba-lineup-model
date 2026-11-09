import json

import pytest

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.season_preparation.assets import official_schedule


@pytest.mark.parametrize("season, succeeds", [("2026-27", True), ("2025-26", False)])
def test_schedule_job_publishes_only_validated_original_bytes(monkeypatch, season, succeeds):
    payload = json.dumps({"leagueSchedule": {
        "seasonYear": season, "leagueId": "00",
        "gameDates": [{"gameDate": "11/08/2026", "games": [{
            "gameId": "0022600200", "gameStatus": 1,
            "gameDateTimeUTC": "2026-11-09T01:00:00Z",
            "homeTeam": {"teamId": 1, "teamTricode": "BOS"},
            "awayTeam": {"teamId": 2, "teamTricode": "NYK"},
            "broadcasters": {"nationalBroadcasters": [{"broadcasterDisplay": "ESPN"}]},
        }, {
            "gameId": "0022601201", "gameStatus": 1,
            "homeTeam": {"teamId": 0}, "awayTeam": {"teamId": 0},
        }]}],
    }}).encode()
    published = []

    def publish(resource, raw, *, fetched_at):
        assert resource.season == "2026-27"
        assert fetched_at.utcoffset().total_seconds() == 0
        published.append(raw)
        return {"uri": "s3://fixture/schedule.json", "sha256": "fixture"}

    monkeypatch.setattr(official_schedule, "_fetch_schedule", lambda: payload)
    monkeypatch.setattr(official_schedule.NBAScheduleResource, "publish", publish)
    result = defs.resolve_job_def("capture_nba_season_schedule").execute_in_process(raise_on_error=False)
    assert result.success is succeeds
    assert published == ([payload] if succeeds else [])
    if succeeds:
        [event] = result.get_asset_materialization_events()
        assert event.event_specific_data.materialization.metadata["assigned_game_count"].value == 1
