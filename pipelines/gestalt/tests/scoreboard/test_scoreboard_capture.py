import json
from datetime import UTC, date, datetime
from hashlib import sha256
from types import SimpleNamespace

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.scoreboard.assets import capture
from gestalt_pipelines.scoreboard.resources import NBAScoreboardArchive, archive

DAY = date(2026, 11, 10)
FETCHED_AT = datetime(2026, 11, 11, 1, 5, tzinfo=UTC)
GAME_ID = "0022600211"


def test_manual_job_publishes_only_a_complete_official_slate(monkeypatch):
    scheduled = [{"game_id": GAME_ID, "home_team_id": 1, "away_team_id": 2}]
    game = {
        "gameId": GAME_ID, "gameStatus": 2, "period": 1, "gameClock": "PT05M20.00S",
        "homeTeam": {"teamId": 1, "score": 16}, "awayTeam": {"teamId": 2, "score": 12},
    }
    payload = json.dumps({"scoreboard": {"gameDate": DAY.isoformat(), "games": [game]}}).encode()
    writes = []

    def games_for_date(self, day, *, allow_empty):
        assert day == DAY and allow_empty
        return scheduled, {"uri": "s3://schedule/immutable.json", "sha256": "verified"}

    monkeypatch.setattr(capture, "datetime", SimpleNamespace(now=lambda _: FETCHED_AT))
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", games_for_date)
    monkeypatch.setattr(NBAScoreboardArchive, "fetch", lambda _: payload)
    monkeypatch.setattr(archive.boto3, "client", lambda _: SimpleNamespace(put_object=lambda **kwargs: writes.append(kwargs)))
    job = defs.resolve_job_def("capture_nba_scoreboard")

    result = job.execute_in_process()
    assert result.success
    [saved] = writes
    assert saved["Bucket"] == "nba-gestalt-data"
    assert saved["Key"] == "lake/raw/cdn/scoreboard/season=2026-27/game_date_pacific=2026-11-10/latest.json"
    assert saved["Body"] == payload
    assert saved["Metadata"]["fetched-at"] == FETCHED_AT.isoformat()
    assert saved["Metadata"]["sha256"] == sha256(payload).hexdigest()
    event = result.get_asset_materialization_events()[0]
    assert event.event_specific_data.materialization.metadata["live_game_count"].value == 1

    payload = json.dumps({"scoreboard": {"gameDate": DAY.isoformat(), "games": []}}).encode()
    assert not job.execute_in_process(raise_on_error=False).success
    assert writes == [saved]
