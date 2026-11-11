from datetime import UTC, date, datetime
from io import BytesIO
from hashlib import sha256
from types import SimpleNamespace
import json

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance, build_schedule_context

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.scoreboard import schedules
from gestalt_pipelines.scoreboard.resources import NBAScoreboardArchive, archive

DAY = date(2026, 11, 10)
GAME = {
    "game_id": "0022600211", "home_team_id": 1, "away_team_id": 2,
    "game_time_utc": datetime(2026, 11, 11, 3, tzinfo=UTC),
}


def test_poll_stops_after_final_and_continues_past_midnight(monkeypatch):
    game = {
        "gameId": GAME["game_id"], "gameStatus": 2, "period": 1, "gameClock": "PT05M00S",
        "homeTeam": {"teamId": 1, "score": 20}, "awayTeam": {"teamId": 2, "score": 18},
    }
    board = {"gameDate": DAY.isoformat(), "games": [game]}
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda self, day, **kw: (
        [GAME] if day == DAY else [], {},
    ))
    monkeypatch.setattr(NBAScoreboardArchive, "read_latest", lambda *args: board)
    clock = SimpleNamespace(now=lambda _: None)
    monkeypatch.setattr(schedules, "datetime", clock)

    with DagsterInstance.local_temp() as instance:
        def tick(hour):
            when = datetime(2026, 11, 11, hour, tzinfo=UTC)
            clock.now = lambda _: when
            with build_schedule_context(
                instance=instance, scheduled_execution_time=when,
                repository_def=defs.get_repository_def(),
                resources={
                    "nba_schedule": NBAScheduleResource(season="2026-27"),
                    "nba_scoreboard_archive": NBAScoreboardArchive(),
                },
            ) as context:
                result = defs.get_schedule_def("scoreboard_poll").evaluate_tick(context)
                return result.run_requests

        assert not tick(2)
        assert len(tick(3)) == 1
        assert len(tick(8)) == 1
        game["gameStatus"] = 3
        assert not tick(8)
        game["gameStatus"] = 2
        assert not tick(9)


def test_saved_scoreboard_reader_checks_checksum_and_missing_key(monkeypatch):
    payload = json.dumps({"scoreboard": {"gameDate": DAY.isoformat(), "games": []}}).encode()
    metadata = {"sha256": sha256(payload).hexdigest()}

    def get_object(**kwargs):
        assert kwargs["Key"] == "lake/raw/cdn/scoreboard/season=2026-27/game_date_pacific=2026-11-10/latest.json"
        return {"Body": BytesIO(payload), "Metadata": metadata}

    monkeypatch.setattr(archive.boto3, "client", lambda _: SimpleNamespace(get_object=get_object))
    resource = NBAScoreboardArchive()
    assert resource.read_latest("2026-27", DAY)["gameDate"] == DAY.isoformat()
    metadata["sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        resource.read_latest("2026-27", DAY)

    def missing(**kwargs):
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    monkeypatch.setattr(archive.boto3, "client", lambda _: SimpleNamespace(get_object=missing))
    assert resource.read_latest("2026-27", DAY) is None
