import json
from datetime import date

import pytest

from gestalt_pipelines.resources import NBAScheduleResource


def test_range_reads_once_and_groups_by_pacific_start_date(monkeypatch):
    def game(gid, start):
        return {
            "gameId": gid, "gameStatus": 3, "gameDateTimeUTC": start, "gameDateEst": "2026-11-08",
            "homeTeam": {"teamId": 1, "teamTricode": "ATL"},
            "awayTeam": {"teamId": 2, "teamTricode": "BOS"},
        }
    payload = json.dumps({"leagueSchedule": {
        "seasonYear": "2026-27", "leagueId": "00",
        "gameDates": [{"gameDate": "11/08/2026", "games": [
            game("0022600199", "2026-11-08T02:00:00Z"),
            game("0022600200", "2026-11-09T02:00:00Z"),
        ]}],
    }}).encode()
    calls = []
    reference = {"uri": "s3://schedule/exact.json", "sha256": "verified"}

    def load(self, selected=None):
        calls.append(selected)
        return payload, reference

    monkeypatch.setattr(NBAScheduleResource, "load", load)
    resource = NBAScheduleResource(season="2026-27")
    games, ref = resource.games_between(date(2026, 11, 7), date(2026, 11, 8))
    assert len(calls) == 1 and ref == reference
    assert [g["game_id"] for g in games] == ["0022600199", "0022600200"]
    games, _ = resource.games_for_date(date(2026, 11, 7))
    assert [g["game_id"] for g in games] == ["0022600199"]
    with pytest.raises(ValueError, match="reversed"):
        resource.games_between(date(2026, 11, 8), date(2026, 11, 7))
