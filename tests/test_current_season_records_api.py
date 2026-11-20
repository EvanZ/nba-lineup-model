"""Official records are independent of projections, complete, and cached ten minutes."""

from datetime import date, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_saved_schedule_api import SavedS3

from nba_lineup_model.web_api import current_season_records as api
from nba_lineup_model.web_api.gateway import MarketGateway
from nba_lineup_model.web_api.schedule import ScheduleReader


class Boards(SavedS3):
    def paginate(self, *, Bucket, Prefix, Delimiter):
        assert Delimiter == "/"
        self.calls.append(("list", Prefix))
        roots = {
            Prefix + k[len(Prefix) :].split("/")[0] + "/"
            for k in self.objects
            if k.startswith(Prefix)
        }
        yield {"CommonPrefixes": [{"Prefix": p} for p in sorted(roots)]}


@pytest.fixture
def boards(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 19, 12, tzinfo=tz)

    monkeypatch.setattr(api, "datetime", Clock)
    monkeypatch.setattr(api, "time", lambda: 3601)
    s3 = Boards()
    reader = ScheduleReader(s3)
    games = []
    docs = {}
    identities = [{"team_id": i, "team": f"T{i:02}"} for i in range(1, 31)]
    for day, n, offset in [("2026-11-17", 30, 100), ("2026-11-18", 24, 200)]:
        rows = []
        for i in range(0, n, 2):
            home, away = i + 1, i + 2
            gid = f"002260{offset + i:04}"
            game = {
                "game_id": gid,
                "game_date": day,
                "start_time": day + "T20:00:00Z",
                "home_team_id": home,
                "away_team_id": away,
                "home_team": f"T{home:02}",
                "away_team": f"T{away:02}",
            }
            games.append(game)
            rows.append(
                {
                    "gameId": gid,
                    "gameStatus": 3,
                    "homeTeam": {
                        "teamId": home,
                        "teamTricode": f"T{home:02}",
                        "wins": 8 if n == 24 else 7,
                        "losses": 7,
                    },
                    "awayTeam": {
                        "teamId": away,
                        "teamTricode": f"T{away:02}",
                        "wins": 7,
                        "losses": 8 if n == 24 else 7,
                    },
                }
            )
        docs[day] = {"scoreboard": {"gameDate": day, "games": rows}}

    def save(day):
        key = f"lake/raw/cdn/scoreboard/season=2026-27/game_date_pacific={day}/latest.json"
        sha = s3.put_fixture(key, docs[day], {"fetched-at": day + "T23:30:00Z"})
        s3.objects[key][1]["sha256"] = sha
        return key

    for day in docs:
        save(day)
    monkeypatch.setattr(
        reader,
        "official_schedule",
        lambda season: (games, identities, {"uri": "saved-calendar", "sha256": "calendar-sha"}),
    )
    monkeypatch.setattr(api, "_reader", lambda: reader)
    api.cached_records.cache_clear()
    yield s3, reader, docs, save
    api.cached_records.cache_clear()


def test_latest_official_records_not_sum_of_daily_snapshots(boards):
    s3, _, _, _ = boards
    legacy = Mock(side_effect=AssertionError("No legacy evaluator"))
    client = TestClient(MarketGateway(legacy))
    r = client.get("/api/current-season-records")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, max-age=599"
    d = r.json()
    assert len(d["teams"]) == 30 and d["as_of"] == "2026-11-18"
    rows = {t["team"]: t for t in d["teams"]}
    assert (rows["T01"]["wins"], rows["T01"]["losses"], rows["T01"]["games"]) == (8, 7, 15)
    assert rows["T25"]["games"] == 14 and rows["T25"]["last_game_date"] == "2026-11-17"
    assert sum(t["wins"] for t in d["teams"]) == sum(t["losses"] for t in d["teams"]) == 222
    calls = len(s3.calls)
    assert client.get("/api/current-season-records").json() == d
    assert len(s3.calls) == calls
    legacy.assert_not_called()
    assert api.cached_records.cache_info().maxsize == 1


def test_cache_rolls_over_after_ten_minutes(boards, monkeypatch):
    _, _, docs, save = boards
    c = TestClient(MarketGateway(None))
    before = c.get("/api/current-season-records").json()
    docs["2026-11-18"]["scoreboard"]["games"][0]["homeTeam"]["wins"] = 9
    save("2026-11-18")
    assert c.get("/api/current-season-records").json() == before
    monkeypatch.setattr(api, "time", lambda: 4201)
    after = c.get("/api/current-season-records").json()
    assert after["teams"][0]["wins"] == 9 and after["teams"][0]["games"] == 16


def test_in_progress_games_do_not_advance_record(boards):
    _, _, docs, save = boards
    docs["2026-11-18"]["scoreboard"]["games"][0]["gameStatus"] = 2
    save("2026-11-18")
    d = api.read_records(api._reader(), date(2026, 11, 19))
    first = d["teams"][0]
    assert first["wins"] == first["losses"] == 7 and first["games"] == 14
    assert first["last_game_date"] == "2026-11-17"


@pytest.mark.parametrize("damage", ["checksum", "incomplete", "negative", "oversize", "identity"])
def test_invalid_or_partial_records_fail_without_cached_fallback(boards, damage):
    s3, _, docs, save = boards
    day = "2026-11-18"
    if damage == "incomplete":
        docs[day]["scoreboard"]["games"].pop()
    elif damage == "negative":
        docs[day]["scoreboard"]["games"][0]["homeTeam"]["wins"] = -1
    elif damage == "oversize":
        docs[day]["scoreboard"]["games"][0]["homeTeam"]["wins"] = 100
    elif damage == "identity":
        docs[day]["scoreboard"]["games"][0]["homeTeam"]["teamTricode"] = "WRONG"
    key = save(day)
    if damage == "checksum":
        raw, metadata = s3.objects[key]
        s3.objects[key] = (raw + b" ", metadata)
    r = TestClient(MarketGateway(None)).get("/api/current-season-records")
    assert r.status_code == 503 and r.headers["cache-control"] == "no-store"
    assert api.cached_records.cache_info().currsize == 0
    if damage == "checksum":
        s3.objects[key] = (raw, metadata)
        assert TestClient(MarketGateway(None)).get("/api/current-season-records").status_code == 200


def test_future_partitions_are_not_read(boards):
    s3, reader, docs, save = boards
    docs["2026-11-20"] = {"scoreboard": {"gameDate": "2026-11-20", "games": []}}
    save("2026-11-20")
    result = api.read_records(reader, date(2026, 11, 19))
    assert len(result["teams"]) == 30 and result["as_of"] == "2026-11-18"
    assert not any("2026-11-20" in key for kind, key in s3.calls if kind == "get")


def test_missing_team_coverage_is_explicit(boards):
    s3, _, _, _ = boards
    for key in list(s3.objects):
        if "2026-11-17" in key:
            del s3.objects[key]
    r = TestClient(MarketGateway(None)).get("/api/current-season-records")
    assert r.status_code == 503 and r.headers["cache-control"] == "no-store"


def test_non_regular_season_results_do_not_change_records(boards):
    _, reader, docs, save = boards
    games, teams, ref = reader.official_schedule("2026-27")
    game = {**games[0], "game_id": "0062600001", "game_date": "2026-11-19"}
    games.append(game)
    docs["2026-11-19"] = {
        "scoreboard": {
            "gameDate": "2026-11-19",
            "games": [
                {
                    "gameId": "0062600001",
                    "gameStatus": 3,
                    "homeTeam": {"teamId": 1, "teamTricode": "T01", "wins": 100, "losses": 100},
                    "awayTeam": {"teamId": 2, "teamTricode": "T02", "wins": 100, "losses": 100},
                }
            ],
        }
    }
    save("2026-11-19")
    d = api.read_records(reader, date(2026, 11, 19))
    assert d["as_of"] == "2026-11-18" and d["teams"][0]["games"] == 15
