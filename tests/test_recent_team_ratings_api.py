"""L5 serving reads only the five selected games and preserves the preseason forecast."""

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_saved_schedule_api import SavedS3, prediction
from test_team_rating_history_api import publish

from nba_lineup_model.web_api import recent_team_ratings as api
from nba_lineup_model.web_api.gateway import MarketGateway
from nba_lineup_model.web_api.schedule import ScheduleReader


def make_game(day, pair):
    day = f"2026-11-{day:02}"
    start = (
        datetime.combine(date.fromisoformat(day) + timedelta(days=1), datetime.min.time(), UTC)
        .replace(hour=1)
        .isoformat()
    )
    return {
        "game_id": f"00226{date.fromisoformat(day).day:03}{pair:02}",
        "game_date": day,
        "start_time": start,
        "home_team": f"T{pair * 2 + 1:02}",
        "away_team": f"T{pair * 2 + 2:02}",
        "home_team_id": pair * 2 + 1,
        "away_team_id": pair * 2 + 2,
    }


@pytest.fixture
def source(monkeypatch):
    s3 = SavedS3()
    teams = [{"team_id": i, "team": f"T{i:02}"} for i in range(1, 31)]
    games = [make_game(day, pair) for day in [10, 11, 12, 13, 14, 15, 16, 19] for pair in range(15)]
    records = {}
    for game in games:
        record = prediction()
        record.update(
            game_id=game["game_id"],
            game_date_pacific=game["game_date"],
            scheduled_start_utc=game["start_time"],
            generated_at_utc=game["game_date"] + "T12:00:00Z",
        )
        for side in ["home", "away"]:
            record[side].update(team_id=game[side + "_team_id"], team=game[side + "_team"])
        d = int(game["game_date"][-2:])
        pair = (game["home_team_id"] - 1) // 2
        record["components"].update(
            home_player_total=1.123456789 + pair * 0.1 + d * 0.01,
            away_player_total=-2.234567891 + pair * 0.1 - d * 0.02,
            hca=99.9,
            b2b_adjustment=88.8,
        )
        records[game["game_id"]] = record
        publish(s3, record)
    reader = ScheduleReader(s3)
    reader.official_schedule = lambda season: (
        deepcopy(games),
        teams,
        {"uri": "s3://nba-gestalt-data/schedule.json", "sha256": "a" * 64},
    )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 19, 12, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(api, "_reader", lambda: reader)
    monkeypatch.setattr(api, "time", lambda: 3601)
    monkeypatch.setattr(api, "datetime", Clock)
    api.cached_recent_ratings.cache_clear()
    legacy = Mock(side_effect=AssertionError("Legacy must not load"))
    yield s3, reader, TestClient(MarketGateway(legacy)), records, legacy
    api.cached_recent_ratings.cache_clear()


def test_exact_mean_only_five_started_games_and_one_pointer_per_game(source):
    s3, _, client, records, legacy = source
    before = deepcopy(s3.objects)
    response = client.get("/api/recent-team-ratings")
    assert (
        response.status_code == 200 and response.headers["cache-control"] == "private, max-age=599"
    )
    data = response.json()
    assert data["complete"] and data["window"] == 5 and len(data["teams"]) == 30
    for row in data["teams"]:
        assert row["availability"] == "available" and row["game_count"] == 5
        assert [p["game_date_pacific"] for p in row["points"]] == [
            f"2026-11-{d}" for d in range(12, 17)
        ]
        side = "home" if row["team_id"] % 2 else "away"
        values = [
            records[p["game_id"]]["components"][side + "_player_total"] for p in row["points"]
        ]
        assert row["nail"] == sum(values) / 5
        assert all(p["source"]["sha256"] in p["source"]["uri"] for p in row["points"])
    pointers = [key for op, key in s3.calls if key.endswith("latest.json")]
    assert len(pointers) == len(set(pointers)) == 75
    assert all(
        op == "get" and key.startswith("lake/artifacts/game_predictions/") for op, key in s3.calls
    )
    assert not any(
        "game_date_pacific=2026-11-10" in key
        or "game_date_pacific=2026-11-11" in key
        or "game_date_pacific=2026-11-19" in key
        for _, key in s3.calls
    )
    count = len(s3.calls)
    assert client.get("/api/recent-team-ratings").json() == data and len(s3.calls) == count
    assert s3.objects == before
    legacy.assert_not_called()


def test_missing_prediction_is_not_replaced_with_an_older_game(source):
    s3, _, client, records, _ = source
    game = next(
        r
        for r in records.values()
        if r["game_date_pacific"] == "2026-11-16" and r["home"]["team"] == "T01"
    )
    root, _ = publish(s3, game)
    del s3.objects[root + "latest.json"]
    data = client.get("/api/recent-team-ratings").json()
    assert data["complete"] is False
    missing = [r for r in data["teams"] if r["availability"] != "available"]
    assert [r["team"] for r in missing] == ["T01", "T02"]
    assert all(
        r["nail"] is None
        and r["game_count"] == 5
        and r["missing_prediction_game_ids"] == [game["game_id"]]
        for r in missing
    )
    assert not any("game_date_pacific=2026-11-11" in key for _, key in s3.calls)


def test_corrected_pointer_is_revalidated_after_ten_minutes(source, monkeypatch):
    s3, _, client, records, _ = source
    before = client.get("/api/recent-team-ratings").json()
    corrected = next(
        deepcopy(r)
        for r in records.values()
        if r["game_date_pacific"] == "2026-11-16" and r["home"]["team"] == "T01"
    )
    corrected["components"]["home_player_total"] += 10
    publish(s3, corrected)
    assert client.get("/api/recent-team-ratings").json() == before
    monkeypatch.setattr(api, "time", lambda: 4201)
    after = client.get("/api/recent-team-ratings").json()
    assert after["teams"][0]["nail"] == pytest.approx(before["teams"][0]["nail"] + 2)
    assert after["teams"][0]["points"][-1]["source"] != before["teams"][0]["points"][-1]["source"]
    assert (
        after["teams"][1] == before["teams"][1]
        or after["teams"][1]["nail"] == before["teams"][1]["nail"]
    )


@pytest.mark.parametrize("damage", ["checksum", "team", "nan", "late"])
def test_corrupt_prediction_fails_explicitly_and_is_not_cached(source, damage):
    s3, _, client, records, _ = source
    record = next(
        deepcopy(r)
        for r in records.values()
        if r["game_date_pacific"] == "2026-11-16" and r["home"]["team"] == "T01"
    )
    if damage == "team":
        record["home"]["team_id"] = 999
    elif damage == "nan":
        record["components"]["home_player_total"] = float("nan")
    elif damage == "late":
        record["generated_at_utc"] = "2026-11-17T02:00:00Z"
    root, sha = publish(s3, record)
    if damage == "checksum":
        s3.objects[root + sha + ".json"] = (b"changed", {})
    response = client.get("/api/recent-team-ratings")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    assert api.cached_recent_ratings.cache_info().currsize == 0


def test_fewer_than_five_games_is_explicitly_unavailable(source):
    _, reader, _, _, _ = source
    data = api.read_recent_ratings(reader, datetime(2026, 11, 14, 2, tzinfo=UTC))
    assert not data["complete"]
    assert all(
        r["availability"] == "insufficient_history" and r["game_count"] == 4 and r["nail"] is None
        for r in data["teams"]
    )


def test_historical_replays_retain_purpose_and_full_precision(source):
    s3, _, client, records, _ = source
    record = next(
        deepcopy(r)
        for r in records.values()
        if r["game_date_pacific"] == "2026-11-16" and r["home"]["team"] == "T01"
    )
    record.update(purpose="historical_replay", generated_at_utc="2026-11-18T12:00:00Z")
    publish(s3, record)
    response = client.get("/api/recent-team-ratings")
    assert response.status_code == 200
    point = response.json()["teams"][0]["points"][-1]
    assert (
        point["purpose"] == "historical_replay"
        and point["nail"] == record["components"]["home_player_total"]
    )
