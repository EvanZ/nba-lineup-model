"""Team curves consume exact accepted totals without ratings or forecast work."""

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from test_saved_schedule_api import DAY, SEASON, encoded, fixtures, prediction

from nba_lineup_model.web_api import team_rating_history as api
from nba_lineup_model.web_api.gateway import MarketGateway
from nba_lineup_model.web_api.schedule import BUCKET, ScheduleReader


def publish(s3, record, *, consistent=True):
    if consistent:
        c = record["components"]
        record["predicted_home_margin"] = (
            c["home_player_total"] - c["away_player_total"] + c["hca"] + c["b2b_adjustment"]
        )
    root = (
        f"lake/artifacts/game_predictions/season={SEASON}/model_version=v1/"
        f"game_date_pacific={record['game_date_pacific']}/game_id={record['game_id']}/"
    )
    sha = s3.put_fixture(root + hashlib.sha256(encoded(record)).hexdigest() + ".json", record)
    pointer = {
        key: record[key]
        for key in ["schema_version", "season", "model_version", "game_date_pacific", "game_id"]
    }
    pointer.update(uri=f"s3://{BUCKET}/{root}{sha}.json", sha256=sha)
    s3.put_fixture(root + "latest.json", pointer)
    return root, sha


@pytest.fixture
def source(monkeypatch):
    s3, row = fixtures()
    before = deepcopy(row)
    before.update(gameId="0022600230", gameDateTimeUTC="2026-11-15T00:00:00Z")
    before["homeTeam"], before["awayTeam"] = before["awayTeam"], before["homeTeam"]
    missing = deepcopy(row)
    missing.update(gameId="0022600220", gameDateTimeUTC="2026-11-14T00:00:00Z")
    future = deepcopy(row)
    future.update(gameId="0022600240", gameDateTimeUTC="2026-11-17T00:00:00Z")
    other = deepcopy(before)
    other.update(
        gameId="0022600231",
        homeTeam={"teamId": 3, "teamTricode": "OTH"},
        awayTeam={"teamId": 4, "teamTricode": "ELSE"},
    )
    doc = {
        "leagueSchedule": {
            "seasonYear": SEASON,
            "leagueId": "00",
            "gameDates": [{"gameDate": DAY, "games": [row, before, missing, future, other]}],
        }
    }
    root = f"lake/raw/scheduleleaguev2/season={SEASON}/"
    sha = hashlib.sha256(encoded(doc)).hexdigest()
    s3.put_fixture(root + f"sha256={sha}/schedule.json", doc)
    s3.put_fixture(
        root + "latest.json",
        {"sha256": sha, "uri": f"s3://{BUCKET}/{root}sha256={sha}/schedule.json"},
    )
    previous = prediction()
    previous.update(
        game_id=before["gameId"],
        game_date_pacific="2026-11-14",
        scheduled_start_utc=before["gameDateTimeUTC"],
        generated_at_utc="2026-11-14T12:00:00Z",
    )
    previous["home"], previous["away"] = previous["away"], previous["home"]
    previous["components"]["away_player_total"] = -1.2345678912345
    publish(s3, previous)
    publish(s3, prediction())
    reader = ScheduleReader(s3)
    monkeypatch.setattr(api, "_reader", lambda: reader)
    monkeypatch.setattr(api, "time", lambda: 3601)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 16, 6, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(api, "datetime", Clock)
    api.cached_history.cache_clear()
    api.cached_team_logo.cache_clear()
    legacy = Mock(side_effect=AssertionError("Legacy must not load"))
    yield s3, reader, TestClient(MarketGateway(legacy)), previous, legacy
    api.cached_history.cache_clear()
    api.cached_team_logo.cache_clear()


def test_exact_totals_order_opponents_and_explicit_missing_game(source):
    s3, _, client, _, legacy = source
    response = client.get("/api/teams/hom/rating-history")
    assert response.status_code == 200
    data = response.json()
    assert data["team"] == "HOM" and data["team_id"] == 1
    assert data["coverage"] == {
        "scheduled_games": 3,
        "available_games": 2,
        "missing_prediction_game_ids": ["0022600220"],
    }
    assert data["complete"] is False
    missing, away, home = data["points"]
    assert missing["nail"] is None and missing["availability"] == "missing"
    assert away["nail"] == -1.2345678912345 and away["home_away"] == "away"
    assert home["nail"] == 3.25 and home["opponent"] == "AWY"
    assert home["projection_state"] == "frozen_pregame"
    assert home["source"]["sha256"] in home["source"]["uri"]
    assert response.headers["cache-control"] == "private, max-age=599"
    assert not any("0022600240" in key or "0022600231" in key for _, key in s3.calls)
    assert all(operation == "get" for operation, _ in s3.calls)
    calls = len(s3.calls)
    assert client.get("/api/teams/HOM/rating-history").json() == data
    assert len(s3.calls) == calls
    legacy.assert_not_called()


def test_changed_pointer_refreshes_after_ten_minutes_preserving_pinned_inputs(source, monkeypatch):
    s3, _, client, _, _ = source
    first = client.get("/api/teams/HOM/rating-history").json()
    corrected = prediction()
    corrected["components"]["home_player_total"] = 4.123456789123
    publish(s3, corrected)
    assert client.get("/api/teams/HOM/rating-history").json() == first
    monkeypatch.setattr(api, "time", lambda: 4201)
    after = client.get("/api/teams/HOM/rating-history").json()
    assert after["points"][-1]["nail"] == 4.123456789123
    assert after["points"][-1]["source"] != first["points"][-1]["source"]
    assert corrected["inputs"] == prediction()["inputs"]


@pytest.mark.parametrize("damage", ["checksum", "team", "date", "late", "nan", "pointer"])
def test_invalid_accepted_prediction_fails_and_is_not_cached(source, damage):
    s3, reader, client, _, _ = source
    record = prediction()
    if damage == "team":
        record["home"]["team_id"] = 999
    elif damage == "date":
        record["game_date_pacific"] = "2026-11-14"
    elif damage == "late":
        record["generated_at_utc"] = "2026-11-16T00:00:01Z"
    elif damage == "nan":
        record["components"]["home_player_total"] = float("nan")
    root, sha = publish(s3, record)
    if damage == "checksum":
        s3.objects[root + sha + ".json"] = (b"changed", {})
    elif damage == "pointer":
        key = root + "latest.json"
        pointer = json.loads(s3.objects[key][0])
        pointer["uri"] = "s3://other/other.json"
        s3.put_fixture(key, pointer)
    elif damage == "date":
        correct_root = root.replace("game_date_pacific=2026-11-14", f"game_date_pacific={DAY}")
        pointer = json.loads(s3.objects[root + "latest.json"][0])
        pointer["uri"] = f"s3://{BUCKET}/{correct_root}{sha}.json"
        s3.put_fixture(correct_root + "latest.json", pointer)
        s3.objects[correct_root + sha + ".json"] = s3.objects[root + sha + ".json"]
    response = client.get("/api/teams/HOM/rating-history")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert api.cached_history.cache_info().currsize == 0
    publish(s3, prediction())
    reader.immutable.cache_clear()
    assert client.get("/api/teams/HOM/rating-history").status_code == 200


def test_historical_replay_is_not_labeled_frozen_pregame(source):
    s3, _, client, _, _ = source
    record = prediction("historical_replay")
    record["generated_at_utc"] = "2026-11-18T12:00:00Z"
    publish(s3, record)
    point = client.get("/api/teams/HOM/rating-history").json()["points"][-1]
    assert point["projection_state"] == "historical_replay"
    assert point["purpose"] == "historical_replay" and point["nail"] == 3.25


def test_upcoming_is_provisional_not_frozen(source):
    _, reader, _, _, _ = source
    data = api.read_history(reader, "HOM", SEASON, datetime(2026, 11, 15, 20, tzinfo=UTC))
    assert data["points"][-1]["projection_state"] == "upcoming"


def test_unknown_team_and_unpublished_season_do_not_use_legacy(source):
    s3, _, client, _, legacy = source
    assert client.get("/api/teams/ZZZ/rating-history").status_code == 404
    calls = len(s3.calls)
    response = client.get("/api/teams/HOM/rating-history?season=2025-26")
    assert response.status_code == 404 and response.headers["cache-control"] == "no-store"
    assert len(s3.calls) == calls
    legacy.assert_not_called()


@pytest.mark.parametrize(
    "home_b2b,away_b2b,b2b",
    [(False, False, 0), (True, False, -1.2), (False, True, 1.2), (True, True, 0)],
)
def test_both_team_series_and_signed_saved_adjustments(source, home_b2b, away_b2b, b2b):
    s3, _, client, _, _ = source
    record = prediction()
    record["components"].update(
        hca=1.87654321, home_is_b2b=home_b2b, away_is_b2b=away_b2b, b2b_adjustment=b2b
    )
    publish(s3, record)
    home = client.get("/api/teams/HOM/rating-history").json()["points"][-1]
    away = client.get("/api/teams/AWY/rating-history").json()["points"][-1]
    assert home["nail"] == away["opponent_nail"] == 3.25
    assert away["nail"] == home["opponent_nail"] == 2.125
    assert home["adjustments"] == {
        "hca": 1.87654321,
        "b2b": b2b,
        "team_is_back_to_back": home_b2b,
        "opponent_is_back_to_back": away_b2b,
    }
    assert away["adjustments"] == {
        "hca": -1.87654321,
        "b2b": -b2b,
        "team_is_back_to_back": away_b2b,
        "opponent_is_back_to_back": home_b2b,
    }
    assert home["predicted_margin"] == -away["predicted_margin"] == record["predicted_home_margin"]
    for point in (home, away):
        adjusted = point["nail"] + point["adjustments"]["hca"] + point["adjustments"]["b2b"]
        assert adjusted - point["opponent_nail"] == pytest.approx(point["predicted_margin"])
    assert home["source"] == away["source"]


@pytest.mark.parametrize(
    "field", ["hca", "b2b_adjustment", "away_player_total", "home_is_b2b", "predicted_home_margin"]
)
def test_missing_or_inconsistent_matchup_components_fail_explicitly(source, field):
    s3, _, client, _, _ = source
    record = prediction()
    record["predicted_home_margin"] = 3.125
    if field == "predicted_home_margin":
        record[field] = 99
    else:
        del record["components"][field]
    publish(s3, record, consistent=False)
    response = client.get("/api/teams/HOM/rating-history")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    assert api.cached_history.cache_info().currsize == 0


def test_official_logo_delivery_is_cached_and_bounded(source, monkeypatch):
    _, _, client, _, legacy = source
    raw = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
    fetch = Mock(
        return_value=httpx.Response(
            200,
            content=raw,
            headers={"content-type": "image/svg+xml"},
            request=httpx.Request("GET", "https://cdn.nba.com/logo.svg"),
        )
    )
    monkeypatch.setattr(api.httpx, "get", fetch)
    for _ in range(2):
        response = client.get("/api/team-logos/1610612747.svg")
        assert response.status_code == 200 and response.content == raw
        assert response.headers["cache-control"] == "public, max-age=86400"
    fetch.assert_called_once_with(
        "https://cdn.nba.com/logos/nba/1610612747/global/L/logo.svg",
        timeout=10.0,
        follow_redirects=False,
    )
    assert client.get("/api/team-logos/123.svg").status_code == 422
    assert client.get("/api/team-logos/1610612767.svg").status_code == 422
    assert fetch.call_count == 1
    legacy.assert_not_called()


@pytest.mark.parametrize(
    "raw,content_type",
    [
        (b"<html>bad</html>", "text/html"),
        (b"bad", "image/svg+xml"),
        (b"<svg" + b" " * 256_000, "image/svg+xml"),
    ],
)
def test_invalid_logos_fail_without_empty_success_or_cache(source, monkeypatch, raw, content_type):
    _, _, client, _, _ = source
    monkeypatch.setattr(
        api.httpx,
        "get",
        Mock(
            return_value=httpx.Response(
                200,
                content=raw,
                headers={"content-type": content_type},
                request=httpx.Request("GET", "https://cdn.nba.com/logo.svg"),
            )
        ),
    )
    response = client.get("/api/team-logos/1610612747.svg")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"
    assert api.cached_team_logo.cache_info().currsize == 0
