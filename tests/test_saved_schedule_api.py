"""Selected-date serving reads only official saved objects, never pipeline code."""

import hashlib
import io
import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from nba_lineup_model.web_api import schedule
from nba_lineup_model.web_api.gateway import MarketGateway

SEASON = "2026-27"
DAY = "2026-11-15"
GID = "0022600236"
START = "2026-11-16T00:00:00Z"
NOW = datetime(2026, 11, 15, 20, tzinfo=UTC)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class SavedS3:
    def __init__(self):
        self.objects = {}
        self.calls = []

    def put_fixture(self, key, value, metadata=None):
        raw = encoded(value)
        self.objects[key] = (raw, metadata or {})
        return hashlib.sha256(raw).hexdigest()

    def get_object(self, *, Bucket, Key):
        assert Bucket == schedule.BUCKET
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw, metadata = self.objects[Key]
        return {"Body": io.BytesIO(raw), "Metadata": metadata}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == schedule.BUCKET
        self.calls.append(("list", Prefix))
        yield {
            "Contents": [
                {"Key": key, "LastModified": NOW} for key in self.objects if key.startswith(Prefix)
            ]
        }


def prediction(purpose="pregame_projection"):
    player = {
        "player_id": 100,
        "name": "Saved Player",
        "injury_status": None,
        "injury_reason": None,
        "play_probability": None,
        "conditional_minutes": 32.125,
        "projected_minutes": 30.875,
        "nail": 2.123456789,
        "rating_contribution": 0.655321,
        "selected_rotation": True,
    }
    return {
        "artifact_kind": "game_prediction",
        "schema_version": 1,
        "season": SEASON,
        "model_version": "v1",
        "purpose": purpose,
        "game_id": GID,
        "game_date_pacific": DAY,
        "scheduled_start_utc": START,
        "generated_at_utc": "2026-11-15T12:00:00Z",
        "predicted_home_margin": 3.123456789,
        "home_win_probability": 0.613456789,
        "assumed_possessions": 100,
        "inputs": {"injury_report": {"sha256": "a" * 64}},
        "components": {
            "home_player_total": 3.25,
            "away_player_total": 2.125,
            "hca": 2,
            "home_is_b2b": False,
            "away_is_b2b": False,
            "b2b_coefficient": -1.2,
            "b2b_adjustment": 0,
        },
        "home": {
            "team_id": 1,
            "team": "HOM",
            "players": [player],
            "submission_status": "pending",
            "total_projected_minutes": 240,
        },
        "away": {
            "team_id": 2,
            "team": "AWY",
            "players": [deepcopy(player)],
            "submission_status": "submitted",
            "total_projected_minutes": 240,
        },
        "injury_report": {"source_url": None, "published_at": None, "pdf_uri": None},
        "ratings_observed_through_game_date_pacific": "2026-11-14",
    }


def select_prediction(s3, record):
    root = (
        f"lake/artifacts/game_predictions/season={SEASON}/model_version=v1/"
        f"game_date_pacific={DAY}/game_id={GID}/"
    )
    raw = encoded(record)
    sha = hashlib.sha256(raw).hexdigest()
    s3.put_fixture(root + sha + ".json", record)
    s3.put_fixture(
        root + "latest.json",
        {
            k: record[k]
            for k in ["schema_version", "season", "model_version", "game_date_pacific", "game_id"]
        }
        | {"uri": f"s3://{schedule.BUCKET}/{root}{sha}.json", "sha256": sha},
    )


def fixtures():
    s3 = SavedS3()
    row = {
        "gameId": GID,
        "gameStatus": 1,
        "gameDateTimeUTC": START,
        "homeTeam": {"teamId": 1, "teamTricode": "HOM"},
        "awayTeam": {"teamId": 2, "teamTricode": "AWY"},
        "broadcasters": {"nationalTvBroadcasters": [{"broadcasterDisplay": "Official TV"}]},
    }
    document = {
        "leagueSchedule": {
            "seasonYear": SEASON,
            "leagueId": "00",
            "gameDates": [{"gameDate": DAY, "games": [row]}],
        }
    }
    root = f"lake/raw/scheduleleaguev2/season={SEASON}/"
    sha = hashlib.sha256(encoded(document)).hexdigest()
    s3.put_fixture(root + f"sha256={sha}/schedule.json", document)
    s3.put_fixture(
        root + "latest.json",
        {"sha256": sha, "uri": f"s3://{schedule.BUCKET}/{root}sha256={sha}/schedule.json"},
    )
    select_prediction(s3, prediction())
    return s3, row


def save_board(s3, row, *, state=2):
    row = deepcopy(row)
    row.update(
        gameStatus=state,
        gameStatusText="Final" if state == 3 else "Q2",
        period=2,
        gameClock="PT03M12.00S",
    )
    row["homeTeam"]["score"], row["awayTeam"]["score"] = 50, 49
    document = {"scoreboard": {"gameDate": DAY, "games": [row]}}
    key = f"lake/raw/cdn/scoreboard/season={SEASON}/game_date_pacific={DAY}/latest.json"
    s3.put_fixture(
        key,
        document,
        {
            "sha256": hashlib.sha256(encoded(document)).hexdigest(),
            "fetched-at": "2026-11-16T00:15:00Z",
        },
    )
    return key


def save_box(s3, row):
    game = deepcopy(row)
    game.update(gameStatus=3, gameStatusText="Final")
    metadata = {"season": SEASON, "game_id": GID}
    for side in ("home", "away"):
        team = game[side + "Team"]
        team.update(score=110, players=[{"personId": 100, "name": "Saved Player"}])
        roster = [{"player_id": 100, "player_name": "Saved Player"}]
        metadata[side + "_team_id"] = team["teamId"]
        metadata[side + "_roster_sha256"] = hashlib.sha256(encoded(roster)).hexdigest()
    document = {"game": game}
    sha = hashlib.sha256(encoded(document)).hexdigest()
    root = f"lake/raw/cdn/boxscore/season={SEASON}/game_id={GID}/"
    key = root + sha + ".json"
    s3.put_fixture(key, document)
    metadata["box_score_sha256"] = sha
    s3.put_fixture(root + sha + ".meta.json", metadata)
    return key


def test_saved_precision_pending_identity_and_calendar():
    s3, _ = fixtures()
    view = schedule.ScheduleReader(s3).view(SEASON, DAY, now=NOW)
    game = view["day"]["games"][0]
    assert game["game_id"] == GID and game["status"]["state"] == "pre"
    assert game["projection"]["home_margin"] == 3.123456789
    assert game["projection"]["home_win_probability"] == 0.613456789
    assert game["projected_minutes"]["home"][0]["status"] is None
    assert game["projected_minutes"]["home"][0]["play_probability"] is None
    assert game["projected_minutes"]["home"][0]["projected_minutes"] == 30.875
    assert game["injury_report"]["team_reports"]["HOM"]["status"] == "pending"
    assert view["day"]["collected_at"] is None
    assert view["calendar"][DAY]["game_count"] == 1
    assert len(view["calendar"]) == 7 and view["calendar"]["2026-11-14"]["game_count"] == 0
    assert (
        len(
            [
                key
                for op, key in s3.calls
                if "/game_predictions/" in key and key.endswith("latest.json")
            ]
        )
        == 1
    )
    assert game["market"] is None and game["availability"]["postgame_details"] == "not_applicable"


def test_missing_prediction_does_not_remove_official_game():
    s3, _ = fixtures()
    for key in list(s3.objects):
        if "/game_predictions/" in key:
            del s3.objects[key]
    view = schedule.ScheduleReader(s3).view(SEASON, DAY, now=NOW)
    assert len(view["day"]["games"]) == 1
    assert view["day"]["games"][0]["projection"] is None
    assert view["day"]["coverage"]["missing_prediction_game_ids"] == [GID]
    assert view["day"]["coverage"]["complete"] is False


def test_missing_observation_is_unknown_not_a_fake_final_or_zero_score():
    s3, _ = fixtures()
    view = schedule.ScheduleReader(s3).view(SEASON, DAY, now=datetime(2026, 11, 17, tzinfo=UTC))
    game = view["day"]["games"][0]
    assert game["status"]["state"] == "unknown"
    assert game["home_score"] is None
    assert view["day"]["coverage"]["missing_observation_game_ids"] == [GID]


def test_live_and_final_scoreboard_use_actual_capture_time():
    for state in (2, 3):
        s3, row = fixtures()
        save_board(s3, row, state=state)
        day = schedule.ScheduleReader(s3).view(SEASON, DAY, now=NOW)["day"]
        assert day["games"][0]["status"]["state"] == {2: "in", 3: "post"}[state]
        assert day["games"][0]["home_score"] == 50
        assert day["collected_at"] == "2026-11-16T00:15:00Z"
        assert not [key for op, key in s3.calls if op == "list"]


def test_historical_box_uses_unchanged_season_game_layout_and_sidecar():
    s3, row = fixtures()
    key = save_box(s3, row)
    record = prediction("historical_replay")
    select_prediction(s3, record)
    day = schedule.ScheduleReader(s3).view(SEASON, DAY, now=datetime(2026, 11, 17, tzinfo=UTC))[
        "day"
    ]
    game = day["games"][0]
    assert game["status"]["completed"] and game["home_score"] == 110
    assert game["prediction_purpose"] == "historical_replay"
    assert game["sources"]["observation"]["uri"].endswith(key)
    assert ("list", f"lake/raw/cdn/boxscore/season={SEASON}/game_id={GID}/") in s3.calls


@pytest.mark.parametrize("damage", ["checksum", "team", "date", "game"])
def test_corrupt_or_mismatched_scoreboard_fails_explicitly(damage):
    s3, row = fixtures()
    key = save_board(s3, row)
    raw, metadata = s3.objects[key]
    doc = json.loads(raw)
    if damage == "team":
        doc["scoreboard"]["games"][0]["homeTeam"]["teamId"] = 999
    if damage == "date":
        doc["scoreboard"]["gameDate"] = "2026-11-14"
    if damage == "game":
        doc["scoreboard"]["games"][0]["gameId"] = "0022600999"
    raw = encoded(doc)
    metadata["sha256"] = "0" * 64 if damage == "checksum" else hashlib.sha256(raw).hexdigest()
    s3.objects[key] = raw, metadata
    with pytest.raises(ValueError):
        schedule.ScheduleReader(s3).view(SEASON, DAY, now=NOW)


def test_bad_box_sidecar_is_not_repaired_or_silently_skipped():
    s3, row = fixtures()
    key = save_box(s3, row)[:-5] + ".meta.json"
    s3.put_fixture(key, {})
    with pytest.raises(ValueError, match="sidecar"):
        schedule.ScheduleReader(s3).view(SEASON, DAY, now=datetime(2026, 11, 17, tzinfo=UTC))


def test_empty_slate_uses_no_predictions_or_observations():
    s3, _ = fixtures()
    view = schedule.ScheduleReader(s3).view(SEASON, "2026-11-03", now=NOW)
    assert view["day"]["available"] and view["day"]["games"] == []
    assert view["day"]["coverage"]["complete"]
    assert all("/scheduleleaguev2/" in key for op, key in s3.calls)


def test_routes_use_same_minute_cache_without_legacy_initialization(monkeypatch):
    s3, _ = fixtures()
    monkeypatch.setattr(schedule, "_reader", lambda: schedule.ScheduleReader(s3))
    clock = [1800000001.0]
    monkeypatch.setattr(schedule, "time", lambda: clock[0])
    schedule._cached_view.cache_clear()

    def legacy():
        pytest.fail("Schedule must never initialize legacy models or release data")

    with TestClient(MarketGateway(legacy)) as client:
        view = client.get("/api/schedule-view?date=" + DAY)
        assert view.status_code == 200 and "max-age=" in view.headers["Cache-Control"]
        calls = len(s3.calls)
        day = client.get("/api/schedule?date=" + DAY)
        assert day.json() == view.json()["day"] and len(s3.calls) == calls
        clock[0] += 60
        assert client.get("/api/schedule-view?date=" + DAY).status_code == 200
        assert len(s3.calls) > calls
        assert client.get("/api/schedule-view?date=bad").status_code == 422
    schedule._cached_view.cache_clear()


def test_missing_schedule_is_503_not_no_games_and_not_cached(monkeypatch):
    s3 = SavedS3()
    monkeypatch.setattr(schedule, "_reader", lambda: schedule.ScheduleReader(s3))
    schedule._cached_view.cache_clear()
    with TestClient(MarketGateway(None)) as client:
        result = client.get("/api/schedule-view?date=" + DAY)
        assert result.status_code == 503
        assert result.headers["Cache-Control"] == "no-store"
        assert schedule._cached_view.cache_info().currsize == 0


@pytest.mark.parametrize(
    "field,value",
    [("game_id", "0022600999"), ("scheduled_start_utc", "2026-11-16T01:00:00Z"), ("home", None)],
)
def test_prediction_identity_does_not_fall_back_to_team_names(field, value):
    s3, _ = fixtures()
    record = prediction()
    if field == "home":
        record["home"]["team_id"] = 999
    else:
        record[field] = value
    select_prediction(s3, record)
    with pytest.raises(ValueError):
        schedule.ScheduleReader(s3).view(SEASON, DAY, now=NOW)


def test_incomplete_scoreboard_preserves_missing_game():
    s3, row = fixtures()
    key = save_board(s3, row)
    doc = {"scoreboard": {"gameDate": DAY, "games": []}}
    s3.put_fixture(
        key,
        doc,
        {"sha256": hashlib.sha256(encoded(doc)).hexdigest(), "fetched-at": "2026-11-16T00:15:00Z"},
    )
    day = schedule.ScheduleReader(s3).view(SEASON, DAY, now=datetime(2026, 11, 17, tzinfo=UTC))[
        "day"
    ]
    assert len(day["games"]) == 1
    assert day["coverage"]["missing_scoreboard_game_ids"] == [GID]
    assert day["coverage"]["missing_observation_game_ids"] == [GID]


def test_mutable_prediction_pointer_refreshes_but_immutable_values_are_cached():
    s3, _ = fixtures()
    reader = schedule.ScheduleReader(s3)
    first = reader.view(SEASON, DAY, now=NOW)
    record = prediction()
    record["predicted_home_margin"] = 7.654321
    select_prediction(s3, record)
    second = reader.view(SEASON, DAY, now=NOW)
    assert first["day"]["games"][0]["projection"]["home_margin"] == 3.123456789
    assert second["day"]["games"][0]["projection"]["home_margin"] == 7.654321
    schedule_reads = [key for op, key in s3.calls if key.endswith("/schedule.json")]
    assert len(schedule_reads) == 1
