import base64
import json
from contextlib import nullcontext
from datetime import UTC, date, datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest

from gestalt_pipelines.box_scores.assets import capture
from gestalt_pipelines.box_scores.resources import archive
from gestalt_pipelines.definitions import defs
from gestalt_pipelines.resources import NBAScheduleResource

DAY = date(2026, 11, 4)
SCHEDULE = {"uri": "s3://schedule/immutable.json", "sha256": "test"}
GAMES = [
    {"game_id": "0022600172", "away_team_id": 1, "away_team_tricode": "MIA",
     "home_team_id": 2, "home_team_tricode": "LAL"},
    {"game_id": "0022600173", "away_team_id": 3, "away_team_tricode": "PHI",
     "home_team_id": 4, "home_team_tricode": "DET"},
]


def _box(game, status):
    return json.dumps({"game": {
        "gameId": game["game_id"],
        "gameStatus": status,
        "awayTeam": {"teamId": game["away_team_id"], "teamTricode": game["away_team_tricode"],
                     "score": 102, "players": [{"personId": 101}]},
        "homeTeam": {"teamId": game["home_team_id"], "teamTricode": game["home_team_tricode"],
                     "score": 104, "players": [{"personId": 102}]},
    }}).encode()


@pytest.fixture
def box_score_job(monkeypatch):
    def games_for_date(self, day, *, allow_empty):
        assert day == DAY
        return GAMES, SCHEDULE

    objects = {}

    def paginate(*, Bucket, Prefix):
        assert Bucket == "nba-gestalt-data"
        return [{"Contents": [
            {"Key": key, "LastModified": datetime(2026, 11, 5, tzinfo=UTC)}
            for key in objects if key.startswith(Prefix)
        ]}]

    def put_object(*, Bucket, Key, Body, IfNoneMatch, ChecksumSHA256, **_):
        assert Bucket == "nba-gestalt-data"
        assert IfNoneMatch == "*" and Key not in objects
        checksum = sha256(Body)
        assert Key.endswith(f"/{checksum.hexdigest()}.json")
        assert ChecksumSHA256 == base64.b64encode(checksum.digest()).decode("ascii")
        objects[Key] = Body

    client = SimpleNamespace(
        get_paginator=lambda _: SimpleNamespace(paginate=paginate),
        get_object=lambda **args: {"Body": BytesIO(objects[args["Key"]])},
        put_object=put_object,
    )
    for index, game in enumerate(GAMES):
        game["game_time_utc"] = datetime(2026, 11, 4, 21 + 2 * index, tzinfo=UTC)
    final_ids = {GAMES[0]["game_id"]}
    requested = []

    def get(url, **kwargs):
        game_id = url.rsplit("_", 1)[-1].removesuffix(".json")
        requested.append(game_id)
        game = next(game for game in GAMES if game["game_id"] == game_id)
        return nullcontext(SimpleNamespace(
            status_code=200, raise_for_status=lambda: None,
            content=_box(game, 3 if game_id in final_ids else 2),
        ))

    monkeypatch.setattr(NBAScheduleResource, "games_for_date", games_for_date)
    monkeypatch.setattr(archive.boto3, "client", lambda _: client)
    monkeypatch.setattr(archive.requests, "get", get)
    job = defs.resolve_job_def("capture_nba_box_scores")
    return job, objects, final_ids, requested


def test_partition_retry_preserves_final_boxes_and_fetches_only_missing_games(box_score_job):
    job, objects, final_ids, requested = box_score_job
    first = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not first.success
    assert requested == [game["game_id"] for game in GAMES]
    assert list(objects.values()) == [_box(GAMES[0], 3)]

    final_ids.add(GAMES[1]["game_id"])
    requested.clear()
    second = job.execute_in_process(partition_key=DAY.isoformat())
    assert second.success
    assert requested == [GAMES[1]["game_id"]]
    assert len(objects) == 2

    requested.clear()
    reused = job.execute_in_process(partition_key=DAY.isoformat())
    assert reused.success and not requested
    metadata = reused.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["reused_count"].value == 2
    assert metadata["captured_count"].value == 0
    assert metadata["schedule"].value == SCHEDULE


def test_today_waits_for_final_boxes_before_materializing(monkeypatch, box_score_job):
    job, objects, final_ids, requested = box_score_job
    clock = SimpleNamespace(now=lambda _: datetime(2026, 11, 4, 20, tzinfo=UTC))
    monkeypatch.setattr(capture, "datetime", clock)

    def run():
        return job.execute_in_process(partition_key=DAY.isoformat())

    def check_pending(result, expected):
        assert result.success
        assert not result.get_asset_materialization_events()
        observations = [event.event_specific_data.asset_observation for event in result.all_events if event.is_asset_observation]
        assert observations[0].metadata["pending_game_ids"].value == expected

    check_pending(run(), [game["game_id"] for game in GAMES])
    assert not requested and not objects

    clock.now = lambda _: datetime(2026, 11, 4, 23, 30, tzinfo=UTC)
    check_pending(run(), [GAMES[1]["game_id"]])
    assert requested == [game["game_id"] for game in GAMES]
    assert list(objects.values()) == [_box(GAMES[0], 3)]

    final_ids.add(GAMES[1]["game_id"])
    requested.clear()
    complete = run()
    assert complete.success and requested == [GAMES[1]["game_id"]]
    metadata = complete.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["pending_game_ids"].value == []
    assert metadata["reused_count"].value == metadata["captured_count"].value == 1


def test_today_still_reports_source_errors(monkeypatch, box_score_job):
    job, objects, _, _ = box_score_job
    monkeypatch.setattr(capture, "datetime", SimpleNamespace(
        now=lambda _: datetime(2026, 11, 4, 23, 30, tzinfo=UTC),
    ))

    def denied(*args, **kwargs):
        raise archive.requests.HTTPError("NBA CDN returned 403.")

    monkeypatch.setattr(archive.requests, "get", denied)
    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    assert not objects and not result.get_asset_materialization_events()
