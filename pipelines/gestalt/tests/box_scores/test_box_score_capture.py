import base64
import json
from contextlib import nullcontext
from datetime import UTC, date, datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from dagster import DagsterInstance, build_schedule_context

from gestalt_pipelines.box_scores import BOX_SCORE_POLL_NAME, schedules
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
                     "score": 102, "players": [{"personId": 101, "name": "Away Player"}]},
        "homeTeam": {"teamId": game["home_team_id"], "teamTricode": game["home_team_tricode"],
                     "score": 104, "players": [{"personId": 102, "name": "Home Player"}]},
    }}).encode()


@pytest.fixture
def box_score_job(monkeypatch):
    def games_for_date(self, day, *, allow_empty):
        return (GAMES if day == DAY else []), SCHEDULE

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
        if not Key.endswith(".meta.json"):
            assert Key.endswith(f"/{checksum.hexdigest()}.json")
        assert ChecksumSHA256 == base64.b64encode(checksum.digest()).decode("ascii")
        objects[Key] = Body

    def get_object(**args):
        key = args["Key"]
        if key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(objects[key])}

    client = SimpleNamespace(
        get_paginator=lambda _: SimpleNamespace(paginate=paginate),
        get_object=get_object,
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
    assert [body for key, body in objects.items() if not key.endswith(".meta.json")] == [_box(GAMES[0], 3)]

    final_ids.add(GAMES[1]["game_id"])
    requested.clear()
    second = job.execute_in_process(partition_key=DAY.isoformat())
    assert second.success
    assert requested == [GAMES[1]["game_id"]]
    assert len(objects) == 4

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
    assert [body for key, body in objects.items() if not key.endswith(".meta.json")] == [_box(GAMES[0], 3)]

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


def test_schedule_polls_only_missing_started_games_and_carries_overnight(monkeypatch, box_score_job):
    job, objects, final_ids, requested = box_score_job
    monkeypatch.setitem(GAMES[1], "game_time_utc", datetime(2026, 11, 5, 3, tzinfo=UTC))
    clock = SimpleNamespace(now=lambda _: None)
    monkeypatch.setattr(capture, "datetime", clock)
    monkeypatch.setattr(schedules, "datetime", clock)
    schedule = defs.get_schedule_def(BOX_SCORE_POLL_NAME)

    with DagsterInstance.local_temp() as instance:
        def tick(when):
            clock.now = lambda _: when
            with build_schedule_context(
                instance=instance,
                scheduled_execution_time=when,
                repository_def=defs.get_repository_def(),
                resources={"nba_schedule": NBAScheduleResource(season="2026-27")},
            ) as context:
                return schedule.evaluate_tick(context).run_requests

        def run(request):
            return job.execute_in_process(
                instance=instance,
                partition_key=request.partition_key,
                tags={"dagster/schedule_name": BOX_SCORE_POLL_NAME},
            )

        assert not tick(datetime(2026, 11, 4, 20, 55, tzinfo=UTC))
        [request] = tick(datetime(2026, 11, 4, 21, 5, tzinfo=UTC))
        first = run(request)
        assert first.success and not first.get_asset_materialization_events()
        assert requested == [GAMES[0]["game_id"]]
        assert not tick(datetime(2026, 11, 4, 21, 10, tzinfo=UTC))

        [request] = tick(datetime(2026, 11, 5, 8, 5, tzinfo=UTC))
        assert request.partition_key == DAY.isoformat()
        pending = run(request)
        assert pending.success and not pending.get_asset_materialization_events()
        assert len(objects) == 2 and requested[-1] == GAMES[1]["game_id"]

        final_ids.add(GAMES[1]["game_id"])
        [request] = tick(datetime(2026, 11, 5, 8, 10, tzinfo=UTC))
        complete = run(request)
        assert complete.success and complete.get_asset_materialization_events()
        assert len(objects) == 4
        assert not tick(datetime(2026, 11, 5, 8, 15, tzinfo=UTC))


def test_backfill_adds_missing_sidecars_without_fetching_or_rewriting_boxes(monkeypatch, box_score_job):
    job, objects, _, requested = box_score_job
    for game in GAMES:
        payload = _box(game, 3)
        digest = sha256(payload).hexdigest()
        objects[f"lake/raw/cdn/boxscore/season=2026-27/game_id={game['game_id']}/{digest}.json"] = payload
    original_boxes = dict(objects)

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("An archived final box must not be fetched again.")

    monkeypatch.setattr(archive.requests, "get", unexpected_fetch)
    result = job.execute_in_process(partition_key=DAY.isoformat())
    metadata = result.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["captured_count"].value == 0
    assert metadata["reused_count"].value == metadata["metadata_created_count"].value == 2
    assert metadata["metadata_reused_count"].value == 0
    assert not requested

    for key, payload in original_boxes.items():
        assert objects[key] == payload
        sidecar = json.loads(objects[key.removesuffix(".json") + ".meta.json"])
        assert sidecar["box_score_sha256"] == sha256(payload).hexdigest()
        game = json.loads(payload)["game"]
        assert sidecar["game_id"] == game["gameId"] and sidecar["season"] == "2026-27"
        assert sidecar["home_team_id"] == game["homeTeam"]["teamId"]
        assert sidecar["away_team_id"] == game["awayTeam"]["teamId"]
        assert len(sidecar["home_roster_sha256"]) == len(sidecar["away_roster_sha256"]) == 64

    saved = dict(objects)
    again = job.execute_in_process(partition_key=DAY.isoformat())
    metadata = again.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert metadata["metadata_created_count"].value == 0
    assert metadata["metadata_reused_count"].value == 2
    assert objects == saved


def test_incorrect_existing_metadata_fails_without_overwriting_it(box_score_job):
    job, objects, final_ids, _ = box_score_job
    final_ids.update(game["game_id"] for game in GAMES)
    assert job.execute_in_process(partition_key=DAY.isoformat()).success
    key = next(key for key in objects if key.endswith(".meta.json"))
    bad = json.loads(objects[key])
    bad["home_roster_sha256"] = "0" * 64
    objects[key] = json.dumps(bad).encode()
    saved = dict(objects)

    result = job.execute_in_process(partition_key=DAY.isoformat(), raise_on_error=False)
    assert not result.success
    assert objects == saved


def test_roster_hash_tracks_identity_and_ignores_order_and_game_statistics():
    team = {
        "teamId": 1, "score": 20,
        "players": [
            {"personId": 12, "name": "Player Two", "statistics": {"minutes": "PT12M"}},
            {"personId": 11, "name": "Player One", "statistics": {"minutes": "PT00M"}},
        ],
    }
    original = archive._roster_hash(team)
    team["score"] = 100
    team["players"].reverse()
    team["players"][0]["statistics"]["minutes"] = "PT30M"
    assert archive._roster_hash(team) == original
    team["players"][0]["name"] = "Corrected Name"
    assert archive._roster_hash(team) != original
    team["players"].pop()
    assert archive._roster_hash(team) != original
    team["players"].append(dict(team["players"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        archive._roster_hash(team)
