import base64
import json
from contextlib import nullcontext
from datetime import UTC, date, datetime, timedelta
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
            {"Key": key, "LastModified": datetime(2026, 11, 5, tzinfo=UTC)+timedelta(microseconds=index)}
            for index, key in enumerate(objects) if key.startswith(Prefix)
        ]}]

    def put_object(*, Bucket, Key, Body, IfNoneMatch, ChecksumSHA256, **_):
        assert Bucket == "nba-gestalt-data"
        assert IfNoneMatch == "*"
        if Key in objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        checksum = sha256(Body)
        if Key.startswith("lake/raw/cdn/boxscore/") and not Key.endswith(".meta.json"):
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


REFRESH_CONFIG = {"ops": {"nba_box_scores": {"config": {"refresh_existing": True}}}}


def test_morning_refresh_fetches_all_games_and_preserves_unchanged_files(box_score_job):
    job, objects, final_ids, requested = box_score_job
    final_ids.update(game["game_id"] for game in GAMES)
    assert job.execute_in_process(partition_key=str(DAY)).success
    before = dict(objects)
    requested.clear()
    result = job.execute_in_process(partition_key=str(DAY), run_config=REFRESH_CONFIG)
    assert result.success and requested == [g["game_id"] for g in GAMES]
    assert objects == before
    meta = result.get_asset_materialization_events()[0].event_specific_data.materialization.metadata
    assert meta["refresh_existing"].value and meta["refreshed_game_count"].value == 2
    assert meta["reused_count"].value == 2 and meta["changed_count"].value == 0


def test_morning_refresh_saves_correction_sidecar_and_warning(monkeypatch, box_score_job):
    job, objects, final_ids, requested = box_score_job
    final_ids.update(game["game_id"] for game in GAMES)
    assert job.execute_in_process(partition_key=str(DAY)).success
    before = dict(objects)
    original_get = archive.requests.get
    changed = json.loads(_box(GAMES[0], 3))
    changed["game"]["homeTeam"]["score"] = 106
    payload = json.dumps(changed).encode()
    def get(url, **kwargs):
        if GAMES[0]["game_id"] in url:
            requested.append(GAMES[0]["game_id"])
            return nullcontext(SimpleNamespace(status_code=200, raise_for_status=lambda: None, content=payload))
        return original_get(url, **kwargs)
    monkeypatch.setattr(archive.requests, "get", get)
    requested.clear()
    result = job.execute_in_process(partition_key=str(DAY), run_config=REFRESH_CONFIG)
    assert result.success and requested == [g["game_id"] for g in GAMES]
    assert all(objects[key] == value for key, value in before.items())
    raw, ref = archive.NBABoxScoreArchive().read_saved("2026-27", GAMES[0])
    assert raw == payload and ref["sha256"] == sha256(payload).hexdigest()
    assert ref["metadata_uri"].removeprefix("s3://nba-gestalt-data/") in objects
    warnings = [(key,json.loads(body)) for key,body in objects.items() if key.endswith(".warning.json")]
    assert len(warnings) == 1
    key, warning = warnings[0]
    assert key.startswith("lake/artifacts/errors/pipeline=box_scores/season=2026-27/occurred_date_pacific=")
    assert warning["game_date_pacific"] == str(DAY) and warning["dagster_run_id"] == result.run_id
    assert warning["details"]["previous"]["sha256"] == sha256(_box(GAMES[0],3)).hexdigest()
    assert warning["details"]["current"]["sha256"] == ref["sha256"]
    before_rerun = dict(objects)
    assert job.execute_in_process(partition_key=str(DAY),run_config=REFRESH_CONFIG).success
    assert objects == before_rerun


@pytest.mark.parametrize("problem", ["missing", "timeout", "nonfinal", "wrong_game", "invalid_json"])
def test_refresh_failure_does_not_silently_reuse_cached_finals(problem, monkeypatch, box_score_job):
    job, objects, final_ids, requested = box_score_job
    final_ids.update(g["game_id"] for g in GAMES)
    assert job.execute_in_process(partition_key=str(DAY)).success
    before = dict(objects)
    original_get = archive.requests.get
    def get(url, **kwargs):
        if GAMES[0]["game_id"] not in url:
            return original_get(url, **kwargs)
        requested.append(GAMES[0]["game_id"])
        if problem == "missing":
            raise archive.requests.HTTPError("NBA CDN returned 404")
        if problem == "timeout":
            raise archive.requests.Timeout("NBA CDN timed out")
        payload = _box(GAMES[0], 2 if problem == "nonfinal" else 3)
        if problem == "wrong_game":
            document = json.loads(payload)
            document["game"]["gameId"] = "0022600999"
            payload = json.dumps(document).encode()
        if problem == "invalid_json":
            payload = b"not NBA JSON"
        return nullcontext(SimpleNamespace(status_code=200, raise_for_status=lambda: None, content=payload))
    monkeypatch.setattr(archive.requests, "get", get)
    result = job.execute_in_process(partition_key=str(DAY),run_config=REFRESH_CONFIG,raise_on_error=False)
    assert not result.success and not result.get_asset_materialization_events()
    assert all(objects[key] == value for key,value in before.items())
    errors = [json.loads(v) for k,v in objects.items() if k.endswith(".failure.json")]
    assert len(errors) == 1
    event = errors[0]
    assert event["status"] == "failed" and event["dagster_run_id"] == result.run_id
    assert set(event["details"]["failed_games"]) == {GAMES[0]["game_id"]}
    assert set(event["details"]["box_scores"]) == {GAMES[1]["game_id"]}
    assert event["input"]["schedule"] == SCHEDULE


def test_morning_refresh_with_partial_missing_archive_keeps_valid_downloads(box_score_job):
    job, objects, _, _ = box_score_job
    result = job.execute_in_process(partition_key=str(DAY),run_config=REFRESH_CONFIG,raise_on_error=False)
    assert not result.success
    raw = [v for k,v in objects.items() if k.startswith("lake/raw/") and not k.endswith(".meta.json")]
    assert raw == [_box(GAMES[0],3)]
    assert any(k.endswith(".failure.json") for k in objects)


@pytest.mark.parametrize("day,hour", [("2026-11-16",2),("2027-03-14",3),("2027-11-07",2)])
def test_morning_schedule_uses_previous_pacific_date_including_dst(day, hour, monkeypatch):
    from zoneinfo import ZoneInfo
    from dagster._utils.schedules import schedule_execution_time_iterator
    zone = ZoneInfo("America/Los_Angeles")
    schedule = defs.get_schedule_def("box_scores_morning_refresh")
    assert schedule.cron_schedule == "45 2 * * *" and schedule.execution_timezone == zone.key
    start = datetime.fromisoformat(day).replace(tzinfo=zone).timestamp()
    tick = next(schedule_execution_time_iterator(start,schedule.cron_schedule,execution_timezone=zone.key))
    assert tick.hour == hour
    assert tick.minute == (0 if hour == 3 else 45)
    monkeypatch.setattr(schedules,"datetime",SimpleNamespace(now=lambda _:tick.astimezone(UTC)))
    with build_schedule_context(scheduled_execution_time=tick,repository_def=defs.get_repository_def()) as context:
        [request] = schedule.evaluate_tick(context).run_requests
    assert request.partition_key == str(tick.date()-timedelta(days=1))
    assert request.run_config == REFRESH_CONFIG


@pytest.mark.parametrize("case",["missed","preseason"])
def test_morning_schedule_skips_missed_and_preseason_dates(case,monkeypatch):
    from zoneinfo import ZoneInfo
    tick = datetime(2026,11,5,2,45,tzinfo=ZoneInfo("America/Los_Angeles"))
    if case == "preseason":
        tick = tick.replace(year=schedules.box_score_game_dates.start.year,
                            month=schedules.box_score_game_dates.start.month,
                            day=schedules.box_score_game_dates.start.day)
    now = tick.astimezone(UTC)+(timedelta(days=2) if case == "missed" else timedelta())
    monkeypatch.setattr(schedules,"datetime",SimpleNamespace(now=lambda _:now))
    with build_schedule_context(scheduled_execution_time=tick,repository_def=defs.get_repository_def()) as context:
        assert not defs.get_schedule_def("box_scores_morning_refresh").evaluate_tick(context).run_requests


def test_refresh_rejects_response_that_would_leave_an_older_version_unselected(monkeypatch, box_score_job):
    job, objects, final_ids, _ = box_score_job
    final_ids.update(g["game_id"] for g in GAMES)
    assert job.execute_in_process(partition_key=str(DAY)).success
    original_get = archive.requests.get
    changed = json.loads(_box(GAMES[0],3))
    changed["game"]["homeTeam"]["score"] = 106
    payload = json.dumps(changed).encode()
    def get(url, **kwargs):
        if GAMES[0]["game_id"] in url:
            return nullcontext(SimpleNamespace(status_code=200, raise_for_status=lambda:None, content=payload))
        return original_get(url, **kwargs)
    monkeypatch.setattr(archive.requests,"get",get)
    assert job.execute_in_process(partition_key=str(DAY),run_config=REFRESH_CONFIG).success
    before = dict(objects)
    monkeypatch.setattr(archive.requests,"get",original_get)
    result = job.execute_in_process(partition_key=str(DAY),run_config=REFRESH_CONFIG,raise_on_error=False)
    assert not result.success
    assert all(objects[key] == value for key,value in before.items())
    assert archive.NBABoxScoreArchive().read_saved("2026-27",GAMES[0])[0] == payload
    error = next(json.loads(v) for k,v in objects.items() if k.endswith(".failure.json"))
    assert "older archived version" in error["details"]["failed_games"][GAMES[0]["game_id"]]


def test_explicit_refresh_requires_final_coverage_even_for_current_date(monkeypatch,box_score_job):
    job,objects,_,_ = box_score_job
    monkeypatch.setattr(capture,"datetime",SimpleNamespace(now=lambda _:datetime(2026,11,4,23,30,tzinfo=UTC)))
    result = job.execute_in_process(partition_key=str(DAY),run_config=REFRESH_CONFIG,raise_on_error=False)
    assert not result.success and not result.get_asset_materialization_events()
    assert any(key.endswith(".failure.json") for key in objects)
