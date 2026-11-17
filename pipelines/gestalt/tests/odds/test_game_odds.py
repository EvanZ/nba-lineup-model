"""Per-game selection, freeze and S3 provenance regression cases."""
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from dagster import Definitions, RunRequest, SkipReason, build_sensor_context

from gestalt_pipelines.odds.assets import game_odds
from gestalt_pipelines.odds.assets import publish
from gestalt_pipelines.odds.jobs import publish_game_odds
from gestalt_pipelines.odds.partitions import game_odds_dates
from gestalt_pipelines.odds.resources import GameOddsArchive, OddsApi
from gestalt_pipelines.odds.sensors import publish_game_odds_sensor
from gestalt_pipelines.odds import sensors
from gestalt_pipelines.odds.utils import eligible_event
from gestalt_pipelines.resources import NBAScheduleResource

DAY = "2026-11-16"
START = datetime(2026, 11, 17, 1, tzinfo=UTC)
GAME = {"game_id": "0022600250", "game_date": date.fromisoformat(DAY), "game_time_utc": START,
        "away_team_id": 1610612757, "home_team_id": 1610612750}
NAMES = {"away": "Trail Blazers", "home": "Timberwolves"}


def snapshot(collected=START-timedelta(minutes=40), book="espnbet"):
    return {"provider": "The Odds API", "sport_key": "basketball_nba", "season": "2026-27",
        "collected_at": collected.isoformat(), "events": [{"id": "provider-game", "sport_key": "basketball_nba",
        "home_team": "Minnesota Timberwolves", "away_team": "Portland Trail Blazers",
        "commence_time": (START+timedelta(minutes=10)).isoformat(), "bookmakers": [{
        "key": book, "last_update": (collected-timedelta(seconds=10)).isoformat(), "markets": [
        {"key": "h2h", "outcomes": [{"name": "Minnesota Timberwolves", "price": -135.12345},
                                       {"name": "Portland Trail Blazers", "price": 120}]},
        {"key": "spreads", "outcomes": [{"name": "Minnesota Timberwolves", "price": -110, "point": -2.5},
                                           {"name": "Portland Trail Blazers", "price": -110, "point": 2.5}]}]}]}]}


class Store:
    def __init__(self):
        self.objects, self.writes, self.race = {}, [], False

    def get_object(self, **kw):
        if kw["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = self.objects[kw["Key"]]
        return {"Body": BytesIO(raw), "ETag": sha256(raw).hexdigest()}

    def head_object(self, **kw):
        return {"ETag": self.get_object(**kw)["ETag"]}

    def put_object(self, **kw):
        old = self.objects.get(kw["Key"])
        if self.race or ("IfNoneMatch" in kw and old is not None) or ("IfMatch" in kw and (old is None or sha256(old).hexdigest() != kw["IfMatch"])):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        assert kw["Key"].startswith(("lake/artifacts/game_odds/", "lake/artifacts/errors/pipeline=odds/"))
        self.objects[kw["Key"]] = kw["Body"]
        self.writes.append(kw["Key"])

    def get_paginator(self, _):
        return SimpleNamespace(paginate=lambda **kw: [{"Contents": [{"Key": key} for key in self.objects if key.startswith(kw["Prefix"])]}])


def add(store, value):
    at = datetime.fromisoformat(value["collected_at"])
    key = f"lake/raw/odds-api/season=2026-27/snapshots/{at:%Y%m%dT%H%M%SZ}.json"
    store.objects[key] = json.dumps(value).encode()
    return key


@pytest.fixture
def setup(monkeypatch):
    store, archive = Store(), GameOddsArchive(season="2026-27")
    monkeypatch.setattr("boto3.client", lambda *a, **kw: store)
    monkeypatch.setattr(OddsApi, "fetch", lambda *a: pytest.fail("Publisher must not call the provider"))
    schedule = NBAScheduleResource(season="2026-27")
    ref = {"uri": "s3://nba-gestalt-data/test/schedule.json", "sha256": "a"*64}
    raw = json.dumps({"leagueSchedule": {"gameDates": [{"games": [{"gameId": GAME["game_id"],
        "homeTeam": {"teamName": NAMES["home"]}, "awayTeam": {"teamName": NAMES["away"]}}]}]}}).encode()
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *a, **kw: ([GAME], ref))
    monkeypatch.setattr(NBAScheduleResource, "games_between", lambda *a, **kw: ([GAME], ref))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda *a, **kw: (raw, ref))
    clock = {"now": START+timedelta(hours=5)}
    fake_datetime = SimpleNamespace(now=lambda _: clock["now"])
    monkeypatch.setattr(publish, "datetime", fake_datetime)
    monkeypatch.setattr(sensors, "datetime", fake_datetime)
    job = Definitions(assets=[game_odds], jobs=[publish_game_odds], resources={
        "nba_schedule": schedule, "game_odds_archive": archive}).resolve_job_def("publish_game_odds")
    add(store, snapshot())
    return job, archive, store, schedule, clock


def test_historical_job_preserves_raw_prices_and_is_idempotent(setup):
    job, archive, store, _, _ = setup
    originals = dict(store.objects)
    assert job.execute_in_process(partition_key=DAY).success
    pointer, _ = archive.selection(DAY, GAME["game_id"])
    assert pointer["frozen"] and pointer["purpose"] == "historical_replay"
    assert pointer["provider_event_id"] == "provider-game"
    assert pointer["cutoff_utc"] == START.isoformat()
    assert pointer["snapshot"]["sha256"] == sha256(next(iter(originals.values()))).hexdigest()
    assert "prices" not in pointer and "outcomes" not in json.dumps(pointer)
    assert len(store.writes) == 1 and store.writes[0].endswith("/game_id=0022600250/latest.json")
    assert job.execute_in_process(partition_key=DAY).success
    assert len(store.writes) == 1
    assert all(store.objects[k] == raw for k, raw in originals.items())


def test_live_updates_keep_last_pregame_source_when_game_disappears_then_freeze(setup):
    job, archive, store, _, clock = setup
    clock["now"] = START-timedelta(minutes=30)
    assert job.execute_in_process(partition_key=DAY).success
    first, _ = archive.selection(DAY, GAME["game_id"])
    assert not first["frozen"] and first["purpose"] == "pregame_selection"
    newer = snapshot(START-timedelta(minutes=20), "draftkings")
    newer_key = add(store, newer)
    clock["now"] = START-timedelta(minutes=10)
    assert job.execute_in_process(partition_key=DAY).success
    second, _ = archive.selection(DAY, GAME["game_id"])
    assert second["snapshot"]["uri"].endswith(newer_key)
    absent = snapshot(START-timedelta(minutes=5)); absent["events"] = []
    add(store, absent)
    clock["now"] = START
    assert job.execute_in_process(partition_key=DAY).success
    frozen, _ = archive.selection(DAY, GAME["game_id"])
    assert frozen["frozen"] and frozen["snapshot"] == second["snapshot"]
    # A later-discovered earlier capture cannot overwrite a valid frozen pointer.
    add(store, snapshot(START-timedelta(minutes=1)))
    writes = len(store.writes)
    assert job.execute_in_process(partition_key=DAY).success
    assert archive.selection(DAY, GAME["game_id"])[0] == frozen and len(store.writes) == writes


@pytest.mark.parametrize("case", ["at_tipoff", "in_play", "earlier_provider", "late_quote", "no_books"])
def test_no_inplay_or_empty_source_is_selected(case):
    record = snapshot()
    if case == "at_tipoff": record["collected_at"] = START.isoformat()
    if case == "in_play": record["collected_at"] = (START+timedelta(minutes=1)).isoformat()
    if case == "earlier_provider": record["events"][0]["commence_time"] = (START-timedelta(hours=1)).isoformat()
    if case == "late_quote": record["events"][0]["bookmakers"][0]["last_update"] = START.isoformat()
    if case == "no_books": record["events"][0]["bookmakers"] = []
    assert eligible_event(record, GAME, NAMES) is None


@pytest.mark.parametrize("case", ["ambiguous", "duplicate_book", "wrong_sport", "one_side", "bad_price", "bad_spread", "naive_time"])
def test_invalid_selected_sources_fail_explicitly(case):
    record = snapshot(); event = record["events"][0]
    if case == "ambiguous": record["events"].append(deepcopy(event))
    if case == "duplicate_book": event["bookmakers"].append(deepcopy(event["bookmakers"][0]))
    if case == "wrong_sport": event["sport_key"] = "football"
    if case == "one_side": event["bookmakers"][0]["markets"][0]["outcomes"].pop()
    if case == "bad_price": event["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 0
    if case == "bad_spread": event["bookmakers"][0]["markets"][1]["outcomes"][0]["point"] = 99
    if case == "naive_time": record["collected_at"] = "2026-11-16T16:20:00"
    with pytest.raises(ValueError): eligible_event(record, GAME, NAMES)


def test_missing_markets_are_reported_without_fabrication(setup):
    job, archive, store, _, _ = setup
    store.objects.clear()
    result = job.execute_in_process(partition_key=DAY)
    assert result.success and archive.selection(DAY, GAME["game_id"])[0] is None
    metadata = result.asset_materializations_for_node("game_odds")[0].metadata
    assert metadata["missing_game_ids"].value == [GAME["game_id"]]


@pytest.mark.parametrize("case", ["checksum", "wrong_bucket", "wrong_game", "wrong_event", "race"])
def test_bad_pinned_inputs_and_publication_races_are_rejected(setup, case):
    job, archive, store, _, clock = setup
    clock["now"] = START-timedelta(minutes=30)
    assert job.execute_in_process(partition_key=DAY).success
    pointer, _ = archive.selection(DAY, GAME["game_id"])
    if case == "checksum": pointer["snapshot"]["sha256"] = "b"*64
    if case == "wrong_bucket": pointer["snapshot"]["uri"] = pointer["snapshot"]["uri"].replace("nba-gestalt-data", "other")
    if case == "wrong_game": pointer["game_id"] = "other"
    if case == "wrong_event": pointer["provider_event_id"] = "other"
    store.objects[archive.key(DAY, GAME["game_id"])] = json.dumps(pointer).encode()
    store.race = case == "race"
    clock["now"] = START
    assert not job.execute_in_process(partition_key=DAY, raise_on_error=False).success


def test_sensor_only_uses_saved_source_and_clock_and_does_not_scan_archives(setup, monkeypatch):
    job, archive, store, schedule, clock = setup
    clock["now"] = START-timedelta(minutes=30)
    raw = next(iter(store.objects.values()))
    store.objects[archive.raw_root()+"/latest.json"] = raw
    monkeypatch.setattr(GameOddsArchive, "snapshots", lambda *a: pytest.fail("Sensor must not scan the raw archive"))
    with build_sensor_context(resources={"nba_schedule": schedule, "game_odds_archive": archive}) as context:
        requests = list(publish_game_odds_sensor(context)); cursor = context.cursor
    assert len(requests) == 1 and isinstance(requests[0], RunRequest) and requests[0].partition_key == DAY
    with build_sensor_context(cursor=cursor, resources={"nba_schedule": schedule, "game_odds_archive": archive}) as context:
        assert isinstance(list(publish_game_odds_sensor(context))[0], SkipReason)
    clock["now"] = START
    with build_sensor_context(cursor=cursor, resources={"nba_schedule": schedule, "game_odds_archive": archive}) as context:
        final = list(publish_game_odds_sensor(context))
    assert len(final) == 1 and final[0].run_key != requests[0].run_key


def test_partition_window_contains_today_and_seven_future_dates():
    now = START-timedelta(hours=2)
    keys = game_odds_dates.get_partition_keys(current_time=now)
    assert keys[0] == "2026-10-20" and keys[-1] == "2026-11-23"


@pytest.mark.parametrize("stamp", ["2026-11-01T08:30:00+00:00", "2026-11-01T09:30:00+00:00", "2027-03-14T09:30:00+00:00", "2027-03-14T10:30:00+00:00"])
def test_cutoff_uses_real_instants_across_dst(stamp):
    start = datetime.fromisoformat(stamp)
    game = dict(GAME, game_time_utc=start)
    record = snapshot(start-timedelta(minutes=1))
    record["events"][0]["commence_time"] = start.isoformat()
    assert eligible_event(record, game, NAMES)[1] == start
    record["collected_at"] = start.isoformat()
    assert eligible_event(record, game, NAMES) is None


def test_sensor_new_capture_changes_run_key_and_handles_seven_day_game(setup, monkeypatch):
    _, archive, store, schedule, clock = setup
    clock["now"] = START-timedelta(minutes=30)
    future_game = dict(GAME, game_id="future", game_time_utc=START+timedelta(days=7))
    monkeypatch.setattr(NBAScheduleResource, "games_between", lambda *a, **kw: ([future_game], {}))
    latest_key = archive.raw_root()+"/latest.json"
    store.objects[latest_key] = b"first"
    with build_sensor_context(resources={"nba_schedule": schedule, "game_odds_archive": archive}) as context:
        first = list(publish_game_odds_sensor(context))[0]
        cursor = context.cursor
    assert first.partition_key == "2026-11-23"
    store.objects[latest_key] = b"second"
    with build_sensor_context(cursor=cursor, resources={"nba_schedule": schedule, "game_odds_archive": archive}) as context:
        second = list(publish_game_odds_sensor(context))[0]
    assert first.run_key != second.run_key


def test_sensor_does_not_implicitly_backfill_yesterday(setup, monkeypatch):
    _, archive, store, schedule, clock = setup
    clock["now"] = START+timedelta(days=1)
    store.objects[archive.raw_root()+"/latest.json"] = b"first"
    with build_sensor_context(resources={"nba_schedule": schedule, "game_odds_archive": archive}) as context:
        requests = list(publish_game_odds_sensor(context))
    assert not requests


def test_changed_source_or_schedule_identity_is_not_silently_accepted(setup):
    _, archive, store, _, _ = setup
    key = next(iter(store.objects))
    for field, value in [("season", "2025-26"), ("provider", "Other source"), ("collected_at", "2026-11-16T23:00:00Z")]:
        bad = snapshot(); bad[field] = value
        store.objects[key] = json.dumps(bad).encode()
        with pytest.raises(ValueError): archive.snapshot(key)


def test_market_with_only_polymarket_or_only_espnbet_is_eligible():
    for book in ["polymarket", "espnbet"]:
        record = snapshot(book=book)
        assert eligible_event(record, GAME, NAMES)[0]["bookmakers"][0]["key"] == book


@pytest.mark.parametrize("seconds", [-1, 0, 60])
def test_missing_odds_warning_is_saved_only_at_or_after_tipoff(setup, seconds):
    job, archive, store, _, clock = setup
    store.objects.clear()
    clock["now"] = START+timedelta(seconds=seconds)
    result = job.execute_in_process(partition_key=DAY)
    assert result.success and archive.selection(DAY, GAME["game_id"])[0] is None
    warnings = [(key, raw) for key, raw in store.objects.items() if key.endswith(".warning.json")]
    metadata = result.asset_materializations_for_node("game_odds")[0].metadata
    assert len(warnings) == len(metadata["warnings"].value) == (1 if seconds >= 0 else 0)
    if warnings:
        key, raw = warnings[0]
        assert key.startswith("lake/artifacts/errors/pipeline=odds/season=2026-27/occurred_date_pacific=2026-11-16/")
        warning = json.loads(raw)
        assert warning["pipeline"] == "odds" and warning["stage"] == "game_odds"
        assert warning["status"] == "warning" and warning["warning_type"] == "missing_pregame_odds_at_tipoff"
        assert warning["game_id"] == GAME["game_id"] and warning["game_date_pacific"] == DAY
        assert warning["dagster_run_id"] == result.run_id
        assert warning["details"]["scheduled_start_utc"] == START.isoformat()
        assert warning["inputs"]["schedule"]["sha256"] == "a"*64
        assert metadata["warnings"].value[0]["sha256"] == sha256(raw).hexdigest()


def test_warning_observation_date_is_separate_from_historical_game_date(setup):
    job, _, store, _, clock = setup
    store.objects.clear()
    clock["now"] = START+timedelta(days=1)
    assert job.execute_in_process(partition_key=DAY).success
    [(key, raw)] = store.objects.items()
    assert "/occurred_date_pacific=2026-11-17/" in key
    assert json.loads(raw)["game_date_pacific"] == "2026-11-16"


def test_pregame_selection_does_not_emit_missing_odds_warning(setup):
    job, _, store, _, _ = setup
    result = job.execute_in_process(partition_key=DAY)
    assert result.success
    assert not any(key.endswith(".warning.json") for key in store.objects)
    assert result.asset_materializations_for_node("game_odds")[0].metadata["warnings"].value == []


def test_missing_odds_warning_preserves_other_games(setup, monkeypatch):
    job, archive, store, _, _ = setup
    other = dict(GAME, game_id="0022600251", home_team_id=1610612761, away_team_id=1610612738)
    ref = {"uri": "s3://nba-gestalt-data/test/schedule.json", "sha256": "a"*64}
    raw = json.dumps({"leagueSchedule": {"gameDates": [{"games": [
        {"gameId": GAME["game_id"], "homeTeam": {"teamName": NAMES["home"]}, "awayTeam": {"teamName": NAMES["away"]}},
        {"gameId": other["game_id"], "homeTeam": {"teamName": "Raptors"}, "awayTeam": {"teamName": "Celtics"}},
    ]}]}}).encode()
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", lambda *a, **kw: ([GAME, other], ref))
    monkeypatch.setattr(NBAScheduleResource, "load", lambda *a, **kw: (raw, ref))
    result = job.execute_in_process(partition_key=DAY)
    assert result.success and archive.selection(DAY, GAME["game_id"])[0]["frozen"]
    assert archive.selection(DAY, other["game_id"])[0] is None
    warnings = result.asset_materializations_for_node("game_odds")[0].metadata["warnings"].value
    assert len(warnings) == 1 and warnings[0]["game_id"] == other["game_id"]


def test_warning_is_immutable_and_idempotent_within_same_run(setup):
    _, archive, store, _, _ = setup
    kwargs = dict(game=GAME, names=NAMES, schedule={"uri": "saved", "sha256": "a"*64}, run_id="same-run")
    first = archive.save_missing_odds_warning(**kwargs, occurred_at=START)
    original = dict(store.objects)
    second = archive.save_missing_odds_warning(**kwargs, occurred_at=START+timedelta(seconds=20))
    assert second == first and store.objects == original
    kwargs["schedule"] = {"uri": "other", "sha256": "b"*64}
    with pytest.raises(ValueError, match="differs"):
        archive.save_missing_odds_warning(**kwargs, occurred_at=START+timedelta(seconds=30))


def test_warning_write_failure_does_not_silently_succeed(setup, monkeypatch):
    job, _, store, _, _ = setup
    store.objects.clear()
    def fail(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
    monkeypatch.setattr(store, "put_object", fail)
    assert not job.execute_in_process(partition_key=DAY, raise_on_error=False).success


def test_warning_readback_is_verified(setup, monkeypatch):
    _, archive, store, _, _ = setup
    original = store.get_object
    def corrupt(**kwargs):
        result = original(**kwargs)
        if kwargs["Key"].endswith(".warning.json"):
            result["Body"] = BytesIO(b"{}")
        return result
    monkeypatch.setattr(store, "get_object", corrupt)
    with pytest.raises(ValueError, match="read-back"):
        archive.save_missing_odds_warning(game=GAME, names=NAMES, schedule={}, run_id="run", occurred_at=START)


def test_sensor_still_triggers_at_tipoff_when_latest_capture_is_absent(setup):
    _, archive, store, schedule, clock = setup
    store.objects.clear()
    clock["now"] = START-timedelta(seconds=1)
    resources = {"nba_schedule": schedule, "game_odds_archive": archive}
    with build_sensor_context(resources=resources) as context:
        first = list(publish_game_odds_sensor(context)); cursor = context.cursor
    assert len(first) == 1 and isinstance(first[0], RunRequest)
    clock["now"] = START
    with build_sensor_context(cursor=cursor, resources=resources) as context:
        due = list(publish_game_odds_sensor(context))
    assert len(due) == 1 and isinstance(due[0], RunRequest) and due[0].run_key != first[0].run_key
