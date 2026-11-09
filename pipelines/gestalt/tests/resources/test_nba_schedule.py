import json
from datetime import UTC, date, datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from gestalt_pipelines.resources import nba_schedule

BUCKET = "nba-gestalt-data"
PREFIX = "lake/raw/scheduleleaguev2/season=2026-27"
LATEST_KEY = f"{PREFIX}/latest.json"
FETCHED_AT = datetime(2026, 11, 8, tzinfo=UTC)


@pytest.fixture
def objects(monkeypatch):
    stored = {}

    def put_object(*, Bucket, Key, Body, IfNoneMatch=None, Metadata=None, **_):
        assert Bucket == BUCKET
        if IfNoneMatch == "*" and Key in stored:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        stored[Key] = Body, Metadata

    def get_object(*, Bucket, Key):
        assert Bucket == BUCKET
        return {"Body": BytesIO(stored[Key][0])}

    client = SimpleNamespace(put_object=put_object, get_object=get_object)
    monkeypatch.setattr(nba_schedule.boto3, "client", lambda _: client)
    return stored


def test_publish_and_load_from_a_fresh_resource(objects):
    producer = nba_schedule.NBAScheduleResource(season="2026-27")
    original = b'{"leagueSchedule":{"gameDates":[]}}'
    original_ref = producer.publish(original, fetched_at=FETCHED_AT)
    digest = sha256(original).hexdigest()
    key = f"{PREFIX}/sha256={digest}/schedule.json"
    assert original_ref == {"uri": f"s3://{BUCKET}/{key}", "sha256": digest}
    assert objects[key] == (original, {
        "source-url": nba_schedule.SOURCE_URL,
        "first-fetched-at": FETCHED_AT.isoformat(),
        "sha256": digest,
    })
    assert json.loads(objects[LATEST_KEY][0]) == original_ref

    updated = b'{"leagueSchedule":{"gameDates":[{}]}}'
    updated_ref = producer.publish(updated, fetched_at=FETCHED_AT)
    consumer = nba_schedule.NBAScheduleResource(season="2026-27")
    assert consumer.load() == (updated, updated_ref)
    assert consumer.load(original_ref) == (original, original_ref)
    other_season = nba_schedule.NBAScheduleResource(season="2027-28")
    with pytest.raises(ValueError, match="for 2027-28"):
        other_season.load(original_ref)


def test_publish_keeps_current_reference_if_archive_verification_fails(objects):
    resource = nba_schedule.NBAScheduleResource(season="2026-27")
    original_ref = resource.publish(b"{}", fetched_at=FETCHED_AT)
    payload = b'{"leagueSchedule":{}}'
    digest = sha256(payload).hexdigest()
    objects[f"{PREFIX}/sha256={digest}/schedule.json"] = b"corrupt bytes", {}
    with pytest.raises(ValueError, match="do not match their checksum"):
        resource.publish(payload, fetched_at=FETCHED_AT)
    assert json.loads(objects[LATEST_KEY][0]) == original_ref


def test_games_for_date_uses_pacific_tipoff_date(objects):
    season = "2026-27"
    payload = json.dumps({"leagueSchedule": {
        "seasonYear": season, "leagueId": "00",
        "gameDates": [{"gameDate": "11/08/2026", "games": [{
            "gameId": "late",
            "gameDateEst": "2026-11-08T00:00:00Z",
            "gameDateTimeUTC": "2026-11-09T01:00:00Z",
            "awayTeam": {"teamId": 3, "teamTricode": "CCC"},
            "homeTeam": {"teamId": 4, "teamTricode": "DDD"},
            "broadcasters": {"nationalBroadcasters": [{"broadcasterDisplay": "ESPN"}]},
        }, {
            "gameId": "cup_slot", "gameStatus": 1,
            "awayTeam": {"teamId": 0}, "homeTeam": {"teamId": 0},
        }]}],
    }}).encode()
    resource = nba_schedule.NBAScheduleResource(season=season)
    day = date(2026, 11, 8)
    reference = resource.publish(payload, fetched_at=FETCHED_AT)
    games, selected = resource.games_for_date(day)
    assert selected == reference
    assert len(games) == 1
    assert games[0]["game_id"] == "late" and games[0]["game_date"] == day
    assert "broadcasters" not in games[0]

    bad_payload = payload.replace(b'"broadcasters"', b'"espnData"')
    bad_ref = resource.publish(bad_payload, fetched_at=FETCHED_AT)
    with pytest.raises(ValueError, match="Prohibited field"):
        resource.games_for_date(day, bad_ref)
