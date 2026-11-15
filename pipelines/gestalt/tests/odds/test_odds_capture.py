import json
from datetime import UTC, datetime, timedelta
from io import BytesIO

import boto3
import pytest
from types import SimpleNamespace
from botocore.stub import Stubber
from dagster import (
    DagsterInstance,
    ReexecutionOptions,
    execute_job,
    reconstructable,
)

from gestalt_pipelines.definitions import defs
from gestalt_pipelines.odds.resources import OddsApi
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.utils import PACIFIC


def _capture_job():
    return defs.resolve_job_def("archive_nba_market_odds")


def test_quota_reserve_prevents_paid_request(monkeypatch):
    calls = []

    def get(self, url, **params):
        calls.append(url)
        return [], {"x-requests-remaining": "51"}

    monkeypatch.setattr(OddsApi, "_get", get)
    assert OddsApi(api_key="test").fetch("2026-27") is None
    assert calls == ["https://api.the-odds-api.com/v4/sports"]


@pytest.mark.parametrize("crosses_midnight", [False, True])
def test_failed_latest_write_reuses_staged_odds(monkeypatch, tmp_path, crosses_midnight):
    calls = []
    eligibility_checked = []

    def get(self, url, **params):
        calls.append((url, params))
        return [{"bookmakers": []}], {
            "x-requests-remaining": "100",
            "x-requests-used": "400",
            "x-requests-last": "2",
        }

    def games_for_date(self, *args, **kwargs):
        eligibility_checked.append(True)
        return [{"game_time_utc": datetime.now(UTC) + timedelta(hours=1)}], {}

    monkeypatch.setenv("GESTALT_ODDS_API_KEY", "test")
    monkeypatch.setattr(OddsApi, "_get", get)
    monkeypatch.setattr(NBAScheduleResource, "games_for_date", games_for_date)
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test",
    )
    monkeypatch.setattr(boto3, "client", lambda _: client)
    stub = Stubber(client)
    stub.add_response("put_object", {})
    stub.add_client_error("get_object", "NoSuchKey")
    stub.add_client_error("put_object", "InternalError", http_status_code=500)

    with stub, DagsterInstance.local_temp(str(tmp_path)) as instance:
        with execute_job(reconstructable(_capture_job), instance=instance, tags={"dagster/partition": str(datetime.now(PACIFIC).date())}) as failed:
            assert not failed.success
            options = ReexecutionOptions.from_failure(failed.run_id, instance)
        pending = next(tmp_path.rglob(f"{failed.run_id}.json"))
        payload = pending.read_bytes()
        bucket = "nba-gestalt-data"
        prefix = "lake/raw/odds-api/season=2026-27"
        collected = datetime.fromisoformat(json.loads(payload)["collected_at"])
        archive = f"{prefix}/snapshots/{collected:%Y%m%dT%H%M%SZ}.json"
        latest = f"{prefix}/latest.json"
        stub.add_client_error("put_object", "PreconditionFailed", http_status_code=412)
        stub.add_response(
            "get_object", {"Body": BytesIO(payload)}, {"Bucket": bucket, "Key": archive},
        )
        stub.add_client_error(
            "get_object", "NoSuchKey", expected_params={"Bucket": bucket, "Key": latest},
        )
        stub.add_response("put_object", {}, {
            "Bucket": bucket, "Key": latest, "Body": payload, "ContentType": "application/json",
        })

        partition = str(collected.astimezone(PACIFIC).date())
        if crosses_midnight:
            from gestalt_pipelines.odds.assets import capture
            next_day = collected.astimezone(PACIFIC)+timedelta(days=1)
            monkeypatch.setattr(capture, "datetime", SimpleNamespace(
                now=lambda _: next_day, fromisoformat=datetime.fromisoformat,
            ))
        with execute_job(
            reconstructable(_capture_job), instance=instance, reexecution_options=options,
            tags={"dagster/partition": partition},
        ) as recovered:
            assert recovered.success
        stub.assert_no_pending_responses()
        assert not pending.exists()

    assert len(eligibility_checked) == 1
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["sports", "odds"]
    assert calls[1][1]["markets"] == "spreads,h2h"


def test_tenth_bookmaker_espnbet_is_requested_and_published_unchanged(monkeypatch):
    from copy import deepcopy
    from gestalt_pipelines.odds.resources.odds_api import BOOKMAKERS, MARKETS, SPORTS_URL

    events = [{
        "id": "nba-game", "bookmakers": [{
            "key": "espnbet", "title": "ESPN BET", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "point": -3.5, "price": -110}]},
                {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -150}]},
            ],
        }],
    }]
    calls = []

    def get(self, url, **params):
        calls.append((url, params))
        if url == SPORTS_URL:
            return [], {"x-requests-remaining": "52"}
        return deepcopy(events), {
            "x-requests-remaining": "50", "x-requests-used": "450", "x-requests-last": "2",
        }

    monkeypatch.setattr(OddsApi, "_get", get)
    resource = OddsApi(api_key="test")
    snapshot = resource.fetch("2026-27")
    assert len(BOOKMAKERS) == len(set(BOOKMAKERS)) == 10
    assert "espnbet" in BOOKMAKERS
    assert snapshot["events"] == events
    assert snapshot["request"]["bookmakers"] == list(BOOKMAKERS)
    assert snapshot["quota"]["requests_remaining"] == snapshot["quota"]["minimum_remaining_credits"] == 50
    assert snapshot["quota"]["request_cost"] == len(MARKETS) == 2
    assert calls == [
        (SPORTS_URL, {}),
        (f"{SPORTS_URL}/basketball_nba/odds", {
            "bookmakers": ",".join(BOOKMAKERS), "markets": "spreads,h2h",
            "oddsFormat": "american", "dateFormat": "iso",
        }),
    ]

    payload = json.dumps(snapshot).encode()
    collected = datetime.fromisoformat(snapshot["collected_at"])
    prefix = "lake/raw/odds-api/season=2026-27"
    archive_key = f"{prefix}/snapshots/{collected:%Y%m%dT%H%M%SZ}.json"
    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test")
    monkeypatch.setattr(boto3, "client", lambda _: client)
    with Stubber(client) as stub:
        stub.add_response("put_object", {}, {
            "Bucket": "nba-gestalt-data", "Key": archive_key, "Body": payload,
            "ContentType": "application/json", "IfNoneMatch": "*",
        })
        stub.add_client_error("get_object", "NoSuchKey", expected_params={
            "Bucket": "nba-gestalt-data", "Key": f"{prefix}/latest.json",
        })
        stub.add_response("put_object", {}, {
            "Bucket": "nba-gestalt-data", "Key": f"{prefix}/latest.json", "Body": payload,
            "ContentType": "application/json",
        })
        assert resource.publish("2026-27", payload) == f"s3://nba-gestalt-data/{archive_key}"
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("field,value", [("season", "2025-26"), ("provider", "unapproved provider")])
def test_publish_still_rejects_the_wrong_season_or_provider(monkeypatch, field, value):
    snapshot = {"season": "2026-27", "provider": "The Odds API", field: value}
    monkeypatch.setattr(boto3, "client", lambda _: pytest.fail("Invalid sources must not write S3"))
    with pytest.raises(ValueError, match="season or source validation"):
        OddsApi(api_key="unused").publish("2026-27", json.dumps(snapshot).encode())
