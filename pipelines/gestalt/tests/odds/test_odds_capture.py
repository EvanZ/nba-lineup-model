import json
from datetime import UTC, datetime, timedelta
from io import BytesIO

import boto3
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


def test_failed_latest_write_reuses_staged_odds(monkeypatch, tmp_path):
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
        with execute_job(reconstructable(_capture_job), instance=instance) as failed:
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

        with execute_job(
            reconstructable(_capture_job), instance=instance, reexecution_options=options,
        ) as recovered:
            assert recovered.success
        stub.assert_no_pending_responses()
        assert not pending.exists()

    assert len(eligibility_checked) == 1
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["sports", "odds"]
    assert calls[1][1]["markets"] == "spreads,h2h"
