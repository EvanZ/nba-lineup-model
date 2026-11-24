"""The saved archive is independent of the legacy report and alert state."""

import hashlib
import io
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from fastapi.testclient import TestClient

from nba_lineup_model.web_api import saved_data_quality as quality
from nba_lineup_model.web_api.gateway import MarketGateway


def test_report_requires_exact_recovery_link_and_shares_summary_cache(monkeypatch):
    yesterday = datetime.now(quality.PACIFIC) - timedelta(days=1)
    today = datetime.now(quality.PACIFIC)
    objects = {}

    def publish(kind, digit, timestamp, **extra):
        event_id = digit * 64
        key = (
            f"{quality.ROOT}pipeline=ratings/season={quality.SEASON}/"
            f"occurred_date_pacific={timestamp.date()}/{event_id}.{kind}.json"
        )
        raw = json.dumps(
            {
                "schema_version": 1,
                "event_id": event_id,
                "season": quality.SEASON,
                "stage": "player_ratings",
                "status": quality.EVENT_STATUS[kind],
                "timestamp": timestamp.isoformat(),
                "message": "Saved test event.",
                **extra,
            }
        ).encode()
        objects[key] = raw
        return {"uri": f"s3://{quality.BUCKET}/{key}", "sha256": hashlib.sha256(raw).hexdigest()}

    first = publish("failure", "a", yesterday)
    second = publish("failure", "b", today)
    publish("recovery", "c", today, original_failure=first)
    publish("recovery", "d", today, original_failure={**second, "sha256": first["sha256"]})
    publish("warning", "e", today)

    class SavedEvents:
        reads = 0

        def get_paginator(self, method):
            assert method == "list_objects_v2"
            return self

        def paginate(self, **params):
            assert params == {"Bucket": quality.BUCKET, "Prefix": quality.ROOT}
            return iter([{"Contents": [{"Key": key, "ETag": '"saved"'} for key in objects]}])

        def get_object(self, **params):
            assert params["Bucket"] == quality.BUCKET
            assert params["IfMatch"] == '"saved"'
            self.reads += 1
            return {"Body": io.BytesIO(objects[params["Key"]])}

    store = SavedEvents()
    monkeypatch.setattr(quality, "_reader", lambda: SimpleNamespace(client=store))
    monkeypatch.setattr(quality, "time", lambda: 301)
    quality.archive.cache_clear()
    quality.read_event.cache_clear()
    try:
        client = TestClient(MarketGateway(None))
        url = "/api/reports/data-quality"
        summary = client.get(url, params={"summary": "true"})
        assert summary.status_code == 200
        assert summary.json()["failures_today_without_recovery"] == 1
        assert "events" not in summary.json()

        detail = client.get(url)
        assert detail.status_code == 200
        assert detail.headers["cache-control"] == "private, max-age=299"
        assert detail.json()["counts"] == {"failure": 2, "warning": 1, "recovery": 2}
        rows = {row["event_id"][0]: row for row in detail.json()["events"]}
        assert rows["a"]["recovered_at"] == today.isoformat()
        assert rows["b"]["recovered_at"] is None
        assert rows["c"]["original_failure_id"] == "a" * 64
        assert rows["d"]["original_failure_id"] is None
        assert store.reads == len(objects)
    finally:
        quality.archive.cache_clear()
        quality.read_event.cache_clear()
