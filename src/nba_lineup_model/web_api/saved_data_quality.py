"""Read the saved pipeline event archive without changing operational state."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Response

from .schedule import BUCKET, PACIFIC, _instant, _reader

router = APIRouter()
_logger = logging.getLogger(__name__)
_lock = threading.Lock()

SEASON = "2026-27"
ROOT = "lake/artifacts/errors/"
TTL_SECONDS = 300
EVENT_STATUS = {"failure": "failed", "warning": "warning", "recovery": "recovered"}
EVENT_PATH = re.compile(
    ROOT + rf"pipeline=([a-z0-9_]+)/season={SEASON}/"
    r"occurred_date_pacific=(\d{4}-\d{2}-\d{2})/([a-f0-9]{64})\.(failure|warning|recovery)\.json"
)


@lru_cache(maxsize=2048)
def read_event(key, etag, pipeline, occurred_on, event_id, kind):
    with _reader().client.get_object(Bucket=BUCKET, Key=key, IfMatch=etag)["Body"] as body:
        raw = body.read()
    event = json.loads(raw)
    timestamp = _instant(event["timestamp"])
    if (
        type(event.get("schema_version")) is not int
        or event["schema_version"] != 1
        or event["season"] != SEASON
        or event.get("pipeline", pipeline) != pipeline
        or timestamp.astimezone(PACIFIC).date().isoformat() != occurred_on
        or event["event_id"] != event_id
        or event["status"] != EVENT_STATUS[kind]
    ):
        raise ValueError("Saved data quality event does not match its key")
    message, stage = event["message"], event["stage"]
    if (
        not isinstance(message, str)
        or not message.strip()
        or not isinstance(stage, str)
        or not stage
    ):
        raise ValueError("Saved data quality event lacks a message or stage")
    game_date = event.get("game_date_pacific")
    if game_date is not None and date.fromisoformat(game_date).isoformat() != game_date:
        raise ValueError("Invalid affected game date")
    row = {
        "event_id": event_id,
        "occurred_date": occurred_on,
        "timestamp": timestamp.isoformat(),
        "game_date": game_date,
        "pipeline": pipeline,
        "stage": stage,
        "kind": kind,
        "message": message,
    }
    return row, hashlib.sha256(raw).hexdigest(), event.get("original_failure")


def failure_links(events):
    matching_failures = {}
    recoveries = defaultdict(list)
    for uri, (event, _, original) in events.items():
        if event["kind"] != "recovery" or not isinstance(original, dict):
            continue
        original_uri = original.get("uri")
        if not isinstance(original_uri, str) or original_uri not in events:
            continue
        failure, digest, _ = events[original_uri]
        if (
            failure["kind"] == "failure"
            and failure["pipeline"] == event["pipeline"]
            and failure["stage"] == event["stage"]
            and original.get("sha256") == digest
            and original.get("event_id", failure["event_id"]) == failure["event_id"]
            and _instant(event["timestamp"]) >= _instant(failure["timestamp"])
        ):
            matching_failures[uri] = failure["event_id"]
            recoveries[original_uri].append(event["timestamp"])
    return matching_failures, recoveries


@lru_cache(maxsize=1)
def archive(_period):
    client = _reader().client
    candidates = [
        (key, item["ETag"], *match.groups())
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=ROOT)
        for item in page.get("Contents", ())
        if (match := EVENT_PATH.fullmatch(key := item["Key"]))
    ]
    with ThreadPoolExecutor(max_workers=8) as workers:
        values = workers.map(lambda candidate: read_event(*candidate), candidates)
        events = {
            f"s3://{BUCKET}/{candidate[0]}": item
            for candidate, item in zip(candidates, values, strict=True)
        }
    matching_failures, recoveries = failure_links(events)
    rows = []
    for uri, (event, _, _) in events.items():
        row = dict(event)
        if row["kind"] == "failure":
            row["recovered_at"] = min(recoveries.get(uri, ()), key=_instant, default=None)
        elif row["kind"] == "recovery":
            row["original_failure_id"] = matching_failures.get(uri)
        rows.append(row)
    rows.sort(key=lambda row: (_instant(row["timestamp"]), row["event_id"]), reverse=True)
    checked = datetime.now(PACIFIC)
    today = checked.date().isoformat()
    totals = Counter(row["kind"] for row in rows)
    return {
        "season": SEASON,
        "checked_at": checked.isoformat(),
        "counts": {kind: totals[kind] for kind in EVENT_STATUS},
        "failures_today_without_recovery": sum(
            row["kind"] == "failure" and row["occurred_date"] == today and not row["recovered_at"]
            for row in rows
        ),
        "events": rows,
    }


@router.get("/api/reports/data-quality")
def data_quality(response: Response, summary: bool = False):
    period = int(time() // TTL_SECONDS)
    try:
        with _lock:
            data = archive(period)
    except Exception as error:
        _logger.warning("Saved data quality archive unavailable: %s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Saved data quality events are unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None
    remaining = max(0, int((period + 1) * TTL_SECONDS - time()))
    response.headers["Cache-Control"] = f"private, max-age={remaining}"
    return {key: value for key, value in data.items() if key != "events"} if summary else data
