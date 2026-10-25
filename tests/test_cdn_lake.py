from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nba_lineup_model.ingest.nba_cdn import CachedResponse, NbaCdnEndpoint
from nba_lineup_model.season.cdn_lake import (
    CdnBatchError,
    CdnLakeArchive,
    ingest_completed_cdn,
    pending_completed_cdn_game_ids,
)
from nba_lineup_model.season.stats_v3_lake import CompletedScheduleGame


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def list_keys(self, prefix_uri: str) -> list[str]:
        bucket, key = prefix_uri.removeprefix("s3://").split("/", 1)
        prefix = f"s3://{bucket}/{key.rstrip('/')}/"
        return [
            uri.removeprefix(f"s3://{bucket}/") for uri in self.objects if uri.startswith(prefix)
        ]

    def put_bytes(self, uri: str, payload: bytes) -> bool:
        if uri in self.objects:
            if self.objects[uri] != payload:
                raise RuntimeError(f"immutable mismatch: {uri}")
            return False
        self.objects[uri] = payload
        return True


def _write_final_slate(root: Path) -> None:
    path = root / "latest" / "2026-10-20.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "season": "2026-27",
                "game_date": "2026-10-20",
                "games": [
                    {
                        "game_id": "0022600001",
                        "espn_game_id": "401909088",
                        "start_time": "2026-10-21T00:00:00Z",
                        "final_observed_at": "2026-10-21T03:00:00Z",
                        "away_team": "BOS",
                        "home_team": "DET",
                        "status": {"state": "post", "completed": True},
                    }
                ],
            }
        )
    )


def _game() -> CompletedScheduleGame:
    return CompletedScheduleGame(
        game_id="0022600001",
        espn_game_id="401909088",
        season="2026-27",
        game_date=date(2026, 10, 20),
        start_time=datetime(2026, 10, 21, 0, 0, tzinfo=UTC),
        final_observed_at=datetime(2026, 10, 21, 3, 0, tzinfo=UTC),
        away_team="BOS",
        home_team="DET",
    )


def _response(suffix: str = "") -> CachedResponse:
    payload = {
        "game": {
            "gameId": "0022600001",
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 10000,
                    "description": suffix,
                }
            ],
        }
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return CachedResponse(
        endpoint=NbaCdnEndpoint.PLAY_BY_PLAY,
        game_id="0022600001",
        url=("https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_0022600001.json"),
        fetched_at=datetime(2026, 10, 21, 4, 0, tzinfo=UTC),
        response_headers={"content-type": "application/json"},
        payload=payload,
        raw_body=raw,
    )


def test_archive_uses_distinct_cdn_namespace_and_retains_revisions() -> None:
    store = MemoryObjectStore()
    archive = CdnLakeArchive(store=store)

    first = archive.archive(
        _game(),
        _response(),
        run_id="run-1",
        attempt_number=1,
        known_sha256=[],
    )
    revised = archive.archive(
        _game(),
        _response("corrected"),
        run_id="run-2",
        attempt_number=1,
        known_sha256=[first.sha256],
    )

    assert first.disposition == "archived"
    assert revised.disposition == "revised"
    assert revised.prior_sha256 == first.sha256
    raw_objects = [
        uri
        for uri in store.objects
        if "/raw/cdn/playbyplay/" in uri and not uri.endswith(".meta.json")
    ]
    assert len(raw_objects) == 2


def test_concurrent_identical_archive_is_an_idempotent_checksum_hit() -> None:
    store = MemoryObjectStore()
    archive = CdnLakeArchive(store=store)
    response = _response()

    first = archive.archive(
        _game(),
        response,
        run_id="run-1",
        attempt_number=1,
        known_sha256=[],
    )
    raced = archive.archive(
        _game(),
        response,
        run_id="run-2",
        attempt_number=1,
        known_sha256=[],
    )

    assert first.disposition == "archived"
    assert raced.disposition == "unchanged"
    assert raced.prior_sha256 == first.sha256
    assert len([uri for uri in store.objects if "/success-" in uri]) == 1


def test_ingest_is_resumable_and_fetches_only_play_by_play(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = CdnLakeArchive(store=store)
    calls: list[str] = []

    def fetch(game_id: str) -> CachedResponse:
        calls.append(game_id)
        return _response()

    first = ingest_completed_cdn(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        fetch_response=fetch,
    )
    second = ingest_completed_cdn(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        fetch_response=fetch,
    )

    assert first["archived_count"] == 1
    assert second["requested_endpoint_count"] == 0
    assert second["skipped_count"] == 1
    assert calls == ["0022600001"]


def test_refresh_with_same_bytes_does_not_add_success_record(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = CdnLakeArchive(store=store)

    ingest_completed_cdn(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        fetch_response=lambda _game_id: _response(),
    )
    refreshed = ingest_completed_cdn(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        refresh=True,
        fetch_response=lambda _game_id: _response(),
    )

    assert refreshed["unchanged_count"] == 1
    success_records = [uri for uri in store.objects if "/success-" in uri]
    assert len(success_records) == 1


def test_pending_uses_small_local_coverage_index(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)

    pending = pending_completed_cdn_game_ids(
        state_dir=tmp_path,
        season="2026-27",
        coverage_path=tmp_path / "coverage.json",
        now=datetime(2026, 10, 21, 5, 0, tzinfo=UTC),
    )

    assert pending == ["0022600001"]


def test_failure_is_manifested_and_fails_batch(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = CdnLakeArchive(store=store)

    def fail(_game_id: str) -> CachedResponse:
        raise ValueError("bad response")

    with pytest.raises(CdnBatchError) as raised:
        ingest_completed_cdn(
            state_dir=tmp_path,
            coverage_path=tmp_path / "coverage.json",
            archive=archive,
            season="2026-27",
            ignore_grace=True,
            fetch_response=fail,
        )

    assert raised.value.summary["failed_count"] == 1
    assert len([uri for uri in store.objects if "/failure-" in uri]) == 1
