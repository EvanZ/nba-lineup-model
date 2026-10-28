from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from nba_lineup_model.ingest.nba_stats import NbaStatsEndpoint, StatsCachedResponse
from nba_lineup_model.season.stats_v3_lake import (
    AwsCliImmutableObjectStore,
    CompletedScheduleGame,
    StatsV3BatchError,
    StatsV3LakeArchive,
    discover_completed_schedule_games,
    ingest_completed_stats_v3,
    pending_completed_game_ids,
)


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


def test_aws_store_verifies_with_s3api_file_download(monkeypatch) -> None:
    payload = b"immutable payload"
    calls: list[list[str]] = []

    def fake_run(args, *, capture_output, check=False):
        calls.append(list(args))
        if args[:3] == ["aws", "s3api", "put-object"]:
            return subprocess.CompletedProcess(args, 0, stdout=b"{}", stderr=b"")
        assert args[:3] == ["aws", "s3api", "get-object"]
        Path(args[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(args, 0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    created = AwsCliImmutableObjectStore().put_bytes(
        "s3://example/lake/object.bin", payload
    )

    assert created is True
    assert calls[1][:3] == ["aws", "s3api", "get-object"]
    assert "--checksum-mode" in calls[1]


def _write_final_slate(
    root: Path,
    *,
    final_observed_at: str = "2026-10-21T03:00:00Z",
) -> None:
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
                        "final_observed_at": final_observed_at,
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


def _response(endpoint: NbaStatsEndpoint, suffix: str = "") -> StatsCachedResponse:
    if endpoint is NbaStatsEndpoint.PLAY_BY_PLAY_V3:
        payload = {
            "game": {
                "gameId": "0022600001",
                "actions": [{"actionNumber": 1, "description": suffix}],
            }
        }
    else:
        payload = {
            "boxScoreTraditional": {
                "gameId": "0022600001",
                "homeTeam": {"teamTricode": "DET"},
                "awayTeam": {"teamTricode": "BOS"},
                "revision": suffix,
            }
        }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return StatsCachedResponse(
        endpoint=endpoint,
        game_id="0022600001",
        url=f"https://stats.nba.com/stats/{endpoint.value}?GameID=0022600001",
        fetched_at=datetime(2026, 10, 21, 4, 0, tzinfo=UTC),
        response_headers={"content-type": "application/json"},
        payload=payload,
        raw_body=raw,
    )


def test_completed_game_waits_for_one_hour_grace_period(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)

    too_soon = discover_completed_schedule_games(
        tmp_path,
        season="2026-27",
        now=datetime(2026, 10, 21, 3, 59, tzinfo=UTC),
    )
    eligible = discover_completed_schedule_games(
        tmp_path,
        season="2026-27",
        now=datetime(2026, 10, 21, 4, 0, tzinfo=UTC),
    )

    assert too_soon == []
    assert [game.game_id for game in eligible] == ["0022600001"]


def test_archive_is_content_addressed_and_retains_revisions() -> None:
    store = MemoryObjectStore()
    archive = StatsV3LakeArchive(store=store)
    game = _game()
    first_response = _response(NbaStatsEndpoint.PLAY_BY_PLAY_V3)

    first = archive.archive(
        game,
        first_response,
        run_id="run-1",
        attempt_number=1,
        known_sha256=[],
    )
    unchanged = archive.archive(
        game,
        first_response,
        run_id="run-2",
        attempt_number=1,
        known_sha256=[first.sha256],
    )
    revised = archive.archive(
        game,
        _response(NbaStatsEndpoint.PLAY_BY_PLAY_V3, "corrected"),
        run_id="run-3",
        attempt_number=1,
        known_sha256=[first.sha256],
    )

    assert first.disposition == "archived"
    assert unchanged.disposition == "unchanged"
    assert unchanged.record_uri is None
    assert revised.disposition == "revised"
    assert revised.prior_sha256 == first.sha256
    raw_objects = [
        uri
        for uri in store.objects
        if "/raw/stats/playbyplayv3/" in uri and not uri.endswith(".meta.json")
    ]
    assert len(raw_objects) == 2


def test_batch_backfill_is_resumable_and_noop_rerun_is_idempotent(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = StatsV3LakeArchive(store=store)
    calls: list[NbaStatsEndpoint] = []

    def fetch(endpoint: NbaStatsEndpoint, _game_id: str) -> StatsCachedResponse:
        calls.append(endpoint)
        return _response(endpoint)

    first = ingest_completed_stats_v3(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        fetch_response=fetch,
        now=datetime(2026, 10, 21, 5, 0, tzinfo=UTC),
    )
    second = ingest_completed_stats_v3(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        fetch_response=fetch,
        now=datetime(2026, 10, 21, 5, 5, tzinfo=UTC),
    )

    assert first["archived_count"] == 2
    assert second["requested_endpoint_count"] == 0
    assert second["skipped_count"] == 2
    assert calls == [
        NbaStatsEndpoint.PLAY_BY_PLAY_V3,
        NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
    ]
    success_records = [uri for uri in store.objects if "/records/" in uri and "/success-" in uri]
    assert len(success_records) == 2


def test_refresh_with_identical_bytes_does_not_add_artifact_records(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = StatsV3LakeArchive(store=store)

    def fetch(endpoint: NbaStatsEndpoint, _game_id: str) -> StatsCachedResponse:
        return _response(endpoint)

    ingest_completed_stats_v3(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        fetch_response=fetch,
    )
    refreshed = ingest_completed_stats_v3(
        state_dir=tmp_path,
        coverage_path=tmp_path / "coverage.json",
        archive=archive,
        season="2026-27",
        ignore_grace=True,
        refresh=True,
        fetch_response=fetch,
    )

    assert refreshed["unchanged_count"] == 2
    success_records = [uri for uri in store.objects if "/records/" in uri and "/success-" in uri]
    assert len(success_records) == 2


def test_pending_requires_both_endpoint_artifacts(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = StatsV3LakeArchive(store=store)
    game = _game()
    archive.archive(
        game,
        _response(NbaStatsEndpoint.PLAY_BY_PLAY_V3),
        run_id="run-1",
        attempt_number=1,
        known_sha256=[],
    )

    pending = pending_completed_game_ids(
        state_dir=tmp_path,
        lake_uri="s3://nba-gestalt-data/lake",
        season="2026-27",
        coverage_path=tmp_path / "coverage.json",
        now=datetime(2026, 10, 21, 5, 0, tzinfo=UTC),
    )

    assert pending == ["0022600001"]


def test_failure_is_manifested_and_fails_the_batch(tmp_path: Path) -> None:
    _write_final_slate(tmp_path)
    store = MemoryObjectStore()
    archive = StatsV3LakeArchive(store=store)

    def fail(_endpoint: NbaStatsEndpoint, _game_id: str) -> StatsCachedResponse:
        raise ValueError("bad response")

    with pytest.raises(StatsV3BatchError) as raised:
        ingest_completed_stats_v3(
            state_dir=tmp_path,
            coverage_path=tmp_path / "coverage.json",
            archive=archive,
            season="2026-27",
            ignore_grace=True,
            fetch_response=fail,
        )

    assert raised.value.summary["failed_count"] == 2
    failure_records = [uri for uri in store.objects if "/failure-" in uri]
    assert len(failure_records) == 2
