from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nba_lineup_model.ingest.nba_cdn import (
    NbaCdnClient,
    NbaCdnEndpoint,
    NbaCdnError,
    RawJsonCache,
)
from nba_lineup_model.season.fetch import (
    failed_fetch_record,
    fetch_game_raw,
    is_transient_fetch_error,
    select_catalog_games,
)
from nba_lineup_model.season.schema import CatalogGame, GameCatalog
from nba_lineup_model.season.storage import (
    append_fetch_records,
    read_fetch_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures"
PLAY_BY_PLAY_BODY = (FIXTURES / "playbyplay_minimal.json").read_bytes()
BOXSCORE_BODY = (FIXTURES / "boxscore_minimal.json").read_bytes()


def catalog_game(
    *,
    game_id: str = "0022000180",
    season: str = "2020-21",
    season_type: str = "regular",
    game_date: date = date(2020, 12, 27),
    game_status: str = "final",
) -> CatalogGame:
    return CatalogGame(
        game_id=game_id,
        season=season,
        season_type=season_type,
        game_date=game_date,
        game_status=game_status,
        home_team_id=1610612753,
        home_team_tricode="ORL",
        away_team_id=1610612764,
        away_team_tricode="WAS",
        period_count=4 if game_status == "final" else None,
        is_overtime=False if game_status == "final" else None,
        source_status_code=3 if game_status == "final" else 1,
        source_status_text="Final" if game_status == "final" else "Scheduled",
        source_url="https://stats.nba.com/stats/scheduleleaguev2",
        source_fetched_at=datetime(2026, 7, 27, tzinfo=UTC),
    )


def mock_nba_client(
    raw_dir: Path,
    *,
    boxscore_status: int = 200,
    play_by_play_body: bytes = PLAY_BY_PLAY_BODY,
) -> tuple[NbaCdnClient, httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/playbyplay/" in request.url.path:
            return httpx.Response(200, content=play_by_play_body, request=request)
        if "/boxscore/" in request.url.path:
            return httpx.Response(
                boxscore_status,
                content=BOXSCORE_BODY,
                request=request,
            )
        raise AssertionError(f"Unexpected NBA test URL: {request.url}")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return (
        NbaCdnClient(cache=RawJsonCache(raw_dir), http_client=http_client),
        http_client,
    )


def test_fetch_game_raw_downloads_then_skips_valid_cache(tmp_path: Path):
    game = catalog_game()
    client, http_client = mock_nba_client(tmp_path)
    try:
        downloaded = fetch_game_raw(
            game,
            run_id="download",
            raw_dir=tmp_path,
            client=client,
        )
    finally:
        http_client.close()

    cached = fetch_game_raw(game, run_id="cached", raw_dir=tmp_path)

    assert downloaded.status == "succeeded"
    assert downloaded.play_by_play_cache_hit is False
    assert downloaded.boxscore_cache_hit is False
    assert downloaded.play_by_play_sha256 == hashlib.sha256(PLAY_BY_PLAY_BODY).hexdigest()
    assert downloaded.boxscore_sha256 == hashlib.sha256(BOXSCORE_BODY).hexdigest()
    assert downloaded.play_by_play_bytes == len(PLAY_BY_PLAY_BODY)
    assert downloaded.boxscore_bytes == len(BOXSCORE_BODY)
    assert cached.status == "skipped"
    assert cached.skip_reason == "already_cached"
    assert cached.play_by_play_cache_hit is True
    assert cached.boxscore_cache_hit is True


def test_fetch_game_raw_replaces_invalid_cache(tmp_path: Path):
    game = catalog_game()
    cache = RawJsonCache(tmp_path)
    invalid_path = cache.path_for(NbaCdnEndpoint.PLAY_BY_PLAY, game.game_id)
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text("{invalid")
    client, http_client = mock_nba_client(tmp_path)
    try:
        record = fetch_game_raw(
            game,
            run_id="replace-invalid",
            raw_dir=tmp_path,
            client=client,
        )
    finally:
        http_client.close()

    assert record.status == "succeeded"
    assert record.play_by_play_cache_hit is False
    assert invalid_path.read_bytes() == PLAY_BY_PLAY_BODY


def test_fetch_game_raw_rejects_mismatched_source_game(tmp_path: Path):
    game = catalog_game()
    mismatched_body = PLAY_BY_PLAY_BODY.replace(
        b'"gameId": "0022000180"',
        b'"gameId": "0022000181"',
    )
    client, http_client = mock_nba_client(
        tmp_path,
        play_by_play_body=mismatched_body,
    )
    try:
        with pytest.raises(NbaCdnError, match="gameId does not match"):
            fetch_game_raw(
                game,
                run_id="mismatched-source",
                raw_dir=tmp_path,
                client=client,
            )
    finally:
        http_client.close()


def test_failed_fetch_record_retains_partial_artifact(tmp_path: Path):
    game = catalog_game()
    client, http_client = mock_nba_client(tmp_path, boxscore_status=503)
    started_at = datetime.now(UTC)
    try:
        with pytest.raises(NbaCdnError) as error:
            fetch_game_raw(
                game,
                run_id="partial",
                raw_dir=tmp_path,
                client=client,
                started_at=started_at,
            )
    finally:
        http_client.close()

    record = failed_fetch_record(
        game,
        run_id="partial",
        started_at=started_at,
        error=error.value,
        raw_dir=tmp_path,
    )

    assert record.status == "failed"
    assert record.play_by_play_sha256 == hashlib.sha256(PLAY_BY_PLAY_BODY).hexdigest()
    assert record.boxscore_sha256 is None
    assert record.error_type == "NbaCdnError"
    assert is_transient_fetch_error(error.value) is True
    assert is_transient_fetch_error(NbaCdnError("timeout", status_code=408, transient=True))
    assert is_transient_fetch_error(NbaCdnError("not found", status_code=404)) is False


def test_fetch_manifest_round_trips_with_identifier_types(tmp_path: Path):
    game = catalog_game()
    client, http_client = mock_nba_client(tmp_path / "raw")
    try:
        record = fetch_game_raw(
            game,
            run_id="manifest",
            raw_dir=tmp_path / "raw",
            client=client,
        )
    finally:
        http_client.close()
    path = tmp_path / "manifests" / "fetches.parquet"

    manifest = append_fetch_records([record], path)

    assert read_fetch_manifest(path) == manifest
    schema = pq.read_schema(path)
    assert pa.types.is_string(schema.field("game_id").type) or pa.types.is_large_string(
        schema.field("game_id").type
    )
    assert schema.field("attempt_number").type == pa.int64()
    assert schema.field("play_by_play_bytes").type == pa.int64()
    assert schema.field("started_at").type.tz == "UTC"


def test_catalog_selection_is_deterministic_and_final_only():
    games = [
        catalog_game(game_id="0022000182", game_date=date(2020, 12, 29)),
        catalog_game(game_id="0022000181", game_date=date(2020, 12, 28)),
        catalog_game(
            game_id="0042000401",
            season_type="playoffs",
            game_date=date(2021, 7, 6),
        ),
        catalog_game(
            game_id="0022000183",
            game_date=date(2020, 12, 30),
            game_status="scheduled",
        ),
    ]
    catalog = GameCatalog(games=games)

    selected = select_catalog_games(
        catalog,
        season="2020-21",
        season_types=["regular"],
        limit=1,
    )

    assert [game.game_id for game in selected] == ["0022000181"]
