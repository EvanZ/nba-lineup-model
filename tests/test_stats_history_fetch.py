from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nba_lineup_model.ingest.nba_stats import (
    NbaStatsClient,
    NbaStatsEndpoint,
    NbaStatsError,
    NbaStatsRawCache,
    NbaStatsRequestGate,
)
from nba_lineup_model.season.schema import CatalogGame
from nba_lineup_model.season.stats import (
    append_stats_fetch_records,
    fetch_stats_endpoint_raw,
    is_transient_stats_fetch_error,
    read_stats_fetch_manifest,
    stats_play_by_play_final_score,
)

GAME_ID = "0021900194"
PLAY_BY_PLAY_BODY = (
    b'{"meta":{"request":"http://nba.cloud/games/0021900194/playbyplay?Format=json"},'
    b'"game":{"gameId":"0021900194","actions":[{"actionNumber":2,"actionType":"period",'
    b'"scoreHome":"100","scoreAway":"97"}]}}'
)
BOXSCORE_BODY = (
    b'{"meta":{"request":"http://nba.cloud/games/0021900194/boxscoretraditional?'
    b'Format=json"},"boxScoreTraditional":{"gameId":"0021900194",'
    b'"homeTeam":{"teamId":1610612742},"awayTeam":{"teamId":1610612759}}}'
)
GAME_ROTATION_BODY = (
    b'{"resource":"gamerotation","parameters":{"GameID":"0021900194","LeagueID":"00"},'
    b'"resultSets":[{"name":"AwayTeam","headers":["GAME_ID"],'
    b'"rowSet":[["0021900194"]]},{"name":"HomeTeam","headers":["GAME_ID"],'
    b'"rowSet":[["0021900194"]]}]}'
)


def catalog_game() -> CatalogGame:
    return CatalogGame(
        game_id=GAME_ID,
        season="2019-20",
        season_type="regular",
        game_date=date(2019, 11, 18),
        game_status="final",
        home_team_id=1610612742,
        home_team_tricode="DAL",
        away_team_id=1610612759,
        away_team_tricode="SAS",
        period_count=4,
        is_overtime=False,
        source_status_code=3,
        source_status_text="Final",
        source_url="https://stats.nba.com/stats/scheduleleaguev2",
        source_fetched_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("endpoint", "body", "expected_parameters"),
    [
        (
            NbaStatsEndpoint.PLAY_BY_PLAY_V3,
            PLAY_BY_PLAY_BODY,
            {"GameID": GAME_ID, "StartPeriod": "0", "EndPeriod": "14"},
        ),
        (
            NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
            BOXSCORE_BODY,
            {
                "GameID": GAME_ID,
                "StartPeriod": "0",
                "EndPeriod": "0",
                "StartRange": "0",
                "EndRange": "0",
                "RangeType": "0",
            },
        ),
        (
            NbaStatsEndpoint.GAME_ROTATION,
            GAME_ROTATION_BODY,
            {"GameID": GAME_ID, "LeagueID": "00"},
        ),
    ],
)
def test_stats_client_preserves_endpoint_bytes_and_parameters(
    tmp_path: Path,
    endpoint: NbaStatsEndpoint,
    body: bytes,
    expected_parameters: dict[str, str],
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=body,
            headers={"Content-Type": "application/json", "x-datasource": "S3"},
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    cache = NbaStatsRawCache(tmp_path)
    client = NbaStatsClient(
        cache=cache,
        http_client=http_client,
        request_gate=NbaStatsRequestGate(),
    )
    try:
        downloaded = client.fetch(endpoint, GAME_ID, use_cache=False)
        cached = client.fetch(endpoint, GAME_ID)
    finally:
        http_client.close()

    assert len(requests) == 1
    assert dict(requests[0].url.params) == expected_parameters
    assert requests[0].headers["referer"] == "https://www.nba.com/"
    assert "Chrome/145.0.0.0" in requests[0].headers["user-agent"]
    assert downloaded.raw_body == body
    assert cached.raw_body == body
    assert cached.response_headers["x-datasource"] == "S3"
    assert cache.path_for(endpoint, GAME_ID).read_bytes() == body
    assert cache.metadata_path_for(endpoint, GAME_ID).exists()


def test_stats_client_rejects_mismatched_game_before_cache_write(tmp_path: Path):
    mismatched = PLAY_BY_PLAY_BODY.replace(b"0021900194", b"0021900195")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=mismatched, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    cache = NbaStatsRawCache(tmp_path)
    client = NbaStatsClient(
        cache=cache,
        http_client=http_client,
        request_gate=NbaStatsRequestGate(),
    )
    try:
        with pytest.raises(NbaStatsError, match="gameId does not match"):
            client.fetch(NbaStatsEndpoint.PLAY_BY_PLAY_V3, GAME_ID, use_cache=False)
    finally:
        http_client.close()

    assert not cache.path_for(NbaStatsEndpoint.PLAY_BY_PLAY_V3, GAME_ID).exists()


def test_stats_fetch_downloads_then_skips_and_round_trips_manifest(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=PLAY_BY_PLAY_BODY,
            headers={"x-datasource": "S3"},
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    raw_dir = tmp_path / "raw" / "stats"
    client = NbaStatsClient(
        cache=NbaStatsRawCache(raw_dir),
        http_client=http_client,
        request_gate=NbaStatsRequestGate(),
    )
    try:
        downloaded = fetch_stats_endpoint_raw(
            catalog_game(),
            NbaStatsEndpoint.PLAY_BY_PLAY_V3,
            run_id="download",
            raw_dir=raw_dir,
            client=client,
        )
    finally:
        http_client.close()
    skipped = fetch_stats_endpoint_raw(
        catalog_game(),
        NbaStatsEndpoint.PLAY_BY_PLAY_V3,
        run_id="skip",
        raw_dir=raw_dir,
    )
    manifest_path = tmp_path / "manifests" / "stats_fetches.parquet"

    manifest = append_stats_fetch_records([downloaded, skipped], manifest_path)

    assert downloaded.status == "succeeded"
    assert downloaded.sha256 == hashlib.sha256(PLAY_BY_PLAY_BODY).hexdigest()
    assert downloaded.byte_count == len(PLAY_BY_PLAY_BODY)
    assert skipped.status == "skipped"
    assert skipped.cache_hit is True
    assert read_stats_fetch_manifest(manifest_path) == manifest
    schema = pq.read_schema(manifest_path)
    assert pa.types.is_string(schema.field("game_id").type) or pa.types.is_large_string(
        schema.field("game_id").type
    )
    assert schema.field("byte_count").type == pa.int64()
    assert schema.field("started_at").type.tz == "UTC"


def test_stats_play_by_play_final_score_uses_last_valid_score_pair():
    assert stats_play_by_play_final_score(
        {
            "game": {
                "actions": [
                    {"scoreHome": "100", "scoreAway": "97"},
                    {"scoreHome": "", "scoreAway": ""},
                ]
            }
        }
    ) == (100, 97)


def test_stats_server_error_is_transient_and_not_cached(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            content=b'{"message":"upstream failed"}',
            headers={"Content-Type": "application/json"},
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    cache = NbaStatsRawCache(tmp_path)
    client = NbaStatsClient(
        cache=cache,
        http_client=http_client,
        request_gate=NbaStatsRequestGate(),
    )
    try:
        with pytest.raises(NbaStatsError) as error:
            client.fetch(NbaStatsEndpoint.GAME_ROTATION, GAME_ID, use_cache=False)
    finally:
        http_client.close()

    assert error.value.status_code == 500
    assert error.value.transient is True
    assert is_transient_stats_fetch_error(error.value)
    assert not cache.path_for(NbaStatsEndpoint.GAME_ROTATION, GAME_ID).exists()
