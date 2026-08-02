from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nba_lineup_model.players.collect import collect_player_bios
from nba_lineup_model.players.normalize import (
    player_catalog_from_response,
    player_season_bios_from_response,
)
from nba_lineup_model.players.source import (
    PlayerStatsCache,
    PlayerStatsClient,
    PlayerStatsEndpoint,
    PlayerStatsResponse,
)
from nba_lineup_model.players.storage import (
    merge_player_catalogs,
    player_catalog_frame,
    player_season_bio_frame,
    read_player_catalog,
    read_player_season_bios,
    validate_player_season_partition,
    write_player_catalog,
    write_player_season_bios,
)

INDEX_HEADERS = [
    "PERSON_ID",
    "PLAYER_LAST_NAME",
    "PLAYER_FIRST_NAME",
    "PLAYER_SLUG",
    "TEAM_ID",
    "TEAM_SLUG",
    "IS_DEFUNCT",
    "TEAM_ABBREVIATION",
    "JERSEY_NUMBER",
    "POSITION",
    "HEIGHT",
    "WEIGHT",
    "COLLEGE",
    "COUNTRY",
    "DRAFT_YEAR",
    "DRAFT_ROUND",
    "DRAFT_NUMBER",
    "ROSTER_STATUS",
    "FROM_YEAR",
    "TO_YEAR",
    "SUPPLEMENTAL_STATUS",
]
BIO_HEADERS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "AGE",
    "PLAYER_HEIGHT",
    "PLAYER_HEIGHT_INCHES",
    "PLAYER_WEIGHT",
    "COLLEGE",
    "COUNTRY",
    "DRAFT_YEAR",
    "DRAFT_ROUND",
    "DRAFT_NUMBER",
    "GP",
    "PTS",
    "USG_PCT",
]


def player_index_payload() -> dict:
    return {
        "resultSets": [
            {
                "name": "PlayerIndex",
                "headers": INDEX_HEADERS,
                "rowSet": [
                    [
                        1630639,
                        "Lawson",
                        "A.J.",
                        "aj-lawson",
                        1610612761,
                        "raptors",
                        0,
                        "TOR",
                        "0",
                        "G",
                        "6-6",
                        "179",
                        "South Carolina",
                        "Canada",
                        "Undrafted",
                        "Undrafted",
                        "Undrafted",
                        1,
                        "2021",
                        "2025",
                        None,
                    ],
                    [
                        204001,
                        "Porziņģis",
                        "Kristaps",
                        "kristaps-porzingis",
                        1610612744,
                        "warriors",
                        0,
                        "GSW",
                        "7",
                        "F-C",
                        "7-2",
                        "240",
                        None,
                        "Latvia",
                        "2015",
                        "1",
                        "4",
                        1,
                        "2015",
                        "2025",
                        None,
                    ],
                ],
            }
        ]
    }


def player_bio_payload(*, height_inches: int = 78) -> dict:
    return {
        "resultSets": [
            {
                "name": "LeagueDashPlayerBioStats",
                "headers": BIO_HEADERS,
                "rowSet": [
                    [
                        1630639,
                        "A.J. Lawson",
                        1610612761,
                        "TOR",
                        25.0,
                        "6-6",
                        height_inches,
                        "179",
                        "South Carolina",
                        "Canada",
                        "Undrafted",
                        "Undrafted",
                        "Undrafted",
                        7,
                        1.7,
                        0.157,
                    ],
                    [
                        204001,
                        "Kristaps Porziņģis",
                        1610612744,
                        "GSW",
                        30.0,
                        "7-2",
                        86,
                        "240",
                        None,
                        "Latvia",
                        "2015",
                        "1",
                        "4",
                        17,
                        16.7,
                        0.24,
                    ],
                ],
            }
        ]
    }


def response(
    endpoint: PlayerStatsEndpoint,
    payload: dict,
    *,
    season_type: str | None,
) -> PlayerStatsResponse:
    raw_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return PlayerStatsResponse(
        endpoint=endpoint,
        season="2025-26",
        season_type=season_type,
        params={"Season": "2025-26"},
        url=f"https://stats.nba.com/stats/{endpoint.value}",
        payload=payload,
        raw_body=raw_body,
    )


def test_player_stats_client_preserves_raw_bytes_and_reuses_cache(tmp_path: Path):
    index_body = json.dumps(player_index_payload(), separators=(",", ":")).encode()
    bio_body = json.dumps(player_bio_payload(), separators=(",", ":")).encode()
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        body = index_body if request.url.path.endswith("playerindex") else bio_body
        return httpx.Response(200, content=body, request=request)

    cache = PlayerStatsCache(tmp_path / "raw")
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = PlayerStatsClient(cache=cache, http_client=http_client)

    index = client.fetch_player_index("2025-26")
    bios = client.fetch_player_season_bios("2025-26")
    cached_index = client.fetch_player_index("2025-26")
    cached_bios = client.fetch_player_season_bios("2025-26")

    assert requests == [
        "/stats/playerindex",
        "/stats/leaguedashplayerbiostats",
    ]
    assert index.raw_body == cached_index.raw_body == index_body
    assert bios.raw_body == cached_bios.raw_body == bio_body
    assert cache.path_for(PlayerStatsEndpoint.PLAYER_INDEX, "2025-26").read_bytes() == index_body
    assert (
        cache.path_for(
            PlayerStatsEndpoint.PLAYER_BIO_STATS,
            "2025-26",
            "regular",
        ).read_bytes()
        == bio_body
    )


def test_normalizes_identity_and_leakage_safe_player_seasons():
    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            player_index_payload(),
            season_type=None,
        )
    )
    bios = player_season_bios_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_BIO_STATS,
            player_bio_payload(),
            season_type="regular",
        ),
        catalog,
    )

    porzingis = next(player for player in catalog.players if player.player_id == 204001)
    lawson = next(player for player in bios.players if player.player_id == 1630639)
    assert porzingis.display_name == "Kristaps Porziņģis"
    assert porzingis.height_inches == 86
    assert porzingis.draft_number == 4
    assert lawson.is_undrafted is True
    assert lawson.draft_year is None
    assert lawson.listed_position == "G"
    assert lawson.height_inches == 78
    assert set(player_season_bio_frame(bios).columns).isdisjoint(
        {"games_played", "points", "usage_percentage", "GP", "PTS", "USG_PCT"}
    )


def test_player_index_collapses_duplicate_team_listings():
    payload = player_index_payload()
    duplicate = list(payload["resultSets"][0]["rowSet"][1])
    duplicate[INDEX_HEADERS.index("TEAM_ID")] = 1610612738
    duplicate[INDEX_HEADERS.index("TEAM_ABBREVIATION")] = "BOS"
    payload["resultSets"][0]["rowSet"].append(duplicate)

    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            payload,
            season_type=None,
        )
    )

    assert [player.player_id for player in catalog.players] == [204001, 1630639]
    porzingis = next(player for player in catalog.players if player.player_id == 204001)
    assert porzingis.latest_team_abbreviation == "GSW"


def test_player_index_rejects_conflicting_duplicate_identities():
    payload = player_index_payload()
    duplicate = list(payload["resultSets"][0]["rowSet"][1])
    duplicate[INDEX_HEADERS.index("PLAYER_LAST_NAME")] = "Other"
    payload["resultSets"][0]["rowSet"].append(duplicate)

    with pytest.raises(ValueError, match="conflicting identity data"):
        player_catalog_from_response(
            response(
                PlayerStatsEndpoint.PLAYER_INDEX,
                payload,
                season_type=None,
            )
        )


def test_rejects_disagreeing_player_bio_height():
    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            player_index_payload(),
            season_type=None,
        )
    )

    with pytest.raises(ValueError, match="height fields disagree"):
        player_season_bios_from_response(
            response(
                PlayerStatsEndpoint.PLAYER_BIO_STATS,
                player_bio_payload(height_inches=79),
                season_type="regular",
            ),
            catalog,
        )


def test_player_index_treats_zero_pick_values_as_undrafted():
    payload = player_index_payload()
    row = payload["resultSets"][0]["rowSet"][1]
    row[INDEX_HEADERS.index("DRAFT_ROUND")] = 0
    row[INDEX_HEADERS.index("DRAFT_NUMBER")] = 0

    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            payload,
            season_type=None,
        )
    )
    player = next(player for player in catalog.players if player.player_id == 204001)

    assert player.is_undrafted is True
    assert player.draft_year == 2015
    assert player.draft_round is None
    assert player.draft_number is None


def test_player_bios_preserve_missing_weight():
    payload = player_bio_payload()
    payload["resultSets"][0]["rowSet"][0][BIO_HEADERS.index("PLAYER_WEIGHT")] = None
    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            player_index_payload(),
            season_type=None,
        )
    )

    bios = player_season_bios_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_BIO_STATS,
            payload,
            season_type="regular",
        ),
        catalog,
    )

    assert bios.players[1].weight_pounds is None


def test_player_bios_preserve_missing_historical_height():
    payload = player_bio_payload()
    row = payload["resultSets"][0]["rowSet"][0]
    row[BIO_HEADERS.index("PLAYER_HEIGHT")] = None
    row[BIO_HEADERS.index("PLAYER_HEIGHT_INCHES")] = None
    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            player_index_payload(),
            season_type=None,
        )
    )

    bios = player_season_bios_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_BIO_STATS,
            payload,
            season_type="regular",
        ),
        catalog,
    )

    player = next(player for player in bios.players if player.player_id == 1630639)
    assert player.height_raw is None
    assert player.height_inches is None


def test_player_catalog_and_season_bios_round_trip_with_stable_types(
    tmp_path: Path,
):
    catalog = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            player_index_payload(),
            season_type=None,
        )
    )
    bios = player_season_bios_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_BIO_STATS,
            player_bio_payload(),
            season_type="regular",
        ),
        catalog,
    )
    catalog_path = write_player_catalog(
        catalog,
        tmp_path / "catalog" / "players.parquet",
    )
    manifest = write_player_season_bios(bios, tmp_path / "curated")

    assert read_player_catalog(catalog_path) == catalog
    assert (
        read_player_season_bios(
            "2025-26",
            "regular",
            tmp_path / "curated",
        )
        == bios
    )
    assert (
        validate_player_season_partition(
            "2025-26",
            "regular",
            tmp_path / "curated",
        )
        == manifest
    )
    catalog_schema = pq.read_schema(catalog_path)
    bio_path = (
        tmp_path / "curated" / "player_seasons" / "2025-26" / "regular" / "part-00000.parquet"
    )
    bio_schema = pq.read_schema(bio_path)
    assert catalog_schema.field("player_id").type == pa.int64()
    assert catalog_schema.field("height_inches").type == pa.int64()
    assert bio_schema.field("team_id").type == pa.int64()
    assert bio_schema.field("bio_source_fetched_at").type.tz == "UTC"
    root_frame = pd.read_parquet(tmp_path / "curated" / "player_seasons")
    assert set(root_frame["season"].astype(str)) == {"2025-26"}
    assert set(root_frame["season_type"].astype(str)) == {"regular"}


def test_player_catalog_merge_preserves_newer_player_universe():
    latest = player_catalog_from_response(
        response(
            PlayerStatsEndpoint.PLAYER_INDEX,
            player_index_payload(),
            season_type=None,
        )
    )
    older_players = [
        player.model_copy(
            update={
                "source_season": "2019-20",
                "to_year": min(player.to_year or 2019, 2019),
            }
        )
        for player in latest.players[:1]
    ]
    older = latest.model_copy(update={"players": older_players})

    merged = merge_player_catalogs(latest, older)

    assert [player.player_id for player in merged.players] == [204001, 1630639]
    assert all(player.source_season == "2025-26" for player in merged.players)


def test_collect_player_bios_writes_both_normalized_datasets(tmp_path: Path):
    bodies = {
        "/stats/playerindex": json.dumps(player_index_payload()).encode(),
        "/stats/leaguedashplayerbiostats": json.dumps(player_bio_payload()).encode(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=bodies[request.url.path],
            request=request,
        )

    cache = PlayerStatsCache(tmp_path / "raw")
    client = PlayerStatsClient(
        cache=cache,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    summary = collect_player_bios(
        "2025-26",
        raw_dir=tmp_path / "raw",
        player_catalog_path=tmp_path / "catalog" / "players.parquet",
        curated_dir=tmp_path / "curated",
        client=client,
    )

    assert summary.historical_player_count == 2
    assert summary.player_season_count == 2
    assert summary.player_index_cache_hit is False
    assert summary.player_bios_cache_hit is False
    assert Path(summary.player_catalog_path).exists()
    assert Path(summary.player_season_partition_path).exists()
    assert player_catalog_frame(read_player_catalog(summary.player_catalog_path))[
        "player_id"
    ].tolist() == [204001, 1630639]
