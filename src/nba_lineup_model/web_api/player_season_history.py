"""Query saved career ratings directly on S3; cache only per-player SQL results.

Uses DuckDB from the web dependency group, at the version in uv.lock. The dev
API environment needs that package and its httpfs extension installed. No pandas,
artifact downloads, persistent database, or DuckDB file/block cache is used by this reader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from functools import lru_cache
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Path, Response

router = APIRouter()
_lock = threading.Lock()

# Existing, checksum-verified historical objects; no deployment release dependency.
SOURCES = {
    "context": {
        "etag": '"62bba2ffc8a668022a8be143e5fe44aa"',
        "sha256": "a33ae47fd3e98e674945216136acb725c04ee58cc92d292392e39e321a3e4656",
        "size_bytes": 259535,
        "uri": "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/response_curve_cache/forward_nail_rapm_v1212_residualized_lambda/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25-player-context.parquet",
        "version_id": "0BJHup5Kh4OpplXOCcsZxyM6CGoNeir2",
    },
    "panel": {
        "etag": '"2b66b860a9a0eeab2f900b9e3c83e68e"',
        "sha256": "b18ed0d44a607be74e077fa7115779d5e10d4e86e0501fb3580210d5b75c432c",
        "size_bytes": 3505317,
        "uri": "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/data/analytical/player_season_panel/player_seasons.parquet",
        "version_id": "RocI0ZhNpjAr2W5fWykk5aWi1P2QnX2Q",
    },
    "ratings": {
        "etag": '"a4e4bbce38d95a50b3a5d0548e27f894"',
        "sha256": "823b3fc7a5cee5355fc4886fcf4d9b81d958b8adde179a18a8992bf98256c3c7",
        "size_bytes": 1141736,
        "uri": "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/lineup_rankings/forward_nail_rapm_v1212_residualized_lambda/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/player_ratings.parquet",
        "version_id": "v_YFLB3htZ9qB7PWBVK.255if2yXrTOZ",
    },
    "splits": {
        "etag": '"13032b86f7edcbbb37d353fc2463bf1c"',
        "sha256": "e559cad5fac1bd8e4a96791e98cc2b03c83a90fe90b3a6bd438832ec7553c08b",
        "size_bytes": 470274,
        "uri": "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/constrained_split_ratings/forward_nail_rapm_v1212_residualized_lambda/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/constrained-split-nail-2025-26-20260829T130012Z-81a5fad3/player_ratings.parquet",
        "version_id": "g02_Yj4s.7fXBmCxXsNbrwR8witPyNfy",
    },
    "teams": {
        "etag": '"0aa697f0b0eac321803670dcf7deb89b"',
        "sha256": "aefafa589675add7d1cdf486c8eba3f8a7898827ed5b992406a94fe982752f70",
        "size_bytes": 243941,
        "uri": "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/player_team_splits/forward_nail_rapm_v1212_residualized_lambda/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25.parquet",
        "version_id": "41g6bv0dLoLmxwqbkxSSbMjlAur.nwy.",
    },
}
SOURCE_ID = hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest()

QUERY = """
WITH catalog AS (
    SELECT *, min(rapm) OVER (PARTITION BY season) AS season_min_rating,
           max(rapm) OVER (PARTITION BY season) AS season_max_rating,
           first_value(player_id) OVER leader AS season_max_player_id,
           first_value(player_name) OVER leader AS season_max_player_name
    FROM read_parquet(?)
    WINDOW leader AS (PARTITION BY season ORDER BY rapm DESC, player_name, player_id)
), ratings AS (
    SELECT season, player_id, player_name, age, rapm AS rating,
           prior_rapm AS prior_rating, rapm_adjustment_from_prior AS season_update,
           additive_profile_adjustment, rank_all_players AS nail_rank,
           season_min_rating, season_max_rating, season_max_player_id, season_max_player_name
    FROM catalog WHERE player_id = ?
), panel AS (
    SELECT season, player_id, games, games_started, rapm_possessions AS possessions
    FROM read_parquet(?) WHERE player_id = ?
), splits AS (
    SELECT season, player_id, total_nail, offense_rating, defense_rating
    FROM read_parquet(?) WHERE player_id = ?
), context AS (
    SELECT season, player_id, observed_context_exposure
    FROM read_parquet(?) WHERE player_id = ?
)
SELECT r.*, p.games, p.games_started, p.possessions,
       s.total_nail, s.offense_rating, s.defense_rating, c.observed_context_exposure
FROM ratings r
LEFT JOIN panel p USING (season, player_id)
LEFT JOIN splits s USING (season, player_id)
LEFT JOIN context c USING (season, player_id)
ORDER BY r.season
"""
TEAM_QUERY = """
SELECT season, team_id, team, possessions, games, is_primary_team, is_latest_team
FROM read_parquet(?) WHERE player_id = ?
ORDER BY season, last_game_time_utc, team_id
"""


def records(result):
    columns = [c[0] for c in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def query_player(connection, uris, player_id):
    params = [
        value
        for name in ("ratings", "panel", "splits", "context")
        for value in (uris[name], player_id)
    ]
    seasons = records(connection.execute(QUERY, params))
    teams = records(connection.execute(TEAM_QUERY, [uris["teams"], player_id]))
    if len({s["season"] for s in seasons}) != len(seasons):
        raise ValueError("Duplicate saved player-season rows")
    for row in seasons:
        if row["player_id"] != player_id or row["season"] > "2025-26":
            raise ValueError("Historical player/season identity mismatch")
        if not math.isfinite(row["rating"]):
            raise ValueError("Invalid historical NAIL")
        if row["total_nail"] is not None and (
            not math.isclose(row["total_nail"], row["rating"], abs_tol=1e-8)
            or not math.isclose(
                row["offense_rating"] + row["defense_rating"], row["rating"], abs_tol=1e-8
            )
        ):
            raise ValueError("Saved offensive/defensive allocation differs from NAIL")
        row.pop("total_nail")
        row["team_splits"] = [
            {k: v for k, v in t.items() if k != "season"}
            for t in teams
            if t["season"] == row["season"]
        ]
        latest = next((t for t in row["team_splits"] if t["is_latest_team"]), None)
        row["team"] = latest["team"] if latest else "-"
        row["team_id"] = latest["team_id"] if latest else None
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite saved season value: " + key)
    return seasons


class SeasonReader:
    def __init__(self):
        import boto3
        import duckdb

        self.session = boto3.Session(profile_name="gestalt-dev-web-readonly")
        self.client = self.session.client("s3")
        self.connection = duckdb.connect(
            config={
                "memory_limit": "128MB",
                "threads": 1,
                "enable_external_file_cache": "false",
                "enable_http_metadata_cache": "false",
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
                "temp_directory": "",
            }
        )
        self.connection.execute("LOAD httpfs")

    def check_sources(self, sources=None):
        # These exact object versions were checksum-verified when registered.
        # Detect replacement before/after querying; never silently select newer data.
        for source in (SOURCES if sources is None else sources).values():
            uri = urlparse(source["uri"])
            head = self.client.head_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))
            if (
                head.get("VersionId") != source["version_id"]
                or head["ETag"] != source["etag"]
                or head["ContentLength"] != source["size_bytes"]
            ):
                raise ValueError("Registered historical source object changed")

    def authenticate(self):
        credentials = self.session.get_credentials().get_frozen_credentials()
        # Bind the existing read-only role credentials; never use the instance role
        # or log credentials/query arguments. Boto3 refreshes expired sessions.
        self.connection.execute(
            "CREATE OR REPLACE SECRET player_career (TYPE s3, KEY_ID $1, SECRET $2, "
            "SESSION_TOKEN $3, REGION 'us-west-1', SCOPE 's3://nba-gestalt-data/')",
            [credentials.access_key, credentials.secret_key, credentials.token],
        )

    def history(self, player_id):
        self.check_sources()
        self.authenticate()
        seasons = query_player(
            self.connection, {k: v["uri"] for k, v in SOURCES.items()}, player_id
        )
        self.check_sources()
        return {
            "kind": "saved_player_season_history",
            "schema_version": 1,
            "player_id": player_id,
            "player_name": seasons[0]["player_name"] if seasons else None,
            "through_season": "2025-26",
            "seasons": seasons,
            "sources": SOURCES,
            "source_id": SOURCE_ID,
        }


@lru_cache(maxsize=1)
def _reader():
    return SeasonReader()


@lru_cache(maxsize=256)
def _history(player_id, source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Historical source selection differs from this code")
    return _reader().history(player_id)


@router.get("/api/players/{player_id}/season-history")
def season_history(response: Response, player_id: int = Path(gt=0)):
    try:
        with _lock:
            result = _history(player_id, SOURCE_ID)
        response.headers["Cache-Control"] = "private, max-age=3600"
        return result
    except Exception as error:
        # DuckDB errors can include query arguments: keep credential-bearing SQL
        # out of logs and responses, and never cache errors or partial histories.
        logging.getLogger(__name__).error(
            "Saved season history unavailable: %s", type(error).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="Saved season history is unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
