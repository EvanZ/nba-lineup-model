"""Serve frozen historical lineup tables with direct, read-only S3 queries."""

import json
import logging
from functools import lru_cache
from time import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response

from .historical_lineup_sources import OBJECTS
from .player_season_history import SOURCES, _lock, _reader
from .saved_current_lineups import SEASON as CURRENT_SEASON
from .saved_current_lineups import current_lineups

router = APIRouter()
SEASONS = sorted(OBJECTS, reverse=True)
AVAILABLE_SEASONS = [CURRENT_SEASON, *SEASONS]
ARCHIVE = SOURCES["ratings"]["uri"].rsplit("/", 1)[0]
CACHE_SECONDS = 3600

QUERY = """
SELECT
    row_number() OVER (
        ORDER BY context_edge DESC, possessions DESC, team ASC, lineup_label ASC
    ) AS rank,
    team_id, team, player_ids, player_names, lineup_label, possessions, games,
    player_rating, player_edge, offensive_edge, defensive_edge, composition_rating,
    composition_edge, matchup_bonus, context_edge, gestalt_score, actual_net_rating,
    actual_offensive_rating, actual_defensive_rating
FROM read_parquet(?)
WHERE possessions >= ? AND list_has_all(player_ids, CAST(? AS BIGINT[]))
ORDER BY rank
"""


def source_for(season):
    version, etag, size = OBJECTS[season]
    return {
        "uri": f"{ARCHIVE}/{season}.parquet",
        "version_id": version,
        "etag": etag,
        "size_bytes": size,
    }


@lru_cache(maxsize=8)
def _query(season, minimum_possessions, player_ids, period):
    source = source_for(season)
    reader = _reader()
    reader.check_sources({"lineups": source})
    reader.authenticate()
    cursor = reader.connection.execute(
        QUERY, [source["uri"], minimum_possessions, list(player_ids)]
    )
    fields = [column[0] for column in cursor.description]
    rows = [dict(zip(fields, values, strict=True)) for values in cursor.fetchall()]
    reader.check_sources({"lineups": source})
    return json.dumps(
        {
            "kind": "saved_historical_lineups",
            "season": season,
            "run_id": ARCHIVE.rsplit("/", 1)[-1],
            "available_seasons": AVAILABLE_SEASONS,
            "lineup_state": {"kind": "completed_fit", "season": season},
            "rating_state": {"kind": "completed_fit", "season": season},
            "minimum_possessions": minimum_possessions,
            "lineups": rows,
            "source": source,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


@router.get("/api/lineups")
def historical_lineups(
    season: str = Query(default=SEASONS[0], pattern=r"^\d{4}-\d{2}$"),
    minimum_possessions: float = Query(default=500.0, ge=0, allow_inf_nan=False),
    player_id: Annotated[tuple[int, ...], Query()] = (),
):
    if season not in AVAILABLE_SEASONS:
        raise HTTPException(
            404,
            "No saved historical lineup table for this season.",
            headers={"Cache-Control": "no-store"},
        )
    period = int(time() // CACHE_SECONDS)
    try:
        if season == CURRENT_SEASON:
            payload = current_lineups(minimum_possessions, player_id)
            payload.update(
                available_seasons=AVAILABLE_SEASONS, minimum_possessions=minimum_possessions
            )
            body = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
        else:
            with _lock:
                body = _query(season, minimum_possessions, tuple(sorted(set(player_id))), period)
    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger(__name__).warning("Saved lineups unavailable: %s", type(error).__name__)
        raise HTTPException(
            503,
            "Saved lineups are temporarily unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None
    max_age = max(0, int((period + 1) * CACHE_SECONDS - time()))
    return Response(
        body,
        media_type="application/json",
        headers={"Cache-Control": f"private, max-age={max_age}"},
    )
