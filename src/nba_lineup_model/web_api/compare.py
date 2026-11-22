"""Player lookup for comparisons of accepted historical and current ratings."""

import logging
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Query, Response

from .player_discovery import _normalize
from .player_season_history import SOURCES, SOURCE_ID, _lock, _reader, records
from .rankings import SEASON, _lock_rankings, current_rankings

router = APIRouter()

CATALOG_QUERY = """
WITH latest AS (
    SELECT player_id, player_name, season,
           count(DISTINCT season) OVER (PARTITION BY player_id) AS history_seasons
    FROM read_parquet(?)
    QUALIFY row_number() OVER (PARTITION BY player_id ORDER BY season DESC) = 1
), teams AS (
    SELECT player_id, season, team
    FROM read_parquet(?) WHERE is_latest_team
)
SELECT l.player_id, l.player_name, l.season AS latest_season,
       t.team AS latest_team, l.history_seasons
FROM latest l LEFT JOIN teams t USING (player_id, season)
"""


@lru_cache(maxsize=1)
def historical_catalog(source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Historical source selection differs from this code")
    reader = _reader()
    sources = {name: SOURCES[name] for name in ("ratings", "teams")}
    reader.check_sources(sources)
    reader.authenticate()
    rows = records(reader.connection.execute(
        CATALOG_QUERY, [source["uri"] for source in sources.values()]
    ))
    reader.check_sources(sources)
    catalog = {player["player_id"]: player for player in rows}
    if not catalog or len(catalog) != len(rows):
        raise ValueError("Invalid saved comparison catalog")
    return catalog


def match_players(historical, current, query, limit):
    normalized = _normalize(query)
    if not normalized:
        return []
    tokens = normalized.split()
    catalog = historical.copy()
    for player in current:
        player_id = player["player_id"]
        catalog[player_id] = {
            "player_id": player_id,
            "player_name": player["player_name"],
            "latest_season": SEASON,
            "latest_team": player["team"],
            "history_seasons": historical.get(player_id, {}).get("history_seasons", 0),
        }
    matches = []
    for player_id, player in catalog.items():
        name = _normalize(player["player_name"] or "")
        exact = normalized in (str(player_id), name)
        if exact or all(token in name for token in tokens):
            priority = (
                0 if exact else 1 if name.startswith(normalized) else 2,
                -player["history_seasons"],
                name,
                player_id,
            )
            matches.append((priority, player))
    return [player for _, player in sorted(matches)[:limit]]


@router.get("/api/compare/players")
def comparison_search(
    response: Response,
    q: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=12, ge=1, le=25),
):
    historical, current, unavailable = {}, [], []
    hour = int(time() // 3600)
    try:
        with _lock:
            historical = historical_catalog(SOURCE_ID)
    except Exception as error:
        logging.getLogger(__name__).warning("Historical comparison catalog unavailable: %s", type(error).__name__)
        unavailable.append("historical")
    try:
        with _lock_rankings:
            snapshot = current_rankings(hour)
            current = snapshot["players"]
            if snapshot["metadata_warning"]:
                current_rankings.cache_clear()
    except Exception as error:
        logging.getLogger(__name__).warning("Current comparison catalog unavailable: %s", type(error).__name__)
        unavailable.append("current-season")
    if len(unavailable) == 2:
        raise HTTPException(
            status_code=503,
            detail="Saved comparison search is unavailable.",
            headers={"Cache-Control": "no-store"},
        )
    response.headers["Cache-Control"] = (
        "no-store" if unavailable else f"private, max-age={max(0, int((hour + 1) * 3600 - time()))}"
    )
    return {"players": match_players(historical, current, q, limit), "unavailable": unavailable}
