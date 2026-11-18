"""Saved availability/minutes history: direct DuckDB query, per-player result cache."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Path, Response

from .player_season_history import _lock, _reader, records

router = APIRouter()

SOURCE = {
    "etag": '"159ae060590c727edb5ef7fd04bf0975"',
    "sha256": "b2ed3a20a5e16833f7674070b357962e9a028fcde99662481b5701cbf45a1cca",
    "size_bytes": 313582,
    "uri": "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/player_rotation_history/forward_nail_rapm_v1212_residualized_lambda/forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25.parquet",
    "version_id": "h810dQKyfjuAWEFiPwcUtr3Bqjaw22vc",
}
SOURCE_ID = hashlib.sha256(json.dumps(SOURCE, sort_keys=True).encode()).hexdigest()
QUERY = """
SELECT season, season_start_year, player_id, player_name, age,
       actual_available_games, known_roster_games, actual_availability_share,
       injury_or_illness_games, rest_games, actual_minutes_per_available_game,
       actual_total_minutes, predicted_available_share,
       predicted_minutes_per_available_game, projected_total_minutes, is_preseason_forecast
FROM read_parquet(?) WHERE player_id = ? ORDER BY season_start_year
"""


def query_player(connection, uri, player_id):
    rows = records(connection.execute(QUERY, [uri, player_id]))
    if len({row["season"] for row in rows}) != len(rows):
        raise ValueError("Duplicate saved availability/minutes season")
    for row in rows:
        if (
            row["player_id"] != player_id
            or row["season"]
            != f"{row['season_start_year']}-{str(row['season_start_year'] + 1)[-2:]}"
        ):
            raise ValueError("Saved availability/minutes player/season identity mismatch")
        if row["season"] > "2026-27" or (
            row["season"] == "2026-27" and not row["is_preseason_forecast"]
        ):
            raise ValueError("This archive does not select current-season observed state")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Invalid saved availability/minutes value")
            if key.endswith("share") and value is not None and not 0 <= value <= 1:
                raise ValueError("Invalid saved availability share")
        for key in (
            "actual_available_games",
            "known_roster_games",
            "injury_or_illness_games",
            "rest_games",
            "actual_minutes_per_available_game",
            "actual_total_minutes",
            "predicted_minutes_per_available_game",
            "projected_total_minutes",
        ):
            if row[key] is not None and row[key] < 0:
                raise ValueError("Negative saved availability/minutes value")
    return rows


@lru_cache(maxsize=256)
def _history(player_id, source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Availability/minutes source selection differs from this code")
    reader = _reader()
    reader.check_sources({"rotation": SOURCE})
    reader.authenticate()
    rows = query_player(reader.connection, SOURCE["uri"], player_id)
    reader.check_sources({"rotation": SOURCE})
    return {
        "kind": "saved_player_rotation_history",
        "schema_version": 1,
        "player_id": player_id,
        "player_name": rows[0]["player_name"] if rows else None,
        "rows": rows,
        "coverage_from_season": "2015-16",
        "preseason_forecast_season": "2026-27",
        "source": SOURCE,
        "source_id": SOURCE_ID,
    }


@router.get("/api/players/{player_id}/rotation-history")
def rotation_history(response: Response, player_id: int = Path(gt=0)):
    try:
        with _lock:
            result = _history(player_id, SOURCE_ID)
        response.headers["Cache-Control"] = "private, max-age=3600"
        return result
    except Exception as error:
        logging.getLogger(__name__).error(
            "Saved availability/minutes history unavailable: %s", type(error).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="Saved availability/minutes history is unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
