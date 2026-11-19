"""Team game-day NAIL from accepted predictions; never recalculate model inputs."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Path, Query, Response

from .schedule import PACIFIC, _instant, _number, _reader

router = APIRouter()
SEASON = "2026-27"
CACHE_SECONDS = 600
_lock = threading.Lock()


def read_history(reader, team, season, now):
    games, teams, schedule_ref = reader.official_schedule(season)
    identity = next((row for row in teams if row["team"] == team), None)
    if identity is None:
        raise HTTPException(404, "Team not found.", headers={"Cache-Control": "no-store"})
    today = str(now.astimezone(PACIFIC).date())
    selected = sorted(
        (
            game
            for game in games
            if game["game_id"].startswith("002")
            and game["game_date"] <= today
            and team in (game["home_team"], game["away_team"])
        ),
        key=lambda game: (game["game_date"], game["start_time"] or "", game["game_id"]),
    )

    def point(game):
        side = "home" if game["home_team"] == team else "away"
        other = "away" if side == "home" else "home"
        row = {
            "game_id": game["game_id"],
            "game_date_pacific": game["game_date"],
            "scheduled_start_utc": game["start_time"],
            "home_away": side,
            "opponent": game[other + "_team"],
            "opponent_team_id": game[other + "_team_id"],
            "nail": None,
            "availability": "missing",
            "purpose": None,
            "projection_state": None,
            "ratings_observed_through_game_date_pacific": None,
            "generated_at_utc": None,
            "source": None,
        }
        saved = reader.prediction(season, game["game_date"], game)
        if saved is None:
            return row
        record, reference = saved
        if record[side]["team_id"] != identity["team_id"]:
            raise ValueError("Saved prediction has inconsistent team identity")
        start = _instant(record["scheduled_start_utc"])
        generated = _instant(record["generated_at_utc"])
        if record["purpose"] == "pregame_projection" and generated > start:
            raise ValueError("Pregame prediction was generated after tipoff")
        row.update(
            nail=_number(record["components"][side + "_player_total"]),
            availability="available",
            purpose=record["purpose"],
            projection_state=(
                "historical_replay"
                if record["purpose"] == "historical_replay"
                else "upcoming"
                if start > now
                else "frozen_pregame"
            ),
            ratings_observed_through_game_date_pacific=record.get(
                "ratings_observed_through_game_date_pacific"
            ),
            generated_at_utc=record["generated_at_utc"],
            source=reference,
        )
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        points = list(pool.map(point, selected))
    missing = [row["game_id"] for row in points if row["availability"] == "missing"]
    return {
        "kind": "saved_team_rating_history",
        "season": season,
        "model_version": "v1",
        "team": team,
        "team_id": identity["team_id"],
        "as_of": today,
        "complete": not missing,
        "points": points,
        "coverage": {
            "scheduled_games": len(selected),
            "available_games": len(points) - len(missing),
            "missing_prediction_game_ids": missing,
        },
        "sources": {"schedule": schedule_ref},
    }


@lru_cache(maxsize=64)
def cached_history(team, season, period):
    return read_history(_reader(), team, season, datetime.now(PACIFIC))


@router.get("/api/teams/{team}/rating-history")
def team_rating_history(
    response: Response,
    team: str = Path(pattern=r"^[A-Za-z0-9]{2,4}$"),
    season: str = Query(default=SEASON, pattern=r"^\d{4}-\d{2}$"),
):
    if season != SEASON:
        raise HTTPException(
            404,
            "No saved game-prediction history is published for this season.",
            headers={"Cache-Control": "no-store"},
        )
    try:
        period = int(time() // CACHE_SECONDS)
        with _lock:
            result = cached_history(team.upper(), season, period)
        remaining = max(0, int((period + 1) * CACHE_SECONDS - time()))
        response.headers["Cache-Control"] = f"private, max-age={remaining}"
        return result
    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger(__name__).error("Team NAIL history unavailable: %s", type(error).__name__)
        raise HTTPException(
            503,
            "Saved team NAIL history is unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
