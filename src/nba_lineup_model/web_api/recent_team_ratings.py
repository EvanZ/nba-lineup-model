"""Last-five raw pregame team strengths; no stints, model updates or forecast changes."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Response

from .schedule import PACIFIC, _instant, _number, _reader

router = APIRouter()
SEASON = "2026-27"
WINDOW = 5
CACHE_SECONDS = 600
_lock = threading.Lock()


def read_recent_ratings(reader, now):
    games, teams, schedule_ref = reader.official_schedule(SEASON)
    if (
        len(teams) != 30
        or len({t["team_id"] for t in teams}) != 30
        or len({t["team"] for t in teams}) != 30
    ):
        raise ValueError("Official schedule must identify all 30 teams")
    eligible = sorted(
        (
            g
            for g in games
            if g["game_id"].startswith("002")
            and g["start_time"]
            and _instant(g["start_time"]) <= now
        ),
        key=lambda g: (_instant(g["start_time"]), g["game_id"]),
        reverse=True,
    )
    windows = {
        t["team"]: [g for g in eligible if t["team_id"] in (g["home_team_id"], g["away_team_id"])][
            :WINDOW
        ]
        for t in teams
    }
    requested = {g["game_id"]: g for window in windows.values() for g in window}

    def accepted(game):
        saved = reader.prediction(SEASON, game["game_date"], game)
        if saved is None:
            return game["game_id"], None
        record, source = saved
        if record["purpose"] == "pregame_projection" and _instant(
            record["generated_at_utc"]
        ) > _instant(record["scheduled_start_utc"]):
            raise ValueError("Pregame prediction was generated after tipoff")
        values = {
            side: _number(record["components"][side + "_player_total"]) for side in ("home", "away")
        }
        return game["game_id"], (values, record["purpose"], source)

    with ThreadPoolExecutor(max_workers=4) as pool:
        saved = dict(pool.map(accepted, requested.values()))
    rows = []
    for team in sorted(teams, key=lambda t: t["team"]):
        window = windows[team["team"]]
        points = []
        for game in reversed(window):
            selection = saved[game["game_id"]]
            side = "home" if game["home_team_id"] == team["team_id"] else "away"
            points.append(
                {
                    "game_id": game["game_id"],
                    "game_date_pacific": game["game_date"],
                    "nail": selection[0][side] if selection else None,
                    "purpose": selection[1] if selection else None,
                    "source": selection[2] if selection else None,
                }
            )
        missing = [p["game_id"] for p in points if p["nail"] is None]
        available = len(window) == WINDOW and not missing
        rows.append(
            {
                **team,
                "nail": sum(p["nail"] for p in points) / WINDOW if available else None,
                "availability": "available"
                if available
                else "missing_predictions"
                if missing
                else "insufficient_history",
                "game_count": len(window),
                "last_game_date": points[-1]["game_date_pacific"] if points else None,
                "missing_prediction_game_ids": missing,
                "points": points,
            }
        )
    return {
        "kind": "saved_recent_team_ratings",
        "season": SEASON,
        "model_version": "v1",
        "window": WINDOW,
        "complete": all(r["availability"] == "available" for r in rows),
        "definition": (
            "Simple mean of raw pregame team NAIL over the last five started "
            "regular-season games; excludes HCA and B2B adjustments."
        ),
        "teams": rows,
        "sources": {"schedule": schedule_ref},
    }


@lru_cache(maxsize=1)
def cached_recent_ratings(period):
    return read_recent_ratings(_reader(), datetime.now(PACIFIC))


@router.get("/api/recent-team-ratings")
def recent_team_ratings(response: Response):
    try:
        period = int(time() // CACHE_SECONDS)
        with _lock:
            result = cached_recent_ratings(period)
        remaining = max(0, int((period + 1) * CACHE_SECONDS - time()))
        response.headers["Cache-Control"] = f"private, max-age={remaining}"
        return result
    except Exception as error:
        logging.getLogger(__name__).error("Recent team NAIL unavailable: %s", type(error).__name__)
        raise HTTPException(
            503,
            "L5 NAIL is unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
