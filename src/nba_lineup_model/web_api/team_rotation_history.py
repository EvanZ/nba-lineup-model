"""Read the producer's accepted team rotation history for game-by-game playback."""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import lru_cache
from time import time

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query, Response

from .rotation_ordering import rotation_order
from .team_rotation import (
    BUCKET,
    CACHE_SECONDS,
    PACIFIC,
    SEASON,
    _reader,
    catalog,
    validate_summary,
)

router = APIRouter()
_lock = threading.Lock()


def unavailable(message):
    return HTTPException(404, message, headers={"Cache-Control": "no-store"})


@lru_cache(maxsize=16)
def cached_history(team, season, period):
    (entries, _, _), _ = catalog()
    entry = entries.get(team)
    if season != SEASON or entry is None or season not in entry["seasons"]:
        raise unavailable("No saved rotation history is published for this team-season.")
    team_id = entry["team_id"]
    root = f"lake/artifacts/team_rotation/season={season}/team_id={team_id}/"
    reader = _reader()
    selected = reader._get(root + "history.json", missing=True)
    if selected is None:
        raise unavailable("No saved rotation history is published for this team-season.")
    index = json.loads(selected[0])
    identity = {
        "schema_version": 1,
        "processing_version": "v1",
        "season": season,
        "team_id": team_id,
    }
    if index["artifact_kind"] != "team_rotation_history_index" or any(
        index[key] != value for key, value in identity.items()
    ):
        raise ValueError("Invalid rotation history index identity")
    points = {}
    today = datetime.now(PACIFIC).date()
    for point in index["points"]:
        day, sha = point["game_date_pacific"], point["sha256"]
        parsed = date.fromisoformat(day)
        key = root + f"game_date_pacific={day}/{sha}.json"
        if (
            set(point) != {"game_date_pacific", "uri", "sha256"}
            or not re.fullmatch(r"[a-f0-9]{64}", sha)
            or point["uri"] != f"s3://{BUCKET}/{key}"
            or parsed.isoformat() != day
            or parsed >= today
            or (points and day <= next(reversed(points)))
        ):
            raise ValueError("Invalid, duplicate, or unordered rotation history point")
        points[day] = (key, point)
    through = index["through_game_date_pacific"]
    if not points or through != next(reversed(points)):
        raise ValueError("Rotation history head differs from the final indexed point")

    def load(day):
        key, source = points[day]
        doc = reader.immutable(key, source["sha256"])
        if (
            doc["artifact_kind"] != "team_rotation"
            or doc["game_date_pacific"] != day
            or any(doc[key] != value for key, value in identity.items())
            or set(doc["coverage"]["game_ids"]) != set(doc["inputs"]["box_scores"])
        ):
            raise ValueError("Rotation history summary has the wrong identity or box coverage")
        return doc, validate_summary(doc, team)

    head, head_matrices = load(through)
    game_days = sorted(head["inputs"]["stints"])
    if not head["coverage"]["game_ids"]:
        raise unavailable("Rotation playback needs at least one saved game.")
    if (
        not game_days
        or game_days[-1] != head["last_team_game_date_pacific"]
        or len(game_days) != head["coverage"]["game_count"]
        or any(day not in points for day in game_days)
    ):
        raise ValueError("Rotation history does not contain one snapshot for every covered game")

    # The head pins the team's game dates. Read only those accepted points and the head.
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(load, (day for day in game_days if day != through)))
    if through in game_days:
        loaded.append((head, head_matrices))

    players = [{"id": p["player_id"], "name": p["player_name"]} for p in head["players"]]
    positions = {p["id"]: i for i, p in enumerate(players)}
    previous_games = set()
    previous_order = list(range(len(players)))
    frames = []
    for day, (doc, matrices) in zip(game_days, loaded, strict=True):
        games = set(doc["coverage"]["game_ids"])
        added = games - previous_games
        if (
            not previous_games <= games
            or len(added) != 1
            or doc["last_team_game_date_pacific"] != day
            or not set(doc["player_order"]) <= positions.keys()
        ):
            raise ValueError(
                "Rotation history cannot be played as one cumulative snapshot per game"
            )
        local_positions = {pid: i for i, pid in enumerate(doc["player_order"])}
        players_for_game = [local_positions.get(p["id"]) for p in players]
        matrix = matrices["floor_correlation"]
        exposure = np.array([p["on_court_possessions"] for p in doc["players"]])
        order, _ = rotation_order(matrix, exposure, "hclust")
        active = [positions[doc["player_order"][i]] for i in order]
        previous_order = active + [i for i in previous_order if i not in active]
        frames.append(
            {
                "date": day,
                "label": date.fromisoformat(day).strftime("%b %d, %Y"),
                "game": len(games),
                "game_id": next(iter(added)),
                "order": previous_order,
                "matrix": [
                    [
                        float(matrix[a, b]) if a is not None and b is not None else None
                        for b in players_for_game
                    ]
                    for a in players_for_game
                ],
                "minutes": [
                    doc["players"][i]["on_court_minutes"] if i is not None else None
                    for i in players_for_game
                ],
                "source": points[day][1],
            }
        )
        previous_games = games
    if previous_games != set(head["coverage"]["game_ids"]):
        raise ValueError("Rotation playback does not reach the published game coverage")
    return {
        "kind": "saved_team_rotation_history",
        "team": team,
        "season": season,
        "players": players,
        "frames": frames,
        "through_game_date_pacific": through,
    }


@router.get("/api/teams/{team}/rotation-history")
def team_rotation_history(
    response: Response,
    team: str = Path(pattern=r"^[A-Za-z0-9]{2,4}$"),
    season: str = Query(default=SEASON, pattern=r"^\d{4}-\d{2}$"),
):
    try:
        period = int(time() // CACHE_SECONDS)
        with _lock:
            data = cached_history(team.upper(), season, period)
        max_age = max(0, int((period + 1) * CACHE_SECONDS - time()))
        response.headers["Cache-Control"] = f"private, max-age={max_age}"
        return data
    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger(__name__).warning("Saved team rotation history unavailable: %s", error)
        raise HTTPException(
            503,
            "Saved rotation history is temporarily unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None
