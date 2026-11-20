"""Read accepted team-only rotation summaries; never load raw stints or models."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, datetime
from functools import lru_cache
from time import time

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query, Response

from .historical_team_rotation import historical_summary
from .rotation_ordering import rotation_order
from .schedule import BUCKET, PACIFIC, _number, _reader
from .team_navigation import SEASON, catalog

router = APIRouter()
CACHE_SECONDS = 600
_lock = threading.Lock()


def validate_summary(doc, team):
    calculation = doc["calculation"]
    for field, expected in {
        "population": "team_regular_season_stints_only",
        "possessions": "positive_only",
        "floor_signal": "stint_possessions_when_on_court_else_zero",
        "correlation": "ordinary_pearson",
        "similarity": "shared_over_union",
    }.items():
        if calculation[field] != expected:
            raise ValueError("Unsupported rotation definition")
    coverage = doc["coverage"]
    games = coverage["game_ids"]
    if (
        coverage["complete"] is not True
        or coverage["game_count"] != len(games)
        or len(set(games)) != len(games)
        or any(not re.fullmatch(r"002\d{7}", gid) for gid in games)
        or type(coverage["stint_count"]) is not int
        or coverage["stint_count"] < 0
    ):
        raise ValueError("Incomplete rotation coverage")
    last = doc["last_team_game_date_pacific"]
    if games and (
        last is None or date.fromisoformat(last) > date.fromisoformat(doc["game_date_pacific"])
    ):
        raise ValueError("Invalid last team game date")
    order, players = doc["player_order"], doc["players"]
    if (
        len(set(order)) != len(order)
        or any(type(pid) is not int or pid <= 0 for pid in order)
        or [p["player_id"] for p in players] != order
        or doc["team"] != team
    ):
        raise ValueError("Rotation player/team identity mismatch")
    for player in players:
        if (
            not isinstance(player["player_name"], str)
            or not player["player_name"].strip()
            or _number(player["on_court_possessions"]) <= 0
            or _number(player["on_court_minutes"]) < 0
            or type(player["games"]) is not int
            or not 0 <= player["games"] <= len(games)
        ):
            raise ValueError("Invalid saved rotation player")
    if players != sorted(
        players, key=lambda p: (-p["on_court_possessions"], p["player_name"], p["player_id"])
    ):
        raise ValueError("Rotation candidates must use exposure order")
    n = len(order)
    matrices = {}
    for key in ["shared_possessions", "shared_court_similarity", "floor_correlation"]:
        raw = doc[key]
        if len(raw) != n or any(len(row) != n for row in raw):
            raise ValueError("Rotation matrix dimensions differ from players")
        matrix = np.array([[_number(value) for value in row] for row in raw]).reshape(n, n)
        if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=0):
            raise ValueError("Rotation matrix must be symmetric")
        if (
            n
            and key != "shared_possessions"
            and (
                matrix.min() < (-1 if key == "floor_correlation" else 0) - 1e-10
                or matrix.max() > 1 + 1e-10
                or not np.allclose(np.diag(matrix), 1, atol=1e-9, rtol=0)
            )
        ):
            raise ValueError("Rotation matrix has invalid bounds or diagonal")
        matrices[key] = matrix
    exposure = np.array([p["on_court_possessions"] for p in players])
    shared = matrices["shared_possessions"]
    if n:
        if (
            shared.min() < 0
            or not np.allclose(np.diag(shared), exposure, atol=1e-8, rtol=0)
            or np.any(shared > np.minimum(exposure[:, None], exposure[None, :]) + 1e-8)
        ):
            raise ValueError("Shared possessions differ from exposure")
        expected = shared / (exposure[:, None] + exposure[None, :] - shared)
        if not np.allclose(expected, matrices["shared_court_similarity"], atol=1e-9, rtol=0):
            raise ValueError("Saved similarity differs from shared possessions")
    return matrices


@lru_cache(maxsize=32)
def cached_summary(team, season, period):
    (entries, _, _), _ = catalog()
    if team not in entries or season not in entries[team]["seasons"]:
        raise HTTPException(
            404,
            "No published rotation summary for this team-season.",
            headers={"Cache-Control": "no-store"},
        )
    if season != SEASON:
        doc, source = historical_summary(team, season, period)
        return doc, source, validate_summary(doc, team)
    tid = entries[team]["team_id"]
    root = f"lake/artifacts/team_rotation/season={season}/team_id={tid}/"
    reader = _reader()
    selected = reader._get(root + "latest.json", missing=True)
    if selected is None:
        raise HTTPException(
            404,
            "No published rotation summary for this team-season.",
            headers={"Cache-Control": "no-store"},
        )
    pointer = json.loads(selected[0])
    day = pointer["game_date_pacific"]
    if (
        str(date.fromisoformat(day)) != day
        or date.fromisoformat(day) >= datetime.now(PACIFIC).date()
    ):
        raise ValueError("Rotation summary must cover a completed Pacific date")
    sha = pointer["sha256"]
    key = root + f"game_date_pacific={day}/{sha}.json"
    if not re.fullmatch(r"[a-f0-9]{64}", sha) or pointer["uri"] != f"s3://{BUCKET}/{key}":
        raise ValueError("Rotation selection escapes team/date")
    doc = reader.immutable(key, sha)
    for value in [pointer, doc]:
        for field, expected in {
            "schema_version": 1,
            "processing_version": "v1",
            "season": season,
            "team_id": tid,
            "game_date_pacific": day,
        }.items():
            if value[field] != expected:
                raise ValueError("Rotation summary identity mismatch")
    if set(doc["inputs"]["box_scores"]) != set(doc["coverage"]["game_ids"]):
        raise ValueError("Published rotation input coverage differs from its games")
    if doc["artifact_kind"] != "team_rotation":
        raise ValueError("Wrong rotation artifact kind")
    return doc, {"uri": pointer["uri"], "sha256": sha}, validate_summary(doc, team)


@router.get("/api/teams/{team}/rotation")
def team_rotation(
    response: Response,
    team: str = Path(pattern=r"^[A-Za-z0-9]{2,4}$"),
    season: str = Query(default=SEASON, pattern=r"^\d{4}-\d{2}$"),
    max_players: int = Query(default=30, ge=2, le=30),
    ordering: str = Query(default="hclust", pattern=r"^(hclust|aoe|fpc)$"),
    order_metric: str = Query(default="floor", pattern=r"^(floor|overlap)$"),
):
    try:
        period = int(time() // CACHE_SECONDS)
        with _lock:
            doc, source, matrices = cached_summary(team.upper(), season, period)
        count = min(max_players, len(doc["players"]))
        if count < 2:
            raise HTTPException(
                404,
                "Not enough saved court exposure for a rotation chart.",
                headers={"Cache-Control": "no-store"},
            )
        exposure = np.array([p["on_court_possessions"] for p in doc["players"][:count]])
        metric = "floor_correlation" if order_metric == "floor" else "shared_court_similarity"
        indices, branches = rotation_order(matrices[metric][:count, :count], exposure, ordering)
        result = {
            "kind": "saved_team_rotation",
            "team": doc["team"],
            "season": season,
            "game_date_pacific": doc["game_date_pacific"],
            "last_team_game_date_pacific": doc["last_team_game_date_pacific"],
            "game_count": doc["coverage"]["game_count"],
            "players": [doc["players"][i] for i in indices],
            **{key: matrix[np.ix_(indices, indices)].tolist() for key, matrix in matrices.items()},
            "dendrogram": {"branches": branches},
            "ordering": ordering,
            "order_metric": order_metric,
            "calculation": doc["calculation"],
            "coverage": doc["coverage"],
            "source": source,
        }
        response.headers["Cache-Control"] = (
            f"private, max-age={max(0, int((period + 1) * CACHE_SECONDS - time()))}"
        )
        return result
    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger(__name__).error(
            "Saved team rotation unavailable: %s", type(error).__name__
        )
        raise HTTPException(
            503,
            "Saved rotation data is unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
