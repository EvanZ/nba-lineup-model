"""Serve the saved preseason Wins snapshot without loading the legacy evaluator."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections import Counter
from datetime import date
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Response

from .schedule import BUCKET, _reader

router = APIRouter()
SOURCE = {
    "uri": (
        "s3://nba-gestalt-data/nba-gestalt/releases/gestalt-web-d739c28/artifacts/web/"
        "win_projections/forward_nail_rapm_v1212_residualized_lambda/"
        "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25.json"
    ),
    "sha256": "0800c16b1f5e90d213e4268b22af23a37fdc602ef897dbc816d6e19ccb373cc3",
    "version_id": "G70yQXFtzwdP.yOKetj_WljCSHLF5Hx.",
    "etag": '"cd895e6ed119dfb9fe7548f2cb1329ec"',
    "size": 822867,
}
SOURCE_ID = hashlib.sha256(json.dumps(SOURCE, sort_keys=True).encode()).hexdigest()
_lock = threading.Lock()


def finite(value, *, low=None, high=None):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or low is not None
        and value < low
        or high is not None
        and value > high
    ):
        raise ValueError("Invalid saved preseason forecast value")
    return value


def validate(document):
    minutes, forecast = document["minutes"], document["win_projection"]
    if minutes["season"] != "2026-27" or forecast["season"] != "2026-27":
        raise ValueError("Wrong preseason forecast season")
    teams = minutes["teams"]
    if (
        not isinstance(teams, list)
        or len(teams) != 30
        or len(set(teams)) != 30
        or any(not isinstance(t, str) or not t for t in teams)
    ):
        raise ValueError("Preseason forecast must cover all 30 teams")
    if (
        minutes["regulation_team_minutes"] != 240
        or type(minutes["initial_rotation_size"]) is not int
        or minutes["initial_rotation_size"] <= 0
    ):
        raise ValueError("Invalid saved rotation contract")
    players = minutes["players"]
    if not isinstance(players, list) or len(players) != 521:
        raise ValueError("Incomplete saved preseason player coverage")
    ids = set()
    for player in players:
        pid = player["player_id"]
        if type(pid) is not int or pid <= 0 or pid in ids or player["team"] not in teams:
            raise ValueError("Invalid preseason player/team identity")
        ids.add(pid)
        if not isinstance(player["player_name"], str) or not player["player_name"].strip():
            raise ValueError("Missing saved preseason player name")
        finite(player["projected_nail"])
        finite(player["availability_probability"], low=0, high=1)
        for field in (
            "conditional_minutes_per_game",
            "baseline_minutes_per_game",
            "baseline_minute_share",
        ):
            finite(player[field], low=0)
    if set(p["team"] for p in players) != set(teams):
        raise ValueError("Preseason player coverage omits a team")
    rows = forecast["teams"]
    if len(rows) != 30 or {r["team"] for r in rows} != set(teams):
        raise ValueError("Incomplete or ambiguous preseason team forecasts")
    for row in rows:
        finite(row["team_strength"])
        for key in ("projected_wins", "projected_losses", "scheduled_wins", "unassigned_wins"):
            finite(row[key], low=0, high=82)
        if not math.isclose(row["projected_wins"] + row["projected_losses"], 82, abs_tol=1e-8):
            raise ValueError("Saved season projection does not cover 82 games")
        bands = [finite(row[f"win_total_p{p}"], low=0, high=82) for p in (1, 5, 50, 95, 99)]
        if bands != sorted(bands):
            raise ValueError("Invalid saved win projection intervals")
    games = forecast["scheduled_games"]
    if len(games) != 1200 or forecast["scheduled_game_count"] != 1200:
        raise ValueError("Incomplete saved preseason schedule")
    if (
        forecast["scheduled_games_per_team"] != 80
        or forecast["unassigned_regular_games_per_team"] != 2
    ):
        raise ValueError("Wrong saved NBA Cup schedule contract")
    game_ids, counts = set(), Counter()
    for game in games:
        gid = game["game_id"]
        home, away = game["home_team"], game["away_team"]
        if (
            not isinstance(gid, str)
            or not gid
            or gid in game_ids
            or home not in teams
            or away not in teams
            or home == away
        ):
            raise ValueError("Invalid saved schedule game identity")
        game_ids.add(gid)
        date.fromisoformat(game["game_date"])
        for field in ("home_back_to_back", "away_back_to_back"):
            if game[field] not in (0, 1):
                raise ValueError("Invalid saved schedule controls")
        counts.update((home, away))
    if any(counts[t] != 80 for t in teams):
        raise ValueError("Saved schedule does not cover every team")
    for field in ("win_probability_scale", "home_court", "back_to_back"):
        finite(forecast[field])
    for field in ("calibration_beta", "production_beta", "holdout_brier", "holdout_accuracy"):
        finite(forecast["calibration"][field])


@lru_cache(maxsize=1)
def snapshot(source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Preseason source selection changed")
    key = SOURCE["uri"].removeprefix(f"s3://{BUCKET}/")
    response = _reader().client.get_object(Bucket=BUCKET, Key=key, VersionId=SOURCE["version_id"])
    raw = response["Body"].read()
    if (
        response.get("VersionId") != SOURCE["version_id"]
        or response.get("ETag") != SOURCE["etag"]
        or len(raw) != SOURCE["size"]
        or hashlib.sha256(raw).hexdigest() != SOURCE["sha256"]
    ):
        raise ValueError("Saved preseason snapshot failed verification")
    document = json.loads(raw)
    validate(document)
    # Unwrap the existing page payload; saved players, forecasts and controls stay unchanged.
    return {
        **document["minutes"],
        "win_projection": document["win_projection"],
        "kind": "saved_preseason_win_projections",
        "state_kind": "preseason_preview",
        "source": SOURCE,
        "source_id": SOURCE_ID,
    }


@router.get("/api/win-projections")
def preseason_win_projections(response: Response):
    try:
        with _lock:
            result = snapshot(SOURCE_ID)
        response.headers["Cache-Control"] = "private, max-age=3600"
        return result
    except Exception as error:
        logging.getLogger(__name__).error(
            "Saved preseason Wins unavailable: %s", type(error).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="Saved preseason win projections are unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
