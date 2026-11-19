"""Season rankings from saved JSON states and filtered, cached S3 DuckDB queries."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Query, Response

from . import player_summary as profiles
from .player_season_history import SOURCES, _lock, _reader, records
from .schedule import PACIFIC

router = APIRouter()
SEASON = profiles.SEASON
SOURCE_ID = hashlib.sha256(json.dumps(SOURCES, sort_keys=True).encode()).hexdigest()
_lock_rankings = threading.Lock()
COMMON_COLUMNS = [
    "player_name",
    "team",
    "position",
    "draft_number",
    "rapm",
    "prior_rating",
    "season_update",
    "possessions",
]
HISTORICAL_COLUMNS = COMMON_COLUMNS + [
    "offense_rating",
    "defense_rating",
    "additive_profile_adjustment",
    "observed_context_exposure",
    "games",
]
QUERY = """
WITH ratings AS (SELECT * FROM read_parquet(?) WHERE season = ?),
teams AS (SELECT * FROM read_parquet(?) WHERE season = ? AND is_latest_team),
panel AS (SELECT * FROM read_parquet(?) WHERE season = ?),
splits AS (SELECT * FROM read_parquet(?) WHERE season = ?),
context AS (SELECT * FROM read_parquet(?) WHERE season = ?)
SELECT r.player_id,
       coalesce(nullif(trim(r.player_name), ''), nullif(trim(p.player_name), '')) AS player_name,
       r.age, r.rapm, r.rank_all_players AS rank,
       r.prior_rapm AS prior_rating, r.rapm_adjustment_from_prior AS season_update,
       r.additive_profile_adjustment, t.team, t.team_id, p.listed_position AS position,
       p.draft_year, p.draft_round, p.draft_number, p.is_undrafted,
       p.games, p.rapm_possessions AS possessions,
       s.total_nail, s.offense_rating, s.defense_rating, c.observed_context_exposure
FROM ratings r
LEFT JOIN teams t USING (season, player_id)
LEFT JOIN panel p USING (season, player_id)
LEFT JOIN splits s USING (season, player_id)
LEFT JOIN context c USING (season, player_id)
"""


def draft_integer(value):
    if value in (None, "", "Undrafted", "UNDRAFTED", 0, "0"):
        return None
    number = int(value)
    if isinstance(value, bool) or number <= 0 or float(value) != number:
        raise ValueError("Invalid saved draft metadata")
    return number


def draft_metadata(bio):
    draft = bio.get("draft", {})
    year = draft.get("year")
    undrafted = True if str(year).casefold() == "undrafted" else False if year else None
    number = draft_integer(year)
    return {
        "draft_year": number,
        "draft_class_year": number,
        "draft_round": draft_integer(draft.get("round")),
        "draft_number": draft_integer(draft.get("pick")),
        "is_undrafted": undrafted,
    }


@lru_cache(maxsize=1)
def historical_seasons(source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Ranking source identity changed")
    with _lock:
        reader = _reader()
        reader.check_sources({"ratings": SOURCES["ratings"]})
        reader.authenticate()
        seasons = [
            r[0]
            for r in reader.connection.execute(
                "SELECT DISTINCT season FROM read_parquet(?) ORDER BY season DESC",
                [SOURCES["ratings"]["uri"]],
            ).fetchall()
        ]
        reader.check_sources({"ratings": SOURCES["ratings"]})
    if not seasons:
        raise ValueError("No historical rankings published")
    for season in seasons:
        year = int(season[:4])
        if season != f"{year}-{(year + 1) % 100:02d}" or season >= SEASON:
            raise ValueError("Invalid saved rankings season")
    return seasons


@lru_cache(maxsize=1)
def current_rankings(hour):
    client = _reader().client
    cutoff = datetime.now(PACIFIC).date() - timedelta(days=1)
    day = profiles._latest_day(
        client, f"lake/artifacts/player_ratings/season={SEASON}/model_version=v1/", cutoff
    )
    doc, ref = profiles._state(client, "player_ratings", SEASON, day)
    ratings = profiles._rating_players(doc)
    warning = None
    try:
        biographies, teams, biography_ref = profiles._biographies(client, SEASON)
        drafts = {pid: draft_metadata(biographies.get(pid, {})) for pid in ratings}
    except profiles.SOURCE_ERRORS as error:
        logging.getLogger(__name__).error(
            "Ranking identity metadata unavailable: %s", type(error).__name__
        )
        biographies, teams, biography_ref, drafts = {}, {}, None, {}
        warning = (
            "Saved team labels, positions, and draft information are unavailable. "
            "Ratings remain available."
        )
    players = []
    for rank, rating in enumerate(
        sorted(ratings.values(), key=lambda p: (-p["nail"], p["player_name"], p["player_id"])), 1
    ):
        pid = rating["player_id"]
        tid = rating["team_id"]
        if tid is not None:
            profiles._id(tid)
        players.append(
            {
                "player_id": pid,
                "player_name": rating["player_name"],
                "team_id": tid,
                "team": teams.get(tid, "-"),
                "position": biographies.get(pid, {}).get("position"),
                **drafts.get(pid, draft_metadata({})),
                "age": None,
                "rank": rank,
                "rapm": rating["nail"],
                "prior_rating": rating["starting_nail"],
                "season_update": rating["residual"],
                "possessions": rating["on_court_possessions"],
                "games": None,
                "offense_rating": None,
                "defense_rating": None,
                "additive_profile_adjustment": None,
                "observed_context_exposure": None,
                "initialization": rating.get("initialization"),
                "initialized_on": rating.get("initialized_on"),
                "rating_season": SEASON,
                "rating_state": "in_season_updated",
            }
        )
    return {
        "players": players,
        "season": SEASON,
        "available_columns": COMMON_COLUMNS,
        "field_labels": {
            "prior_rating": "Starting NAIL",
            "season_update": "Change since start",
            "possessions": "Rating poss.",
        },
        "rating_state": {
            "kind": "in_season_updated",
            "season": SEASON,
            "evidence_through_date": str(day),
            "effective_from_date": str(day + timedelta(days=1)),
            "model_version": "v1",
        },
        "sources": {"ratings": ref, "biography": biography_ref},
        "metadata_warning": warning,
    }


@lru_cache(maxsize=8)
def historical_rankings(season, source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Ranking source identity changed")
    params = [
        v
        for key in ("ratings", "teams", "panel", "splits", "context")
        for v in (SOURCES[key]["uri"], season)
    ]
    with _lock:
        reader = _reader()
        reader.check_sources(SOURCES)
        reader.authenticate()
        players = records(reader.connection.execute(QUERY, params))
        reader.check_sources(SOURCES)
    if not players or len({p["player_id"] for p in players}) != len(players):
        raise ValueError("Empty or ambiguous saved rankings population")
    for p in players:
        profiles._id(p["player_id"])
        profiles._id(p["rank"])
        if p["rapm"] is None:
            raise ValueError("Missing saved NAIL")
        if p["player_name"] is not None and not isinstance(p["player_name"], str):
            raise ValueError("Invalid saved player name")
        p["name_missing"] = not bool(p["player_name"])
        if p["name_missing"]:
            p["player_name"] = f"NBA player {p['player_id']}"
        for field in (
            "age",
            "rapm",
            "prior_rating",
            "season_update",
            "additive_profile_adjustment",
            "offense_rating",
            "defense_rating",
            "total_nail",
            "observed_context_exposure",
            "possessions",
            "games",
        ):
            value = p[field]
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                raise ValueError("Nonfinite saved ranking statistic")
        for field in ("possessions", "games"):
            if p[field] is not None and p[field] < 0:
                raise ValueError("Negative saved exposure")
        total = p.pop("total_nail")
        split = (total, p["offense_rating"], p["defense_rating"])
        if any(v is not None for v in split) and (
            any(v is None for v in split)
            or not (
                math.isclose(total, p["rapm"], rel_tol=0, abs_tol=1e-8)
                and math.isclose(
                    p["offense_rating"] + p["defense_rating"], p["rapm"], rel_tol=0, abs_tol=1e-8
                )
            )
        ):
            raise ValueError("Saved offense/defense allocation differs from NAIL")
        if p["team_id"] is not None:
            profiles._id(p["team_id"])
        p["team"] = p["team"] or "-"
        if p["position"] is not None and not isinstance(p["position"], str):
            raise ValueError("Invalid saved player position")
        for field in ("draft_year", "draft_round", "draft_number"):
            p[field] = draft_integer(p[field])
        p["draft_class_year"] = p["draft_year"]
        if p["is_undrafted"] is not None and type(p["is_undrafted"]) is not bool:
            raise ValueError("Invalid saved draft status")
        p.update(rating_season=season, rating_state="completed_fit")
    return {
        "season": season,
        "players": sorted(players, key=lambda p: p["rank"]),
        "available_columns": HISTORICAL_COLUMNS,
        "field_labels": {},
        "rating_state": {"kind": "completed_fit", "season": season},
        "metadata_warning": None,
        "sources": SOURCES,
    }


@router.get("/api/rankings")
def rankings(response: Response, season: str = Query(default=SEASON, pattern=r"^\d{4}-\d{2}$")):
    try:
        hour = int(time() // 3600)
        with _lock_rankings:
            if season == SEASON:
                data = current_rankings(hour)
                if data["metadata_warning"]:
                    current_rankings.cache_clear()
            else:
                if season not in historical_seasons(SOURCE_ID):
                    raise HTTPException(
                        status_code=404,
                        detail="No saved rankings for this season.",
                        headers={"Cache-Control": "no-store"},
                    )
                data = historical_rankings(season, SOURCE_ID)
            navigation_warning = None
            try:
                available = [SEASON, *historical_seasons(SOURCE_ID)]
            except Exception:
                if season != SEASON:
                    raise
                available = [SEASON]
                navigation_warning = "Historical season navigation is temporarily unavailable."
        response.headers["Cache-Control"] = (
            "no-store"
            if data["metadata_warning"] or navigation_warning
            else f"private, max-age={max(0, int((hour + 1) * 3600 - time()))}"
        )
        return {**data, "available_seasons": available, "navigation_warning": navigation_warning}
    except HTTPException:
        raise
    except Exception as error:
        logging.getLogger(__name__).error("Saved rankings unavailable: %s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Saved rankings are unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
