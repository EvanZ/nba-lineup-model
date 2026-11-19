"""Saved-identity player search and bounded official NBA headshot delivery."""

from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from time import time

import httpx
from fastapi import APIRouter, HTTPException, Path, Query, Response

from . import player_summary as profiles
from .team_navigation import search_teams

router = APIRouter()


def _normalize(value):
    return " ".join(
        "".join(
            char if char.isalnum() else " "
            for char in unicodedata.normalize("NFKD", value.casefold())
            if not unicodedata.combining(char)
        ).split()
    )


def search_players(inputs, query, limit):
    normalized = _normalize(query)
    if not normalized:
        return []
    tokens = normalized.split()
    matches = []
    for pid in inputs["biographies"].keys() | inputs["summaries"].keys():
        biography = inputs["biographies"].get(pid)
        summary = inputs["summaries"].get(pid)
        name = biography["name"] if biography else inputs["names"][pid]
        key = _normalize(name)
        if str(pid) != normalized and not all(token in key for token in tokens):
            continue
        rank = (
            0
            if key == normalized or str(pid) == normalized
            else 1
            if key.startswith(normalized)
            else 2,
            0 if summary else 1,
            key,
            pid,
        )
        matches.append(
            (
                rank,
                {
                    "player_id": pid,
                    "player_name": name,
                    "latest_team": summary["team"] if summary else None,
                    "summary_available": summary is not None,
                },
            )
        )
    return [player for _, player in sorted(matches, key=lambda pair: pair[0])[:limit]]


@router.get("/api/search")
def player_search(
    response: Response,
    q: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=8, ge=1, le=25),
):
    unavailable_sections = []
    players, teams = [], []
    cache_control = "no-store"
    try:
        inputs, cache_control = profiles.shared_inputs()
        players = search_players(inputs, q, limit)
    except profiles.SOURCE_ERRORS:
        unavailable_sections.append("players")
    try:
        teams, _ = search_teams(q, limit)
    except Exception as error:
        logging.getLogger(__name__).error("Team search unavailable: %s", type(error).__name__)
        unavailable_sections.append("teams")
    if len(unavailable_sections) == 2:
        raise HTTPException(
            status_code=503,
            detail="Saved search is unavailable.",
            headers={"Cache-Control": "no-store"},
        )
    response.headers["Cache-Control"] = "no-store" if unavailable_sections else cache_control
    return {
        "players": players,
        "teams": teams,
        "scope": "players_and_teams",
        "unavailable_sections": unavailable_sections,
    }


@lru_cache(maxsize=64)
def _headshot(player_id, _period):
    response = httpx.get(
        f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png",
        timeout=10.0,
        follow_redirects=False,
    )
    response.raise_for_status()
    raw = response.content
    if (
        response.headers.get("content-type", "").split(";", 1)[0] != "image/png"
        or not raw.startswith(b"\x89PNG\r\n\x1a\n")
        or len(raw) > 2_000_000
    ):
        raise ValueError("Invalid official NBA headshot response")
    return raw


@router.get("/api/headshots/{player_id}.png")
def player_headshot(player_id: int = Path(gt=0)):
    try:
        inputs, _ = profiles.shared_inputs()
        if player_id not in inputs["biographies"] and player_id not in inputs["summaries"]:
            raise HTTPException(status_code=404, detail="Player not found.")
        raw = _headshot(player_id, int(time() // 43200))
    except (httpx.HTTPError, *profiles.SOURCE_ERRORS) as error:
        raise HTTPException(
            status_code=404,
            detail="Player photo is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from error
    return Response(
        content=raw, media_type="image/png", headers={"Cache-Control": "public, max-age=43200"}
    )
