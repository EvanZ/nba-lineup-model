"""Cached official team records from saved NBA scoreboards; no forecasting work."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Response

from .schedule import BUCKET, PACIFIC, _instant, _reader

router = APIRouter()
SEASON = "2026-27"
CACHE_SECONDS = 600
_lock = threading.Lock()


def scoreboard_dates(client, today):
    prefix = f"lake/raw/cdn/scoreboard/season={SEASON}/"
    days = set()
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter="/"
    ):
        for row in page.get("CommonPrefixes", []):
            partition = row["Prefix"].removeprefix(prefix).rstrip("/")
            if not partition.startswith("game_date_pacific="):
                continue
            text = partition.removeprefix("game_date_pacific=")
            day = date.fromisoformat(text)
            if str(day) != text:
                raise ValueError("Invalid saved scoreboard partition")
            if day <= today:
                days.add(text)
    return sorted(days, reverse=True)


def read_records(reader, today):
    games, identities, calendar_ref = reader.official_schedule(SEASON)
    teams = {row["team_id"]: row["team"] for row in identities}
    if len(teams) != 30 or len(set(teams.values())) != 30:
        raise ValueError("Official calendar must identify all 30 teams")
    by_date = defaultdict(list)
    for game in games:
        by_date[game["game_date"]].append(game)
    records, sources = {}, {}
    for day in scoreboard_dates(reader.client, today):
        scheduled = by_date[day]
        rows, ref = reader.scoreboard(SEASON, day, scheduled)
        expected = {g["game_id"] for g in scheduled if g["game_id"].startswith("002")}
        observed = {gid for gid in rows if gid.startswith("002")}
        if ref is None or observed != expected:
            raise ValueError("Saved scoreboard does not cover its official regular-season slate")
        sources[day] = ref
        starts = {g["game_id"]: g["start_time"] or "" for g in scheduled}
        for gid, row in sorted(
            rows.items(), key=lambda item: (starts[item[0]], item[0]), reverse=True
        ):
            if not gid.startswith("002") or row["gameStatus"] != 3:
                continue
            for side in ("homeTeam", "awayTeam"):
                team = row[side]
                tid = int(team["teamId"])
                if tid in records:
                    continue
                if tid not in teams or team["teamTricode"] != teams[tid]:
                    raise ValueError("Scoreboard team differs from official calendar")
                wins, losses = team["wins"], team["losses"]
                if (
                    any(type(v) is not int or v < 0 for v in (wins, losses))
                    or not 1 <= wins + losses <= 82
                ):
                    raise ValueError("Invalid official completed-game team record")
                records[tid] = {
                    "team": teams[tid],
                    "team_id": tid,
                    "wins": wins,
                    "losses": losses,
                    "games": wins + losses,
                    "last_game_date": day,
                }
        if set(records) == set(teams):
            break
    if set(records) != set(teams):
        missing = sorted(teams[tid] for tid in teams.keys() - records.keys())
        logging.getLogger(__name__).error("Saved records missing teams: %s", ", ".join(missing))
        raise ValueError("Saved final scoreboards do not cover every team")
    latest = max(sources.values(), key=lambda ref: _instant(ref["captured_at"]))
    return {
        "kind": "saved_current_season_records",
        "season": SEASON,
        "as_of": max(r["last_game_date"] for r in records.values()),
        "updated_at": latest["captured_at"],
        "source": "official NBA CDN scoreboards",
        "teams": sorted(records.values(), key=lambda r: r["team"]),
        "sources": {"schedule": calendar_ref, "scoreboards": sources},
    }


@lru_cache(maxsize=1)
def cached_records(period, today):
    return read_records(_reader(), today)


@router.get("/api/current-season-records")
def current_records(response: Response):
    try:
        period = int(time() // CACHE_SECONDS)
        with _lock:
            result = cached_records(period, datetime.now(PACIFIC).date())
        remaining = max(0, int((period + 1) * CACHE_SECONDS - time()))
        response.headers["Cache-Control"] = f"private, max-age={remaining}"
        return result
    except Exception as error:
        logging.getLogger(__name__).error("Saved NBA records unavailable: %s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Current W-L records are unavailable or failed validation.",
            headers={"Cache-Control": "no-store"},
        ) from None
