"""Read-only roster movement from the saved opening snapshot and official transactions."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from datetime import date, datetime, timedelta
from functools import lru_cache
from time import time
from typing import Literal

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Response

from .player_season_history import _lock, _reader
from .roster_movements import movement_payload
from .schedule import BUCKET, PACIFIC
from .team_navigation import selected

router = APIRouter()
_logger = logging.getLogger(__name__)
_moves_lock = threading.Lock()
SEASON = "2026-27"
PRIOR_SEASON = "2025-26"
OPENING_DATE = date(2026, 10, 20)
TRANSACTION_ROOT = f"lake/raw/nba/transactions/season={SEASON}/"
TRANSACTION_SOURCE = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
TRANSACTION_TYPES = {"Trade": "trade", "Signing": "signing", "Waive": "waiver"}


def source(key, version_id, etag, size):
    return {
        "uri": f"s3://{BUCKET}/{key}",
        "version_id": version_id,
        "etag": f'"{etag}"',
        "size_bytes": size,
    }


# Exact versions of the October 20 opening roster and the saved historical tables.
SOURCES = {
    "roster": source(
        "lake/curated/team_rosters/2026-27/part-00000.parquet",
        "Nu5MTw9QXm0_7QZ.C8lmoKhmSzIl3WIF",
        "09e6d9b1a061ca8b57a23795f5cc807e",
        33079,
    ),
    "origin": source(
        "lake/curated/team_rosters/2025-26/final_regular_season_snapshot.parquet",
        "7j_eoslUMjI4BVBh1DAN0qrooQxiNsmI",
        "339d65695223f2a26d2f22b229cdd925",
        17734,
    ),
    "minutes": source(
        "lake/analytical/player_season_panel/player_seasons.parquet",
        "bw0y8G2Rgz8D0yOzh2OjgCqQoeQhyVn0",
        "2b66b860a9a0eeab2f900b9e3c83e68e",
        3505317,
    ),
    "catalog": source(
        "lake/catalog/players.parquet",
        "oAhO7eIIVT6agACESCNIhGSxoKU2ceTB",
        "5515940c58561ff54759a116d69afc04",
        295680,
    ),
}

QUERY = """
SELECT cast(r.player_id AS VARCHAR) AS player_id, r.player_name,
       cast(r.team_id AS BIGINT) AS team_id,
       r.team_abbreviation, r.experience, r.school, r.how_acquired, o.source_team,
       p.minutes AS prior_season_minutes, b.college, b.country
FROM read_parquet(?) r
LEFT JOIN read_parquet(?) o ON cast(r.player_id AS BIGINT) = cast(o.player_id AS BIGINT)
LEFT JOIN read_parquet(?) p
  ON cast(r.player_id AS BIGINT) = cast(p.player_id AS BIGINT) AND p.season = ?
LEFT JOIN read_parquet(?) b ON cast(r.player_id AS BIGINT) = b.player_id
"""


def nba_id(value):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or int(value) != value
        or value <= 0
    ):
        raise ValueError("Invalid transaction player/team ID")
    return int(value)


@lru_cache(maxsize=1)
def offseason():
    reader = _reader()
    with _lock:
        reader.check_sources(SOURCES)
        reader.authenticate()
        rows = reader.connection.execute(
            QUERY,
            [
                SOURCES["roster"]["uri"],
                SOURCES["origin"]["uri"],
                SOURCES["minutes"]["uri"],
                PRIOR_SEASON,
                SOURCES["catalog"]["uri"],
            ],
        ).fetchdf()
        reader.check_sources(SOURCES)
    teams = dict(zip(rows["team_id"], rows["team_abbreviation"], strict=True))
    if len(teams) != 30 or len(set(teams.values())) != 30 or rows["player_id"].duplicated().any():
        raise ValueError("Invalid saved opening roster coverage")
    root = f"lake/artifacts/player_ratings/season={SEASON}/model_version=v1/priors/"
    prior, _ = selected(
        reader.client,
        root + "latest.json",
        root,
        {"schema_version": 1, "season": SEASON, "model_version": "v1"},
    )
    if prior["artifact_kind"] != "player_rating_priors":
        raise ValueError("Invalid saved preseason rating kind")
    ratings, names = {}, {}
    for player in prior["players"]:
        player_id = str(player["player_id"])
        if player_id in ratings or not math.isfinite(player["rapm"]):
            raise ValueError("Invalid saved preseason rating")
        ratings[player_id] = player["rapm"]
        names[player_id] = player["player_name"]
    names.update(zip(rows["player_id"], rows["player_name"], strict=True))
    payload = movement_payload(rows, ratings, SEASON, PRIOR_SEASON)
    payload["as_of_date"] = OPENING_DATE.isoformat()
    return payload, teams, names, ratings


@lru_cache(maxsize=512)
def transactions_on_day(key, etag, day):
    client = _reader().client
    with client.get_object(Bucket=BUCKET, Key=key, IfMatch=etag)["Body"] as body:
        document = json.load(body)
    if (
        document["season"] != SEASON
        or document["game_date_pacific"] != day.isoformat()
        or document["source_url"] != TRANSACTION_SOURCE
        or not isinstance(document["transactions"], list)
    ):
        raise ValueError("Invalid saved transaction partition")
    for row in document["transactions"]:
        if date.fromisoformat(row["TRANSACTION_DATE"][:10]) != day:
            raise ValueError("Transaction outside its Pacific date partition")
    return document["transactions"]


def accepted_roster(client, day):
    root = f"lake/artifacts/rosters/season={SEASON}/game_date_pacific={day}/"
    try:
        roster, _ = selected(
            client,
            root + "latest.json",
            root,
            {"schema_version": 1, "season": SEASON, "game_date_pacific": str(day)},
        )
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
            raise
        return {}
    return roster["teams"]


def trade_origin(team_id, player_id, before, after, teams):
    if player_id not in after.get(str(team_id), ()):
        return None
    origins = [teams[int(tid)] for tid, players in before.items() if player_id in players]
    if len(origins) == 1 and origins[0] != teams[team_id]:
        return origins[0]
    return None


@lru_cache(maxsize=1)
def in_season(hour):
    _, teams, names, ratings = offseason()
    client = _reader().client
    today = datetime.now(PACIFIC).date()
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=TRANSACTION_ROOT)
    partitions = {}
    for page in pages:
        for item in page.get("Contents", ()):
            key = item["Key"]
            match = re.fullmatch(
                r"game_date_pacific=(\d{4}-\d{2}-\d{2})/latest\.json",
                key.removeprefix(TRANSACTION_ROOT),
            )
            if match:
                day = date.fromisoformat(match[1])
                if OPENING_DATE <= day <= today:
                    partitions[day] = (key, item["ETag"])
    if not partitions:
        raise ValueError("No saved current-season transactions")
    transactions = []
    rosters = {}
    for day, (key, etag) in sorted(partitions.items(), reverse=True):
        for row in transactions_on_day(key, etag, day):
            player_id, team_id = nba_id(row["PLAYER_ID"]), nba_id(row["TEAM_ID"])
            team = teams[team_id]
            kind = row["Transaction_Type"]
            move_type = TRANSACTION_TYPES.get(kind, "other")
            source_team = team if move_type == "waiver" else None
            target_team = None if move_type == "waiver" else team
            if move_type == "trade":
                previous = day - timedelta(days=1)
                for roster_day in (previous, day):
                    if roster_day not in rosters:
                        rosters[roster_day] = accepted_roster(client, roster_day)
                source_team = trade_origin(
                    team_id, player_id, rosters[previous], rosters[day], teams
                )
            name = names.get(str(player_id))
            if not name:
                name = " ".join(row["PLAYER_SLUG"].split("-")).title() or f"Player #{player_id}"
            identity = {k: v for k, v in row.items() if k != "first_observed_at"}
            event_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            transactions.append(
                {
                    "event_id": event_id,
                    "transaction_date": str(day),
                    "transaction_type": kind,
                    "move_type": move_type,
                    "player_id": player_id,
                    "player_name": name,
                    "source_team": source_team,
                    "target_team": target_team,
                    "team": team,
                    "how_acquired": row["TRANSACTION_DESCRIPTION"],
                    "projected_rating": ratings.get(str(player_id)),
                    "source_uri": f"s3://{BUCKET}/{key}",
                }
            )
    through = max(partitions)
    missing = [
        str(OPENING_DATE + timedelta(days=i))
        for i in range((through - OPENING_DATE).days)
        if OPENING_DATE + timedelta(days=i) not in partitions
    ]
    return {
        "current_season": SEASON,
        "prior_season": PRIOR_SEASON,
        "teams": sorted(teams.values()),
        "from_date": str(min(partitions)),
        "through_date": str(through),
        "missing_dates": missing,
        "transactions": transactions,
    }


@router.get("/api/roster-moves")
def roster_moves(response: Response, period: Literal["offseason", "in-season"] = "offseason"):
    hour = int(time() // 3600)
    try:
        with _moves_lock:
            data = in_season(hour) if period == "in-season" else offseason()[0]
    except Exception as error:
        _logger.warning("Saved Moves %s unavailable: %s", period, type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="Saved roster movement data is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from None
    response.headers["Cache-Control"] = (
        f"private, max-age={max(0, int((hour + 1) * 3600 - time()))}"
    )
    return data
