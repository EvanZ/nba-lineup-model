"""Aggregate verified current-season stints and the latest published NAIL in memory."""

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from time import time

from fastapi import HTTPException

from . import player_summary as profiles
from .schedule import PACIFIC, _instant, _number, _reader
from .team_navigation import SEASON

CACHE_SECONDS = 3600
_lock = threading.Lock()

STINT_SCHEMA = """
[
    {
        "game_id":"VARCHAR", "possessions":"DOUBLE",
        "home_team_id":"BIGINT", "away_team_id":"BIGINT",
        "home_team":"VARCHAR", "away_team":"VARCHAR",
        "home_player_ids":["BIGINT"], "away_player_ids":["BIGINT"],
        "margin":"DOUBLE"
    }
]
"""
RATING_SCHEMA = '[{"player_id":"BIGINT","nail":"DOUBLE"}]'
QUERY = f"""
WITH stints AS (
    SELECT unnest(from_json(?, '{STINT_SCHEMA}'), recursive := true)
), ratings AS (
    SELECT unnest(from_json(?, '{RATING_SCHEMA}'), recursive := true)
), units AS (
    SELECT game_id, home_team_id AS team_id, home_team AS team,
           list_sort(home_player_ids) AS player_ids, possessions, margin
    FROM stints
    UNION ALL
    SELECT game_id, away_team_id, away_team, list_sort(away_player_ids), possessions, -margin
    FROM stints
), observed AS (
    SELECT team_id, team, player_ids,
           sum(possessions) AS possessions,
           count(DISTINCT game_id) AS games,
           100.0 * sum(margin) / sum(possessions) AS actual_net_rating
    FROM units
    GROUP BY team_id, team, player_ids
    HAVING sum(possessions) > 0
)
SELECT o.*, CASE WHEN count(r.nail) = 5 THEN sum(r.nail) END AS lineup_nail
FROM observed o
CROSS JOIN unnest(o.player_ids) AS p(player_id)
LEFT JOIN ratings r ON r.player_id = p.player_id
GROUP BY ALL
ORDER BY possessions DESC, games DESC, team ASC, player_ids ASC
"""


def _date_stints(reader, day, games):
    root = f"lake/artifacts/stints/season={SEASON}/game_date_pacific={day}/"
    selected = reader.selection(
        root + "latest.json",
        root,
        {
            "season": SEASON,
            "game_date_pacific": day,
            "schema_version": 1,
            "processing_version": "nba_stints_v1",
        },
        missing=True,
    )
    if selected is None:
        return day, None
    doc, _ = selected
    expected = set(games)
    validation = doc["validation"]
    if not (
        validation["complete"] is True
        and set(validation["processed_game_ids"])
        == set(validation["expected_game_ids"])
        == expected
        and {row["game_id"] for row in doc["rows"]} == expected
    ):
        raise ValueError(f"Incomplete stint game coverage for {day}")
    rows = []
    seen = set()
    for row in doc["rows"]:
        gid, index = row["game_id"], row["stint_index"]
        if type(index) is not int or index < 0 or (gid, index) in seen:
            raise ValueError("Invalid or duplicate current lineup stint index")
        seen.add((gid, index))
        game = games[gid]
        possessions = _number(row["possessions"])
        if possessions < 0 or _instant(row["game_time_utc"]) != _instant(game["start_time"]):
            raise ValueError("Invalid stint possessions or schedule identity")
        normalized = {
            "game_id": gid,
            "possessions": possessions,
            "margin": _number(row["points_home"]) - _number(row["points_away"]),
        }
        for side in ("home", "away"):
            ids = row[side + "_player_ids"]
            team_id = row[side + "_team_id"]
            if (
                team_id != game[side + "_team_id"]
                or len(ids) != 5
                or len(set(ids)) != 5
                or any(type(pid) is not int or pid <= 0 for pid in ids)
            ):
                raise ValueError("Stint does not identify five official players on each team")
            normalized[side + "_team_id"] = team_id
            normalized[side + "_team"] = game[side + "_team"]
            normalized[side + "_player_ids"] = ids
        rows.append(normalized)
    return day, rows


def aggregate(connection, stints, ratings):
    nails = [{"player_id": pid, "nail": player["nail"]} for pid, player in ratings.items()]
    cursor = connection.execute(QUERY, [json.dumps(stints), json.dumps(nails)])
    fields = [col[0] for col in cursor.description]
    rows = []
    for values in cursor.fetchall():
        row = dict(zip(fields, values, strict=True))
        row["player_names"] = [
            ratings[pid]["player_name"] if pid in ratings else f"NBA player {pid}"
            for pid in row["player_ids"]
        ]
        row["lineup_label"] = ", ".join(row["player_names"])
        rows.append(row)
    return rows


@lru_cache(maxsize=1)
def _load(period):
    import duckdb

    reader = _reader()
    schedule, _, _ = reader.official_schedule(SEASON)
    today = datetime.now(PACIFIC).date()
    dates = defaultdict(dict)
    for game in schedule:
        day = game["game_date"]
        if game["game_id"].startswith("002") and day < today.isoformat():
            dates[day][game["game_id"]] = game

    def load_date(day):
        return _date_stints(reader, day, dates[day])

    with ThreadPoolExecutor(max_workers=8) as pool:
        daily = dict(pool.map(load_date, sorted(dates)))
    published = [day for day, rows in daily.items() if rows is not None]
    if not published:
        raise HTTPException(404, "No current-season stints have been published yet.")
    through = published[-1]
    if any(rows is None for day, rows in daily.items() if day <= through):
        raise ValueError("Current-season lineup stints have a missing scheduled date")

    stints = [row for day in published for row in daily[day]]
    ratings_root = f"lake/artifacts/player_ratings/season={SEASON}/model_version=v1/"
    ratings_day = profiles._latest_day(reader.client, ratings_root, today - timedelta(days=1))
    state, rating_ref = profiles._state(reader.client, "player_ratings", SEASON, ratings_day)
    ratings = profiles._rating_players(state)

    with duckdb.connect(
        config={"memory_limit": "128MB", "threads": 1, "temp_directory": ""}
    ) as conn:
        rows = aggregate(conn, stints, ratings)
    return {
        "coverage": {
            "through_game_date_pacific": through,
            "games": sum(len(dates[day]) for day in published),
            "lineups": len(rows),
            "rated_lineups": sum(row["lineup_nail"] is not None for row in rows),
            "pending_game_dates": [day for day in daily if day > through],
        },
        "rating_state": {
            "kind": "in_season_updated",
            "evidence_through_date": str(ratings_day),
            "effective_from_date": state["effective_from_game_date_pacific"],
            "model_version": "v1",
        },
        "sources": {"ratings": rating_ref},
        "lineups": rows,
    }


def current_lineups(minimum_possessions, player_ids):
    with _lock:
        selection = _load(int(time() // CACHE_SECONDS))
    wanted = set(player_ids)
    lineups = (
        row
        for row in selection["lineups"]
        if row["possessions"] >= minimum_possessions and wanted.issubset(row["player_ids"])
    )
    return {
        "kind": "saved_current_lineups",
        "season": SEASON,
        "lineup_state": {"kind": "observed", "season": SEASON},
        "coverage": selection["coverage"],
        "rating_state": selection["rating_state"],
        "sources": selection["sources"],
        "lineups": [dict(row, rank=i) for i, row in enumerate(lineups, 1)],
    }
