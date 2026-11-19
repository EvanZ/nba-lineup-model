"""Read-only team/season navigation and search, independent of roster ratings."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime
from functools import lru_cache
from time import time

from fastapi import APIRouter, HTTPException, Path, Query, Response

from .player_season_history import SOURCES, _lock, _reader, records
from .player_summary import _latest_day
from .schedule import BUCKET, PACIFIC

router = APIRouter()
SEASON = "2026-27"
_lock_navigation = threading.Lock()
SOURCE = SOURCES["teams"]
SOURCE_ID = hashlib.sha256(json.dumps(SOURCE, sort_keys=True).encode()).hexdigest()
QUERY = """SELECT DISTINCT season, team_id, team FROM read_parquet(?)
           WHERE is_latest_team ORDER BY season, team"""
DEFERRED = ["win_history", "rotation", "lineups", "lineup_lab"]


def _season(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError("Invalid saved season")
    if int(value[-2:]) != (int(value[:4]) + 1) % 100:
        raise ValueError("Invalid saved season range")
    return value


def _team(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9]{2,4}", value):
        raise ValueError("Invalid saved team code")
    return value


def _json(client, key):
    return client.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def selected(client, key, root, identity, suffix=".json"):
    pointer = json.loads(_json(client, key))
    sha = pointer["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid team-navigation source checksum")
    target = root + sha + suffix
    if pointer["uri"] != f"s3://{BUCKET}/{target}":
        raise ValueError("Navigation selection escapes its source scope")
    raw = _json(client, target)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Navigation source checksum mismatch")
    doc = json.loads(raw)
    if "schema_version" in identity and (
        type(pointer.get("schema_version")) is not int or type(doc.get("schema_version")) is not int
    ):
        raise ValueError("Invalid navigation schema identity")
    for name, value in identity.items():
        if pointer.get(name) != value or doc.get(name) != value:
            raise ValueError("Navigation source identity mismatch")
    return doc, {"uri": pointer["uri"], "sha256": sha}


@lru_cache(maxsize=1)
def historical_catalog(source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Historical team source selection changed")
    with _lock:
        reader = _reader()
        reader.check_sources({"teams": SOURCE})
        reader.authenticate()
        rows = records(reader.connection.execute(QUERY, [SOURCE["uri"]]))
        reader.check_sources({"teams": SOURCE})
    if not rows:
        raise ValueError("No historical team navigation is published")
    for row in rows:
        _season(row["season"])
        _team(row["team"])
        if row["season"] >= SEASON or type(row["team_id"]) is not int or row["team_id"] <= 0:
            raise ValueError("Historical team/season identity mismatch")
    return rows


def read_current(client, today):
    root = f"lake/artifacts/rosters/season={SEASON}/"
    day = _latest_day(client, root, today)
    date_root = root + f"game_date_pacific={day}/"
    doc, roster_ref = selected(
        client,
        date_root + "latest.json",
        date_root,
        {"schema_version": 1, "season": SEASON, "game_date_pacific": str(day)},
    )
    teams = doc["teams"]
    if not isinstance(teams, dict) or len(teams) != 30:
        raise ValueError("Navigation roster must cover all 30 teams")
    ids = set()
    for tid, players in teams.items():
        if (
            not tid.isdigit()
            or int(tid) <= 0
            or str(int(tid)) != tid
            or not isinstance(players, list)
        ):
            raise ValueError("Invalid roster team identity")
        if len(set(players)) != len(players) or any(
            type(pid) is not int or pid <= 0 for pid in players
        ):
            raise ValueError("Invalid roster player identities")
        if ids.intersection(players):
            raise ValueError("Roster player is assigned to multiple teams")
        ids.update(players)
    schedule_root = f"lake/raw/scheduleleaguev2/season={SEASON}/"
    schedule, schedule_ref = selected(
        client,
        schedule_root + "latest.json",
        schedule_root + "sha256=",
        {},
        suffix="/schedule.json",
    )
    league = schedule["leagueSchedule"]
    if league["seasonYear"] != SEASON:
        raise ValueError("Official schedule season mismatch")
    identities = {}
    for group in league["gameDates"]:
        for game in group["games"]:
            for team in (game["homeTeam"], game["awayTeam"]):
                tid = str(team["teamId"])
                if tid not in teams:
                    continue
                code = _team(team["teamTricode"])
                name = " ".join((team["teamCity"].strip(), team["teamName"].strip()))
                item = {"team_id": int(tid), "team": code, "display_name": name}
                if not name.strip() or tid in identities and identities[tid] != item:
                    raise ValueError("Ambiguous official team identity")
                identities[tid] = item
    if set(identities) != set(teams) or len({v["team"] for v in identities.values()}) != 30:
        raise ValueError("Official schedule and roster team coverage differ")
    return list(identities.values()), str(day), {"roster": roster_ref, "schedule": schedule_ref}


def merge_catalog(current, historical):
    catalog = {}
    seen = set()
    for row in historical:
        code, season = row["team"], row["season"]
        key = (code, season)
        if key in seen:
            raise ValueError("Historical team code has ambiguous season identity")
        seen.add(key)
        entry = catalog.setdefault(code, {"team": code, "display_name": code, "seasons": []})
        entry["seasons"].append(season)
    for row in current:
        entry = catalog.setdefault(row["team"], {"team": row["team"], "seasons": []})
        entry.update(display_name=row["display_name"], team_id=row["team_id"])
        entry["seasons"].append(SEASON)
    for entry in catalog.values():
        entry["seasons"] = sorted(set(entry["seasons"]), reverse=True)
        entry["latest_season"] = entry["seasons"][0]
        entry["season_count"] = len(entry["seasons"])
    return catalog


@lru_cache(maxsize=1)
def _catalog(hour, today):
    reader = _reader()
    current, day, sources = read_current(reader.client, today)
    catalog = merge_catalog(current, historical_catalog(SOURCE_ID))
    return catalog, day, {**sources, "historical_teams": SOURCE}


def catalog():
    hour = int(time() // 3600)
    with _lock_navigation:
        result = _catalog(hour, datetime.now(PACIFIC).date())
    return result, f"private, max-age={max(0, int((hour + 1) * 3600 - time()))}"


def team_list(entries, season):
    selected = sorted(code for code, row in entries.items() if season in row["seasons"])
    if not selected:
        raise HTTPException(
            status_code=404,
            detail="No saved teams are available for this season.",
            headers={"Cache-Control": "no-store"},
        )
    return selected


def search_teams(query, limit):
    # Import the text normalizer lazily to keep discovery/navigation independent.
    from .player_discovery import _normalize

    (entries, _, _), cache_control = catalog()
    text = _normalize(query)
    if not text:
        return [], cache_control
    tokens = text.split()
    hits = [
        row
        for row in entries.values()
        if all(token in _normalize(row["team"] + " " + row["display_name"]) for token in tokens)
    ]
    hits.sort(
        key=lambda row: (
            0 if _normalize(row["team"]) == text else 1,
            0 if row["latest_season"] == SEASON else 1,
            row["display_name"],
            row["team"],
        )
    )
    return [
        {k: row[k] for k in ("team", "display_name", "latest_season", "season_count")}
        for row in hits[:limit]
    ], cache_control


def unavailable(error):
    logging.getLogger(__name__).error("Saved team navigation unavailable: %s", type(error).__name__)
    return HTTPException(
        status_code=503,
        detail="Saved team navigation is unavailable or failed validation.",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/teams")
def teams(response: Response, season: str = Query(default=SEASON, pattern=r"^\d{4}-\d{2}$")):
    try:
        (entries, day, sources), cache_control = catalog()
        selected = team_list(entries, season)
        response.headers["Cache-Control"] = cache_control
        return {
            "season": season,
            "teams": selected,
            "available_seasons": sorted(
                {s for row in entries.values() for s in row["seasons"]}, reverse=True
            ),
            "team_details": [entries[code] for code in selected],
            "roster_as_of": day,
            "sources": sources,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise unavailable(error) from None


@router.get("/api/teams/{team}")
def team_navigation(
    response: Response,
    team: str = Path(pattern=r"^[A-Za-z0-9]{2,4}$"),
    season: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$"),
    minimum_possessions: float = Query(default=50, ge=0),
):
    try:
        (entries, day, sources), cache_control = catalog()
        code = team.upper()
        if code not in entries:
            raise HTTPException(
                status_code=404, detail="Team not found.", headers={"Cache-Control": "no-store"}
            )
        chosen = season or entries[code]["latest_season"]
        if chosen not in entries[code]["seasons"]:
            raise HTTPException(
                status_code=404,
                detail="This team has no saved selection for that season.",
                headers={"Cache-Control": "no-store"},
            )
        from .team_roster import roster

        saved_roster = roster(entries[code], chosen, day, sources)
        response.headers["Cache-Control"] = (
            cache_control
            if saved_roster["status"] == "available" and not saved_roster.get("name_source_errors")
            else "no-store"
        )
        return {
            "kind": "saved_team_navigation",
            "roster": saved_roster,
            "team": code,
            "display_name": entries[code]["display_name"],
            "season": chosen,
            "available_seasons": entries[code]["seasons"],
            "teams": [entries[t] for t in team_list(entries, chosen)],
            "roster_as_of": day if chosen == SEASON else None,
            "deferred_sections": DEFERRED,
            "sources": sources,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise unavailable(error) from None
