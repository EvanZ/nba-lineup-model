"""Player identity and current daily state, read directly from accepted S3 outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import date, datetime, timedelta
from functools import lru_cache
from time import time
from urllib.parse import parse_qs, urlparse

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, Path, Response

from .schedule import BUCKET, PACIFIC, _number

router = APIRouter()
SEASON = "2026-27"
VERSION = "v1"
_cache_lock = threading.Lock()
SOURCE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, BotoCoreError, ClientError)


def _read(client, key):
    raw = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return raw, json.loads(raw)


def _id(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid official player/team ID")
    return value


def _latest_day(client, root, cutoff):
    days = set()
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=root, Delimiter="/"
    ):
        for item in page.get("CommonPrefixes", []):
            match = re.fullmatch(
                re.escape(root) + r"game_date_pacific=(\d{4}-\d{2}-\d{2})/", item["Prefix"]
            )
            if match and date.fromisoformat(match[1]) <= cutoff:
                days.add(date.fromisoformat(match[1]))
    if not days:
        raise ValueError("No completed daily player state is published")
    return max(days)


def _state(client, kind, season, day):
    root = f"lake/artifacts/{kind}/season={season}/model_version={VERSION}/game_date_pacific={day}/"
    _, pointer = _read(client, root + "latest.json")
    sha = pointer["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid player-state checksum")
    key = root + sha + ".json"
    if pointer["uri"] != f"s3://{BUCKET}/{key}":
        raise ValueError("Player selection escapes its date/season/version")
    raw, doc = _read(client, key)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Player state failed checksum verification")
    expected = {"schema_version": 1, "season": season, "game_date_pacific": str(day)}
    for record in (pointer, doc):
        if type(record["schema_version"]) is not int or any(
            record[k] != v for k, v in expected.items()
        ):
            raise ValueError("Player-state identity/schema mismatch")
    version = doc["model_version"] if kind == "player_ratings" else doc["model"]["model_version"]
    artifact = "player_ratings" if kind == "player_ratings" else "conditional_minutes_state"
    if (
        pointer["model_version"] != VERSION
        or version != VERSION
        or doc["artifact_kind"] != artifact
        or doc["date_contract"] != "completed_game_date_v1"
        or doc["observed_through_game_date_pacific"] != str(day)
        or doc["effective_from_game_date_pacific"] != str(day + timedelta(days=1))
    ):
        raise ValueError("Wrong daily player-state version/date contract")
    return doc, {
        "uri": pointer["uri"],
        "sha256": sha,
        "selection_uri": f"s3://{BUCKET}/{root}latest.json",
    }


def _biographies(client, season):
    root = f"lake/raw/playerindex/{season}"
    raw, doc = _read(client, root + ".json")
    _, meta = _read(client, root + ".meta.json")
    sha = hashlib.sha256(raw).hexdigest()
    source = urlparse(meta["url"])
    if (
        meta["sha256"] != sha
        or meta["season"] != season
        or meta["endpoint"] != "playerindex"
        or source.scheme != "https"
        or source.hostname != "stats.nba.com"
        or source.path.rstrip("/") != "/stats/playerindex"
        or parse_qs(source.query).get("Season") != [season]
    ):
        raise ValueError("Official PlayerIndex source/checksum mismatch")
    if doc["resource"] != "playerindex" or doc["parameters"]["Season"] != season:
        raise ValueError("Wrong official PlayerIndex document identity")
    sets = doc["resultSets"]
    if not isinstance(sets, list) or len(sets) != 1:
        raise ValueError("Ambiguous official PlayerIndex table")
    table = sets[0]
    headers = table["headers"]
    required = {
        "PERSON_ID",
        "PLAYER_FIRST_NAME",
        "PLAYER_LAST_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "POSITION",
        "DRAFT_YEAR",
        "DRAFT_ROUND",
        "DRAFT_NUMBER",
    }
    if len(set(headers)) != len(headers) or not required.issubset(headers):
        raise ValueError("Official PlayerIndex columns changed")
    players, teams = {}, {}
    for values in table["rowSet"]:
        if len(values) != len(headers):
            raise ValueError("Malformed official PlayerIndex row")
        row = dict(zip(headers, values, strict=True))
        pid = _id(row["PERSON_ID"])
        if pid in players:
            raise ValueError("Duplicate official PlayerIndex ID")
        name = " ".join(
            str(row[k] or "").strip() for k in ("PLAYER_FIRST_NAME", "PLAYER_LAST_NAME")
        ).strip()
        if not name:
            raise ValueError("Missing official player name")
        players[pid] = {
            "name": name,
            "position": row["POSITION"] or None,
            "draft": {
                "year": row["DRAFT_YEAR"],
                "round": row["DRAFT_ROUND"],
                "pick": row["DRAFT_NUMBER"],
            },
        }
        team_id, abbreviation = row["TEAM_ID"], row["TEAM_ABBREVIATION"]
        if team_id and abbreviation:
            _id(team_id)
            if team_id in teams and teams[team_id] != abbreviation:
                raise ValueError("Ambiguous official team identity")
            teams[team_id] = abbreviation
    return (
        players,
        teams,
        {
            "uri": f"s3://{BUCKET}/{root}.json",
            "sha256": sha,
            "metadata_uri": f"s3://{BUCKET}/{root}.meta.json",
            "source_url": meta["url"],
            "uploaded_at": meta.get("uploaded_at"),
            "fetched_at": meta.get("fetched_at"),
            "acquisition_method": meta.get("acquisition_method"),
        },
    )


def read_player_inputs(client, *, season, cutoff):
    """Load one shared pair of daily selections, never the full daily history."""
    roots = {
        kind: f"lake/artifacts/{kind}/season={season}/model_version={VERSION}/"
        for kind in ("player_ratings", "player_minutes")
    }
    days = {kind: _latest_day(client, root, cutoff) for kind, root in roots.items()}
    if len(set(days.values())) != 1:
        raise ValueError("Latest completed ratings and minutes dates are not aligned")
    day = days["player_ratings"]
    ratings, rating_ref = _state(client, "player_ratings", season, day)
    minutes, minutes_ref = _state(client, "player_minutes", season, day)
    biographies, teams, biography_ref = _biographies(client, season)
    rating_players = {}
    for player in ratings["players"]:
        pid = _id(player["player_id"])
        if pid in rating_players:
            raise ValueError("Duplicate daily rating player ID")
        if not isinstance(player["player_name"], str) or not player["player_name"].strip():
            raise ValueError("Missing saved player name")
        for field in (
            "nail",
            "starting_nail",
            "residual",
            "on_court_possessions",
            "standard_error",
        ):
            _number(player[field])
        if player["on_court_possessions"] < 0 or player["standard_error"] < 0:
            raise ValueError("Invalid saved possession count or rating uncertainty")
        if abs(player["nail"] - player["starting_nail"] - player["residual"]) > 1e-9:
            raise ValueError("Saved rating and delta disagree")
        rating_players[pid] = player
    minute_players = {}
    for key, player in minutes["players"].items():
        pid = _id(player["player_id"])
        if str(pid) != key or pid in minute_players:
            raise ValueError("Daily minutes player-key identity mismatch")
        if any(_number(player[field]) < 0 for field in ("estimate", "starting_minutes")):
            raise ValueError("Invalid saved conditional-minutes estimate")
        if not isinstance(player["history"], list):
            raise ValueError("Invalid saved minutes window")
        minute_players[pid] = player
    if not rating_players or rating_players.keys() != minute_players.keys():
        raise ValueError("Daily rating/minutes player coverage differs")
    summaries = {}
    for pid, rating in rating_players.items():
        minute = minute_players[pid]
        team_id = rating["team_id"]
        if team_id != minute["current_team_id"]:
            raise ValueError("Daily rating/minutes team identities differ")
        if team_id is not None and (_id(team_id) not in teams):
            raise ValueError("Daily state team lacks an official abbreviation")
        summaries[pid] = {
            "team_id": team_id,
            "team": teams.get(team_id),
            "rating": {
                "nail": rating["nail"],
                "starting_nail": rating["starting_nail"],
                "delta": rating["residual"],
                "on_court_possessions": rating["on_court_possessions"],
                "standard_error": rating["standard_error"],
                "initialization_source": rating["forecast_source"],
            },
            "minutes": {
                "conditional_estimate": minute["estimate"],
                "starting_estimate": minute["starting_minutes"],
                "retained_observation_count": len(minute["history"]),
            },
        }
    return {
        "season": season,
        "day": str(day),
        "effective_from": str(day + timedelta(days=1)),
        "biographies": biographies,
        "summaries": summaries,
        "names": {pid: row["player_name"] for pid, row in rating_players.items()},
        "sources": {"ratings": rating_ref, "minutes": minutes_ref, "biography": biography_ref},
        "coverage": {
            "rating_player_count": len(rating_players),
            "minutes_player_count": len(minute_players),
            "missing_biography_player_ids": sorted(rating_players.keys() - biographies.keys()),
        },
    }


def player_summary(inputs, player_id):
    biography = inputs["biographies"].get(player_id)
    summary = inputs["summaries"].get(player_id)
    if biography is None and summary is None:
        raise HTTPException(
            status_code=404, detail="Player not found.", headers={"Cache-Control": "no-store"}
        )
    return {
        "kind": "saved_player_summary",
        "schema_version": 1,
        "season": inputs["season"],
        "player_id": player_id,
        "player_name": biography["name"] if biography else inputs["names"][player_id],
        "team_id": summary["team_id"] if summary else None,
        "team": summary["team"] if summary else None,
        "position": biography["position"] if biography else None,
        "draft": biography["draft"] if biography else None,
        "biography_available": biography is not None,
        "observed_through": inputs["day"],
        "effective_from": inputs["effective_from"],
        "rating": summary["rating"] if summary else None,
        "minutes": summary["minutes"] if summary else None,
        "sources": inputs["sources"],
        "coverage": inputs["coverage"],
        "deferred_sections": ["career_ratings", "rotation_history", "attribution", "game_log"],
    }


@lru_cache(maxsize=1)
def _cached_inputs(season, cutoff, _hour):
    import boto3

    return read_player_inputs(boto3.client("s3"), season=season, cutoff=cutoff)


def shared_inputs():
    cutoff = datetime.now(PACIFIC).date() - timedelta(days=1)
    hour = int(time() // 3600)
    with _cache_lock:
        inputs = _cached_inputs(SEASON, cutoff, hour)
    return inputs, f"private, max-age={max(0, int((hour + 1) * 3600 - time()))}"


@router.get("/api/players/{player_id}")
def player_profile(response: Response, player_id: int = Path(gt=0)):
    try:
        inputs, cache_control = shared_inputs()
        result = player_summary(inputs, player_id)
        response.headers["Cache-Control"] = cache_control
        return result
    except SOURCE_ERRORS as error:
        logging.getLogger(__name__).exception("Saved player summary unavailable")
        raise HTTPException(
            status_code=503,
            detail="Saved player data is unavailable. Please try again later.",
            headers={"Cache-Control": "no-store"},
        ) from error
