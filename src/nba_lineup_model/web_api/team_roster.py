"""Read-only team rosters from accepted daily states and filtered S3 Parquets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from time import time

from . import player_summary as profiles
from .player_season_history import SOURCES, _lock, _reader, records
from .schedule import BUCKET, PACIFIC

SEASON = "2026-27"
HISTORICAL = {key: SOURCES[key] for key in ("teams", "ratings", "panel", "splits")}
SOURCE_ID = hashlib.sha256(json.dumps(HISTORICAL, sort_keys=True).encode()).hexdigest()
_lock_roster = threading.Lock()
TRANSACTIONS_URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"

QUERY = """
WITH assignments AS (
    SELECT * FROM read_parquet(?) WHERE season = ?
), members AS (
    SELECT player_id, team_id FROM assignments WHERE team = ? AND is_latest_team
), trades AS (
    SELECT player_id, list(struct_pack(team_id := team_id, team := team,
        possessions := possessions, games := games, is_latest_team := is_latest_team,
        is_primary_team := is_primary_team) ORDER BY last_game_time_utc, team_id) AS team_splits
    FROM assignments WHERE player_id IN (SELECT player_id FROM members) GROUP BY player_id
), ratings AS (SELECT * FROM read_parquet(?) WHERE season = ?),
panel AS (SELECT * FROM read_parquet(?) WHERE season = ?),
splits AS (SELECT * FROM read_parquet(?) WHERE season = ?)
SELECT m.player_id, m.team_id,
       coalesce(nullif(r.player_name, ''), nullif(p.player_name, '')) AS player_name,
       p.listed_position AS position, r.rapm AS nail, r.rank_all_players AS rank,
       s.offense_rating, s.defense_rating, s.total_nail, p.rapm_possessions AS possessions,
       p.games, t.team_splits
FROM members m
LEFT JOIN ratings r USING (player_id)
LEFT JOIN panel p USING (player_id)
LEFT JOIN splits s USING (player_id)
LEFT JOIN trades t USING (player_id)
"""


def number(value, *, nonnegative=False):
    if type(value) not in (float, int) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("Invalid saved roster statistic")
    return value


def read_reference(client, ref, root, *, suffix=".json"):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid roster source checksum")
    key = ref["uri"].removeprefix(f"s3://{BUCKET}/")
    if ref["uri"] != f"s3://{BUCKET}/{key}" or not key.startswith(root) or not key.endswith(suffix):
        raise ValueError("Roster reference escapes its source scope")
    raw = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("Pinned roster input failed checksum verification")
    return json.loads(raw)


def transaction_name(row):
    # Same documented name-span rule as the shared NBA name resource.
    def key(value):
        text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
        return re.sub(r"[^a-z0-9]", "", text)

    target = key(row.get("PLAYER_SLUG") or "")
    if not target:
        return None
    text = row.get("TRANSACTION_DESCRIPTION") or ""
    words = list(re.finditer(r"\S+", text))
    for start in range(len(words)):
        for end in range(start, len(words)):
            candidate = text[words[start].start() : words[end].end()].strip(" ,;:")
            normalized = key(candidate)
            if normalized == target:
                return candidate.rstrip(".")
            if len(normalized) > len(target):
                break
    return None


def transaction_names(client, roster, wanted):
    names, sources = {}, {}
    if not wanted:
        return names, sources
    for day, ref in sorted(roster["inputs"]["transactions"].items(), reverse=True):
        if str(date.fromisoformat(day)) != day or day > roster["game_date_pacific"]:
            raise ValueError("Invalid pinned transaction date")
        root = f"lake/raw/nba/transactions/season={SEASON}/game_date_pacific={day}/"
        if ref["uri"] != f"s3://{BUCKET}/{root}latest.json":
            raise ValueError("Invalid transaction reference identity")
        doc = read_reference(client, ref, root)
        if (
            doc["season"] != SEASON
            or doc["game_date_pacific"] != day
            or doc["source_url"] != TRANSACTIONS_URL
            or not isinstance(doc["transactions"], list)
        ):
            raise ValueError("Invalid official transaction name source")
        used = False
        for row in doc["transactions"]:
            value = row["PLAYER_ID"]
            if isinstance(value, bool) or int(value) <= 0 or int(value) != float(value):
                raise ValueError("Invalid transaction player identity")
            pid = int(value)
            if (
                str(date.fromisoformat(row["TRANSACTION_DATE"][:10])) != day
                or datetime.fromisoformat(row["first_observed_at"]).tzinfo is None
            ):
                raise ValueError("Invalid transaction date or observation identity")
            if pid not in wanted or pid in names:
                continue
            used = True
            name = transaction_name(row)
            if name:
                names[pid] = name
        if used:
            sources[day] = ref
        if wanted <= names.keys():
            break
    return names, sources


@lru_cache(maxsize=1)
def current_inputs(hour, roster_uri, roster_sha, roster_day):
    client = _reader().client
    root = f"lake/artifacts/rosters/season={SEASON}/game_date_pacific={roster_day}/"
    if roster_uri != f"s3://{BUCKET}/{root}{roster_sha}.json":
        raise ValueError("Invalid roster partition/selection identity")
    ref = {"uri": roster_uri, "sha256": roster_sha}
    roster = read_reference(client, ref, root)
    if (
        type(roster["schema_version"]) is not int
        or roster["schema_version"] != 1
        or roster["season"] != SEASON
        or roster["game_date_pacific"] != roster_day
    ):
        raise ValueError("Wrong roster identity/schema")
    teams = roster["teams"]
    if not isinstance(teams, dict) or len(teams) != 30:
        raise ValueError("Current roster must cover all 30 teams")
    members = set()
    for tid, ids in teams.items():
        if not tid.isdigit() or int(tid) <= 0 or str(int(tid)) != tid or not isinstance(ids, list):
            raise ValueError("Invalid roster team membership")
        for pid in ids:
            profiles._id(pid)
        if len(ids) != len(set(ids)) or members.intersection(ids):
            raise ValueError("Duplicate roster membership")
        members.update(ids)
    cutoff = datetime.now(PACIFIC).date() - timedelta(days=1)
    day = profiles._latest_day(
        client, f"lake/artifacts/player_ratings/season={SEASON}/model_version=v1/", cutoff
    )
    state, rating_ref = profiles._state(client, "player_ratings", SEASON, day)
    ratings = profiles._rating_players(state)
    biographies, _, biography_ref = profiles._biographies(client, SEASON)
    names = {pid: bio["name"] for pid, bio in biographies.items()}
    names.update({pid: row["player_name"] for pid, row in ratings.items() if pid not in names})
    ranked = sorted(ratings, key=lambda pid: (-ratings[pid]["nail"], names[pid], pid))
    return {
        "teams": teams,
        "ratings": ratings,
        "biographies": biographies,
        "names": names,
        "name_inputs": roster,
        "ranks": {pid: index for index, pid in enumerate(ranked, 1)},
        "rating_state": {
            "kind": "daily",
            "evidence_through_date": str(day),
            "effective_from_date": state["effective_from_game_date_pacific"],
            "rank_population_count": len(ratings),
        },
        "sources": {
            "roster": ref,
            "ratings": rating_ref,
            "biography": biography_ref,
            "transaction_names": {},
        },
    }


def row_coverage(players):
    return {
        "player_count": len(players),
        **{
            f"missing_{field}_player_ids": [p["player_id"] for p in players if p[field] is None]
            for field in ("nail", "position", "rank")
        },
        "missing_name_player_ids": [p["player_id"] for p in players if p["name_missing"]],
    }


def sort_players(players):
    return sorted(
        players,
        key=lambda p: (p["nail"] is None, -(p["nail"] or 0), p["player_name"], p["player_id"]),
    )


@lru_cache(maxsize=128)
def supplemental_names(hour, roster_uri, roster_sha, roster_day, player_ids):
    inputs = current_inputs(hour, roster_uri, roster_sha, roster_day)
    return transaction_names(_reader().client, inputs["name_inputs"], set(player_ids))


def current_roster(entry, day, sources):
    ref = sources["roster"]
    inputs = current_inputs(int(time() // 3600), ref["uri"], ref["sha256"], day)
    ids = inputs["teams"].get(str(entry["team_id"]))
    if ids is None:
        raise ValueError("Selected team is absent from authoritative roster")
    wanted = set(ids) - inputs["names"].keys()
    supplemental, transaction_refs, name_errors = {}, {}, []
    if wanted:
        try:
            supplemental, transaction_refs = supplemental_names(
                int(time() // 3600), ref["uri"], ref["sha256"], day, tuple(sorted(wanted))
            )
        except Exception as error:
            logging.getLogger(__name__).error(
                "Supplemental roster names unavailable: %s", type(error).__name__
            )
            name_errors = [{"player_ids": sorted(wanted), "reason": "source_validation_failed"}]
    players = []
    for pid in ids:
        rating = inputs["ratings"].get(pid)
        bio = inputs["biographies"].get(pid, {})
        players.append(
            {
                "player_id": pid,
                "player_name": inputs["names"].get(pid)
                or supplemental.get(pid)
                or f"NBA player {pid}",
                "name_missing": pid not in inputs["names"] and pid not in supplemental,
                "position": bio.get("position"),
                "nail": rating["nail"] if rating else None,
                "rank": inputs["ranks"].get(pid),
                "offense_rating": None,
                "defense_rating": None,
                "possessions": rating["on_court_possessions"] if rating else None,
                "games": None,
                "team_splits": [],
            }
        )
    return {
        "status": "available",
        "players": sort_players(players),
        "coverage": row_coverage(players),
        "roster_as_of": day,
        "rating_state": inputs["rating_state"],
        "sources": {**inputs["sources"], "transaction_names": transaction_refs},
        "name_source_errors": name_errors,
        "identity_warning": (
            "Some player names could not be verified. Affected players are shown by NBA ID."
            if name_errors
            else None
        ),
        "possessions_label": "Rating poss.",
        "games_available": False,
        "splits_available": False,
        "membership": "current_roster",
    }


@lru_cache(maxsize=128)
def historical_roster(team, season, source_id):
    if source_id != SOURCE_ID:
        raise ValueError("Historical roster source identity changed")
    uris = {key: ref["uri"] for key, ref in HISTORICAL.items()}
    params = [
        uris["teams"],
        season,
        team,
        uris["ratings"],
        season,
        uris["panel"],
        season,
        uris["splits"],
        season,
    ]
    with _lock:
        reader = _reader()
        reader.check_sources(HISTORICAL)
        reader.authenticate()
        players = records(reader.connection.execute(QUERY, params))
        reader.check_sources(HISTORICAL)
    if not players or len({p["player_id"] for p in players}) != len(players):
        raise ValueError("Empty or ambiguous saved historical roster")
    for p in players:
        profiles._id(p["player_id"])
        profiles._id(p["team_id"])
        for field in (
            "nail",
            "offense_rating",
            "defense_rating",
            "total_nail",
            "possessions",
            "games",
            "rank",
        ):
            if p[field] is not None:
                number(p[field], nonnegative=field in ("possessions", "games", "rank"))
        if p["rank"] is not None and (p["rank"] < 1 or p["rank"] != int(p["rank"])):
            raise ValueError("Invalid saved league rank")
        total, offense, defense = p.pop("total_nail"), p["offense_rating"], p["defense_rating"]
        if any(v is not None for v in (total, offense, defense)):
            if any(v is None for v in (total, offense, defense, p["nail"])) or not (
                math.isclose(total, p["nail"], rel_tol=0, abs_tol=1e-8)
                and math.isclose(offense + defense, p["nail"], rel_tol=0, abs_tol=1e-8)
            ):
                raise ValueError("Historical offense/defense allocation disagrees with NAIL")
        p["name_missing"] = not bool(p["player_name"])
        if p["name_missing"]:
            p["player_name"] = f"NBA player {p['player_id']}"
        if p["position"] is not None and not isinstance(p["position"], str):
            raise ValueError("Invalid saved player position")
        latest = [t for t in p["team_splits"] if t["is_latest_team"]]
        if len(latest) != 1 or latest[0]["team"] != team or latest[0]["team_id"] != p["team_id"]:
            raise ValueError("Historical season-end team identity mismatch")
    return {
        "status": "available",
        "players": sort_players(players),
        "coverage": row_coverage(players),
        "roster_as_of": None,
        "rating_state": {"kind": "completed_fit", "season": season},
        "sources": HISTORICAL,
        "possessions_label": "Poss",
        "games_available": True,
        "splits_available": True,
        "membership": "season_end_assignment",
    }


def roster(entry, season, day, sources):
    try:
        with _lock_roster:
            return (
                current_roster(entry, day, sources)
                if season == SEASON
                else historical_roster(entry["team"], season, SOURCE_ID)
            )
    except Exception as error:
        logging.getLogger(__name__).error("Saved team roster unavailable: %s", type(error).__name__)
        return {
            "status": "unavailable",
            "players": [],
            "message": "Saved roster data is unavailable or failed validation. Please try again.",
        }
