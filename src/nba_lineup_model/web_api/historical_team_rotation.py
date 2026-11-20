"""Query canonical historical stints on S3; cache compact team results only."""

from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np
from fastapi import HTTPException

from .player_season_history import SOURCES, _lock, _reader

ROOT = "s3://nba-gestalt-data/lake/analytical/rapm_stints/"
NAMES_URI = "s3://nba-gestalt-data/lake/analytical/player_season_panel/player_seasons.parquet"

QUERY = """
WITH selected AS MATERIALIZED (
    SELECT game_id, home_team_id AS team_id, home_player_ids AS player_ids,
           possessions, duration_seconds, game_time_utc
    FROM read_parquet(?) WHERE season = ? AND season_type = 'regular' AND home_team_id = ?
    UNION ALL
    SELECT game_id, away_team_id, away_player_ids, possessions, duration_seconds, game_time_utc
    FROM read_parquet(?) WHERE season = ? AND season_type = 'regular' AND away_team_id = ?
), names AS (
    SELECT player_id, min(player_name) AS player_name, count(DISTINCT player_name) AS name_count
    FROM read_parquet(?) WHERE season = ? AND player_name IS NOT NULL AND player_name <> ''
      AND player_id IN (SELECT unnest(player_ids) FROM selected)
    GROUP BY player_id
)
SELECT
    (SELECT list(struct_pack(game_id := game_id, player_ids := player_ids,
                              possessions := possessions, duration_seconds := duration_seconds))
     FROM selected),
    (SELECT list(struct_pack(player_id := player_id, player_name := player_name,
                              name_count := name_count)) FROM names),
    (SELECT min(team_id) FROM selected),
    (SELECT count(DISTINCT team_id) FROM selected),
    (SELECT max(CAST(timezone('America/Los_Angeles', game_time_utc) AS DATE)) FROM selected)
"""

STATS_QUERY = """
SELECT count(*), count(DISTINCT game_id),
       max(CAST(timezone('America/Los_Angeles', game_time_utc) AS DATE)),
       count(*) FILTER (WHERE season <> ? OR season_type <> 'regular'
         OR NOT regexp_full_match(game_id, '002[0-9]{7}')
         OR NOT isfinite(possessions) OR possessions <= 0
         OR NOT isfinite(duration_seconds) OR duration_seconds < 0 OR game_time_utc IS NULL),
       count(*) FILTER (WHERE contains(
           lower(source_build_run_id || quality_issue_codes_json), 'espn')),
       list(DISTINCT struct_pack(team := home_team_tricode, team_id := home_team_id)),
       list(DISTINCT struct_pack(team := away_team_tricode, team_id := away_team_id))
FROM read_parquet(?)
"""


def _head(client, uri, expected_sha, expected_size=None):
    parsed = urlparse(uri)
    response = client.head_object(
        Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), ChecksumMode="ENABLED"
    )
    if (
        response.get("ChecksumType") != "FULL_OBJECT"
        or base64.b64decode(response.get("ChecksumSHA256", "")).hex() != expected_sha
        or (expected_size is not None and response["ContentLength"] != expected_size)
    ):
        raise ValueError("Canonical historical object differs from its accepted checksum")
    return {
        "uri": uri,
        "sha256": expected_sha,
        "version_id": response["VersionId"],
        "etag": response["ETag"],
        "size_bytes": response["ContentLength"],
    }


@lru_cache(maxsize=64)
def _sources(season, period):
    if season not in {f"{year}-{(year + 1) % 100:02}" for year in range(1996, 2026)}:
        raise HTTPException(
            404,
            "No saved historical rotation dataset for this season.",
            headers={"Cache-Control": "no-store"},
        )
    client = _reader().client
    manifest_uri = ROOT + season + "/regular/_manifest.json"
    parsed = urlparse(manifest_uri)
    response = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    with response["Body"] as body:
        raw = body.read()
    manifest = json.loads(raw)
    if (
        manifest["season"] != season
        or manifest["season_type"] != "regular"
        or manifest["part_filename"] != "part-00000.parquet"
    ):
        raise ValueError("Wrong historical stint dataset identity")
    if manifest.get("excluded_nonconserving_game_ids"):
        raise ValueError("Historical stint dataset excludes games requiring review")
    stints = _head(
        client,
        ROOT + season + "/regular/part-00000.parquet",
        manifest["part_sha256"],
        manifest["byte_count"],
    )
    names = _head(client, NAMES_URI, SOURCES["panel"]["sha256"], SOURCES["panel"]["size_bytes"])
    meta = {
        "uri": manifest_uri,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "version_id": response["VersionId"],
        "etag": response["ETag"],
        "size_bytes": len(raw),
    }
    return json.dumps(
        {
            "season": season,
            "stints": stints,
            "names": names,
            "manifest": meta,
            "row_count": manifest["row_count"],
            "source_game_count": manifest["source_game_count"],
        },
        sort_keys=True,
    )


@lru_cache(maxsize=64)
def _dataset_stats(source_key):
    sources = json.loads(source_key)
    reader = _reader()
    # Called while the existing DuckDB connection lock is held.
    season = sources["season"]
    values = reader.connection.execute(STATS_QUERY, [season, sources["stints"]["uri"]]).fetchone()
    if (
        values[0] != sources["row_count"]
        or values[1] != sources["source_game_count"]
        or values[2] is None
        or values[3]
        or values[4]
    ):
        raise ValueError("Historical source coverage, identity or provenance failed validation")
    identities = {}
    for row in values[5] + values[6]:
        code, tid = row["team"], row["team_id"]
        if (
            not code
            or not isinstance(tid, int)
            or tid <= 0
            or (code in identities and identities[code] != tid)
        ):
            raise ValueError("Historical team code has ambiguous NBA identity")
        identities[code] = tid
    return {"game_count": int(values[1]), "through": str(values[2]), "team_ids": identities}


@lru_cache(maxsize=128)
def _query_team(team, season, source_key):
    sources = json.loads(source_key)
    descriptors = {key: sources[key] for key in ("stints", "names", "manifest")}
    with _lock:
        reader = _reader()
        reader.check_sources(descriptors)
        reader.authenticate()
        stats = _dataset_stats(source_key)
        if team not in stats["team_ids"]:
            raise HTTPException(
                404,
                "No saved historical rotation stints for this team-season.",
                headers={"Cache-Control": "no-store"},
            )
        selected_id = stats["team_ids"][team]
        result = reader.connection.execute(
            QUERY,
            [
                sources["stints"]["uri"],
                season,
                selected_id,
                sources["stints"]["uri"],
                season,
                selected_id,
                sources["names"]["uri"],
                season,
            ],
        ).fetchone()
        reader.check_sources(descriptors)
    rows, name_rows, team_id, identity_count, last_day = result
    if not rows:
        raise HTTPException(
            404,
            "No saved historical rotation stints for this team-season.",
            headers={"Cache-Control": "no-store"},
        )
    if identity_count != 1 or not team_id or last_day is None:
        raise ValueError("Historical team identity is ambiguous")
    names = {}
    for row in name_rows or []:
        if row["name_count"] != 1:
            raise ValueError("Historical official player name is ambiguous")
        names[row["player_id"]] = row["player_name"]
    ids = sorted({pid for row in rows for pid in row["player_ids"]})
    if any(pid not in names for pid in ids):
        raise ValueError("Historical rotation players lack official names")
    positions = {pid: i for i, pid in enumerate(ids)}
    active = np.zeros((len(rows), len(ids)))
    weights = np.array([r["possessions"] for r in rows], dtype=float)
    seconds = np.array([r["duration_seconds"] for r in rows], dtype=float)
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.isfinite(seconds).all()
        or np.any(seconds < 0)
    ):
        raise ValueError("Historical rotation exposure is invalid")
    games = {}
    for index, row in enumerate(rows):
        lineup = row["player_ids"]
        if (
            len(lineup) != 5
            or len(set(lineup)) != 5
            or any(type(pid) is not int or pid <= 0 for pid in lineup)
        ):
            raise ValueError("Historical rotation lineup is invalid")
        indices = [positions[pid] for pid in lineup]
        active[index, indices] = 1
        minutes = games.setdefault(row["game_id"], np.zeros(len(ids)))
        minutes[indices] += seconds[index] / 60
    exposure = active.T @ weights
    minutes = active.T @ (seconds / 60)
    game_count = np.sum(np.array(list(games.values())) > 0, axis=0)
    shared = active.T @ (active * weights[:, None])
    similarity = shared / (exposure[:, None] + exposure[None, :] - shared)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(active * weights[:, None], rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0, posinf=0, neginf=0)
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    np.fill_diagonal(similarity, 1)
    order = sorted(range(len(ids)), key=lambda i: (-exposure[i], names[ids[i]], ids[i]))
    doc = {
        "artifact_kind": "historical_rotation_query",
        "team": team,
        "team_id": int(team_id),
        "season": season,
        "game_date_pacific": stats["through"],
        "last_team_game_date_pacific": str(last_day),
        "player_order": [ids[i] for i in order],
        "players": [
            {
                "player_id": ids[i],
                "player_name": names[ids[i]],
                "on_court_possessions": float(exposure[i]),
                "on_court_minutes": float(minutes[i]),
                "games": int(game_count[i]),
            }
            for i in order
        ],
        "shared_possessions": shared[np.ix_(order, order)].tolist(),
        "shared_court_similarity": similarity[np.ix_(order, order)].tolist(),
        "floor_correlation": correlation[np.ix_(order, order)].tolist(),
        "calculation": {
            "population": "team_regular_season_stints_only",
            "possessions": "positive_only",
            "floor_signal": "stint_possessions_when_on_court_else_zero",
            "correlation": "ordinary_pearson",
            "similarity": "shared_over_union",
        },
        "coverage": {
            "complete": True,
            "scope": "saved_historical_regular_season_dataset",
            "game_ids": sorted(games),
            "game_count": len(games),
            "stint_count": len(rows),
            "dataset_game_count": stats["game_count"],
        },
    }
    source = {
        "kind": "historical_parquet_query",
        **sources["stints"],
        "names": sources["names"],
        "manifest": sources["manifest"],
    }
    return doc, source


def historical_summary(team, season, period):
    source_key = _sources(season, period)
    return _query_team(team, season, source_key)
