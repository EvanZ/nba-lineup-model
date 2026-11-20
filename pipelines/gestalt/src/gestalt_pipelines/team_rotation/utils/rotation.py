"""Pure, team-only rotation math; no models or web dependencies."""
import json
import math
from collections import defaultdict

import numpy as np

PROCESSING_VERSION = "v1"


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def official_json(raw):
    value = json.loads(raw)
    if "espn" in json.dumps(value).casefold():
        raise ValueError("Prohibited basketball source in team rotation inputs.")
    return value


def validate_rows(rows, games):
    """Reject damaged lineups and incomplete game coverage before publication."""
    by_id = {g["game_id"]: g for g in games}
    seen, positive, indices = set(), set(), set()
    for row in rows:
        gid = row["game_id"]
        if gid not in by_id:
            raise ValueError(f"Unexpected stint game {gid}.")
        game = by_id[gid]
        identity = (gid, row["stint_index"])
        if identity in indices:
            raise ValueError(f"Duplicate stint {identity}.")
        indices.add(identity)
        ids = []
        for side in ("home", "away"):
            if row[side+"_team_id"] != game[side+"_team_id"]:
                raise ValueError(f"Stint team differs from schedule for {gid}.")
            lineup = row[side+"_player_ids"]
            if len(lineup) != 5 or any(type(p) is not int or p <= 0 for p in lineup):
                raise ValueError(f"Invalid five-player lineup for {gid}.")
            ids.extend(lineup)
        if len(set(ids)) != 10:
            raise ValueError(f"Duplicate player in game {gid} lineup.")
        for field in ("possessions", "duration_seconds"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {field} for {gid}.")
        seen.add(gid)
        if row["possessions"] > 0:
            positive.add(gid)
    if seen != set(by_id) or positive != set(by_id):
        raise ValueError("Every scheduled game must have positive-possession stints.")


def summarize_team(*, season, day, team_id, team, games, rows, names, inputs):
    selected_games = [g for g in games if team_id in (g["home_team_id"], g["away_team_id"])]
    game_ids = {g["game_id"] for g in selected_games}
    selected = [r for r in rows if r["game_id"] in game_ids and r["possessions"] > 0]
    exposure = defaultdict(float)
    minutes = defaultdict(float)
    game_minutes = defaultdict(lambda: defaultdict(float))
    lineups = []
    for row in selected:
        side = "home" if row["home_team_id"] == team_id else "away"
        lineup = row[side+"_player_ids"]
        lineups.append(lineup)
        for pid in lineup:
            if not names.get(pid):
                raise ValueError(f"No official pinned-box name for player {pid}.")
            exposure[pid] += row["possessions"]
            minutes[pid] += row["duration_seconds"]/60
            game_minutes[pid][row["game_id"]] += row["duration_seconds"]/60
    order = sorted(exposure, key=lambda pid: (-exposure[pid], names[pid], pid))
    positions = {pid: i for i, pid in enumerate(order)}
    active = np.zeros((len(selected), len(order)))
    for r, lineup in enumerate(lineups):
        for pid in lineup:
            active[r, positions[pid]] = 1
    weights = np.array([r["possessions"] for r in selected])
    shared = active.T @ (active*weights[:, None])
    p = np.array([exposure[pid] for pid in order])
    union = p[:, None]+p[None, :]-shared
    similarity = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
    correlation = np.eye(len(order))
    if len(selected) >= 2 and order:
        signal = active*weights[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(signal, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0, posinf=0, neginf=0)
        correlation = (correlation+correlation.T)/2
    np.fill_diagonal(correlation, 1)
    np.fill_diagonal(similarity, 1)
    return {
        "schema_version": 1, "artifact_kind": "team_rotation", "processing_version": PROCESSING_VERSION,
        "season": season, "team_id": team_id, "team": team, "game_date_pacific": day,
        "last_team_game_date_pacific": max((g["game_date_pacific"] for g in selected_games), default=None),
        "calculation": {"population": "team_regular_season_stints_only", "possessions": "positive_only",
                        "floor_signal": "stint_possessions_when_on_court_else_zero", "correlation": "ordinary_pearson",
                        "undefined_correlation": 0, "diagonal": 1, "similarity": "shared_over_union",
                        "games": "positive_stint_duration_games", "duration": "full_stint_including_overtime"},
        "player_order": order,
        "players": [{"player_id": pid, "player_name": names[pid], "on_court_possessions": exposure[pid],
                     "on_court_minutes": minutes[pid], "games": sum(m > 0 for m in game_minutes[pid].values())}
                    for pid in order],
        "shared_possessions": shared.tolist(), "shared_court_similarity": similarity.tolist(),
        "floor_correlation": correlation.tolist(),
        "coverage": {"complete": True, "game_ids": sorted(game_ids), "game_count": len(game_ids), "stint_count": len(selected)},
        "inputs": inputs,
    }
