"""Advance one date from an exact predecessor without counting a game twice."""

from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from math import isclose

from .model import (
    MODEL_VERSION, SETTINGS, MinutesError, implementation_identity, model_metadata,
    number, player_id, validate_model,
)
from .observations import game_observations
from .priors import initialize_player, validate_priors

from ...utils import NBA_TEAM_IDS, PACIFIC



def estimate(start: float, history: list[dict]) -> float:
    prior = SETTINGS["prior_weight_games"]
    return (prior*start + sum(row["weight"]*row["minutes"] for row in history)) / (
        prior + sum(row["weight"] for row in history)
    )


def validate_state(record: dict, season: str, version: str = MODEL_VERSION) -> None:
    validate_model(record, season, version, kind="conditional_minutes_state")
    day = date.fromisoformat(record["game_date_pacific"])
    if record.get("validation", {}).get("complete") is not True:
        raise MinutesError("Previous minutes state did not pass full-slate validation.")
    players = record["players"]
    if not isinstance(players, dict) or not players:
        raise MinutesError("Minutes state has no player catalog.")
    included = record["included_game_ids"]
    if len(set(included)) != len(included):
        raise MinutesError("Minutes state repeats an included game.")
    for key, row in players.items():
        if player_id(key) != row["player_id"]:
            raise MinutesError("Player ID differs from its state key.")
        start = number(row["starting_minutes"], label="starting minutes", maximum=48)
        history = row["history"]
        if len(history) > SETTINGS["eligible_game_window"] or len({h["game_id"] for h in history}) != len(history):
            raise MinutesError("Invalid or duplicated three-game window.")
        order = [(h["game_date_pacific"], h["game_time_utc"], h["game_id"]) for h in history]
        if order != sorted(order):
            raise MinutesError("Minutes history is not chronological.")
        for h in history:
            if date.fromisoformat(h["game_date_pacific"]) > day or h["game_id"] not in included:
                raise MinutesError("Future or unprocessed evidence in minutes history.")
            number(h["minutes"], label="observed minutes", maximum=48)
            number(h["weight"], label="observation weight", maximum=1)
            if h["classification"] not in {"played", "available_zero", "g_league_zero"}:
                raise MinutesError("An excluded game entered the eligible window.")
            if h["classification"] != "played" and h["minutes"] != 0:
                raise MinutesError("A real-zero observation is nonzero.")
            if h["game_id"] not in record["game_inputs"]:
                raise MinutesError("A window is missing its exact game-source references.")
        value = number(row["estimate"], label="minutes estimate", maximum=48)
        if not isclose(value, estimate(start, history), abs_tol=1e-10, rel_tol=0):
            raise MinutesError("Saved estimate does not match its historical prior and window.")
        if row["current_team_id"] is not None and row["current_team_id"] not in NBA_TEAM_IDS:
            raise MinutesError("Invalid current team in minutes state.")


def update_state(
    *, day: date, first_date: date, season: str, priors: dict, parent: dict | None,
    rosters: dict, injuries: dict, stints: dict, boxes: dict, games: list[dict],
    inputs: dict, version: str = MODEL_VERSION,
) -> dict:
    validate_priors(priors, season, version)
    if day < first_date:
        raise MinutesError("Minutes date precedes the configured season start.")
    if day == first_date:
        if parent is not None:
            raise MinutesError("First date must initialize from historical priors.")
        players, included, game_inputs = {}, set(), {}
    else:
        if parent is None:
            raise MinutesError("The exact previous date's minutes state is required.")
        validate_state(parent, season, version)
        if parent["game_date_pacific"] != (day-timedelta(days=1)).isoformat():
            raise MinutesError("Minutes state must be from the immediately preceding date.")
        if parent["inputs"]["priors"] != inputs["priors"]:
            raise MinutesError("Historical starting inputs changed; replay must start from the first date.")
        players = deepcopy(parent["players"])
        included = set(parent["included_game_ids"])
        game_inputs = deepcopy(parent["game_inputs"])
    if (
        rosters.get("season") != season or rosters.get("game_date_pacific") != day.isoformat()
        or set(rosters["teams"]) != set(map(str, NBA_TEAM_IDS))
    ):
        raise MinutesError("Selected rosters must cover the exact season/date and all 30 NBA teams.")
    owners = {}
    for tid, ids in rosters["teams"].items():
        if not ids:
            raise MinutesError("A selected roster is empty.")
        for raw_id in ids:
            pid = player_id(raw_id)
            if pid in owners:
                raise MinutesError("A player appears on multiple current rosters.")
            owners[pid] = int(tid)
    for row in players.values():
        row["current_team_id"] = None

    def ensure(pid):
        key = str(pid)
        if key not in players:
            players[key] = initialize_player(pid, priors, day=day.isoformat())
        return players[key]

    for pid, tid in owners.items():
        ensure(pid)["current_team_id"] = tid
    expected = sorted(g["game_id"] for g in games)
    if (
        len(set(expected)) != len(expected)
        or stints.get("season") != season or stints.get("game_date_pacific") != day.isoformat()
        or stints.get("validation", {}).get("complete") is not True
        or stints["validation"]["expected_game_ids"] != expected
        or stints["validation"]["processed_game_ids"] != expected
        or set(boxes) != set(expected) or set(injuries["games"]) != set(expected)
        or injuries.get("season") != season or injuries.get("game_date_pacific") != day.isoformat()
    ):
        raise MinutesError("Selected stints, boxes, and injuries do not cover the full official slate.")
    if set(r["game_id"] for r in stints["rows"]) != set(expected):
        raise MinutesError("Stint rows omit a scheduled game or include another date.")
    observations = []
    for game in sorted(games, key=lambda g: (g["game_time_utc"], g["game_id"])):
        gid = game["game_id"]
        if gid in included:
            raise MinutesError(f"Game {gid} has already been applied to the parent state.")
        if game["game_time_utc"].astimezone(PACIFIC).date() != day:
            raise MinutesError("Official game belongs to a different Pacific date.")
        game_rows = [r for r in stints["rows"] if r["game_id"] == gid]
        rows = game_observations(game, rosters["teams"], injuries["games"][gid], boxes[gid], game_rows)
        game_inputs[gid] = {
            "stints": inputs["stints"], "rosters": inputs["rosters"],
            "injuries": inputs["injuries"], "box_score": inputs["boxes"][gid],
            "schedule": inputs["schedule"],
        }
        for row in rows:
            state = ensure(row["player_id"])
            if row["classification"] == "medical_exclusion":
                observations.append({**row, "game_date_pacific": day.isoformat(), "minutes": None, "weight": 0})
                continue
            value = 0.0
            if row["classification"] == "played":
                prior, delta = state["starting_minutes"], SETTINGS["played_clip_delta_minutes"]
                value = min(min(48, prior+delta), max(max(0, prior-delta), row["role_minutes"]))
            item = {**row, "game_date_pacific": day.isoformat(), "minutes": value, "weight": 1.0}
            state["history"] = (state["history"]+[item])[-SETTINGS["eligible_game_window"]:]
            observations.append(item)
        included.add(gid)
    for row in players.values():
        row["estimate"] = estimate(row["starting_minutes"], row["history"])
    retained = {h["game_id"] for p in players.values() for h in p["history"]} | set(expected)
    record = {
        "schema_version": 1, "artifact_kind": "conditional_minutes_state",
        "purpose": "postgame_model_state", "season": season, "game_date_pacific": day.isoformat(),
        "model": model_metadata(version), "implementation": implementation_identity(),
        "inputs": inputs, "players": dict(sorted(players.items(), key=lambda kv: int(kv[0]))),
        "included_game_ids": sorted(included),
        "game_inputs": {gid: ref for gid, ref in sorted(game_inputs.items()) if gid in retained},
        "observations": observations,
        "validation": {
            "complete": True, "expected_game_ids": expected, "processed_game_ids": expected,
            "rostered_player_count": len(owners), "retained_player_count": len(players),
            "team_count": 30, "classifications": dict(Counter(r["classification"] for r in observations)),
        },
    }
    validate_state(record, season, version)
    return record


__all__ = ["estimate", "update_state", "validate_state"]
