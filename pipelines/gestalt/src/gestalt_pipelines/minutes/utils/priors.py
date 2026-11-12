"""Preserve accepted conditional values and historical new-player initialization."""

import math
from hashlib import sha256
from statistics import median

from .model import (
    MODEL_VERSION, SETTINGS, MinutesError, model_metadata, number, official_json,
    player_id, validate_model,
)


def build_priors(
    frozen: bytes, historical_rows: list[dict], *, season: str,
    history_reference: dict, draft_reference: dict,
) -> dict:
    payload = official_json(frozen)["minutes"]
    if payload["season"] != season or int(payload["completed_season"][:4]) >= int(season[:4]):
        raise MinutesError("Historical starting minutes belong to a different or unfinished season.")
    players = {}
    cold = []
    for row in payload["players"]:
        pid = str(player_id(row["player_id"]))
        if pid in players:
            raise MinutesError("Duplicate player in frozen minutes catalog.")
        minutes = number(row["conditional_minutes_per_game"], label="historical starting minutes", maximum=48)
        is_cold = bool(row.get("uses_draft_cold_start_prior")) or not bool(row.get("has_prior_minutes_state", True))
        if is_cold:
            cold.append(minutes)
        players[pid] = {
            "name": str(row["player_name"]), "starting_minutes": minutes,
            "branch": "frozen_cold_start" if is_cold else "frozen_historical_role",
            "uses_draft_cold_start_prior": bool(row.get("uses_draft_cold_start_prior")),
            "uses_incumbent_team_strength": bool(row.get("uses_incumbent_team_strength")),
        }
    if not players or not historical_rows:
        raise MinutesError("The frozen prior and verified historical role inputs must not be empty.")
    roles = {}
    ignored_forecasts = 0
    for row in sorted(historical_rows, key=lambda r: (int(r["season_start_year"]), str(r["season"]))):
        if row.get("is_preseason_forecast", False):
            ignored_forecasts += 1
            continue
        if int(row["season_start_year"]) >= int(season[:4]):
            raise MinutesError("Current/future-season evidence in historical role input.")
        pid = str(player_id(row["player_id"]))
        value = row["actual_minutes_per_available_game"]
        usable = value is not None and math.isfinite(float(value)) and 0 <= float(value) <= 48
        roles[pid] = {
            "name": str(row["player_name"]), "season": str(row["season"]),
            "starting_minutes": float(value) if usable else None,
        }
    if not roles:
        raise MinutesError("Historical input contains no completed observed seasons.")
    typical = float(median(cold)) if cold else 0.0
    if typical <= 0:
        typical = float(SETTINGS["cold_start_fallback_minutes"])
    return {
        "schema_version": 1, "artifact_kind": "conditional_minutes_priors",
        "season": season, "model": model_metadata(),
        "sources": {
            "frozen_catalog": {
                "sha256": sha256(frozen).hexdigest(), "player_count": len(players),
                "historical_source_through_season": payload["completed_season"],
                "membership_is_not_opening_roster": True,
            },
            "historical_roles": {**history_reference, "ignored_forecast_rows": ignored_forecasts},
            "draft_history": draft_reference,
        },
        "players": players, "historical_roles": roles,
        "cold_start_minutes": typical, "cold_start_cohort_size": len(cold),
    }


def validate_priors(record: dict, season: str, version: str = MODEL_VERSION) -> None:
    validate_model(record, season, version, kind="conditional_minutes_priors")
    if not isinstance(record.get("players"), dict) or not record["players"]:
        raise MinutesError("Starting-prior catalog is empty.")
    if not isinstance(record.get("historical_roles"), dict) or not record["historical_roles"]:
        raise MinutesError("Verified historical role catalog is missing; cannot substitute cold starts.")
    for key, row in record["players"].items():
        player_id(key)
        number(row["starting_minutes"], label="starting minutes", maximum=48)
    number(record["cold_start_minutes"], label="cold-start minutes", maximum=48)


def initialize_player(pid: int, priors: dict, *, day: str) -> dict:
    key = str(player_id(pid))
    row = priors["players"].get(key)
    if row is not None:
        minutes, branch, name = row["starting_minutes"], row["branch"], row["name"]
    else:
        history = priors["historical_roles"].get(key)
        if history is not None and history["starting_minutes"] is not None:
            minutes, branch, name = history["starting_minutes"], "last_historical_available_game_role", history["name"]
        else:
            minutes, branch, name = priors["cold_start_minutes"], "frozen_cohort_cold_start", None
    minutes = number(minutes, label="initial conditional minutes", maximum=48)
    return {
        "player_id": pid, "name": name, "introduced_on": day,
        "current_team_id": None, "starting_minutes": minutes,
        "initialization": {"branch": branch, "prior_player_id": key},
        "history": [], "estimate": minutes,
    }


__all__ = ["build_priors", "initialize_player", "validate_priors"]
