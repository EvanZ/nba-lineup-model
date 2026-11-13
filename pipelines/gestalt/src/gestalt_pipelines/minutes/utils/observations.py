"""Classify complete official player-games and measure the first 44 minutes."""

import math
import re
from collections import defaultdict

from .model import SETTINGS, MinutesError, number, player_id


def classify(minutes: float, report: dict | None) -> str:
    if minutes > 0:
        return "played"
    if report is not None:
        status = report["status"]
        if status not in SETTINGS["restricted_statuses"] + ["Available"]:
            raise MinutesError(f"Unsupported player injury status: {status}.")
        reason = re.sub(r"[^a-z]", "", report.get("reason", "").casefold())
        if "gleague" in reason or "twoway" in reason or "assignment" in reason:
            return "g_league_zero"
        if status in SETTINGS["restricted_statuses"]:
            return "medical_exclusion"
        return "available_zero"
    return "available_zero"


def _measure_minutes(rows: list[dict], game: dict) -> tuple[dict, dict, dict]:
    if not rows:
        raise MinutesError("Required stint timeline is empty.")
    exposure = defaultdict(float)
    actual = defaultdict(float)
    played_teams = {}
    end = 0.0
    cutoff = SETTINGS["role_cutoff_minutes"]*60
    for row in sorted(rows, key=lambda r: (r["start_elapsed_game_seconds"], r["stint_index"])):
        start = number(row["start_elapsed_game_seconds"], label="stint start")
        stop = number(row["end_elapsed_game_seconds"], label="stint end")
        if row["game_id"] != game["game_id"] or stop < start or not math.isclose(start, end, abs_tol=1e-6, rel_tol=0):
            raise MinutesError("Stint game identity, overlap, or full-timeline gap.")
        end = stop
        home, away = row["home_player_ids"], row["away_player_ids"]
        if (
            row["home_team_id"] != game["home_team_id"] or row["away_team_id"] != game["away_team_id"]
            or len(home) != 5 or len(away) != 5
            or len(set(home+away)) != 10
        ):
            raise MinutesError("Stint team/lineup identities are invalid.")
        seconds = max(0.0, min(stop, cutoff)-min(start, cutoff))
        for ids, side in ((home, "home"), (away, "away")):
            for raw_id in ids:
                pid = player_id(raw_id)
                exposure[pid] += seconds*48/cutoff
                actual[pid] += (stop-start)/60
                if stop > start:
                    played_teams[pid] = game[side+"_team_id"]
    if not math.isclose(sum(exposure.values()), 480, abs_tol=1e-6, rel_tol=0):
        raise MinutesError("First-44-minute player exposure is incomplete.")
    return dict(actual), dict(exposure), played_teams


def role_minutes(rows: list[dict], game: dict) -> dict[int, float]:
    return _measure_minutes(rows, game)[1]


def game_observations(
    game: dict, roster_teams: dict, injury_game: dict, stints: list[dict],
) -> list[dict]:
    gid = game["game_id"]
    if (
        not injury_game.get("report_used")
        or injury_game.get("scheduled_start_utc") != game["game_time_utc"].isoformat()
    ):
        raise MinutesError("No verified pre-tipoff injury selection for the scheduled game.")
    actual, roles, played_teams = _measure_minutes(stints, game)
    owners, reports = {}, {}
    for side in ("home", "away"):
        tid = str(game[side+"_team_id"])
        injury = injury_game["teams"][tid]
        if injury["team"] != game[side+"_team_tricode"] or injury.get("unresolved_players"):
            raise MinutesError("Game/team or reconciled injury identities differ.")
        if injury["submission_status"] not in {"submitted", "missing", "not_yet_submitted"}:
            raise MinutesError("Invalid injury team submission status.")
        for raw_id in roster_teams[tid]:
            owners[player_id(raw_id)] = int(tid)
        for raw_id, row in injury["players"].items():
            reports[player_id(raw_id)] = row
    result = []
    for pid, roster_team_id in sorted(owners.items()):
        minutes = actual.get(pid, 0.0)
        report = reports.get(pid)
        result.append({
            "player_id": pid, "team_id": played_teams.get(pid, roster_team_id), "game_id": gid,
            "game_time_utc": game["game_time_utc"].isoformat(),
            "actual_minutes": minutes, "role_minutes": roles.get(pid, 0.0),
            "classification": classify(minutes, report),
            "injury_status": report["status"] if report else None,
            "injury_reason": report.get("reason") if report else None,
        })
    return result


__all__ = ["classify", "game_observations", "role_minutes"]
