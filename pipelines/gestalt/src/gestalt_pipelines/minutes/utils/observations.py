"""Classify complete official player-games and measure the first 44 minutes."""

import math
import re
from collections import defaultdict

from .model import SETTINGS, MinutesError, number, player_id


def actual_minutes(player: dict) -> float:
    value = player.get("statistics", {}).get("minutes")
    match = re.fullmatch(r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?", str(value))
    if match is None or not any(match.groups()):
        raise MinutesError("Missing or malformed final box-score minutes.")
    return sum(float(x or 0)*factor for x, factor in zip(match.groups(), (60, 1, 1/60)))


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


def role_minutes(rows: list[dict], game: dict, box: dict) -> dict[int, float]:
    if not rows:
        raise MinutesError("Required stint timeline is empty.")
    exposure = defaultdict(float)
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
        for pid in home+away:
            exposure[player_id(pid)] += seconds*48/cutoff
    expected = 2880 + max(0, int(box["game"]["period"])-4)*300
    if not math.isclose(end, expected, abs_tol=1e-6, rel_tol=0):
        raise MinutesError("Stints do not cover the completed game including overtime.")
    if not math.isclose(sum(exposure.values()), 480, abs_tol=1e-6, rel_tol=0):
        raise MinutesError("First-44-minute player exposure is incomplete.")
    return dict(exposure)


def game_observations(
    game: dict, roster_teams: dict, injury_game: dict, box: dict, stints: list[dict],
) -> list[dict]:
    gid = game["game_id"]
    body = box["game"]
    if body["gameId"] != gid or body["gameStatus"] != 3:
        raise MinutesError("A matching final box score is required.")
    if (
        not injury_game.get("report_used")
        or injury_game.get("scheduled_start_utc") != game["game_time_utc"].isoformat()
    ):
        raise MinutesError("No verified pre-tipoff injury selection for the scheduled game.")
    roles = role_minutes(stints, game, box)
    owners, explicit_owners, players, reports = {}, {}, {}, {}
    for side in ("home", "away"):
        tid = str(game[side+"_team_id"])
        team = body[side+"Team"]
        injury = injury_game["teams"][tid]
        if (
            int(team["teamId"]) != int(tid) or team["teamTricode"] != game[side+"_team_tricode"]
            or injury["team"] != game[side+"_team_tricode"] or injury.get("unresolved_players")
        ):
            raise MinutesError("Game/team or reconciled injury identities differ.")
        if injury["submission_status"] not in {"submitted", "missing", "not_yet_submitted"}:
            raise MinutesError("Invalid injury team submission status.")
        for pid in roster_teams[tid]:
            owners[player_id(pid)] = int(tid)
        seen = set()
        for p in team["players"]:
            pid = player_id(p["personId"])
            if pid in seen:
                raise MinutesError("Duplicate final box-score player.")
            seen.add(pid)
            players[pid] = p
        for raw_id, r in injury["players"].items():
            pid = player_id(raw_id)
            reports[pid] = r
        # The audit's union preserves explicit game membership even after a same-day
        # waiver/trade changes the end-of-day roster. Current membership is separate.
        for pid in seen | {player_id(p) for p in injury["players"]}:
            if pid in explicit_owners and explicit_owners[pid] != int(tid):
                raise MinutesError("Official game sources assign a player to both teams.")
            explicit_owners[pid] = int(tid)
    owners.update(explicit_owners)
    if set(roles)-set(players):
        raise MinutesError("A stint player is missing from the referenced final box score.")
    result = []
    for pid, team_id in sorted(owners.items()):
        p = players.get(pid)
        minutes = actual_minutes(p) if p is not None else 0.0
        report = reports.get(pid)
        result.append({
            "player_id": pid, "team_id": team_id, "game_id": gid,
            "game_time_utc": game["game_time_utc"].isoformat(),
            "actual_minutes": minutes, "role_minutes": roles.get(pid, 0.0),
            "classification": classify(minutes, report),
            "injury_status": report["status"] if report else None,
            "injury_reason": report.get("reason") if report else None,
            "in_box_score": p is not None,
        })
    return result


__all__ = ["actual_minutes", "classify", "game_observations", "role_minutes"]
