"""The Schedule tab's injury-weighted, active-15 proportional allocation."""

from copy import deepcopy
from datetime import date, datetime
from hashlib import sha256
from math import isclose

from ...utils import NBA_TEAM_IDS, PACIFIC
from .model import MinutesError, implementation_identity, json_bytes, number, player_id
from .priors import initialize_player

PROJECTION_VERSION = "v1"
PROJECTION_SETTINGS = {
    "allocation": "injury_weighted_top15_proportional",
    "rotation_size": 15,
    "team_minutes": 240,
    "individual_cap": None,
    "pending_report_multiplier": 1,
    "status_multipliers": {
        "Out": 0, "Doubtful": .25, "Questionable": .5, "Probable": .75, "Available": 1,
    },
    "tie_break": "nba_player_id",
}


def projection_model(version=PROJECTION_VERSION):
    return {
        "name": "injury_adjusted_game_minutes", "model_version": version,
        "settings": deepcopy(PROJECTION_SETTINGS),
        "configuration_sha256": sha256(json_bytes(PROJECTION_SETTINGS)).hexdigest(),
    }


def projection_fingerprint(game: dict, inputs: dict, version=PROJECTION_VERSION, *, implementation=None) -> str:
    return sha256(json_bytes({
        "game_id": game["game_id"], "start": game["game_time_utc"].isoformat(),
        "home": game["home_team_id"], "away": game["away_team_id"],
        "inputs": inputs, "model": projection_model(version),
        "implementation": implementation if implementation is not None else implementation_identity(),
    })).hexdigest()


def build_game_projection(
    *, game: dict, day: str, season: str, baseline: dict, rosters: dict,
    injuries: dict, priors: dict, inputs: dict, version=PROJECTION_VERSION,
) -> dict:
    gid = game["game_id"]
    if (
        game["game_time_utc"].astimezone(PACIFIC).date().isoformat() != day
        or (baseline is not None and baseline["season"] != season) or rosters["season"] != season
        or injuries["season"] != season or injuries["game_date_pacific"] != day
        or (baseline is not None and date.fromisoformat(baseline["game_date_pacific"]) >= date.fromisoformat(day))
        or rosters["game_date_pacific"] > day
        or set(rosters["teams"]) != set(map(str, NBA_TEAM_IDS))
    ):
        raise MinutesError("Game projection inputs have mismatched season/date or roster coverage.")
    owners = [player_id(p) for ids in rosters["teams"].values() for p in ids]
    if len(owners) != len(set(owners)) or any(not ids for ids in rosters["teams"].values()):
        raise MinutesError("The accepted roster contains duplicate players or an empty team.")
    injury = injuries["games"].get(gid)
    if injury is None or injury["scheduled_start_utc"] != game["game_time_utc"].isoformat():
        raise MinutesError("Missing matching game injury selection.")
    teams = {}
    for side in ("home", "away"):
        tid = str(game[side+"_team_id"])
        report = injury["teams"][tid]
        if report.get("unresolved_players") or report["team"] != game[side+"_team_tricode"]:
            raise MinutesError("Unresolved or mismatched injury identities.", {"game_id": gid, "team_id": tid})
        state = report["submission_status"]
        if state not in {"submitted", "missing", "not_yet_submitted"}:
            raise MinutesError("Unknown injury team submission state.")
        roster = {player_id(p) for p in rosters["teams"][tid]}
        reports = {player_id(p): row for p,row in report["players"].items()}
        rows = []
        for pid in sorted(roster):
            player = baseline["players"].get(str(pid)) if baseline is not None else None
            initialized = player is None
            if initialized:
                player = initialize_player(pid, priors, day=day)
            minutes = number(player["estimate"], label="baseline minutes", maximum=48)
            r = reports.get(pid)
            status = r["status"] if r else ("Available" if state == "submitted" else "Report pending")
            if status != "Report pending" and status not in PROJECTION_SETTINGS["status_multipliers"]:
                raise MinutesError("Unknown game injury status.", {"player_id": pid, "status": status})
            probability = None if status == "Report pending" else PROJECTION_SETTINGS["status_multipliers"][status]
            multiplier = 1 if probability is None else probability
            rows.append({
                "player_id": pid, "name": player.get("name"), "conditional_minutes": minutes,
                "injury_status": status, "injury_reason": r.get("reason") if r else None,
                "play_probability": probability, "allocation_multiplier": multiplier,
                "injury_adjusted_minutes": minutes*multiplier,
                "initialized_for_projection": initialized,
                "initialization": player["initialization"],
            })
        selected = sorted(rows, key=lambda p: (-p["injury_adjusted_minutes"],p["player_id"]))[:15]
        selected_ids = {p["player_id"] for p in selected}
        total = sum(p["injury_adjusted_minutes"] for p in selected)
        if total <= 0:
            raise MinutesError("Injury assumptions leave no positive team minutes.", {"game_id":gid,"team_id":tid})
        for p in rows:
            p["selected_rotation"] = p["player_id"] in selected_ids
            p["projected_minutes"] = 240*p["injury_adjusted_minutes"]/total if p["selected_rotation"] else 0.0
        teams[tid] = {
            "team_id": int(tid), "team": game[side+"_team_tricode"],
            "submission_status": state, "players": rows,
            "total_projected_minutes": sum(p["projected_minutes"] for p in rows),
        }
    record = {
        "schema_version": 1, "artifact_kind": "game_minutes",
        "purpose": "historical_replay" if inputs.get("historical_replay") else "pregame_projection",
        "season": season, "game_date_pacific": day, "game_id": gid,
        "scheduled_start_utc": game["game_time_utc"].isoformat(),
        "home_team_id": game["home_team_id"], "away_team_id": game["away_team_id"],
        "baseline_through_date": baseline["game_date_pacific"] if baseline is not None else None,
        "baseline_kind": "daily_state" if baseline is not None else "frozen_priors",
        "roster_evidence": "reconstructed_game_date" if inputs.get("historical_replay") else "latest_known_snapshot",
        "roster_date_pacific": rosters["game_date_pacific"],
        "model": projection_model(version), "implementation": implementation_identity(),
        "input_fingerprint": projection_fingerprint(game,inputs,version),
        "inputs": inputs, "teams": teams,
    }
    validate_game_projection(record,season,version)
    return record


def validate_game_projection(record: dict, season: str, version=PROJECTION_VERSION) -> None:
    if (
        record.get("schema_version") != 1 or record.get("artifact_kind") != "game_minutes"
        or record.get("purpose") not in {"pregame_projection","historical_replay"} or record.get("season") != season
        or record.get("model") != projection_model(version) or len(record["teams"]) != 2
    ):
        raise MinutesError("Invalid game-minutes schema, season or model.")
    start = datetime.fromisoformat(record["scheduled_start_utc"])
    home,away = record["home_team_id"],record["away_team_id"]
    if (
        start.tzinfo is None or home == away
        or start.astimezone(PACIFIC).date().isoformat() != record["game_date_pacific"]
        or set(record["teams"]) != {str(home),str(away)}
        or (record["baseline_through_date"] is not None and record["baseline_through_date"] >= record["game_date_pacific"])
        or (record["baseline_through_date"] is None and record["inputs"]["daily_minutes"] is not None)
        or record["roster_date_pacific"] > record["game_date_pacific"]
    ):
        raise MinutesError("Invalid game projection date or matchup.")
    expected = projection_fingerprint(
        {"game_id":record["game_id"],"game_time_utc":start,
         "home_team_id":home,"away_team_id":away},record["inputs"],version,
        implementation=record["implementation"],
    )
    if record["input_fingerprint"] != expected:
        raise MinutesError("Game projection source fingerprint differs from its contents.")
    ids = set()
    for tid,team in record["teams"].items():
        if int(tid) not in NBA_TEAM_IDS or team["team_id"] != int(tid) or not team["players"]:
            raise MinutesError("Invalid game-minutes team identity.")
        rows = team["players"]
        selected = {p["player_id"] for p in sorted(
            rows,key=lambda p:(-p["injury_adjusted_minutes"],p["player_id"]),
        )[:15]}
        weight = sum(p["injury_adjusted_minutes"] for p in rows if p["player_id"] in selected)
        if weight <= 0:
            raise MinutesError("Game-minutes team has no positive weight.")
        for p in rows:
            pid = player_id(p["player_id"])
            if pid in ids:
                raise MinutesError("Game projection repeats a player.")
            ids.add(pid)
            baseline = number(p["conditional_minutes"],label="conditional minutes",maximum=48)
            status = p["injury_status"]
            if status == "Report pending":
                probability,multiplier = None,1
            else:
                if status not in PROJECTION_SETTINGS["status_multipliers"]:
                    raise MinutesError("Invalid saved injury status.")
                probability = multiplier = PROJECTION_SETTINGS["status_multipliers"][status]
            if p["play_probability"] != probability or p["allocation_multiplier"] != multiplier:
                raise MinutesError("Saved injury multiplier differs from the approved policy.")
            adjusted = number(p["injury_adjusted_minutes"],label="injury-adjusted minutes")
            final = number(p["projected_minutes"],label="projected minutes")
            expected = 240*adjusted/weight if pid in selected else 0
            if (
                not isclose(adjusted,baseline*multiplier,abs_tol=1e-10,rel_tol=0)
                or not isclose(final,expected,abs_tol=1e-9,rel_tol=0)
                or p["selected_rotation"] != (pid in selected)
            ):
                raise MinutesError("Saved game allocation does not match its inputs.")
        total = sum(p["projected_minutes"] for p in rows)
        if not isclose(total,240,abs_tol=1e-7,rel_tol=0) or not isclose(team["total_projected_minutes"],total,abs_tol=1e-9):
            raise MinutesError("Game allocation does not conserve 240 team minutes.")
