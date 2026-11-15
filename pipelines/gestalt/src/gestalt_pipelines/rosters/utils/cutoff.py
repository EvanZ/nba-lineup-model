"""Select game eligibility from actual observation evidence or the explicit date-only D+1 rule."""
import json
from datetime import date, datetime, timedelta
from hashlib import sha256

from ...utils import PACIFIC
from .build import RosterError, build_rosters, repair_date

ROSTER_CUTOFF_RULE = "scheduled_tipoff_roster_v1"


def aware_time(value):
    if value is None:
        return None
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise RosterError("Roster timing evidence must include a timezone.", {"value": value})
    return result


def _include(reported, observed, cutoff, confirmed=None):
    day = cutoff.astimezone(PACIFIC).date()
    if reported > day:
        return False, False
    if confirmed is not None:
        return confirmed < cutoff, False
    if observed is not None and observed < cutoff:
        return True, False
    return day >= reported+timedelta(days=1), True


def game_roster(*, season, game, opening, transactions, repairs, inputs, contract_games=None, repair_observed_at=None):
    cutoff = game["game_time_utc"]
    if cutoff.tzinfo is None:
        raise RosterError("Scheduled tipoff needs a timezone.", {})
    day = cutoff.astimezone(PACIFIC).date()
    inputs = {**inputs, "transactions": {key: value for key, value in inputs["transactions"].items() if key <= day.isoformat()}}
    selected, selected_repairs, warnings = [], [], []
    def decide(row, source, reported, observed, confirmed=None):
        included, assumed = _include(reported, observed, cutoff, confirmed)
        if assumed:
            warnings.append({
                "transaction_identity": sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "source": source, "player_id": int(row["PLAYER_ID"] if source == "nba_transactions" else row["player_id"]),
                "team_id": int(row["TEAM_ID"] if source == "nba_transactions" else row["team_id"]),
                "transaction_type": row.get("Transaction_Type", row.get("action")),
                "reported_date_pacific": str(reported), "recorded_observation_time": observed.isoformat() if observed else None,
                "game_id": game["game_id"], "scheduled_start_utc": cutoff.isoformat(),
                "assumed_application_date_pacific": str(reported+timedelta(days=1)), "included_for_game": included,
                "reason": "No trustworthy pre-tipoff timing evidence; date-only availability begins on the following Pacific date.",
            })
        return included
    for row in transactions:
        reported = date.fromisoformat(row["TRANSACTION_DATE"][:10])
        if decide(row, "nba_transactions", reported, aware_time(row.get("first_observed_at"))):
            selected.append(row)
    for row in repairs:
        reported = repair_date(row, opening["date"])
        if decide(row, "repair_transactions", reported, aware_time(repair_observed_at), aware_time(row.get("effective_at"))):
            selected_repairs.append(row)
    record = build_rosters(season=season, day=day, opening=opening, transactions=selected,
                           repair_transactions=selected_repairs, inputs=inputs, games=contract_games)
    target = [str(game[side+"_team_id"]) for side in ("home", "away")]
    selection = {"rule": ROSTER_CUTOFF_RULE, "selection_kind": "reconstructed", "game_id": game["game_id"],
                 "scheduled_start_utc": cutoff.isoformat(), "inputs": inputs,
                 "teams": {team: record["teams"][team] for team in target}, "assumptions": warnings}
    return {**record, "reconstruction_rule": ROSTER_CUTOFF_RULE, "cutoff_selection": selection}


def saved_pregame_roster(*, season, game, projection, reference):
    """The producer's immutable pregame projection already contains every rostered player, including zeros."""
    if (projection.get("purpose") != "pregame_projection" or projection.get("season") != season
            or projection.get("game_id") != game["game_id"]
            or aware_time(projection["scheduled_start_utc"]) != game["game_time_utc"]):
        raise RosterError("Saved pregame roster does not match this game's cutoff.", {"game_id": game["game_id"]})
    teams = {str(game[side+"_team_id"]): [int(row["player_id"]) for row in projection["teams"][str(game[side+"_team_id"])]["players"]]
             for side in ("home", "away")}
    prior = projection["inputs"].get("roster_cutoff")
    assumptions = prior.get("assumptions", []) if prior else []
    return {"season": season, "game_date_pacific": str(game["game_time_utc"].astimezone(PACIFIC).date()), "teams": teams,
            "cutoff_selection": {"rule": ROSTER_CUTOFF_RULE, "selection_kind": "saved_pregame", "game_id": game["game_id"],
                "scheduled_start_utc": game["game_time_utc"].isoformat(), "teams": teams,
                "inputs": {"game_minutes": reference, "original_rosters": projection["inputs"]["rosters"]},
                "assumptions": assumptions}}


def confirm_same_day_box_roster(record, *, game, transactions, box, box_reference, transaction_reference):
    """Historical membership evidence for same-day transaction players only; never fabricate a publication time."""
    from copy import deepcopy
    from .inputs import nba_id, roster_hash

    day = game["game_time_utc"].astimezone(PACIFIC).date()
    moves = [row for row in transactions if row["TRANSACTION_DATE"][:10] == day.isoformat()]
    if not moves:
        return record
    if box.get("gameId") != game["game_id"] or box.get("gameStatus") != 3:
        raise RosterError("Historical roster confirmation requires the matching final NBA box.", {"game_id": game["game_id"]})
    listed, target = {}, []
    for side in ("home", "away"):
        tid = str(game[side+"_team_id"]);target.append(tid)
        team = box[side+"Team"]
        if int(team["teamId"]) != int(tid) or not isinstance(team.get("players"), list) or not team["players"]:
            raise RosterError("Final box has missing players or a different team.", {"game_id": game["game_id"]})
        for player in team["players"]:
            pid = nba_id(player["personId"])
            if pid in listed:
                raise RosterError("Duplicate player in the final box.", {"player_id": pid})
            listed[pid] = tid
    result = deepcopy(record)
    decisions = []
    for pid in sorted({nba_id(row["PLAYER_ID"]) for row in moves}):
        rows = [row for row in moves if nba_id(row["PLAYER_ID"]) == pid]
        before = [tid for tid in target if pid in result["teams"][tid]]
        destination = listed.get(pid)
        if destination is not None:
            for ids in result["teams"].values():
                if pid in ids:ids.remove(pid)
            result["teams"][destination].append(pid)
        else:
            for tid in target:
                if pid in result["teams"][tid]:result["teams"][tid].remove(pid)
        decisions.append({"player_id": pid, "transaction_types": sorted({row["Transaction_Type"] for row in rows}),
                          "reported_date_pacific": str(day), "listed_team_id": int(destination) if destination else None,
                          "included_for_game": destination is not None, "previous_game_teams": before,
                          "timing_assumption": "assumed_known_before_tipoff_from_final_game_box"})
    if not decisions:
        return record
    for tid, ids in result["teams"].items():
        ids.sort()
        if "team_metadata" in result:
            result["team_metadata"][tid]["roster_sha256"] = roster_hash(ids)
    affected = {row["player_id"] for row in decisions}
    selection = result["cutoff_selection"]
    selection["selection_kind"] += "_with_box_confirmation"
    selection["assumptions"] = [row for row in selection["assumptions"]
        if not (row["source"] == "nba_transactions" and row["reported_date_pacific"] == day.isoformat() and row["player_id"] in affected)]
    selection["teams"] = {tid: result["teams"][tid] for tid in target}
    selection["inputs"]["final_box_score"] = box_reference
    selection["inputs"]["box_confirmation_transactions"] = {str(day): transaction_reference}
    selection["historical_box_confirmation"] = {"rule": "same_day_transaction_final_box_v1", "status": "confirmed", "decisions": decisions,
        "note": "Historical reconstruction evidence, not an observed pregame publication timestamp."}
    result["reconstruction_rule"] = "scheduled_tipoff_with_historical_box_confirmation"
    return result
