"""Project accepted ratings and pinned postgame evidence into player game history."""
from copy import deepcopy
from datetime import date
from math import isfinite
import re


STAT_FIELDS = {
    "points": "points", "rebounds": "reboundsTotal", "offensive_rebounds": "reboundsOffensive",
    "defensive_rebounds": "reboundsDefensive", "assists": "assists", "steals": "steals",
    "blocks": "blocks", "turnovers": "turnovers", "personal_fouls": "foulsPersonal",
    "field_goals_made": "fieldGoalsMade", "field_goals_attempted": "fieldGoalsAttempted",
    "field_goal_percentage": "fieldGoalsPercentage", "three_pointers_made": "threePointersMade",
    "three_pointers_attempted": "threePointersAttempted", "three_point_percentage": "threePointersPercentage",
    "free_throws_made": "freeThrowsMade", "free_throws_attempted": "freeThrowsAttempted",
    "free_throw_percentage": "freeThrowsPercentage", "plus_minus": "plusMinusPoints",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("Non-finite or nonnumeric player-history evidence.")
    return value


def game_history_details(game, actual, prediction, box, possessions, inputs):
    """A missing box row does not remove a player present in the saved game roster."""
    gid, day = game["game_id"], str(game["game_date"])
    if box.get("gameId") != gid or box.get("gameStatus") != 3:
        raise ValueError("History requires the pinned final box for its official game.")
    for record, kind in [(actual, "game_prediction_actual_minutes"), (prediction, "game_prediction")]:
        if (record.get("artifact_kind") != kind or record.get("game_id") != gid
                or record.get("game_date_pacific") != day):
            raise ValueError("History game evidence has a different artifact/game/date identity.")
    if actual["baseline_prediction_purpose"] != prediction["purpose"]:
        raise ValueError("Postgame and pinned prediction purposes differ.")
    result, seen = {}, set()
    for side, other in [("home", "away"), ("away", "home")]:
        team, opponent = game[side+"_team_id"], game[other+"_team_id"]
        if any(record[side]["team_id"] != team for record in [actual, prediction]) or box[side+"Team"]["teamId"] != team:
            raise ValueError("Game-history team identity differs from the official schedule.")
        score, other_score = box[side+"Team"]["score"], box[other+"Team"]["score"]
        if actual["actual_"+side+"_score"] != score:
            raise ValueError("Postgame score differs from its pinned final box.")
        forecasts = {p["player_id"]: p for p in prediction[side]["players"]}
        boxes = {p["personId"]: p for p in box[side+"Team"]["players"]}
        rows = {p["player_id"]: p for p in actual[side]["players"]}
        if (len(forecasts) != len(prediction[side]["players"]) or len(boxes) != len(box[side+"Team"]["players"])
                or len(rows) != len(actual[side]["players"]) or set(rows) != set(forecasts) | set(boxes)):
            raise ValueError("History must cover the complete saved prediction/final-box player union.")
        for pid, player in rows.items():
            if pid in seen:
                raise ValueError("Player appears on both teams in the same game.")
            seen.add(pid)
            forecast, row = forecasts.get(pid), boxes.get(pid)
            minutes = _number(player["actual_minutes"])
            if minutes < 0 or bool(player["in_baseline_prediction"]) != (forecast is not None):
                raise ValueError("Invalid saved actual-minute/baseline evidence.")
            played = row is not None and (row.get("played") in (True, 1, "1") or minutes > 0)
            if row is None and (minutes != 0 or possessions.get(pid, 0) != 0):
                raise ValueError("A played player is missing from the pinned final box.")
            if row is not None:
                stamp = row.get("statistics", {}).get("minutes")
                if stamp:
                    match = re.fullmatch(r"PT(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?", stamp)
                    if not match or abs(minutes-float(match[1] or 0)-float(match[2] or 0)/60) > 1e-6:
                        raise ValueError("Saved actual minutes differ from the pinned box.")
                if minutes > 0 and row.get("played") in (False, 0, "0"):
                    raise ValueError("Pinned box contradicts positive played minutes.")
            status = ("not_listed" if row is None else "played" if played else
                      "inactive" if row.get("status") == "INACTIVE" else
                      "did_not_play" if row.get("played") in (False, 0, "0") else "unknown")
            stats = None
            if played:
                raw = row.get("statistics", {})
                stats = {dest: _number(raw[src]) if raw.get(src) is not None else None for dest, src in STAT_FIELDS.items()}
            pregame = None if forecast is None else {k: deepcopy(forecast[k]) for k in ("nail", "projected_minutes", "injury_status", "injury_reason")}
            if pregame is not None:
                _number(pregame["nail"]); _number(pregame["projected_minutes"])
                if player["nail"] != pregame["nail"] or player["projected_minutes"] != pregame["projected_minutes"]:
                    raise ValueError("Postgame player differs from its pinned baseline prediction.")
            result[pid] = {
                "game_id": gid, "team_id": team, "team": game[side+"_team_tricode"],
                "opponent_team_id": opponent, "opponent": game[other+"_team_tricode"], "home_away": side,
                "team_score": score, "opponent_score": other_score, "result": "W" if score > other_score else "L",
                "participation": {"status": status, "reason": row.get("notPlayingDescription") if row else None,
                                  "box_score_row_present": row is not None},
                "prediction_purpose": prediction["purpose"], "pregame": pregame,
                "actual_minutes": minutes, "possessions": _number(possessions.get(pid, 0)),
                "stats": stats, "inputs": deepcopy(inputs),
            }
    if set(possessions)-seen:
        raise ValueError("Pinned stint participant is missing from the saved game roster.")
    return result


def history_records(chain, game_details):
    """Copy only game-day scalars. Current/initial values stay in the player index."""
    histories, retained, initial = {}, set(), {}
    for state, reference in chain:
        day = state["game_date_pacific"]
        players = {p["player_id"]: p for p in state["players"]}
        if not retained <= players.keys():
            raise ValueError("A retained player disappeared from the ratings lineage.")
        retained = players.keys()
        for pid, player in players.items():
            histories.setdefault(pid, [])
            if (date.fromisoformat(player["initialized_on"]) > date.fromisoformat(day)
                    or player["initialization"]["input"] != state["inputs"]["priors"]):
                raise ValueError("Invalid player-history initialization date/input.")
            values = {k: player[k] for k in ("starting_nail", "initialized_on", "initialization", "forecast_source")}
            if pid in initial and initial[pid] != values:
                raise ValueError(f"Saved initialization changed within player {pid}'s lineage.")
            initial[pid] = values
        details = game_details[day]
        if set(details)-players.keys():
            raise ValueError(f"Game-roster players are missing from the accepted ratings on {day}: {sorted(set(details)-players.keys())}")
        for pid, games in details.items():
            player = players[pid]
            if not games:
                raise ValueError("Off-day records are not part of player game history.")
            record = {
                "schema_version": 1, "artifact_kind": "player_game_history",
                **{k: state[k] for k in ("season", "model_version", "model_id", "model_run_id", "date_contract", "calculation_contract")},
                "player_id": pid, "player_name": player.get("player_name"), "game_date_pacific": day,
                "inputs": {"ratings_state": deepcopy(reference), **deepcopy(state["inputs"])},
                "cumulative_possessions": player["on_court_possessions"], "games": deepcopy(games),
            }
            record.update({k: deepcopy(player[k]) for k in ("nail", "starting_nail", "team_id", "initialized_on", "initialization", "forecast_source")})
            histories[pid].append(record)
    return histories


def history_index(records, references, *, first_date, chain, player_id):
    if len(records) != len(references):
        raise ValueError("History points and immutable references differ.")
    head, head_reference = chain[-1]
    first, first_ref = next((state, ref) for state, ref in chain if any(p["player_id"] == player_id for p in state["players"]))
    opening = next(p for p in first["players"] if p["player_id"] == player_id)
    current = next(p for p in head["players"] if p["player_id"] == player_id)
    points = [{"game_date_pacific": row["game_date_pacific"], **ref, "source_ratings": row["inputs"]["ratings_state"]}
              for row, ref in zip(records, references)]
    return {
        "schema_version": 1, "artifact_kind": "player_game_history_index",
        **{k: head[k] for k in ("season", "model_version", "model_id", "model_run_id")},
        "player_id": player_id, "player_name": current["player_name"],
        "through_game_date_pacific": head["game_date_pacific"], "source_ratings": deepcopy(head_reference),
        "initialization_input": deepcopy(head["inputs"]["priors"]),
        "starting_point": {"kind": "opening_prior" if first["game_date_pacific"] == first_date else "introduced_prior",
            "nail": opening["starting_nail"], "initialized_on": opening["initialized_on"],
            "first_state_date": first["game_date_pacific"], "initialization": deepcopy(opening["initialization"]),
            "forecast_source": opening["forecast_source"], "source_ratings": deepcopy(first_ref)},
        "points": points, "last_game": deepcopy(points[-1]) if points else None,
        "current": {"game_date_pacific": head["game_date_pacific"], "nail": current["nail"],
            "cumulative_possessions": current["on_court_possessions"], "team_id": current["team_id"],
            "source_ratings": deepcopy(head_reference)},
    }
