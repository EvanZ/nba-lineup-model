"""Postgame attribution using full-game minutes and the saved forecast's ratings."""
import math
import re
from copy import deepcopy
from datetime import datetime
from hashlib import sha256

from .prediction import (
    MODEL_VERSION, PredictionError, implementation_identity, json_bytes,
    validate_prediction, win_probability,
)


def _minutes(value) -> float:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", str(value))
    if not match or not any(match.groups()):
        raise PredictionError("Final box score has invalid player minutes.")
    hours, minutes, seconds = (float(v or 0) for v in match.groups())
    return 60*hours+minutes+seconds/60


def actual_minutes_fingerprint(inputs: dict, *, version=MODEL_VERSION, implementation=None) -> str:
    return sha256(json_bytes({
        "kind": "postgame_actual_minutes_counterfactual", "inputs": inputs,
        "model_version": version, "minutes_normalization": "five_players",
        "implementation": implementation if implementation is not None else implementation_identity(),
    })).hexdigest()


def extra_actual_player_ids(prediction: dict, box: dict) -> set[int]:
    known = {p["player_id"] for side in ("home", "away") for p in prediction[side]["players"]}
    return {p["personId"] for side in ("home", "away") for p in box[side+"Team"]["players"]}-known


def build_actual_minutes_prediction(*, prediction: dict, box: dict, inputs: dict,
                                    generated_at: datetime, ratings: dict | None = None,
                                    priors: dict | None = None, version=MODEL_VERSION) -> dict:
    season = prediction["season"]
    validate_prediction(prediction, season, version)
    if box["gameId"] != prediction["game_id"] or box["gameStatus"] != 3:
        raise PredictionError("Counterfactual requires the same game's final NBA box score.")
    if ratings is not None and (ratings["season"] != season
            or ratings.get("effective_from_game_date_pacific", ratings["game_date_pacific"]) != prediction["ratings_effective_date_pacific"]
            or ratings["observed_through_game_date_pacific"] != prediction["ratings_observed_through_game_date_pacific"]
            or inputs.get("player_ratings") != prediction["inputs"]["player_ratings"]):
        raise PredictionError("Counterfactual rating input differs from the baseline prediction.")
    daily = {p["player_id"]: p for p in ratings["players"]} if ratings else {}
    catalog = {p["player_id"]: p for p in priors["players"]} if priors else {}
    settings = ratings["settings"] if ratings is not None else priors["settings"] if priors is not None else None
    teams = {}
    seen = set()
    for side in ("home", "away"):
        baseline = prediction[side]
        actual = box[side+"Team"]
        if int(actual["teamId"]) != baseline["team_id"]:
            raise PredictionError("Final box score has a different home/away team.")
        forecast = {p["player_id"]: p for p in baseline["players"]}
        players = {}
        for p in actual["players"]:
            pid = p["personId"]
            if type(pid) is not int or pid <= 0 or pid in players:
                raise PredictionError("Final box score has an invalid or duplicate player ID.")
            minutes = _minutes(p["statistics"]["minutes"])
            players[pid] = {"name": p.get("name"), "minutes": minutes}
        total = sum(p["minutes"] for p in players.values())
        if not math.isfinite(total) or total <= 0:
            raise PredictionError("A final team has no positive actual minutes.")
        rows = []
        for pid in sorted(forecast.keys() | players.keys()):
            if pid in seen:
                raise PredictionError("A counterfactual player appears on both teams.")
            seen.add(pid)
            before = forecast.get(pid)
            observed = players.get(pid)
            if before:
                nail, source = before["nail"], before["rating_source"]
            elif pid in daily:
                nail, source = daily[pid]["nail"], "daily_state"
            elif pid in catalog:
                nail, source = catalog[pid]["rapm"], "frozen_prior"
            elif settings is not None:
                nail, source = settings["replacement_prior"], "replacement"
            else:
                raise PredictionError(f"Actual player {pid} needs the baseline prediction's saved rating input.")
            minutes = observed["minutes"] if observed else 0.0
            projected = before["rating_contribution"] if before else 0.0
            contribution = 5*minutes/total*nail
            name = before.get("name") if before else None
            if not name and observed:
                name = observed["name"]
            rows.append({"player_id": pid, "name": name, "in_baseline_prediction": before is not None,
                         "projected_minutes": before["projected_minutes"] if before else 0.0,
                         "actual_minutes": minutes, "nail": nail, "rating_source": source,
                         "projected_rating_contribution": projected,
                         "actual_rating_contribution": contribution, "contribution_delta": contribution-projected})
        teams[side] = {"team_id": baseline["team_id"], "team": baseline["team"],
                       "total_projected_minutes": baseline["total_projected_minutes"], "total_actual_minutes": total,
                       "projected_player_total": prediction["components"][side+"_player_total"],
                       "actual_player_total": sum(p["actual_rating_contribution"] for p in rows), "players": rows}
    components = deepcopy(prediction["components"])
    for side in ("home", "away"):
        components[side+"_player_total"] = teams[side]["actual_player_total"]
    margin = components["home_player_total"]-components["away_player_total"]+components["hca"]+components["b2b_adjustment"]
    implementation = implementation_identity()
    record = {"schema_version": 1, "artifact_kind": "game_prediction_actual_minutes",
              "purpose": "postgame_actual_minutes_counterfactual", "season": season, "model_version": version,
              **{k: prediction[k] for k in ("game_id", "game_date_pacific", "scheduled_start_utc", "assumed_possessions",
                     "ratings_effective_date_pacific", "ratings_observed_through_game_date_pacific", "win_probability_conversion")},
              "baseline_prediction_purpose": prediction["purpose"],
              "ratings_baseline_kind": prediction.get("ratings_baseline_kind", "daily_state"), "generated_at_utc": generated_at.isoformat(),
              "original_predicted_home_margin": prediction["predicted_home_margin"],
              "original_home_win_probability": prediction["home_win_probability"],
              "counterfactual_home_margin": margin, "counterfactual_home_win_probability": win_probability(margin),
              "actual_home_score": int(box["homeTeam"]["score"]), "actual_away_score": int(box["awayTeam"]["score"]),
              "components": components, **teams, "inputs": inputs, "implementation": implementation,
              "input_fingerprint": actual_minutes_fingerprint(inputs, version=version, implementation=implementation)}
    record["actual_home_margin"] = record["actual_home_score"]-record["actual_away_score"]
    validate_actual_minutes_prediction(record, season, version)
    return record


def validate_actual_minutes_prediction(record: dict, season: str, version=MODEL_VERSION) -> None:
    from .prediction import PROBABILITY_CONVERSION
    if (record.get("schema_version") != 1 or record.get("artifact_kind") != "game_prediction_actual_minutes"
            or record.get("purpose") != "postgame_actual_minutes_counterfactual"
            or record.get("season") != season or record.get("model_version") != version
            or record.get("baseline_prediction_purpose") not in {"pregame_projection", "historical_replay"}
            or record.get("assumed_possessions") != 100
            or record.get("win_probability_conversion") != PROBABILITY_CONVERSION):
        raise PredictionError("Invalid actual-minutes counterfactual schema or settings.")
    start = datetime.fromisoformat(record["scheduled_start_utc"])
    generated = datetime.fromisoformat(record["generated_at_utc"])
    if start.tzinfo is None or generated.tzinfo is None or generated < start:
        raise PredictionError("Counterfactual timestamps must describe a postgame calculation.")
    if ((record["ratings_observed_through_game_date_pacific"] is not None
                and record["ratings_observed_through_game_date_pacific"] >= record["ratings_effective_date_pacific"])
            or record["ratings_effective_date_pacific"] > record["game_date_pacific"]):
        raise PredictionError("Counterfactual contains future ratings evidence.")
    json_bytes(record)
    if "espn" in json_bytes(record).decode().casefold():
        raise PredictionError("Prohibited basketball source in counterfactual inputs.")
    if record["input_fingerprint"] != actual_minutes_fingerprint(record["inputs"], version=version, implementation=record["implementation"]):
        raise PredictionError("Counterfactual fingerprint differs from its saved inputs.")
    ids = set()
    for side in ("home", "away"):
        team = record[side]
        rows = team["players"]
        total = team["total_actual_minutes"]
        if not rows or total <= 0 or not math.isclose(sum(p["actual_minutes"] for p in rows), total, abs_tol=1e-8, rel_tol=0):
            raise PredictionError("Counterfactual team minutes are incomplete.")
        if not math.isclose(sum(p["projected_minutes"] for p in rows), team["total_projected_minutes"], abs_tol=1e-8, rel_tol=0):
            raise PredictionError("Counterfactual projected minutes differ from the baseline.")
        for p in rows:
            if (type(p["player_id"]) is not int or p["player_id"] <= 0 or p["player_id"] in ids
                    or p["actual_minutes"] < 0 or p["projected_minutes"] < 0
                    or p["rating_source"] not in {"daily_state", "frozen_prior", "replacement"}
                    or p["actual_rating_contribution"] != 5*p["actual_minutes"]/total*p["nail"]
                    or p["contribution_delta"] != p["actual_rating_contribution"]-p["projected_rating_contribution"]):
                raise PredictionError("Counterfactual player contribution is invalid or duplicated.")
            ids.add(p["player_id"])
        for kind in ("actual", "projected"):
            if not math.isclose(sum(p[kind+"_rating_contribution"] for p in rows), team[kind+"_player_total"], abs_tol=1e-10, rel_tol=0):
                raise PredictionError("Counterfactual player contributions do not reconstruct the team.")
        if record["components"][side+"_player_total"] != team["actual_player_total"]:
            raise PredictionError("Counterfactual components differ from the team totals.")
    c = record["components"]
    expected = record["home"]["actual_player_total"]-record["away"]["actual_player_total"]+c["hca"]+c["b2b_adjustment"]
    original = record["home"]["projected_player_total"]-record["away"]["projected_player_total"]+c["hca"]+c["b2b_adjustment"]
    if (record["counterfactual_home_margin"] != expected or record["original_predicted_home_margin"] != original
            or record["counterfactual_home_win_probability"] != win_probability(expected)
            or record["original_home_win_probability"] != win_probability(original)
            or record["actual_home_margin"] != record["actual_home_score"]-record["actual_away_score"]
            or record["actual_home_margin"] == 0 or min(record["actual_home_score"], record["actual_away_score"]) < 0
            or record["home"]["team_id"] == record["away"]["team_id"]):
        raise PredictionError("Counterfactual margins, probabilities or result are inconsistent.")
