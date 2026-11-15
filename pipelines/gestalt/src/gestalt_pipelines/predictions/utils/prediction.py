"""The approved scalar-NAIL, fixed-100-possession prediction calculation."""

import json
import math
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from ...utils import COMPLETED_GAME_DATE_CONTRACT, PACIFIC

MODEL_VERSION = "v1"
PROBABILITY_CONVERSION = {
    "method": "fixed_linear_clipped", "intercept": 0.499583,
    "slope": 0.030250, "minimum": 0, "maximum": 1,
}
SETTINGS = {"assumed_possessions": 100, "minutes_normalization": 48,
            "win_probability_conversion": PROBABILITY_CONVERSION}


class PredictionError(ValueError):
    pass


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def implementation_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    inventory = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
    return {"source_sha256": sha256(json_bytes(inventory)).hexdigest()}


def win_probability(margin: float) -> float:
    if not math.isfinite(margin):
        raise PredictionError("The predicted margin must be finite.")
    return min(1.0, max(0.0, 0.499583+0.030250*margin))


def prediction_fingerprint(game: dict, inputs: dict, *, version=MODEL_VERSION, implementation=None) -> str:
    return sha256(json_bytes({
        "game_id": game["game_id"], "start": game["game_time_utc"].isoformat(),
        "home": game["home_team_id"], "away": game["away_team_id"],
        "inputs": inputs, "model_version": version, "settings": SETTINGS,
        "implementation": implementation if implementation is not None else implementation_identity(),
    })).hexdigest()


def build_prediction(*, game: dict, day: str, season: str, ratings: dict, minutes: dict,
                     priors: dict | None, report: dict | None, b2b: dict, inputs: dict,
                     generated_at: datetime, purpose: str, version=MODEL_VERSION, first_date: str | None = None) -> dict:
    bootstrap = ratings.get("artifact_kind") == "player_rating_priors"
    if (ratings["season"] != season or minutes["season"] != season
            or minutes["game_date_pacific"] != day or minutes["game_id"] != game["game_id"]
            or datetime.fromisoformat(minutes["scheduled_start_utc"]) != game["game_time_utc"]
            or game["game_time_utc"].astimezone(PACIFIC).date().isoformat() != day):
        raise PredictionError("Prediction inputs have mismatched identities.")
    if bootstrap:
        if first_date is None or not first_date <= day <= str(date.fromisoformat(first_date)+timedelta(days=1)):
            raise PredictionError("Only the opening-day forecast window may use season priors without a completed state.")
        effective_date, observed_through = first_date, None
    else:
        observed_through = ratings["observed_through_game_date_pacific"]
        effective_date = ratings["effective_from_game_date_pacific"]
        if (ratings.get("date_contract") != COMPLETED_GAME_DATE_CONTRACT
                or ratings["game_date_pacific"] != observed_through or observed_through >= day
                or effective_date != str(date.fromisoformat(observed_through)+timedelta(days=1))):
            raise PredictionError("Prediction ratings have a mismatched date contract or future evidence.")
    rating_by_id = {} if bootstrap else {p["player_id"]: p for p in ratings["players"]}
    prior_source = ratings if bootstrap else priors
    catalog = {p["player_id"]: p for p in prior_source["players"]} if prior_source is not None else {}
    sides, totals = {}, {}
    for side in ("home", "away"):
        team_id = game[side+"_team_id"]
        source = minutes["teams"][str(team_id)]
        players = []
        for player in source["players"]:
            pid = player["player_id"]
            if pid in rating_by_id:
                nail, rating_source = rating_by_id[pid]["nail"], "daily_state"
            elif pid in catalog:
                nail, rating_source = catalog[pid]["rapm"], "frozen_prior"
            else:
                nail, rating_source = ratings["settings"]["replacement_prior"], "replacement"
            if not math.isfinite(nail):
                raise PredictionError(f"Player {pid} has a non-finite rating.")
            row = {key: player[key] for key in (
                "player_id", "name", "conditional_minutes", "projected_minutes", "injury_status",
                "injury_reason", "play_probability", "selected_rotation",
            )}
            row.update(nail=nail, rating_source=rating_source,
                       rating_contribution=player["projected_minutes"]/48*nail)
            players.append(row)
        totals[side] = sum(p["rating_contribution"] for p in players)
        sides[side] = {"team_id": team_id, "team": source["team"], "players": players,
                       "submission_status": source["submission_status"],
                       "total_projected_minutes": source["total_projected_minutes"]}
    coefficient = ratings["settings"]["fixed_back_to_back"]
    adjustment = (int(b2b["home"])-int(b2b["away"]))*coefficient
    hca = ratings["settings"]["hca"]["prior_mean"] if bootstrap else ratings["hca"]["estimate"]
    margin = totals["home"]-totals["away"]+hca+adjustment
    implementation = implementation_identity()
    record = {
        "schema_version": 1, "artifact_kind": "game_prediction", "season": season,
        "model_version": version, "purpose": purpose, "game_id": game["game_id"],
        "game_date_pacific": day, "scheduled_start_utc": game["game_time_utc"].isoformat(),
        "generated_at_utc": generated_at.isoformat(), "assumed_possessions": 100,
        "predicted_home_margin": margin, "home_win_probability": win_probability(margin),
        "win_probability_conversion": dict(PROBABILITY_CONVERSION),
        "ratings_effective_date_pacific": effective_date,
        "ratings_observed_through_game_date_pacific": observed_through,
        "ratings_baseline_kind": "frozen_priors" if bootstrap else "daily_state",
        "minutes_observed_through_game_date_pacific": minutes["baseline_through_date"],
        "components": {"home_player_total": totals["home"], "away_player_total": totals["away"],
                       "hca": hca, "home_is_b2b": b2b["home"], "away_is_b2b": b2b["away"],
                       "b2b_coefficient": coefficient, "b2b_adjustment": adjustment},
        **sides, "injury_report": report, "inputs": inputs, "implementation": implementation,
        "input_fingerprint": prediction_fingerprint(game, inputs, version=version, implementation=implementation),
    }
    validate_prediction(record, season, version)
    return record


def validate_prediction(record: dict, season: str, version=MODEL_VERSION) -> None:
    if (record.get("schema_version") != 1 or record.get("artifact_kind") != "game_prediction"
            or record.get("season") != season or record.get("model_version") != version
            or record.get("purpose") not in {"pregame_projection", "historical_replay"}
            or record.get("assumed_possessions") != 100
            or record.get("win_probability_conversion") != PROBABILITY_CONVERSION):
        raise PredictionError("Invalid prediction schema, season or calculation settings.")
    start = datetime.fromisoformat(record["scheduled_start_utc"])
    generated = datetime.fromisoformat(record["generated_at_utc"])
    day = record["game_date_pacific"]
    if (start.tzinfo is None or generated.tzinfo is None
            or start.astimezone(PACIFIC).date().isoformat() != day
            or (record["purpose"] == "pregame_projection" and generated >= start)
            or record["ratings_effective_date_pacific"] > day
            or (record["ratings_observed_through_game_date_pacific"] is not None
                and record["ratings_observed_through_game_date_pacific"] >= record["ratings_effective_date_pacific"])
            or (record["minutes_observed_through_game_date_pacific"] is not None
                and record["minutes_observed_through_game_date_pacific"] >= day)):
        raise PredictionError("Prediction dates include future evidence or a late pregame calculation.")
    kind = record.get("ratings_baseline_kind", "daily_state")
    if kind not in {"daily_state", "frozen_priors"} or (record["ratings_observed_through_game_date_pacific"] is None and kind != "frozen_priors"):
        raise PredictionError("Prediction rating cutoff requires an explicit frozen-prior baseline.")
    if "espn" in json.dumps(record).casefold():
        raise PredictionError("Prohibited basketball source in prediction inputs.")
    json_bytes(record)
    components = record["components"]
    ids = set()
    for side in ("home", "away"):
        team = record[side]
        rows = team["players"]
        if not rows or team["submission_status"] not in {"submitted", "missing", "not_yet_submitted"}:
            raise PredictionError("Prediction is missing the accepted team allocation.")
        for player in rows:
            if (not isinstance(player["player_id"], int) or player["player_id"] <= 0
                    or player["player_id"] in ids or player["projected_minutes"] < 0
                    or player["conditional_minutes"] < 0
                    or player["rating_source"] not in {"daily_state", "frozen_prior", "replacement"}
                    or player["rating_contribution"] != player["projected_minutes"]/48*player["nail"]):
                raise PredictionError("Invalid or duplicate player contribution.")
            ids.add(player["player_id"])
        total = sum(p["projected_minutes"] for p in rows)
        if (not math.isclose(total, 240, abs_tol=1e-8, rel_tol=0)
                or total != team["total_projected_minutes"]
                or sum(p["rating_contribution"] for p in rows) != components[side+"_player_total"]):
            raise PredictionError("Prediction team totals differ from the accepted minutes.")
    if record["home"]["team_id"] == record["away"]["team_id"]:
        raise PredictionError("Prediction repeats a team.")
    if (type(components["home_is_b2b"]) is not bool or type(components["away_is_b2b"]) is not bool
            or components["b2b_adjustment"] != (int(components["home_is_b2b"])-int(components["away_is_b2b"]))*components["b2b_coefficient"]):
        raise PredictionError("Prediction B2B adjustment differs from its flags.")
    margin = components["home_player_total"]-components["away_player_total"]+components["hca"]+components["b2b_adjustment"]
    if record["predicted_home_margin"] != margin or record["home_win_probability"] != win_probability(margin):
        raise PredictionError("Prediction margin or probability differs from its components.")
    game = {"game_id": record["game_id"], "game_time_utc": start,
            "home_team_id": record["home"]["team_id"], "away_team_id": record["away"]["team_id"]}
    if record["input_fingerprint"] != prediction_fingerprint(game, record["inputs"], version=version, implementation=record["implementation"]):
        raise PredictionError("Prediction fingerprint differs from its saved inputs.")
