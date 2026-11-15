"""Per-game outcomes and additive totals; no model or odds-provider calls."""
import json
import math
from hashlib import sha256
from pathlib import Path

SERIES = ("model", "actual_minutes_counterfactual", "polymarket", "bookmaker_consensus")
CONTRACT = "market_comparison_v1"


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob("utils/*.py"), *root.glob("assets/*.py")])
    return {"calculation_contract": CONTRACT, "source_sha256": sha256(b"".join(
        str(path.relative_to(root)).encode()+path.read_bytes() for path in files)).hexdigest()}


def fingerprint(inputs, season, version, day):
    return sha256(json_bytes({"inputs": inputs, "season": season, "model_version": version,
                              "game_date_pacific": day, "implementation": implementation_identity()})).hexdigest()


def comparison(margin, probability, actual, benchmark, kind):
    for value in (margin, probability):
        if value is not None and not math.isfinite(value):
            raise ValueError("Non-finite performance input.")
    if probability is not None and not 0 <= probability <= 1:
        raise ValueError("Invalid win probability.")
    edge = None if margin is None or benchmark is None else margin-benchmark
    if kind == "market_consensus":
        pick, outcome = None, "not_applicable"
    elif edge is None:
        pick, outcome = None, "unavailable"
    elif math.isclose(edge, 0, abs_tol=1e-12):
        pick, outcome = None, "no_pick"
    else:
        pick = "home" if edge > 0 else "away"
        covered = actual-benchmark
        outcome = "push" if math.isclose(covered, 0, abs_tol=1e-12) else ("win" if edge*covered > 0 else "loss")
    error = None if margin is None else margin-actual
    return {"kind": kind, "predicted_home_margin": margin, "home_win_probability": probability,
            "winner_pick": None if probability is None else ("home" if probability >= .5 else "away"),
            "winner_correct": None if probability is None else ((probability >= .5) == (actual > 0)),
            "home_margin_error": error, "absolute_margin_error": None if error is None else abs(error),
            "squared_margin_error": None if error is None else error*error,
            "brier_loss": None if probability is None else (probability-int(actual > 0))**2,
            "ats": {"benchmark": "bookmaker_consensus", "benchmark_home_spread": None if benchmark is None else -benchmark,
                    "pick": pick, "result": outcome}}


def game_result(game, prediction, counterfactual, box, markets, inputs):
    actual = int(box["homeTeam"]["score"])-int(box["awayTeam"]["score"])
    if actual == 0:
        raise ValueError("A final NBA game cannot be tied.")
    probabilities, spreads = markets.get("h2h", {}), markets.get("spreads", {})
    consensus = spreads.get("consensus")
    comparisons = {
        "model": comparison(prediction["predicted_home_margin"], prediction["home_win_probability"], actual, consensus, prediction["purpose"]),
        "actual_minutes_counterfactual": comparison(counterfactual["counterfactual_home_margin"], counterfactual["counterfactual_home_win_probability"],
                                                    actual, consensus, "postgame_actual_minutes_counterfactual"),
        "polymarket": comparison(spreads.get("polymarket", {}).get("value"), probabilities.get("polymarket", {}).get("value"), actual, consensus, "market_quote"),
        "bookmaker_consensus": comparison(consensus, probabilities.get("consensus"), actual, consensus, "market_consensus"),
    }
    teams = {side: {"team_id": game[side+"_team_id"], "team_abbreviation": game[side+"_team_tricode"],
                    "team_name": box[side+"Team"]["teamName"], "final_score": int(box[side+"Team"]["score"])} for side in ("home", "away")}
    shared = all(row[field] is not None for row in comparisons.values() for field in ("predicted_home_margin", "home_win_probability"))
    missing = [f"{name}/{field}" for name, row in comparisons.items() for field in ("predicted_home_margin", "home_win_probability") if row[field] is None]
    return {"game_id": game["game_id"], "scheduled_start_utc": game["game_time_utc"].isoformat(), **teams,
            "result": {"home_margin": actual, "home_won": actual > 0, "winner": "home" if actual > 0 else "away",
                       "home_cover_margin": None if consensus is None else actual-consensus,
                       "home_covered": None if consensus is None or math.isclose(actual-consensus, 0, abs_tol=1e-12) else actual > consensus},
            "in_shared_comparison": shared, "missing_market_fields": missing, "comparisons": comparisons,
            "markets": {"provider": "The Odds API", "selection_rule": "latest_pregame_polymarket_and_same_snapshot_books", **markets}, "inputs": inputs}


def _totals(rows, name):
    values = [row["comparisons"][name] for row in rows]
    return {"probability_game_count": sum(v["brier_loss"] is not None for v in values),
            "correct_winner_count": sum(v["winner_correct"] is True for v in values),
            "brier_loss_sum": math.fsum(v["brier_loss"] for v in values if v["brier_loss"] is not None),
            "margin_game_count": sum(v["home_margin_error"] is not None for v in values),
            "squared_margin_error_sum": math.fsum(v["squared_margin_error"] for v in values if v["squared_margin_error"] is not None),
            "absolute_margin_error_sum": math.fsum(v["absolute_margin_error"] for v in values if v["absolute_margin_error"] is not None),
            "ats": {label: sum(v["ats"]["result"] == value for v in values) for label, value in (
                ("wins", "win"), ("losses", "loss"), ("pushes", "push"), ("no_picks", "no_pick"), ("not_applicable", "not_applicable"), ("unavailable", "unavailable"))}}


def daily_totals(rows):
    shared = [row for row in rows if row["in_shared_comparison"]]
    different = [row for row in shared if not math.isclose(row["comparisons"]["polymarket"]["predicted_home_margin"],
                                                          row["comparisons"]["bookmaker_consensus"]["predicted_home_margin"], abs_tol=1e-12)]
    return {"cohort": "games_with_model_polymarket_and_bookmaker_consensus", "game_count": len(shared),
            "series": {name: _totals(shared, name) for name in SERIES},
            "polymarket_different_spread": {"game_count": len(different), "series": {name: _totals(different, name) for name in SERIES}}}


def aggregate_totals(days):
    """Combine saved sums/counts, not daily percentages or RMSEs."""
    result = {}
    for name in SERIES:
        selected = [day["series"][name] for day in days]
        count = sum(row["probability_game_count"] for row in selected)
        margins = sum(row["margin_game_count"] for row in selected)
        correct = sum(row["correct_winner_count"] for row in selected)
        ats = {key: sum(row["ats"][key] for row in selected) for key in ("wins", "losses", "pushes", "no_picks", "not_applicable", "unavailable")}
        ats["win_rate_excluding_pushes"] = ats["wins"]/(ats["wins"]+ats["losses"]) if ats["wins"]+ats["losses"] else None
        result[name] = {"games": count, "brier": math.fsum(r["brier_loss_sum"] for r in selected)/count if count else None,
                        "winner_accuracy": correct/count if count else None, "correct_winners": correct,
                        "margin_rmse": math.sqrt(math.fsum(r["squared_margin_error_sum"] for r in selected)/margins) if margins else None,
                        "margin_mae": math.fsum(r["absolute_margin_error_sum"] for r in selected)/margins if margins else None, "ats": ats}
    return result
