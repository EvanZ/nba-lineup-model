"""One date's possession-weighted joint player/HCA update, with fixed priors."""

import math
from datetime import date, timedelta

from .initialization import check_source_policy, implementation_identity, json_bytes
from .sources import MODEL_VERSION
from ...utils import COMPLETED_GAME_DATE_CONTRACT, PACIFIC

CALCULATION_CONTRACT = "frozen_priors_stints_joint_hca_fixed_b2b_v1"


def game_contribution(rows: list[dict], *, player_ids: list[int], starting: dict[int, float],
                      b2b: float) -> dict:
    import numpy as np
    from scipy import sparse

    columns = {player_id: i for i, player_id in enumerate(player_ids)}
    if not math.isfinite(b2b):
        raise ValueError("The fixed B2B adjustment must be finite.")
    positions, indices, signs, weights, targets = [], [], [], [], []
    zero_count = 0
    for row in rows:
        weight = float(row["possessions"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Stint possession weights must be finite and nonnegative.")
        if weight == 0:
            zero_count += 1
            continue
        target = float(row["target_home_net_rating"])
        if not math.isfinite(target):
            raise ValueError("A positive-possession stint has a non-finite scoring target.")
        home, away = row["home_player_ids"], row["away_player_ids"]
        if len(home) != 5 or len(set(home)) != 5 or len(away) != 5 or len(set(away)) != 5:
            raise ValueError("Ratings require five unique players per lineup.")
        if set(home)&set(away):
            raise ValueError("A player appears on both sides of a stint.")
        index = len(weights)
        for lineup, sign in ((home, 1.0), (away, -1.0)):
            for player_id in lineup:
                positions.append(index); indices.append(columns[player_id]); signs.append(sign)
        positions.append(index); indices.append(len(player_ids)); signs.append(1.0)
        weights.append(weight); targets.append(target)
    if not weights:
        raise ValueError("An expected game has no positive-possession ratings evidence.")
    design = sparse.csr_matrix((signs, (positions, indices)), shape=(len(weights), len(player_ids)+1))
    w = np.asarray(weights)
    prior = np.array([starting[i] for i in player_ids]+[0.0])
    # HCA remains in the joint solve. It is not subtracted as a fixed context term.
    residual = np.asarray(targets)-b2b-np.asarray(design@prior).ravel()
    return {
        "gram": (design.T@design.multiply(w[:, None])).toarray(),
        "score": np.asarray(design.T@(w*residual)).ravel(),
        "exposure": np.asarray(abs(design).T@w).ravel(),
        "positive_stint_count": len(weights), "zero_possession_stint_count": zero_count,
        "possessions": float(w.sum()), "fixed_b2b_offset": float(b2b),
    }


def solve_joint(gram, score, settings: dict) -> dict:
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve

    n = len(score)-1
    hca = settings["hca"]
    precision = np.array(gram, dtype=float, copy=True)
    prior_precision = np.array([settings["player_prior_equivalent_possessions"]]*n
                               +[hca["prior_equivalent_possessions"]], dtype=float)
    if not np.isfinite(prior_precision).all() or np.any(prior_precision <= 0):
        raise ValueError("Joint player/HCA prior strengths must be finite and positive.")
    precision[np.diag_indices(n+1)] += prior_precision
    rhs = np.array(score, dtype=float, copy=True)
    rhs[-1] += prior_precision[-1]*hca["prior_mean"]
    if not np.any(gram) and not np.any(score):
        # With no observed games, the posterior is exactly the initialization.
        estimates = np.zeros(n+1)
        estimates[-1] = float(hca["prior_mean"])
        variance = hca["observation_variance_per_100"]/prior_precision
    else:
        factor = cho_factor(precision, lower=True)
        estimates = cho_solve(factor, rhs)
        variance = hca["observation_variance_per_100"]*np.diag(cho_solve(factor, np.eye(n+1)))
    errors = np.sqrt(np.maximum(variance, 0))
    if not np.isfinite(estimates).all() or not np.isfinite(errors).all():
        raise ValueError("The joint ratings/HCA solve produced non-finite outputs.")
    information = float(gram[-1, -1])
    evidence_weight = information/(information+prior_precision[-1])
    return {"estimates": estimates, "standard_errors": errors,
            "hca_information": information, "hca_evidence_weight": evidence_weight}


def update_state(*, day: date, first_date: date, season: str, priors: dict, parent: dict | None,
                 games: list[dict], previous_games: list[dict], stints: dict | None, rosters: dict | None,
                 inputs: dict) -> dict:
    import numpy as np

    if day < first_date:
        raise ValueError("Ratings date precedes the configured season start.")
    observed_day = day
    initialization_only = False
    if day == first_date:
        if parent is not None:
            raise ValueError("First completed date must start from the frozen priors.")
    else:
        if parent is None:
            raise ValueError("The immediately preceding ratings state is required.")
        validate_state(parent, season)
        if (parent["game_date_pacific"] != str(day-timedelta(days=1))
                or parent["inputs"]["priors"] != inputs["priors"]
                or parent["settings"] != priors["settings"]):
            raise ValueError("The ratings parent has the wrong date or frozen initialization.")
    for data in (stints, rosters):
        if data is None or data["season"] != season or data["game_date_pacific"] != str(day):
            raise ValueError("Ratings inputs must belong to the selected completed game date.")
        check_source_policy(data)
    if any(g["game_time_utc"].astimezone(PACIFIC).date() != day for g in games):
        raise ValueError("Ratings inputs contain another game date.")
    expected = sorted(g["game_id"] for g in games)
    if stints is not None:
        coverage = stints["validation"]
        if (not coverage["complete"] or coverage["expected_game_ids"] != expected
                or coverage["processed_game_ids"] != expected):
            raise ValueError("Saved stints do not cover the complete expected slate.")
    rows = stints["rows"] if stints is not None else []
    rows_by_game = {game_id: [] for game_id in expected}
    for row in rows:
        if row["game_id"] not in rows_by_game:
            raise ValueError("Saved stints contain an unexpected game.")
        rows_by_game[row["game_id"]].append(row)
    catalog = {p["player_id"]: p for p in priors["players"]}
    prior_players = {p["player_id"]: p for p in parent["players"]} if parent else {}
    teams = rosters["teams"] if rosters is not None else {}
    team_by_player = {int(i): int(team) for team, ids in teams.items() for i in ids}
    if len(team_by_player) != sum(len(ids) for ids in teams.values()):
        raise ValueError("A selected roster assigns a player to multiple teams.")
    played = {int(i) for row in rows for side in ("home_player_ids", "away_player_ids") for i in row[side]}
    ids = sorted((prior_players.keys() if parent else catalog.keys()) | team_by_player.keys() | played)
    new_replacements = sorted(set(ids)-prior_players.keys()-catalog.keys())
    starting = {
        i: (prior_players[i]["starting_nail"] if i in prior_players else
            catalog[i]["rapm"] if i in catalog else priors["settings"]["replacement_prior"])
        for i in ids
    }
    n = len(ids)
    gram, score, exposure = np.zeros((n+1, n+1)), np.zeros(n+1), np.zeros(n+1)
    included = list(parent["included_game_ids"]) if parent else []
    if set(included)&set(expected):
        raise ValueError("A selected ratings parent already contains this date's games.")
    if parent:
        old = parent["statistics"]; columns = {player_id: i for i, player_id in enumerate(ids)}
        remap = [columns[i] for i in old["player_ids"]]+[n]
        gram[np.ix_(remap, remap)] = old["gram"]
        score[remap] = old["score"]
        exposure[remap] = old["exposure"]
    prior_day_teams = {g[side] for g in previous_games for side in ("home_team_id", "away_team_id")}
    diagnostics = {}
    for game in sorted(games, key=lambda g: (g["game_time_utc"], g["game_id"])):
        flags = {side: game[f"{side}_team_id"] in prior_day_teams for side in ("home", "away")}
        b2b = (int(flags["home"])-int(flags["away"]))*priors["settings"]["fixed_back_to_back"]
        contribution = game_contribution(
            rows_by_game[game["game_id"]], player_ids=ids, starting=starting,
            b2b=b2b,
        )
        gram += contribution["gram"]; score += contribution["score"]; exposure += contribution["exposure"]
        diagnostics[game["game_id"]] = {key: contribution[key] for key in
                                      ("positive_stint_count", "zero_possession_stint_count", "possessions", "fixed_b2b_offset")}
        diagnostics[game["game_id"]].update(
            back_to_back=flags, game_date_pacific=str(observed_day), game_time_utc=game["game_time_utc"].isoformat(),
        )
        included.append(game["game_id"])
    solution = solve_joint(gram, score, priors["settings"])
    players = []
    for index, player_id in enumerate(ids):
        original = prior_players.get(player_id)
        source = original or catalog.get(player_id)
        player = {
            "player_id": player_id, "player_name": source.get("player_name") if source else None,
            "team_id": team_by_player.get(player_id), "starting_nail": float(starting[player_id]),
            "forecast_source": source["forecast_source"] if source else "in_season_replacement_cold_start_prior",
            "initialized_on": original["initialized_on"] if original else day.isoformat(),
            "initialization": original["initialization"] if original else {
                "kind": "frozen_catalog" if player_id in catalog else "replacement",
                "input": inputs["priors"],
            },
            "residual": float(solution["estimates"][index]),
            "nail": float(starting[player_id]+solution["estimates"][index]),
            "on_court_possessions": float(exposure[index]), "information": float(gram[index, index]),
            "standard_error": float(solution["standard_errors"][index]),
        }
        players.append(player)
    record = {
        "schema_version": 1, "artifact_kind": "player_ratings", "season": season,
        "calculation_contract": CALCULATION_CONTRACT,
        "model_version": MODEL_VERSION, "game_date_pacific": day.isoformat(),
        "observed_through_game_date_pacific": str(observed_day),
        "date_contract": COMPLETED_GAME_DATE_CONTRACT,
        "effective_from_game_date_pacific": str(day+timedelta(days=1)),
        "model_id": priors["model_id"], "model_run_id": priors["model_run_id"],
        "implementation": implementation_identity(), "inputs": inputs, "settings": priors["settings"],
        "players": players, "included_game_ids": included,
        "player_initialization_policy": "retain_state_then_frozen_catalog_else_replacement",
        "new_replacement_player_ids": new_replacements,
        "hca": {"estimate": float(solution["estimates"][-1]),
                "standard_error": float(solution["standard_errors"][-1]),
                "information": solution["hca_information"],
                "prior_weight": 1.0-solution["hca_evidence_weight"],
                "evidence_weight": solution["hca_evidence_weight"]},
        "statistics": {"player_ids": ids, "hca_column": n, "gram": gram.tolist(),
                       "score": score.tolist(), "exposure": exposure.tolist()},
        "validation": {"complete": True, "expected_game_ids": expected,
                       "processed_game_ids": sorted(diagnostics), "games": diagnostics,
                       "rostered_player_count": None if initialization_only else len(team_by_player),
                       "retained_player_count": n, "initialization_only": initialization_only},
    }
    validate_state(record, season)
    return record


def validate_state(record: dict, season: str, *, allow_legacy_effective_date: bool = False) -> None:
    import numpy as np

    if (record.get("schema_version"), record.get("artifact_kind"), record.get("season"),
            record.get("model_version")) != (1, "player_ratings", season, MODEL_VERSION):
        raise ValueError("Invalid ratings state identity.")
    if record.get("calculation_contract") != CALCULATION_CONTRACT:
        raise ValueError("Ratings predecessor uses another calculation; a reviewed ratings-only replay is required.")
    data_day = date.fromisoformat(record["game_date_pacific"])
    if record.get("date_contract") == COMPLETED_GAME_DATE_CONTRACT:
        observed_day = data_day
        if record.get("effective_from_game_date_pacific") != str(data_day+timedelta(days=1)):
            raise ValueError("Completed-date ratings must become effective the following day.")
    elif allow_legacy_effective_date and record.get("date_contract") is None:
        observed_day = data_day-timedelta(days=1)
    else:
        raise ValueError("Ratings use another date contract; the approved completed-date replay is required.")
    if record.get("observed_through_game_date_pacific") != str(observed_day):
        raise ValueError("Ratings observation cutoff differs from the recorded date contract.")
    players = record["players"]; ids = [p["player_id"] for p in players]
    stats = record["statistics"]; n = len(ids)
    if not ids or ids != sorted(set(ids)) or stats["player_ids"] != ids or stats["hca_column"] != n:
        raise ValueError("Joint statistics and retained player identities differ.")
    gram, score, exposure = (np.asarray(stats[k], dtype=float) for k in ("gram", "score", "exposure"))
    if (gram.shape != (n+1, n+1) or score.shape != (n+1,) or exposure.shape != (n+1,)
            or not np.isfinite(gram).all() or not np.isfinite(score).all()
            or not np.isfinite(exposure).all() or np.any(exposure < 0)
            or not np.allclose(gram, gram.T, atol=1e-10, rtol=1e-12)):
        raise ValueError("Invalid resumable joint ratings matrix/vectors.")
    if len(set(record["included_game_ids"])) != len(record["included_game_ids"]):
        raise ValueError("Ratings history contains duplicate game evidence.")
    validation = record["validation"]
    if validation["initialization_only"]:
        if (record["included_game_ids"] or np.any(gram) or np.any(score) or np.any(exposure)
                or record["inputs"]["parent"] is not None
                or any(record["inputs"].get(k) is not None for k in ("stints", "rosters", "schedule"))
                or record["hca"]["estimate"] != record["settings"]["hca"]["prior_mean"]
                or any(p["nail"] != p["starting_nail"] for p in players)):
            raise ValueError("Opening-day ratings must contain initialization only.")
    if not validation["complete"] or validation["expected_game_ids"] != validation["processed_game_ids"]:
        raise ValueError("Ratings state does not cover its complete expected slate.")
    for p in players:
        values = [p[k] for k in ("starting_nail", "nail", "residual", "standard_error", "on_court_possessions", "information")]
        if not np.isfinite(values).all() or p["standard_error"] < 0:
            raise ValueError("A player has invalid ratings output.")
        if not math.isclose(p["nail"], p["starting_nail"]+p["residual"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("A player estimate does not match its frozen prior plus residual.")
    check_source_policy(record)
    json_bytes(record)
