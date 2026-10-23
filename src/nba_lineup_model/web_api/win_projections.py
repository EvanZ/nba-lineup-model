"""Frozen preseason team-strength and schedule win projections."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.season.schedule import SeasonScheduleCache
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    MODEL_ARTIFACT,
    TEAM_DISPLAY_NAMES,
    LineupEvaluator,
    _compiled_linear_x3_coefficients,
    _player_rating_center,
)
from nba_lineup_model.web_api.market_win_totals import (
    BETMGM_WIN_TOTALS_2026_27,
    BETMGM_WIN_TOTALS_2026_27_METADATA,
)

DEFAULT_CATALOG_PATH = Path("data/catalog/games.parquet")
CALIBRATION_SEASON = "2024-25"
HOLDOUT_SEASON = "2025-26"
PROJECTION_SEASON = "2026-27"
UNASSIGNED_REGULAR_GAMES = 2
WIN_TOTAL_INTERVAL_TRIALS = 10_000
WIN_TOTAL_INTERVAL_SEED = 20_260_910
DEFAULT_WIN_PROJECTION_CACHE_DIR = Path("artifacts/web/win_projections")
DEFAULT_SCHEDULE_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_SCHEDULE_API_URL = "http://127.0.0.1:8010/api/win-projections"
DEFAULT_SCHEDULE_S3_URI = "s3://nba-gestalt-data/nba-gestalt/schedule"
DEFAULT_SCHEDULE_DAYS_AHEAD = 7
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
POLYMARKET_EVENT_URL = "https://gamma-api.polymarket.com/events/slug"
POLYMARKET_MIDPOINTS_URL = "https://clob.polymarket.com/midpoints"
ESPN_TEAM_ALIASES = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UT": "UTA",
    "UTAH": "UTA",
    "WSH": "WAS",
}
INJURY_STATUS_PLAY_PROBABILITY = {
    "Out": 0.0,
    "Doubtful": 0.25,
    "Questionable": 0.5,
    "Probable": 0.75,
    "Available": 1.0,
}


def build_win_projection_payload(
    *,
    evaluator: LineupEvaluator,
    minutes_payload: dict[str, object],
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> dict[str, object]:
    """Return an auditable 2026-27 baseline win projection.

    Player NAIL values and minute allocations remain frozen.  The only fitted
    win layer is a one-parameter logistic calibration trained on 2024-25,
    then tested without refitting on 2025-26.  The production scale is refit
    after that holdout on the two completed preseason snapshots.
    """

    catalog = pd.read_parquet(catalog_path)
    calibration = _frozen_calibration(evaluator=evaluator, catalog=catalog)
    current_strength = _current_team_strength(minutes_payload)
    controls = evaluator.schedule_controls
    if controls is None:
        raise ValueError("Published NAIL artifact lacks schedule controls")
    games = _season_games(catalog, PROJECTION_SEASON)
    game_rows = _score_games(
        games,
        team_strength=current_strength,
        beta=calibration["production_beta"],
        home_court=float(controls.home_court),
        back_to_back=float(controls.back_to_back),
    )
    teams = _team_win_totals(
        game_rows,
        team_strength=current_strength,
        beta=calibration["production_beta"],
        home_court=float(controls.home_court),
    )
    teams["betmgm_win_total"] = teams["team"].map(BETMGM_WIN_TOTALS_2026_27)
    if teams["betmgm_win_total"].isna().any():
        missing = teams.loc[teams["betmgm_win_total"].isna(), "team"].tolist()
        raise ValueError(f"Missing BetMGM win totals for current teams: {missing}")
    intervals = simulate_baseline_win_total_intervals(
        game_rows,
        team_strength=current_strength,
        beta=calibration["production_beta"],
        home_court=float(controls.home_court),
    )
    teams = teams.merge(intervals, on="team", how="left", validate="one_to_one")
    return {
        "model": "W0 minute-weighted NAIL",
        "season": PROJECTION_SEASON,
        "scheduled_game_count": int(len(games)),
        "scheduled_games_per_team": int(len(games) * 2 / len(teams)),
        "unassigned_regular_games_per_team": UNASSIGNED_REGULAR_GAMES,
        "win_probability_scale": float(calibration["production_beta"]),
        "home_court": float(controls.home_court),
        "back_to_back": float(controls.back_to_back),
        "win_total_interval_trials": WIN_TOTAL_INTERVAL_TRIALS,
        "calibration": calibration,
        "market_win_totals": BETMGM_WIN_TOTALS_2026_27_METADATA,
        "teams": teams.to_dict(orient="records"),
        "scheduled_games": game_rows.assign(
            game_date=lambda frame: pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m-%d"),
            home_back_to_back=lambda frame: frame["home_back_to_back"].astype(int),
            away_back_to_back=lambda frame: frame["away_back_to_back"].astype(int),
        )
        .loc[
            :,
            [
                "game_id",
                "game_date",
                "home_team_tricode",
                "away_team_tricode",
                "home_back_to_back",
                "away_back_to_back",
            ],
        ]
        .rename(
            columns={
                "home_team_tricode": "home_team",
                "away_team_tricode": "away_team",
            }
        )
        .to_dict(orient="records"),
        "contract": (
            "Team strength is the minute-weighted sum of frozen player NAIL forecasts. "
            "Scheduled games use the official home-court and back-to-back controls. "
            "The NBA has published 80 regular-season games per team; two unassigned "
            "games are estimated against a neutral opponent with one home and one away setting."
        ),
    }


def win_projection_cache_path(model_artifact: str, run_id: str) -> Path:
    """Return the immutable release path for a baseline W0 payload."""

    return DEFAULT_WIN_PROJECTION_CACHE_DIR / model_artifact / f"{run_id}.json"


def materialize_win_projection_cache(
    *,
    evaluator: LineupEvaluator,
    output_path: Path | str | None = None,
) -> Path:
    """Build the all-team baseline envelope once for a release bundle."""

    from nba_lineup_model.web_api.inference import MODEL_ARTIFACT
    from nba_lineup_model.web_api.preseason_minutes import (
        build_forward_conditional_preseason_minutes_payload,
    )

    minutes = build_forward_conditional_preseason_minutes_payload(
        preseason_rankings=evaluator.preseason_rankings,
    )
    payload = {
        "minutes": minutes,
        "win_projection": build_win_projection_payload(
            evaluator=evaluator,
            minutes_payload=minutes,
        ),
    }
    path = Path(output_path) if output_path is not None else win_projection_cache_path(
        MODEL_ARTIFACT, evaluator.run_id
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def read_win_projection_cache(*, model_artifact: str, run_id: str) -> dict[str, object]:
    """Read the release-materialized W0 payload without a request-path replay."""

    path = win_projection_cache_path(model_artifact, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing materialized win projection cache: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid win projection cache payload: {path}")
    return payload


def simulate_baseline_win_total_intervals(
    games: pd.DataFrame,
    *,
    team_strength: pd.Series,
    beta: float,
    home_court: float,
    trials: int = WIN_TOTAL_INTERVAL_TRIALS,
    seed: int = WIN_TOTAL_INTERVAL_SEED,
) -> pd.DataFrame:
    """Simulate final win-total quantiles for every baseline team schedule.

    Each team gets an independent deterministic draw stream. This mirrors the
    public envelope's within-team Bernoulli contract while making the published
    P5--P95 table values reproducible in the release bundle.
    """

    if trials <= 0:
        raise ValueError("Win-total interval trials must be positive")
    required = {"home_team_tricode", "away_team_tricode", "home_win_probability"}
    missing = sorted(required - set(games))
    if missing:
        raise ValueError("Win-total intervals require: " + ", ".join(missing))

    probabilities: dict[str, list[float]] = {str(team): [] for team in team_strength.index}
    for row in games.loc[
        :, ["home_team_tricode", "away_team_tricode", "home_win_probability"]
    ].itertuples(index=False):
        home_team = str(row.home_team_tricode)
        away_team = str(row.away_team_tricode)
        home_probability = float(row.home_win_probability)
        probabilities[home_team].append(home_probability)
        probabilities[away_team].append(1.0 - home_probability)

    for team, strength in team_strength.items():
        probabilities[str(team)].extend(
            [
                float(_sigmoid(beta * (float(strength) + home_court))),
                float(_sigmoid(beta * (float(strength) - home_court))),
            ]
        )

    rows: list[dict[str, float | str]] = []
    for team in sorted(probabilities):
        team_probabilities = np.asarray(probabilities[team], dtype=float)
        if len(team_probabilities) != 82:
            raise ValueError(
                f"Expected 82 modeled games for {team}; found {len(team_probabilities)}"
            )
        team_seed = seed + sum(
            (index + 1) * ord(character) for index, character in enumerate(team)
        )
        generator = np.random.default_rng(team_seed)
        totals = (
            generator.random((trials, len(team_probabilities))) < team_probabilities
        ).sum(axis=1)
        ordered = np.sort(totals)
        rows.append(
            {
                "team": team,
                "win_total_p1": float(_empirical_quantile(ordered, 0.01)),
                "win_total_p5": float(_empirical_quantile(ordered, 0.05)),
                "win_total_p50": float(_empirical_quantile(ordered, 0.50)),
                "win_total_p95": float(_empirical_quantile(ordered, 0.95)),
                "win_total_p99": float(_empirical_quantile(ordered, 0.99)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _empirical_quantile(ordered_values: np.ndarray, probability: float) -> int:
    """Match the public envelope's nearest-upper empirical percentile rule."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Empirical percentile must lie between zero and one")
    index = int(np.ceil(probability * (len(ordered_values) - 1)))
    return int(ordered_values[index])


def _json_default(value: Any) -> object:
    """Serialize NumPy scalars without changing the cache's JSON contract."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _frozen_calibration(*, evaluator: LineupEvaluator, catalog: pd.DataFrame) -> dict[str, float]:
    calibration = _historical_game_rows(
        evaluator=evaluator, catalog=catalog, season=CALIBRATION_SEASON
    )
    calibration_beta = fit_win_probability_scale(calibration["edge"], calibration["home_win"])
    holdout = _historical_game_rows(evaluator=evaluator, catalog=catalog, season=HOLDOUT_SEASON)
    holdout_metrics = score_win_probabilities(
        holdout["edge"], holdout["home_win"], beta=calibration_beta
    )
    combined = pd.concat([calibration, holdout], ignore_index=True)
    production_beta = fit_win_probability_scale(combined["edge"], combined["home_win"])
    return {
        "calibration_season": CALIBRATION_SEASON,
        "holdout_season": HOLDOUT_SEASON,
        "calibration_beta": calibration_beta,
        "holdout_brier": holdout_metrics["brier"],
        "holdout_log_loss": holdout_metrics["log_loss"],
        "holdout_accuracy": holdout_metrics["accuracy"],
        "production_beta": production_beta,
    }


def _historical_game_rows(
    *, evaluator: LineupEvaluator,
    catalog: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    strength = _historical_team_strength(evaluator=evaluator, season=season)
    controls = _historical_schedule_controls(evaluator=evaluator, target_season=season)
    games = _season_games(catalog, season)
    predicted = _score_games(
        games,
        team_strength=strength,
        beta=1.0,
        home_court=controls["home_court"],
        back_to_back=controls["back_to_back"],
    )
    outcomes = _actual_home_outcomes(season)
    output = predicted.merge(outcomes, on="game_id", how="inner", validate="one_to_one")
    if len(output) != len(games):
        raise ValueError(f"Missing completed game outcomes for {season}")
    return output


def _historical_team_strength(*, evaluator: LineupEvaluator, season: str) -> pd.Series:
    run_dir = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / evaluator.season / evaluator.run_id
    priors = pd.read_parquet(run_dir / "season_player_priors.parquet")
    priors = priors.loc[priors["season"].eq(season), ["player_id", "prior_rapm"]].rename(
        columns={"prior_rapm": "rapm"}
    )
    profiles = evaluator.historical_profiles.loc[
        evaluator.historical_profiles["season"].eq(season)
    ].copy()
    prior_season = _previous_season(season)
    context_model = evaluator.season_context_models[prior_season]
    exposures = evaluator.exposure_cohort.loc[
        evaluator.exposure_cohort["season"].eq(prior_season),
        ["player_id", "on_court_possessions"],
    ].rename(columns={"on_court_possessions": "possessions"})
    uncentered = _compiled_linear_x3_coefficients(priors, profiles, context_model)
    center = _player_rating_center(uncentered, exposures)
    ratings = _compiled_linear_x3_coefficients(priors, profiles, context_model, center=center)
    rating_by_player = ratings.set_index("player_id")["rapm"]
    candidates = pd.read_parquet(
        Path("data/curated/opening_rosters") / season / "part-00000.parquet"
    ).copy()
    previous_minutes = _prior_minutes(_previous_season(season))
    candidates["player_id"] = candidates["player_id"].astype(int)
    candidates["score"] = candidates["player_id"].map(previous_minutes).fillna(0.0)
    candidates["rating"] = candidates["player_id"].map(rating_by_player).fillna(0.0)
    candidates["share"] = candidates.groupby("team", sort=False)["score"].transform(
        lambda values: _smoothed_shares(values, epsilon=0.32)
    )
    strength = candidates.assign(weight=lambda frame: 5.0 * frame["share"]).groupby(
        "team", sort=True
    ).apply(lambda frame: float((frame["weight"] * frame["rating"]).sum()), include_groups=False)
    return strength - float(strength.mean())


def _current_team_strength(minutes_payload: dict[str, object]) -> pd.Series:
    players = pd.DataFrame(minutes_payload["players"])
    strength = players.assign(
        weight=lambda frame: frame["baseline_minutes_per_game"].astype(float) / 48.0
    ).groupby("team", sort=True).apply(
        lambda frame: float((frame["weight"] * frame["projected_nail"]).sum()),
        include_groups=False,
    )
    return strength - float(strength.mean())


def _historical_schedule_controls(
    *, evaluator: LineupEvaluator, target_season: str
) -> dict[str, float]:
    run_dir = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / evaluator.season / evaluator.run_id
    context = pd.read_parquet(run_dir / "season_context_metadata.parquet")
    schedule = pd.read_parquet(run_dir / "season_schedule_control_metadata.parquet")
    completed = context.loc[context["season"].astype(str).lt(target_season)].copy()
    completed_schedule = schedule.loc[schedule["season"].astype(str).lt(target_season)].copy()
    if completed.empty or completed_schedule.empty:
        raise ValueError(f"No completed schedule-control states before {target_season}")
    return {
        "home_court": float(
            np.average(
                completed["context_home_intercept"],
                weights=completed["context_training_weight_sum"],
            )
        ),
        "back_to_back": float(
            np.average(
                completed_schedule["schedule_control_raw_weight"],
                weights=completed_schedule["schedule_training_stint_count"],
            )
        ),
    }


def _season_games(catalog: pd.DataFrame, season: str) -> pd.DataFrame:
    games = catalog.loc[
        catalog["season"].eq(season) & catalog["season_type"].eq("regular"),
        [
            "game_id",
            "season",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
        ],
    ].copy()
    features = build_back_to_back_game_features(games)
    return games.merge(
        features.loc[:, ["game_id", "home_back_to_back", "away_back_to_back"]],
        on="game_id", how="left", validate="one_to_one"
    )


def _score_games(
    games: pd.DataFrame,
    *, team_strength: pd.Series,
    beta: float,
    home_court: float,
    back_to_back: float,
) -> pd.DataFrame:
    output = games.copy()
    output["home_strength"] = output["home_team_tricode"].map(team_strength)
    output["away_strength"] = output["away_team_tricode"].map(team_strength)
    if output[["home_strength", "away_strength"]].isna().any().any():
        missing = sorted(
            set(output.loc[output["home_strength"].isna(), "home_team_tricode"])
            | set(output.loc[output["away_strength"].isna(), "away_team_tricode"])
        )
        raise ValueError("Team-strength map is missing schedule teams: " + ", ".join(missing))
    output["edge"] = (
        output["home_strength"] - output["away_strength"] + home_court
        + back_to_back * (output["home_back_to_back"] - output["away_back_to_back"])
    )
    output["home_win_probability"] = _sigmoid(beta * output["edge"].to_numpy(dtype=float))
    return output


def _team_win_totals(
    games: pd.DataFrame,
    *, team_strength: pd.Series,
    beta: float,
    home_court: float,
) -> pd.DataFrame:
    home = games.loc[:, ["home_team_tricode", "home_win_probability"]].rename(
        columns={"home_team_tricode": "team", "home_win_probability": "scheduled_wins"}
    )
    away = games.loc[:, ["away_team_tricode", "home_win_probability"]].rename(
        columns={"away_team_tricode": "team"}
    )
    away["scheduled_wins"] = 1.0 - away.pop("home_win_probability")
    scheduled = pd.concat([home, away], ignore_index=True).groupby("team", as_index=False).agg(
        scheduled_wins=("scheduled_wins", "sum"), scheduled_games=("scheduled_wins", "size")
    )
    strengths = pd.DataFrame(
        {"team": team_strength.index, "team_strength": team_strength.values}
    )
    output = strengths.merge(
        scheduled, on="team", how="left", validate="one_to_one"
    )
    neutral_home = _sigmoid(beta * (output["team_strength"].to_numpy(dtype=float) + home_court))
    neutral_away = _sigmoid(beta * (output["team_strength"].to_numpy(dtype=float) - home_court))
    raw_unassigned_wins = UNASSIGNED_REGULAR_GAMES * (neutral_home + neutral_away) / 2.0
    # The two unresolved Cup games must still contribute exactly 30 league wins.
    # Preserve teams' relative neutral-opponent differences, then center their
    # aggregate expectation to one win per team.
    output["unassigned_wins"] = (
        raw_unassigned_wins
        - float(raw_unassigned_wins.mean())
        + UNASSIGNED_REGULAR_GAMES / 2.0
    )
    output["projected_wins"] = output["scheduled_wins"] + output["unassigned_wins"]
    output["projected_losses"] = 82.0 - output["projected_wins"]
    return output.sort_values(
        ["projected_wins", "team"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def fit_win_probability_scale(edges: pd.Series, outcomes: pd.Series) -> float:
    """Fit one no-intercept logistic scale by stable Newton updates."""

    x = np.asarray(edges, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if (
        len(x) == 0
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isin(y, [0.0, 1.0]).all()
    ):
        raise ValueError("Win calibration requires finite edges and binary outcomes")
    beta = 0.1
    for _ in range(100):
        p = _sigmoid(beta * x)
        gradient = float(np.sum((p - y) * x))
        hessian = float(np.sum(p * (1.0 - p) * np.square(x)))
        if hessian <= 1e-12:
            break
        updated = max(0.0, beta - gradient / hessian)
        if abs(updated - beta) <= 1e-10:
            beta = updated
            break
        beta = updated
    return float(beta)


def score_win_probabilities(
    edges: pd.Series, outcomes: pd.Series, *, beta: float
) -> dict[str, float]:
    p = _sigmoid(float(beta) * np.asarray(edges, dtype=float))
    y = np.asarray(outcomes, dtype=float)
    return {
        "brier": float(np.mean(np.square(p - y))),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1.0))),
    }


def build_frozen_season_win_backtest(
    *,
    evaluator: LineupEvaluator,
    catalog: pd.DataFrame,
    target_seasons: list[str],
) -> pd.DataFrame:
    """Create strict preseason W0 win totals for completed target seasons.

    The probability scale for season ``t`` uses only completed game results
    from ``t - 1``.  Player forecasts, minute shares, and schedule controls
    are already frozen by :func:`_historical_game_rows`.
    """

    frames: list[pd.DataFrame] = []
    for target_season in target_seasons:
        calibration_season = _previous_season(target_season)
        calibration = _historical_game_rows(
            evaluator=evaluator, catalog=catalog, season=calibration_season
        )
        beta = fit_win_probability_scale(calibration["edge"], calibration["home_win"])
        target = _historical_game_rows(
            evaluator=evaluator, catalog=catalog, season=target_season
        )
        totals = _completed_team_win_totals(target, beta=beta)
        if len(totals) != 30 or not totals["scheduled_games"].eq(82).all():
            raise ValueError(f"Incomplete completed-season schedule: {target_season}")
        totals.insert(0, "season", target_season)
        totals["calibration_season"] = calibration_season
        totals["win_probability_scale"] = beta
        frames.append(totals)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["season", "team"], kind="stable"
    ).reset_index(drop=True)


def score_season_win_forecasts(
    forecasts: pd.DataFrame, *, prediction_column: str
) -> dict[str, float]:
    """Return team-season point-forecast error measures."""

    required = {prediction_column, "actual_wins"}
    missing = sorted(required - set(forecasts))
    if missing:
        raise ValueError("Season forecast scores require: " + ", ".join(missing))
    error = forecasts[prediction_column].astype(float) - forecasts["actual_wins"].astype(float)
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
    }


def _actual_home_outcomes(season: str) -> pd.DataFrame:
    """Read completed regular-season results from the official schedule cache.

    The schedule endpoint is authoritative for final scores and covers the
    complete slate.  Player box-score partitions are intentionally not used
    here because historical player coverage can be incomplete even where the
    schedule result is known.
    """

    response = SeasonScheduleCache().read(season)
    if response is None:
        raise FileNotFoundError(f"No cached official NBA schedule for {season}")

    records: list[dict[str, object]] = []
    game_dates = response.payload.get("leagueSchedule", {}).get("gameDates", [])
    for game_date in game_dates:
        for game in game_date.get("games", []):
            game_id = str(game.get("gameId", ""))
            if not game_id.startswith("002") or int(game.get("gameStatus", 0)) != 3:
                continue
            home_score = game.get("homeTeam", {}).get("score")
            away_score = game.get("awayTeam", {}).get("score")
            if home_score is None or away_score is None:
                raise ValueError(f"Final schedule game lacks a score: {game_id}")
            if home_score == away_score:
                raise ValueError(f"Tied regular-season outcome in {season}: {game_id}")
            records.append({"game_id": game_id, "home_win": float(home_score > away_score)})

    outcomes = pd.DataFrame.from_records(records)
    if outcomes.empty:
        raise ValueError(f"No completed regular-season outcomes in schedule cache: {season}")
    if outcomes["game_id"].duplicated().any():
        raise ValueError(f"Duplicate regular-season outcome in schedule cache: {season}")
    return outcomes


def _completed_team_win_totals(games: pd.DataFrame, *, beta: float) -> pd.DataFrame:
    required = {"home_team_tricode", "away_team_tricode", "edge", "home_win"}
    missing = sorted(required - set(games))
    if missing:
        raise ValueError("Completed team win totals require: " + ", ".join(missing))
    probabilities = _sigmoid(beta * games["edge"].to_numpy(dtype=float))
    home = pd.DataFrame(
        {
            "team": games["home_team_tricode"].to_numpy(),
            "projected_wins": probabilities,
            "actual_wins": games["home_win"].to_numpy(dtype=float),
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team_tricode"].to_numpy(),
            "projected_wins": 1.0 - probabilities,
            "actual_wins": 1.0 - games["home_win"].to_numpy(dtype=float),
        }
    )
    totals = pd.concat([home, away], ignore_index=True).groupby("team", as_index=False).agg(
        projected_wins=("projected_wins", "sum"),
        actual_wins=("actual_wins", "sum"),
        scheduled_games=("projected_wins", "size"),
    )
    return totals


def _prior_minutes(season: str) -> pd.Series:
    from nba_lineup_model.rotation.l1_minute_share_persistence import read_regular_game_minutes

    return read_regular_game_minutes(season).groupby("player_id")["minutes"].sum()


def _smoothed_shares(scores: pd.Series, *, epsilon: float) -> pd.Series:
    values = scores.to_numpy(dtype=float)
    if float(values.sum()) > 0.0:
        normalized = values / values.sum()
    else:
        normalized = np.full(len(values), 1.0 / len(values))
    return pd.Series((1.0 - epsilon) * normalized + epsilon / len(values), index=scores.index)


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def read_schedule_day(
    game_date: str,
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
) -> dict[str, object]:
    """Read the latest board with immutable pregame and final audit records."""

    path = Path(state_dir) / "latest" / f"{game_date}.json"
    if not path.is_file():
        return {
            "available": False,
            "game_date": game_date,
            "source": "ESPN",
            "games": [],
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid schedule archive payload: {path}")
    games = payload.get("games", [])
    if not isinstance(games, list):
        raise ValueError(f"Invalid schedule games payload: {path}")
    for game in games:
        if not isinstance(game, dict):
            continue
        frozen_path = Path(state_dir) / "frozen" / f"{game.get('espn_game_id')}.json"
        if frozen_path.is_file():
            frozen = json.loads(frozen_path.read_text())
            if isinstance(frozen, dict):
                game["pregame"] = frozen
        final_path = (
            Path(state_dir) / "final" / f"{game.get('espn_game_id')}.summary.json"
        )
        if final_path.is_file():
            final_audit = json.loads(final_path.read_text())
            if isinstance(final_audit, dict):
                game["final_audit"] = final_audit
    payload["available"] = True
    return payload


def read_current_season_records(
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    season: str = PROJECTION_SEASON,
) -> dict[str, object]:
    """Aggregate current regular-season records from immutable final audits."""

    records: dict[str, dict[str, object]] = {}
    latest_game_date: str | None = None
    latest_archive_time: str | None = None
    final_dir = Path(state_dir) / "final"
    if final_dir.is_dir():
        for path in sorted(final_dir.glob("*.summary.json")):
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict) or payload.get("season") != season:
                continue
            game_id = str(payload.get("game_id", ""))
            if not game_id.startswith("002"):
                continue
            final = payload.get("final")
            if not isinstance(final, dict):
                continue
            status = final.get("status")
            if isinstance(status, dict) and status.get("completed") is not True:
                continue
            home_score = _integer_or_none(final.get("home_score"))
            away_score = _integer_or_none(final.get("away_score"))
            if home_score is None or away_score is None or home_score == away_score:
                continue
            home_team = str(payload.get("home_team", ""))
            away_team = str(payload.get("away_team", ""))
            if not home_team or not away_team or home_team == away_team:
                continue
            start_time = str(payload.get("start_time", ""))
            try:
                game_date = (
                    datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    .astimezone(ZoneInfo("America/New_York"))
                    .date()
                    .isoformat()
                )
            except ValueError:
                continue
            winner = home_team if home_score > away_score else away_team
            for team in (away_team, home_team):
                record = records.setdefault(
                    team,
                    {
                        "team": team,
                        "games": 0,
                        "wins": 0,
                        "losses": 0,
                        "last_game_date": game_date,
                    },
                )
                record["games"] = int(record["games"]) + 1
                if team == winner:
                    record["wins"] = int(record["wins"]) + 1
                else:
                    record["losses"] = int(record["losses"]) + 1
                record["last_game_date"] = max(str(record["last_game_date"]), game_date)
            latest_game_date = max(latest_game_date or game_date, game_date)
            archived_at = payload.get("archived_at")
            if isinstance(archived_at, str):
                latest_archive_time = max(latest_archive_time or archived_at, archived_at)
    return {
        "season": season,
        "as_of": latest_game_date,
        "updated_at": latest_archive_time,
        "source": "Immutable completed-game archive (ESPN scoreboard)",
        "teams": sorted(records.values(), key=lambda item: str(item["team"])),
    }


def refresh_schedule_archive(
    *,
    game_date: str,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    projection_url: str = DEFAULT_SCHEDULE_API_URL,
    s3_uri: str | None = DEFAULT_SCHEDULE_S3_URI,
    source_payload: dict[str, object] | None = None,
    injury_report: dict[str, object] | None = None,
    injury_pdf: bytes | None = None,
    forecast_response: dict[str, object] | None = None,
    prediction_market_snapshot: dict[str, object] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Archive one ESPN board, normalize it, and freeze the final pregame view.

    The half-hour collector freezes the first snapshot at or after T-30 minutes.
    If a game turns live between runs, the preceding pregame snapshot is frozen.
    """

    collected_at = (now or datetime.now(UTC)).astimezone(UTC)
    state_root = Path(state_dir)
    latest_path = state_root / "latest" / f"{game_date}.json"
    previous = None
    if latest_path.is_file():
        candidate = json.loads(latest_path.read_text())
        if isinstance(candidate, dict):
            previous = candidate

    raw = source_payload or _fetch_json(f"{ESPN_SCOREBOARD_URL}?dates={game_date.replace('-', '')}")
    forecast_response = forecast_response or _fetch_json(projection_url)
    forecast = forecast_response.get("win_projection", forecast_response)
    minutes_payload = forecast_response.get("minutes", forecast_response)
    if not isinstance(forecast, dict):
        raise ValueError("Win projection endpoint did not return an object")
    if not isinstance(minutes_payload, dict):
        raise ValueError("Win projection endpoint did not return player minutes")
    if injury_report is None:
        injury_report, injury_pdf = _fetch_latest_official_injury_report(
            report_date=game_date,
            now=collected_at,
        )
    if prediction_market_snapshot is None:
        prediction_market_snapshot = _fetch_polymarket_moneyline_snapshot(
            raw,
            game_date=game_date,
            collected_at=collected_at,
        )
    payload = _normalize_espn_schedule(
        raw,
        forecast=forecast,
        minutes=minutes_payload,
        injury_report=injury_report,
        prediction_markets=prediction_market_snapshot.get("games", {}),
        game_date=game_date,
        collected_at=collected_at,
    )
    previous_games = {
        str(game.get("espn_game_id")): game
        for game in (previous or {}).get("games", [])
        if isinstance(game, dict)
    }
    for game in payload["games"]:
        if not isinstance(game, dict):
            continue
        prior = previous_games.get(str(game.get("espn_game_id")))
        if isinstance(prior, dict) and isinstance(prior.get("retrospective"), dict):
            game["retrospective"] = prior["retrospective"]
        if (
            game.get("prediction_market") is None
            and isinstance(prior, dict)
            and isinstance(prior.get("prediction_market"), dict)
        ):
            game["prediction_market"] = {
                **prior["prediction_market"],
                "carried_forward": True,
            }

    timestamp = collected_at.strftime("%Y%m%dT%H%M%SZ")
    archive_dir = state_root / "archive" / game_date
    raw_path = archive_dir / f"{timestamp}.espn.json"
    injury_path = archive_dir / f"{timestamp}.injury.pdf"
    prediction_market_path = archive_dir / f"{timestamp}.polymarket.json"
    normalized_path = archive_dir / f"{timestamp}.json"
    _write_json(raw_path, raw)
    if injury_pdf is not None:
        injury_path.parent.mkdir(parents=True, exist_ok=True)
        injury_path.write_bytes(injury_pdf)
    _write_json(prediction_market_path, prediction_market_snapshot)
    _write_json(normalized_path, payload)

    frozen_count = 0
    for game in payload["games"]:
        if not isinstance(game, dict):
            continue
        frozen_path = state_root / "frozen" / f"{game['espn_game_id']}.json"
        if frozen_path.exists():
            continue
        start = datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
        state = str(game.get("status", {}).get("state", "pre"))
        if collected_at < start - timedelta(minutes=30) and state == "pre":
            continue
        prior = previous_games.get(str(game["espn_game_id"]))
        frozen_game = (
            prior
            if isinstance(prior, dict) and prior.get("status", {}).get("state") == "pre"
            else game
        )
        frozen = {
            **frozen_game,
            "frozen_at": collected_at.isoformat().replace("+00:00", "Z"),
            "freeze_policy": "last half-hour pregame capture",
            "late_freeze": frozen_game is game and state != "pre",
        }
        _write_json_exclusive(frozen_path, frozen)
        frozen_count += 1

    _write_json(latest_path, payload)
    if s3_uri:
        prefix = f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}/game_date={game_date}"
        _aws_s3_cp_exclusive(raw_path, f"{prefix}/snapshots/{timestamp}.espn.json")
        if injury_path.is_file():
            _aws_s3_cp_exclusive(
                injury_path,
                f"{prefix}/snapshots/{timestamp}.injury.pdf",
            )
        _aws_s3_cp_exclusive(
            prediction_market_path,
            f"{prefix}/snapshots/{timestamp}.polymarket.json",
        )
        _aws_s3_cp_exclusive(normalized_path, f"{prefix}/snapshots/{timestamp}.json")
        _aws_s3_cp(latest_path, f"{prefix}/latest.json")
        for game in payload["games"]:
            if not isinstance(game, dict):
                continue
            frozen_path = state_root / "frozen" / f"{game['espn_game_id']}.json"
            if frozen_path.is_file():
                _aws_s3_cp_exclusive(
                    frozen_path,
                    f"{prefix}/frozen/{game['espn_game_id']}.json",
                )
            final_path = state_root / "final" / f"{game['espn_game_id']}.summary.json"
            if final_path.is_file():
                _aws_s3_cp_exclusive(
                    final_path,
                    f"{prefix}/final/{game['espn_game_id']}.summary.json",
                )
            final_espn_path = state_root / "final" / f"{game['espn_game_id']}.espn.json"
            if final_espn_path.is_file():
                _aws_s3_cp_exclusive(
                    final_espn_path,
                    f"{prefix}/final/{game['espn_game_id']}.espn.json",
                )
    return {**payload, "newly_frozen_game_count": frozen_count}


def refresh_schedule_window(
    *,
    start_date: str,
    days_ahead: int = DEFAULT_SCHEDULE_DAYS_AHEAD,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    projection_url: str = DEFAULT_SCHEDULE_API_URL,
    s3_uri: str | None = DEFAULT_SCHEDULE_S3_URI,
    now: datetime | None = None,
) -> dict[str, object]:
    """Archive every available ESPN board in a rolling date window.

    ESPN publishes future boards and market lines before game day, while the
    NBA injury-report filename reflects its publication date rather than the
    dates of every game listed in the document. Fetch the current report and
    projection payload once, then apply that information to each available
    board. Individual games still freeze only at the existing T-30 boundary.
    """

    if days_ahead < 0:
        raise ValueError("Schedule days ahead must be non-negative")
    collected_at = (now or datetime.now(UTC)).astimezone(UTC)
    first_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    eastern_date = collected_at.astimezone(ZoneInfo("America/New_York")).date()
    report_date = min(first_date, eastern_date).isoformat()
    injury_report, injury_pdf = _fetch_latest_official_injury_report(
        report_date=report_date,
        now=collected_at,
    )
    forecast_response = _fetch_json(projection_url)
    archived_days: list[dict[str, object]] = []
    total_games = 0
    newly_frozen = 0
    for offset in range(days_ahead + 1):
        game_date = (first_date + timedelta(days=offset)).isoformat()
        raw = _fetch_json(f"{ESPN_SCOREBOARD_URL}?dates={game_date.replace('-', '')}")
        events = raw.get("events", [])
        if not isinstance(events, list) or not events:
            continue
        prediction_market_snapshot = _fetch_polymarket_moneyline_snapshot(
            raw,
            game_date=game_date,
            collected_at=collected_at,
        )
        payload = refresh_schedule_archive(
            game_date=game_date,
            state_dir=state_dir,
            projection_url=projection_url,
            s3_uri=s3_uri,
            source_payload=raw,
            injury_report=injury_report,
            injury_pdf=injury_pdf,
            forecast_response=forecast_response,
            prediction_market_snapshot=prediction_market_snapshot,
            now=collected_at,
        )
        game_count = len(payload["games"])
        frozen_count = int(payload["newly_frozen_game_count"])
        total_games += game_count
        newly_frozen += frozen_count
        archived_days.append(
            {
                "game_date": game_date,
                "game_count": game_count,
                "newly_frozen_game_count": frozen_count,
            }
        )
    return {
        "start_date": start_date,
        "days_ahead": days_ahead,
        "report_date": report_date,
        "archived_day_count": len(archived_days),
        "game_count": total_games,
        "newly_frozen_game_count": newly_frozen,
        "days": archived_days,
    }


def _fetch_latest_official_injury_report(
    *, report_date: str, now: datetime
) -> tuple[dict[str, object], bytes]:
    eastern = now.astimezone(ZoneInfo("America/New_York"))
    requested_date = datetime.strptime(report_date, "%Y-%m-%d").date()
    if requested_date > eastern.date():
        raise ValueError(f"Official injury report is not available yet for {report_date}")
    if requested_date == eastern.date():
        cursor = eastern.replace(
            minute=(eastern.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
    else:
        cursor = datetime.combine(
            requested_date,
            datetime.max.time().replace(hour=23, minute=45, second=0, microsecond=0),
            tzinfo=ZoneInfo("America/New_York"),
        )
    for _ in range(96):
        if cursor.date() != requested_date:
            break
        suffix = cursor.strftime("%I_%M%p")
        url = (
            "https://ak-static.cms.nba.com/referee/injury/"
            f"Injury-Report_{report_date}_{suffix}.pdf"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nba-gestalt/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                pdf = response.read()
            report = _parse_official_injury_report(pdf)
            report.update(
                {
                    "source_url": url,
                    "published_at": cursor.isoformat(),
                }
            )
            return report, pdf
        except urllib.error.HTTPError as error:
            if error.code not in {403, 404}:
                raise
        cursor -= timedelta(minutes=15)
    raise FileNotFoundError(f"No official NBA injury report found for {report_date}")


def _parse_official_injury_report(pdf: bytes) -> dict[str, object]:
    from pypdf import PdfReader

    statuses: list[dict[str, object]] = []
    team_reports: dict[str, dict[str, object]] = {}
    current_matchup: tuple[str, str] | None = None
    current_team_position = -1
    for page in PdfReader(io.BytesIO(pdf)).pages:
        chunks: list[tuple[float, float, str]] = []

        def visitor(
            text: str,
            _cm: object,
            text_matrix: list[float],
            _font: object,
            _size: float,
            _chunks: list[tuple[float, float, str]] = chunks,
        ) -> None:
            clean = " ".join(text.split())
            if clean:
                _chunks.append(
                    (round(float(text_matrix[5]), 1), float(text_matrix[4]), clean)
                )

        page.extract_text(visitor_text=visitor)
        rows: dict[float, list[tuple[float, str]]] = {}
        for y, x, text in chunks:
            rows.setdefault(y, []).append((x, text))
        ordered_rows = [(y, sorted(parts)) for y, parts in sorted(rows.items())]
        player_row_ys: list[float] = []
        for y, ordered_parts in ordered_rows:
            status = next(
                (
                    text
                    for x, text in ordered_parts
                    if 580.0 <= x < 666.0
                    and text in INJURY_STATUS_PLAY_PROBABILITY
                ),
                None,
            )
            raw_name = " ".join(
                text for x, text in ordered_parts if 420.0 <= x < 580.0
            )
            if status is not None and "," in raw_name:
                player_row_ys.append(y)

        reason_by_player_y: dict[float, str] = {}
        first_reason_boundary: float | None = None
        for index, player_y in enumerate(player_row_ys):
            previous_y = player_row_ys[index - 1] if index > 0 else None
            next_y = (
                player_row_ys[index + 1]
                if index + 1 < len(player_row_ys)
                else None
            )
            before = (
                (player_y - previous_y) / 2.0
                if previous_y is not None
                else (next_y - player_y) / 2.0
                if next_y is not None
                else 16.0
            )
            after = (
                (next_y - player_y) / 2.0
                if next_y is not None
                else (player_y - previous_y) / 2.0
                if previous_y is not None
                else 16.0
            )
            lower = player_y - before
            upper = player_y + after
            if first_reason_boundary is None:
                first_reason_boundary = lower
            reason_parts: list[str] = []
            for reason_y, reason_row in ordered_rows:
                if not lower <= reason_y < upper:
                    continue
                row_text = " ".join(text for _x, text in reason_row)
                upper_row_text = row_text.upper()
                if (
                    "NOT YET SUBMITTED" in upper_row_text
                    or "NO INJURIES TO REPORT" in upper_row_text
                ):
                    continue
                reason_parts.extend(
                    text
                    for x, text in reason_row
                    if x >= 666.0 and text.upper() != "REASON"
                )
            if reason_parts:
                reason_by_player_y[player_y] = " ".join(reason_parts)

        if statuses and first_reason_boundary is not None:
            leading_reason_parts: list[str] = []
            for reason_y, reason_row in ordered_rows:
                if reason_y >= first_reason_boundary:
                    break
                row_text = " ".join(text for _x, text in reason_row)
                upper_row_text = row_text.upper()
                if (
                    "NOT YET SUBMITTED" in upper_row_text
                    or "NO INJURIES TO REPORT" in upper_row_text
                ):
                    continue
                leading_reason_parts.extend(
                    text
                    for x, text in reason_row
                    if x >= 666.0 and text.upper() != "REASON"
                )
            if leading_reason_parts:
                statuses[-1]["reason"] = " ".join(
                    part
                    for part in [
                        statuses[-1].get("reason"),
                        " ".join(leading_reason_parts),
                    ]
                    if part
                )

        for _y, ordered_parts in ordered_rows:
            matchup_text = "".join(
                text for x, text in ordered_parts if 190.0 <= x < 250.0
            )
            matchup = re.search(r"([A-Z]{2,4})@([A-Z]{2,4})", matchup_text)
            if matchup is not None:
                current_matchup = (
                    ESPN_TEAM_ALIASES.get(matchup.group(1), matchup.group(1)),
                    ESPN_TEAM_ALIASES.get(matchup.group(2), matchup.group(2)),
                )
                current_team_position = -1
            team_tokens = [
                text for x, text in ordered_parts if 250.0 <= x < 420.0
            ]
            has_player_or_reason = any(x >= 420.0 for x, _text in ordered_parts)
            next_team_position = current_team_position + 1
            is_expected_team_row = False
            if (
                current_matchup is not None
                and next_team_position < len(current_matchup)
                and team_tokens
            ):
                expected_team = current_matchup[next_team_position]
                expected_name = TEAM_DISPLAY_NAMES.get(expected_team, expected_team)
                normalized_team_tokens = re.sub(
                    r"[^a-z0-9]", "", " ".join(team_tokens).lower()
                )
                normalized_expected_name = re.sub(
                    r"[^a-z0-9]", "", expected_name.lower()
                )
                is_expected_team_row = normalized_team_tokens == normalized_expected_name
            if is_expected_team_row and has_player_or_reason:
                current_team_position += 1
                team = current_matchup[current_team_position]
                row_text = " ".join(text for _x, text in ordered_parts).upper()
                team_reports[team] = {
                    "team": team,
                    "matchup": "@".join(current_matchup),
                    "status": (
                        "not_yet_submitted"
                        if "NOT YET SUBMITTED" in row_text
                        else "submitted"
                    ),
                }
            status = next(
                (
                    text
                    for x, text in ordered_parts
                    if 580.0 <= x < 666.0 and text in INJURY_STATUS_PLAY_PROBABILITY
                ),
                None,
            )
            names = [text for x, text in ordered_parts if 420.0 <= x < 580.0]
            raw_name = " ".join(names)
            if status is None or not raw_name or "," not in raw_name:
                continue
            last_name, first_name = raw_name.split(",", maxsplit=1)
            player_name = f"{first_name.strip()} {last_name.strip()}".strip()
            reason = reason_by_player_y.get(_y)
            statuses.append(
                {
                    "player_name": player_name,
                    "status": status,
                    "reason": reason,
                    "play_probability": INJURY_STATUS_PLAY_PROBABILITY[status],
                }
            )
    if not statuses:
        raise ValueError("Official NBA injury report contained no player statuses")
    return {"players": statuses, "team_reports": team_reports}


def _normalized_player_name(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    ascii_name = re.sub(r"\s+(?:Jr\.?|Sr\.?|II|III|IV)$", "", ascii_name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


def _game_rotation(
    team: str,
    *,
    players: list[dict[str, object]],
    injury_report: dict[str, object],
    rotation_size: int,
) -> dict[str, object]:
    team_players = [row for row in players if str(row.get("team")) == team]
    if not team_players:
        raise ValueError(f"Minutes payload is missing the {team} roster")
    roster_by_name = {
        _normalized_player_name(row.get("player_name")): row for row in team_players
    }
    official = {
        _normalized_player_name(row.get("player_name")): row
        for row in injury_report.get("players", [])
        if isinstance(row, dict)
    }
    team_report = injury_report.get("team_reports", {}).get(team)
    if isinstance(team_report, dict):
        report_status = str(team_report.get("status", "not_in_report"))
    else:
        report_status = "not_in_report"
    report_pending = report_status != "submitted"
    allocations: list[dict[str, object]] = []
    for player in team_players:
        status_row = official.get(_normalized_player_name(player.get("player_name")))
        if status_row:
            status = str(status_row.get("status"))
            public_play_probability: float | None = INJURY_STATUS_PLAY_PROBABILITY.get(
                status, 1.0
            )
        elif report_pending:
            status = "Report pending"
            public_play_probability = None
        else:
            status = "Available"
            public_play_probability = 1.0
        allocation_probability = (
            public_play_probability if public_play_probability is not None else 1.0
        )
        conditional_minutes = float(player.get("conditional_minutes_per_game", 0.0))
        allocations.append(
            {
                "player_id": int(player["player_id"]),
                "player_name": str(player["player_name"]),
                "status": status,
                "play_probability": public_play_probability,
                "conditional_minutes": conditional_minutes,
                "weight": allocation_probability * conditional_minutes,
                "projected_nail": float(player.get("projected_nail", 0.0)),
                "reason": status_row.get("reason") if status_row else None,
            }
        )
    selected = sorted(
        allocations,
        key=lambda row: (-float(row["weight"]), str(row["player_name"])),
    )[:rotation_size]
    total_weight = sum(float(row["weight"]) for row in selected)
    if total_weight <= 0.0:
        raise ValueError(f"Injury assumptions produced no available minutes for {team}")
    for row in allocations:
        row["projected_minutes"] = (
            240.0 * float(row["weight"]) / total_weight if row in selected else 0.0
        )
    team_strength = sum(
        float(row["projected_minutes"]) / 48.0 * float(row["projected_nail"])
        for row in allocations
    )
    assumptions = []
    for normalized_name, status_row in official.items():
        player = roster_by_name.get(normalized_name)
        if player is None:
            continue
        assumptions.append(
            {
                "player_id": int(player["player_id"]),
                "player_name": str(player["player_name"]),
                "team": team,
                "status": status_row.get("status"),
                "reason": status_row.get("reason"),
                "play_probability": status_row.get("play_probability"),
                "projected_minutes": next(
                    float(row["projected_minutes"])
                    for row in allocations
                    if row["player_id"] == int(player["player_id"])
                ),
            }
        )
    public_players = [
        {key: value for key, value in row.items() if key not in {"weight", "projected_nail"}}
        for row in sorted(
            allocations,
            key=lambda item: (-float(item["projected_minutes"]), str(item["player_name"])),
        )
        if float(row["projected_minutes"]) > 0.0
        or row["status"] not in {"Available", "Report pending"}
    ]
    return {
        "team_strength": team_strength,
        "players": public_players,
        "injury_assumptions": sorted(
            assumptions, key=lambda row: (str(row["team"]), str(row["player_name"]))
        ),
        "report_status": report_status,
    }


def _normalize_espn_schedule(
    raw: dict[str, object],
    *,
    forecast: dict[str, object],
    minutes: dict[str, object],
    injury_report: dict[str, object],
    prediction_markets: object,
    game_date: str,
    collected_at: datetime,
) -> dict[str, object]:
    prediction_markets = prediction_markets if isinstance(prediction_markets, dict) else {}
    minute_players = [row for row in minutes.get("players", []) if isinstance(row, dict)]
    schedule = {
        (str(row.get("home_team")), str(row.get("away_team"))): row
        for row in forecast.get("scheduled_games", [])
        if isinstance(row, dict) and row.get("game_date") == game_date
    }
    beta = float(forecast["win_probability_scale"])
    home_court = float(forecast["home_court"])
    back_to_back = float(forecast["back_to_back"])
    games: list[dict[str, object]] = []
    for event in raw.get("events", []):
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions", [])
        if not competitions or not isinstance(competitions[0], dict):
            continue
        competition = competitions[0]
        competitors = {
            str(row.get("homeAway")): row
            for row in competition.get("competitors", [])
            if isinstance(row, dict)
        }
        home = competitors.get("home", {})
        away = competitors.get("away", {})
        home_team = _espn_team(home)
        away_team = _espn_team(away)
        scheduled = schedule.get((home_team, away_team), {})
        home_rotation = _game_rotation(
            home_team,
            players=minute_players,
            injury_report=injury_report,
            rotation_size=int(minutes.get("initial_rotation_size", 15)),
        )
        away_rotation = _game_rotation(
            away_team,
            players=minute_players,
            injury_report=injury_report,
            rotation_size=int(minutes.get("initial_rotation_size", 15)),
        )
        margin = (
            float(home_rotation["team_strength"])
            - float(away_rotation["team_strength"])
            + home_court
            + back_to_back
            * (
                float(scheduled.get("home_back_to_back", 0))
                - float(scheduled.get("away_back_to_back", 0))
            )
        )
        status = event.get("status", {}).get("type", {})
        if not isinstance(status, dict):
            status = {}
        broadcasts = [
            name
            for broadcast in competition.get("broadcasts", [])
            if isinstance(broadcast, dict)
            for name in broadcast.get("names", [])
        ]
        odds = competition.get("odds", [])
        games.append(
            {
                "game_id": scheduled.get("game_id"),
                "espn_game_id": str(event.get("id", "")),
                "start_time": event.get("date"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": _integer_or_none(home.get("score")),
                "away_score": _integer_or_none(away.get("score")),
                "broadcasts": broadcasts,
                "status": {
                    "state": status.get("state", "pre"),
                    "completed": bool(status.get("completed", False)),
                    "description": status.get("description", "Scheduled"),
                    "detail": status.get("detail"),
                    "display_clock": event.get("status", {}).get("displayClock"),
                    "period": event.get("status", {}).get("period"),
                },
                "projection": {
                    "model": forecast.get("model"),
                    "home_margin": margin,
                    "home_win_probability": float(1.0 / (1.0 + np.exp(-beta * margin))),
                },
                "injury_report": {
                    "source": "NBA official injury report",
                    "source_url": injury_report.get("source_url"),
                    "published_at": injury_report.get("published_at"),
                    "status_probabilities": INJURY_STATUS_PLAY_PROBABILITY,
                    "team_reports": {
                        away_team: {"status": away_rotation["report_status"]},
                        home_team: {"status": home_rotation["report_status"]},
                    },
                    "assumptions": [
                        *away_rotation["injury_assumptions"],
                        *home_rotation["injury_assumptions"],
                    ],
                },
                "projected_minutes": {
                    "away": away_rotation["players"],
                    "home": home_rotation["players"],
                },
                "market": (
                    _normalize_espn_odds(odds[0])
                    if odds and isinstance(odds[0], dict)
                    else None
                ),
                "prediction_market": prediction_markets.get(
                    _prediction_market_game_key(away_team, home_team)
                ),
            }
        )
    return {
        "available": True,
        "game_date": game_date,
        "season": forecast.get("season", PROJECTION_SEASON),
        "collected_at": collected_at.isoformat().replace("+00:00", "Z"),
        "source": "ESPN scoreboard API",
        "prediction_market_source": "Polymarket public Gamma and CLOB APIs",
        "injury_report": {
            "source": "NBA official injury report",
            "source_url": injury_report.get("source_url"),
            "published_at": injury_report.get("published_at"),
            "status_probabilities": INJURY_STATUS_PLAY_PROBABILITY,
            "team_reports": injury_report.get("team_reports", {}),
        },
        "games": games,
    }


def _normalize_espn_odds(odds: dict[str, object]) -> dict[str, object]:
    provider = odds.get("provider", {})
    spread = odds.get("pointSpread", {})
    moneyline = odds.get("moneyline", {})
    total = odds.get("total", {})
    return {
        "provider": provider.get("name", "Unknown") if isinstance(provider, dict) else "Unknown",
        "home_spread": _line_value(spread, "home", "close"),
        "away_spread": _line_value(spread, "away", "close"),
        "home_spread_open": _line_value(spread, "home", "open"),
        "away_spread_open": _line_value(spread, "away", "open"),
        "home_moneyline": _odds_value(moneyline, "home", "close"),
        "away_moneyline": _odds_value(moneyline, "away", "close"),
        "home_moneyline_open": _odds_value(moneyline, "home", "open"),
        "away_moneyline_open": _odds_value(moneyline, "away", "open"),
        "total": _line_value(total, "over", "close"),
        "total_open": _line_value(total, "over", "open"),
    }


def _espn_team(competitor: dict[str, object]) -> str:
    team = competitor.get("team", {})
    abbreviation = str(team.get("abbreviation", "")) if isinstance(team, dict) else ""
    return ESPN_TEAM_ALIASES.get(abbreviation, abbreviation)


def _nested(mapping: object, *keys: str) -> object:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _line_value(mapping: object, side: str, moment: str) -> float | None:
    value = _nested(mapping, side, moment, "line")
    if value is None:
        return None
    try:
        return float(str(value).lower().lstrip("ou+"))
    except ValueError:
        return None


def _odds_value(mapping: object, side: str, moment: str) -> int | None:
    value = _nested(mapping, side, moment, "odds")
    try:
        return int(str(value)) if value is not None else None
    except ValueError:
        return None


def _integer_or_none(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _fetch_polymarket_moneyline_snapshot(
    espn_payload: dict[str, object],
    *,
    game_date: str,
    collected_at: datetime,
) -> dict[str, object]:
    """Read each listed NBA moneyline and its current CLOB midpoint.

    Polymarket's game-event slugs are deterministic from the NBA team codes and
    Eastern game date. The event contains many props and alternate lines, so we
    accept only the market whose slug exactly matches the event slug. Every
    selected field and unavailable lookup is retained in the timestamped
    snapshot written by the schedule collector.
    """

    captured_at = collected_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    specifications: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    seen_games: set[str] = set()
    for espn_event in espn_payload.get("events", []):
        if not isinstance(espn_event, dict):
            continue
        competitions = espn_event.get("competitions", [])
        if not competitions or not isinstance(competitions[0], dict):
            continue
        competitors = {
            str(row.get("homeAway")): row
            for row in competitions[0].get("competitors", [])
            if isinstance(row, dict)
        }
        home_team = _espn_team(competitors.get("home", {}))
        away_team = _espn_team(competitors.get("away", {}))
        game_key = _prediction_market_game_key(away_team, home_team)
        if not home_team or not away_team or game_key in seen_games:
            continue
        seen_games.add(game_key)
        event_slug = f"nba-{away_team.lower()}-{home_team.lower()}-{game_date}"
        try:
            event = _fetch_json(f"{POLYMARKET_EVENT_URL}/{event_slug}", timeout=5)
        except (OSError, TimeoutError, ValueError) as error:
            unavailable.append(
                {
                    "game": game_key,
                    "event_slug": event_slug,
                    "reason": f"event lookup failed: {error}",
                }
            )
            continue
        market = _polymarket_moneyline_market(event, event_slug=event_slug)
        if market is None:
            unavailable.append(
                {
                    "game": game_key,
                    "event_slug": event_slug,
                    "reason": "moneyline market unavailable",
                }
            )
            continue
        outcomes = _json_string_list(market.get("outcomes"))
        asset_ids = _polymarket_asset_ids(market)
        fallback_prices = _json_float_list(market.get("outcomePrices"))
        if len(outcomes) != 2 or len(asset_ids) != 2:
            unavailable.append(
                {
                    "game": game_key,
                    "event_slug": event_slug,
                    "reason": "moneyline outcomes or asset IDs are invalid",
                }
            )
            continue
        specifications.append(
            {
                "game": game_key,
                "away_team": away_team,
                "home_team": home_team,
                "event": event,
                "event_slug": event_slug,
                "market": market,
                "outcomes": outcomes,
                "asset_ids": asset_ids,
                "fallback_prices": fallback_prices,
            }
        )

    midpoint_requests = [
        {"token_id": asset_id}
        for specification in specifications
        for asset_id in specification["asset_ids"]
    ]
    midpoint_error: str | None = None
    midpoints: dict[str, object] = {}
    if midpoint_requests:
        try:
            midpoints = _post_json(POLYMARKET_MIDPOINTS_URL, midpoint_requests)
        except (OSError, TimeoutError, ValueError) as error:
            midpoint_error = str(error)

    games: dict[str, object] = {}
    for specification in specifications:
        asset_ids = specification["asset_ids"]
        fallback_prices = specification["fallback_prices"]
        midpoint_prices = [_float_or_none(midpoints.get(str(asset_id))) for asset_id in asset_ids]
        if all(price is not None for price in midpoint_prices):
            raw_prices = [float(price) for price in midpoint_prices]
            pricing_method = "CLOB midpoint"
        elif len(fallback_prices) == 2:
            raw_prices = fallback_prices
            pricing_method = "Gamma outcome price fallback"
        else:
            unavailable.append(
                {
                    "game": specification["game"],
                    "event_slug": specification["event_slug"],
                    "reason": "midpoint and fallback prices are unavailable",
                }
            )
            continue
        raw_sum = float(sum(raw_prices))
        if raw_sum <= 0.0 or any(price < 0.0 or price > 1.0 for price in raw_prices):
            unavailable.append(
                {
                    "game": specification["game"],
                    "event_slug": specification["event_slug"],
                    "reason": "moneyline prices are outside the probability range",
                }
            )
            continue
        market = specification["market"]
        event = specification["event"]
        outcomes = specification["outcomes"]
        games[str(specification["game"])] = {
            "provider": "Polymarket",
            "event_id": str(event.get("id", "")),
            "market_id": str(market.get("id", "")),
            "event_slug": specification["event_slug"],
            "market_slug": market.get("slug"),
            "url": f"https://polymarket.com/event/{specification['event_slug']}",
            "captured_at": captured_at,
            "pricing_method": pricing_method,
            "away_outcome": outcomes[0],
            "home_outcome": outcomes[1],
            "away_asset_id": asset_ids[0],
            "home_asset_id": asset_ids[1],
            "away_probability_raw": raw_prices[0],
            "home_probability_raw": raw_prices[1],
            "raw_probability_sum": raw_sum,
            "away_probability": raw_prices[0] / raw_sum,
            "home_probability": raw_prices[1] / raw_sum,
            "volume": _float_or_none(market.get("volumeNum")),
            "liquidity": _float_or_none(market.get("liquidityNum")),
            "bid_ask_spread": _float_or_none(market.get("spread")),
        }
    return {
        "provider": "Polymarket",
        "captured_at": captured_at,
        "pricing_method": "CLOB midpoint; Gamma outcome price fallback",
        "requested_game_count": len(seen_games),
        "available_game_count": len(games),
        "midpoint_error": midpoint_error,
        "games": games,
        "unavailable": unavailable,
    }


def _prediction_market_game_key(away_team: str, home_team: str) -> str:
    return f"{away_team}-{home_team}"


def _polymarket_moneyline_market(
    event: dict[str, object], *, event_slug: str
) -> dict[str, object] | None:
    for market in event.get("markets", []):
        if not isinstance(market, dict):
            continue
        if market.get("slug") != event_slug:
            continue
        if market.get("closed") is True or market.get("active") is False:
            continue
        return market
    return None


def _polymarket_asset_ids(market: dict[str, object]) -> list[str]:
    if str(market.get("version", "v1")) == "v2":
        value = market.get("positionIds")
        if isinstance(value, list):
            return [str(item) for item in value]
    return _json_string_list(market.get("clobTokenIds"))


def _json_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _json_float_list(value: object) -> list[float]:
    return [
        parsed
        for item in _json_string_list(value)
        if (parsed := _float_or_none(item)) is not None
    ]


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_json(url: str, *, timeout: float = 20) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "nba-gestalt/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON endpoint did not return an object: {url}")
    return payload


def _post_json(url: str, payload: object) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "nba-gestalt/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"JSON endpoint did not return an object: {url}")
    return result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        return


def _aws_s3_cp(source: Path, destination: str) -> None:
    subprocess.run(["aws", "s3", "cp", str(source), destination, "--only-show-errors"], check=True)


def _aws_s3_cp_exclusive(source: Path, destination: str) -> None:
    """Create an immutable S3 key, accepting only byte-identical retries."""

    if not destination.startswith("s3://"):
        raise ValueError(f"S3 destination is invalid: {destination}")
    bucket, separator, key = destination.removeprefix("s3://").partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"S3 destination is invalid: {destination}")
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(source),
            destination,
            "--no-overwrite",
            "--only-show-errors",
        ],
        check=True,
    )
    existing = subprocess.run(
        ["aws", "s3", "cp", destination, "-", "--only-show-errors"],
        capture_output=True,
        check=True,
    ).stdout
    if existing != source.read_bytes():
        raise RuntimeError(
            f"Refusing to overwrite immutable S3 object with different bytes: {destination}"
        )


def refresh_live_score_state(
    *,
    game_date: str,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
) -> dict[str, object]:
    """Update only volatile score, clock, period, and game-status fields."""

    state_root = Path(state_dir)
    latest_path = state_root / "latest" / f"{game_date}.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"No archived schedule state for {game_date}")
    payload = json.loads(latest_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
        raise ValueError(f"Invalid schedule state: {latest_path}")
    raw = _fetch_json(f"{ESPN_SCOREBOARD_URL}?dates={game_date.replace('-', '')}")
    events = {
        str(event.get("id")): event
        for event in raw.get("events", [])
        if isinstance(event, dict)
    }
    newly_completed: list[dict[str, object]] = []
    for game in payload["games"]:
        if not isinstance(game, dict):
            continue
        event = events.get(str(game.get("espn_game_id")))
        if event is None:
            continue
        competitions = event.get("competitions", [])
        if not competitions or not isinstance(competitions[0], dict):
            continue
        competitors = {
            str(row.get("homeAway")): row
            for row in competitions[0].get("competitors", [])
            if isinstance(row, dict)
        }
        status = event.get("status", {})
        status_type = status.get("type", {}) if isinstance(status, dict) else {}
        if not isinstance(status_type, dict):
            status_type = {}
        game["home_score"] = _integer_or_none(competitors.get("home", {}).get("score"))
        game["away_score"] = _integer_or_none(competitors.get("away", {}).get("score"))
        game["status"] = {
            "state": status_type.get("state", "pre"),
            "completed": bool(status_type.get("completed", False)),
            "description": status_type.get("description", "Scheduled"),
            "detail": status_type.get("detail"),
            "display_clock": status.get("displayClock") if isinstance(status, dict) else None,
            "period": status.get("period") if isinstance(status, dict) else None,
        }
        if game["status"]["completed"] and "retrospective" not in game:
            newly_completed.append(game)
    if newly_completed:
        projection_payload = _fetch_json(DEFAULT_SCHEDULE_API_URL)
        for game in newly_completed:
            summary = _fetch_json(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
                f"?event={game['espn_game_id']}"
            )
            retrospective = _build_actual_minutes_retrospective(
                game,
                summary=summary,
                projection_payload=projection_payload,
            )
            game["retrospective"] = retrospective
            frozen_path = state_root / "frozen" / f"{game['espn_game_id']}.json"
            if not frozen_path.is_file():
                raise FileNotFoundError(
                    f"Cannot finalize {game['espn_game_id']} without its frozen pregame record"
                )
            pregame = json.loads(frozen_path.read_text())
            if not isinstance(pregame, dict):
                raise ValueError(f"Invalid frozen pregame record: {frozen_path}")
            final_record = _build_final_game_archive(
                game,
                pregame=pregame,
                retrospective=retrospective,
            )
            _write_json_exclusive(
                state_root / "final" / f"{game['espn_game_id']}.summary.json",
                final_record,
            )
            _write_json_exclusive(
                state_root / "final" / f"{game['espn_game_id']}.espn.json",
                summary,
            )
    collected_at = datetime.now(UTC)
    payload["live_score_collected_at"] = collected_at.isoformat().replace("+00:00", "Z")
    _write_json(latest_path, payload)
    starts = [
        datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
        for game in payload["games"]
        if isinstance(game, dict) and game.get("start_time")
    ]
    states = [
        str(game.get("status", {}).get("state", "pre"))
        for game in payload["games"]
        if isinstance(game, dict)
    ]
    return {
        "game_count": len(states),
        "any_live": "in" in states,
        "all_final": bool(states) and all(state == "post" for state in states),
        "earliest_start": min(starts) if starts else None,
        "latest_start": max(starts) if starts else None,
    }


def _build_final_game_archive(
    game: dict[str, object],
    *,
    pregame: dict[str, object],
    retrospective: dict[str, object],
    archived_at: datetime | None = None,
) -> dict[str, object]:
    """Build the immutable record used to audit a pregame forecast."""

    home_team = str(game["home_team"])
    away_team = str(game["away_team"])
    home_score = _integer_or_none(game.get("home_score"))
    away_score = _integer_or_none(game.get("away_score"))
    if home_score is None or away_score is None:
        raise ValueError(f"Final game {game.get('espn_game_id')} is missing a score")
    actual_home_margin = home_score - away_score

    projection = pregame.get("projection", {})
    if not isinstance(projection, dict) or projection.get("home_margin") is None:
        raise ValueError(f"Frozen game {game.get('espn_game_id')} is missing its projection")
    projected_home_margin = float(projection["home_margin"])
    model_side = home_team if projected_home_margin > 0.0 else away_team
    model_side_correct = (
        actual_home_margin > 0.0 if model_side == home_team else actual_home_margin < 0.0
    )

    market = pregame.get("market")
    market_grading: dict[str, object] | None = None
    if isinstance(market, dict) and market.get("home_spread") is not None:
        home_spread = float(market["home_spread"])
        away_spread = (
            float(market["away_spread"])
            if market.get("away_spread") is not None
            else -home_spread
        )
        market_implied_home_margin = -home_spread
        model_edge = projected_home_margin - market_implied_home_margin
        selection = home_team if model_edge > 0.0 else away_team
        selection_spread = home_spread if selection == home_team else away_spread
        cover_margin = (
            actual_home_margin + home_spread
            if selection == home_team
            else -actual_home_margin + away_spread
        )
        outcome = (
            "push"
            if np.isclose(cover_margin, 0.0)
            else ("win" if cover_margin > 0.0 else "loss")
        )
        market_side = home_team if market_implied_home_margin > 0.0 else away_team
        market_side_correct = (
            actual_home_margin > 0.0
            if market_side == home_team
            else actual_home_margin < 0.0
        )
        market_grading = {
            "provider": market.get("provider"),
            "captured_home_spread": home_spread,
            "captured_away_spread": away_spread,
            "captured_total": market.get("total"),
            "captured_home_moneyline": market.get("home_moneyline"),
            "captured_away_moneyline": market.get("away_moneyline"),
            "market_implied_home_margin": market_implied_home_margin,
            "market_favorite": market_side,
            "market_favorite_won": bool(market_side_correct),
            "model_edge_points": model_edge,
            "model_ats_selection": selection,
            "model_ats_selection_spread": selection_spread,
            "model_ats_cover_margin": cover_margin,
            "model_ats_result": outcome,
        }

    frozen_at = pregame.get("frozen_at")
    return {
        "schema_version": 1,
        "record_type": "immutable_final_game_audit",
        "archived_at": (archived_at or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
        "season": PROJECTION_SEASON,
        "game_date": str(game.get("start_time", ""))[:10],
        "game_id": game.get("game_id"),
        "espn_game_id": str(game.get("espn_game_id", "")),
        "start_time": game.get("start_time"),
        "away_team": away_team,
        "home_team": home_team,
        "final": {
            "away_score": away_score,
            "home_score": home_score,
            "actual_home_margin": actual_home_margin,
            "winner": home_team if actual_home_margin > 0.0 else away_team,
            "status": game.get("status"),
        },
        "pregame": pregame,
        "grading": {
            "pregame_frozen_at": frozen_at,
            "model": projection.get("model"),
            "projected_home_margin": projected_home_margin,
            "model_fair_home_spread": -projected_home_margin,
            "model_fair_line_team": model_side,
            "model_fair_line": -abs(projected_home_margin),
            "projected_home_win_probability": projection.get("home_win_probability"),
            "model_favorite": model_side,
            "model_favorite_won": bool(model_side_correct),
            "margin_absolute_error": abs(actual_home_margin - projected_home_margin),
            "market": market_grading,
        },
        "retrospective": retrospective,
    }


def _build_actual_minutes_retrospective(
    game: dict[str, object],
    *,
    summary: dict[str, object],
    projection_payload: dict[str, object],
) -> dict[str, object]:
    players = [
        row for row in projection_payload.get("players", []) if isinstance(row, dict)
    ]
    ratings = {
        (str(row.get("team")), _normalized_player_name(row.get("player_name"))): row
        for row in players
    }
    actual_by_team: dict[str, list[dict[str, object]]] = {}
    boxscore = summary.get("boxscore", {})
    player_groups = boxscore.get("players", []) if isinstance(boxscore, dict) else []
    for team_group in player_groups:
        if not isinstance(team_group, dict):
            continue
        team_data = team_group.get("team", {})
        team = ESPN_TEAM_ALIASES.get(
            str(team_data.get("abbreviation", "")) if isinstance(team_data, dict) else "",
            str(team_data.get("abbreviation", "")) if isinstance(team_data, dict) else "",
        )
        rows: list[dict[str, object]] = []
        for statistics in team_group.get("statistics", []):
            if not isinstance(statistics, dict):
                continue
            labels = [str(label) for label in statistics.get("labels", [])]
            if "MIN" not in labels:
                continue
            minutes_index = labels.index("MIN")
            for athlete_row in statistics.get("athletes", []):
                if not isinstance(athlete_row, dict):
                    continue
                athlete = athlete_row.get("athlete", {})
                if not isinstance(athlete, dict):
                    continue
                name = str(athlete.get("displayName", ""))
                stats = athlete_row.get("stats", [])
                minutes = (
                    _parse_boxscore_minutes(stats[minutes_index])
                    if isinstance(stats, list) and len(stats) > minutes_index
                    else 0.0
                )
                model_player = ratings.get((team, _normalized_player_name(name)))
                rows.append(
                    {
                        "player_id": (
                            int(model_player["player_id"]) if model_player is not None else None
                        ),
                        "player_name": (
                            str(model_player["player_name"]) if model_player is not None else name
                        ),
                        "actual_minutes": minutes,
                        "projected_nail": (
                            float(model_player.get("projected_nail", 0.0))
                            if model_player is not None
                            else 0.0
                        ),
                        "rating_matched": model_player is not None,
                    }
                )
        actual_by_team[team] = rows

    home_team = str(game["home_team"])
    away_team = str(game["away_team"])
    home_strength = _actual_minutes_team_strength(actual_by_team.get(home_team, []), home_team)
    away_strength = _actual_minutes_team_strength(actual_by_team.get(away_team, []), away_team)
    forecast = projection_payload.get("win_projection", {})
    if not isinstance(forecast, dict):
        raise ValueError("Win projection payload is missing for retrospective")
    scheduled = next(
        (
            row
            for row in forecast.get("scheduled_games", [])
            if isinstance(row, dict) and row.get("game_id") == game.get("game_id")
        ),
        {},
    )
    margin = (
        home_strength
        - away_strength
        + float(forecast["home_court"])
        + float(forecast["back_to_back"])
        * (
            float(scheduled.get("home_back_to_back", 0))
            - float(scheduled.get("away_back_to_back", 0))
        )
    )
    probability = float(
        1.0 / (1.0 + np.exp(-float(forecast["win_probability_scale"]) * margin))
    )
    return {
        "computed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "method": (
            "NAIL strength reweighted by final actual minutes; "
            "team minutes normalized to 240"
        ),
        "home_margin": margin,
        "home_win_probability": probability,
        "actual_minutes": {
            "away": _public_actual_minutes(actual_by_team.get(away_team, [])),
            "home": _public_actual_minutes(actual_by_team.get(home_team, [])),
        },
    }


def _parse_boxscore_minutes(value: object) -> float:
    text = str(value or "0").strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _actual_minutes_team_strength(rows: list[dict[str, object]], team: str) -> float:
    total = sum(float(row["actual_minutes"]) for row in rows)
    if total <= 0.0:
        raise ValueError(f"Final ESPN box score has no actual minutes for {team}")
    return sum(
        5.0 * float(row["actual_minutes"]) / total * float(row["projected_nail"])
        for row in rows
    )


def _public_actual_minutes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in row.items() if key != "projected_nail"}
        for row in sorted(
            rows,
            key=lambda row: (-float(row["actual_minutes"]), str(row["player_name"])),
        )
    ]


def _default_live_score_date(
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    now: datetime | None = None,
) -> str:
    """Prefer yesterday's unfinished slate while its live window is still open."""

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    eastern_date = current_time.astimezone(ZoneInfo("America/New_York")).date()
    candidate_dates = [
        (eastern_date - timedelta(days=1)).isoformat(),
        eastern_date.isoformat(),
    ]
    for game_date in candidate_dates:
        path = Path(state_dir) / "latest" / f"{game_date}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
            continue
        games = [game for game in payload["games"] if isinstance(game, dict)]
        states = [str(game.get("status", {}).get("state", "pre")) for game in games]
        starts = [
            datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
            for game in games
            if game.get("start_time")
        ]
        if not states or not starts or all(state == "post" for state in states):
            continue
        if min(starts) - timedelta(minutes=30) <= current_time <= max(starts) + timedelta(
            hours=6
        ):
            return game_date
    return eastern_date.isoformat()


def live_score_watcher_main() -> int:
    """Poll ESPN only during the bounded daily live-game window."""

    parser = argparse.ArgumentParser(description="Watch live NBA scores until the final game ends")
    parser.add_argument(
        "--date",
        default=None,
        help="Game date to watch; defaults to yesterday when its late slate is unfinished",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("GESTALT_SCHEDULE_STATE_DIR", DEFAULT_SCHEDULE_STATE_DIR)),
    )
    parser.add_argument("--interval-seconds", type=int, default=75)
    args = parser.parse_args()
    if args.interval_seconds < 30:
        parser.error("--interval-seconds must be at least 30")
    game_date = args.date or _default_live_score_date(state_dir=args.state_dir)
    while True:
        state = refresh_live_score_state(game_date=game_date, state_dir=args.state_dir)
        now = datetime.now(UTC)
        earliest = state["earliest_start"]
        latest = state["latest_start"]
        if state["all_final"]:
            print(f"All {state['game_count']} games are final for {game_date}")
            return 0
        if not isinstance(earliest, datetime) or not isinstance(latest, datetime):
            print(f"No games to watch for {game_date}")
            return 0
        if now < earliest - timedelta(minutes=30):
            print(f"Live window has not started for {game_date}")
            return 0
        if now > latest + timedelta(hours=6):
            print(f"Live window expired for {game_date}")
            return 0
        time.sleep(args.interval_seconds)


def refresh_schedule_main() -> int:
    """Refresh the available ESPN schedule/odds window from cron."""

    parser = argparse.ArgumentParser(description="Archive NBA schedule and odds from ESPN")
    parser.add_argument(
        "--date",
        default=datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("GESTALT_SCHEDULE_STATE_DIR", DEFAULT_SCHEDULE_STATE_DIR)),
    )
    parser.add_argument("--projection-url", default=DEFAULT_SCHEDULE_API_URL)
    parser.add_argument("--s3-uri", default=DEFAULT_SCHEDULE_S3_URI)
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=DEFAULT_SCHEDULE_DAYS_AHEAD,
        help="Number of future dates to inspect in addition to --date",
    )
    args = parser.parse_args()
    if args.days_ahead < 0:
        parser.error("--days-ahead must be non-negative")
    payload = refresh_schedule_window(
        start_date=args.date,
        days_ahead=args.days_ahead,
        state_dir=args.state_dir,
        projection_url=args.projection_url,
        s3_uri=None if args.no_s3 else args.s3_uri,
    )
    print(
        f"Archived {payload['game_count']} games across "
        f"{payload['archived_day_count']} available dates from {args.date}; "
        f"newly frozen: {payload['newly_frozen_game_count']}"
    )
    return 0


def main() -> int:
    """Materialize the immutable baseline win-total interval cache."""

    parser = argparse.ArgumentParser(
        description="Build the release-time NBA GESTALT win projection cache"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    evaluator = LineupEvaluator.from_latest_artifact()
    path = materialize_win_projection_cache(evaluator=evaluator, output_path=args.output)
    print(f"Materialized win projection cache: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
