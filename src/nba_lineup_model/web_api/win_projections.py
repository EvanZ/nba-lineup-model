"""Frozen preseason team-strength and schedule win projections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.season.schedule import SeasonScheduleCache
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    MODEL_ARTIFACT,
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
    return {
        "model": "W0 minute-weighted NAIL",
        "season": PROJECTION_SEASON,
        "scheduled_game_count": int(len(games)),
        "scheduled_games_per_team": int(len(games) * 2 / len(teams)),
        "unassigned_regular_games_per_team": UNASSIGNED_REGULAR_GAMES,
        "win_probability_scale": float(calibration["production_beta"]),
        "home_court": float(controls.home_court),
        "back_to_back": float(controls.back_to_back),
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
    output["unassigned_wins"] = UNASSIGNED_REGULAR_GAMES * (neutral_home + neutral_away) / 2.0
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
