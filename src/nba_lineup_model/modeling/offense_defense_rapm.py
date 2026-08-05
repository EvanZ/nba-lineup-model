"""Forward regular-season offense/defense RAPM and frozen preseason scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.evaluation.metrics import mean_absolute_error, mean_squared_error, rmse
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_SEASON,
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _team_net_rating_metrics,
    _team_win_evaluation,
    fit_pythagorean_win_model,
    score_possession_cohort,
)
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.prior_rapm import HISTORICAL_SEASONS
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.train import (
    DEFAULT_LAMBDA_GRID,
    GameSplitPlan,
    chronological_game_splits,
)
from nba_lineup_model.models.baselines import PriorCenteredRidgeLineupModel
from nba_lineup_model.season.schema import validate_season

MODEL_NAME = "frozen_offense_defense_rapm"
RUN_PREFIX = "frozen-offense-defense-rapm"
RANKING_MODEL_NAME = "all_season_offense_defense_rapm"
RANKING_RUN_PREFIX = "all-season-offense-defense-rapm"
DEFAULT_MINIMUM_RANKING_POSSESSIONS = 500.0


@dataclass(frozen=True)
class SideDesign:
    """Two offense-rating observations per lineup stint."""

    player_ids: tuple[int, ...]
    features: sparse.csr_matrix
    target: np.ndarray
    weights: np.ndarray
    game_ids: np.ndarray
    home_offense: np.ndarray


@dataclass(frozen=True)
class ForwardSeasonResult:
    """Completed O/D season used as the next season's frozen prior."""

    season: str
    selected_lambda: float
    coefficients: pd.DataFrame
    cv_results: pd.DataFrame
    league_offensive_rating: float
    home_offense_shift: float


def offense_defense_code_fingerprint() -> str:
    """Hash the sources defining this frozen O/D evaluation."""

    digest = hashlib.sha256()
    paths = (
        Path(__file__),
        Path(__file__).with_name("frozen_prior_evaluation.py"),
        Path(__file__).parents[1] / "evaluation" / "metrics.py",
        Path(__file__).parents[1] / "models" / "baselines.py",
    )
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def build_side_design(stints: pd.DataFrame) -> SideDesign:
    """Encode home and away offense against the opponent's defense.

    Defensive coefficients represent points prevented per 100 possessions. A
    home offensive row therefore contains the home lineup's offensive columns
    with +1 and the away lineup's defensive columns with -1.
    """

    required = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        "points_home",
        "points_away",
        "home_offensive_possessions",
        "away_offensive_possessions",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"O/D RAPM stints missing columns: {sorted(missing)}")
    player_ids = tuple(
        sorted(
            {
                int(player)
                for column in ("home_player_ids", "away_player_ids")
                for lineup in stints[column]
                for player in lineup
            }
        )
    )
    if not player_ids:
        raise ValueError("O/D RAPM requires at least one player")
    player_columns = {player_id: column for column, player_id in enumerate(player_ids)}
    player_count = len(player_ids)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    targets: list[float] = []
    weights: list[float] = []
    game_ids: list[str] = []
    home_flags: list[bool] = []
    for _stint_index, stint in stints.reset_index(drop=True).iterrows():
        for is_home_offense, offense_column, defense_column, points_column, possessions_column in (
            (
                True,
                "home_player_ids",
                "away_player_ids",
                "points_home",
                "home_offensive_possessions",
            ),
            (
                False,
                "away_player_ids",
                "home_player_ids",
                "points_away",
                "away_offensive_possessions",
            ),
        ):
            possessions = float(stint[possessions_column])
            if possessions <= 0:
                continue
            row = len(targets)
            for player_id in stint[offense_column]:
                rows.append(row)
                columns.append(player_columns[int(player_id)])
                values.append(1.0)
            for player_id in stint[defense_column]:
                rows.append(row)
                columns.append(player_count + player_columns[int(player_id)])
                values.append(-1.0)
            rows.append(row)
            columns.append(2 * player_count)
            values.append(1.0 if is_home_offense else -1.0)
            targets.append(100.0 * float(stint[points_column]) / possessions)
            weights.append(possessions)
            game_ids.append(str(stint["game_id"]))
            home_flags.append(is_home_offense)
    if not targets:
        raise ValueError("O/D RAPM has no positive-exposure offense rows")
    return SideDesign(
        player_ids=player_ids,
        features=sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(targets), 2 * player_count + 1),
            dtype=np.float64,
        ).tocsr(),
        target=np.asarray(targets, dtype=float),
        weights=np.asarray(weights, dtype=float),
        game_ids=np.asarray(game_ids, dtype=str),
        home_offense=np.asarray(home_flags, dtype=bool),
    )


def fit_forward_side_season(
    season: str,
    stints: pd.DataFrame,
    prior_coefficients: pd.DataFrame | None,
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> ForwardSeasonResult:
    """Tune and fit one full regular season around the prior O/D state."""

    design = build_side_design(stints)
    split_plan = chronological_game_splits(stints, config=ChronologicalSplitConfig())
    prior = _prior_vector(design.player_ids, prior_coefficients)
    cv_results = _cross_validate_side_model(design, split_plan, prior, lambda_grid)
    selected_lambda = _select_lambda(cv_results)
    model = PriorCenteredRidgeLineupModel(selected_lambda).fit(
        design.features,
        design.target,
        design.weights,
        prior,
    )
    coefficients = _coefficient_frame(
        season,
        design.player_ids,
        model.coef_,
        prior,
        selected_lambda,
    )
    return ForwardSeasonResult(
        season=season,
        selected_lambda=selected_lambda,
        coefficients=coefficients,
        cv_results=cv_results,
        league_offensive_rating=model.intercept_,
        home_offense_shift=float(model.coef_[-1]),
    )


def train_frozen_offense_defense_rapm(
    *,
    season: str = DEFAULT_SEASON,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> Path:
    """Fit forward O/D RAPM through the prior season and score a frozen target."""

    target_season = _validate_target_season(season)
    source_season = _previous_season(target_season)
    if tuple(HISTORICAL_SEASONS)[-1] != source_season:
        raise ValueError("Historical O/D RAPM schedule must end at the target source season")
    analytical_root = Path(analytical_dir)
    prior: pd.DataFrame | None = None
    results: list[ForwardSeasonResult] = []
    for historical_season in HISTORICAL_SEASONS:
        result = fit_forward_side_season(
            historical_season,
            read_rapm_stints(historical_season, analytical_dir=analytical_root),
            prior,
            lambda_grid=lambda_grid,
        )
        results.append(result)
        prior = result.coefficients.loc[
            :, ["player_id", "offense_rapm", "defense_rapm"]
        ].copy()
    source = results[-1]
    evaluation = evaluate_frozen_offense_defense_rapm(
        season=target_season,
        coefficients=source.coefficients,
        league_offensive_rating=source.league_offensive_rating,
        home_offense_shift=source.home_offense_shift,
        analytical_dir=analytical_root,
        curated_dir=Path(curated_dir),
    )
    return _write_run(
        evaluation,
        source=source,
        results=results,
        artifacts_dir=Path(artifacts_dir),
        analytical_dir=analytical_root,
        curated_dir=Path(curated_dir),
        lambda_grid=lambda_grid,
    )


def build_all_season_offense_defense_rankings(
    *,
    season: str = DEFAULT_SEASON,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    frozen_artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    minimum_ranking_possessions: float = DEFAULT_MINIMUM_RANKING_POSSESSIONS,
) -> Path:
    """Refit a completed regular season solely for descriptive O/D rankings.

    The source state is the immutable frozen O/D prior ending with the prior
    regular season. Unlike ``train_frozen_offense_defense_rapm``, this uses all
    target regular-season outcomes and must never be interpreted as a forecast.
    """

    target_season = _validate_target_season(season)
    if minimum_ranking_possessions < 0:
        raise ValueError("Minimum ranking possessions cannot be negative")
    source_dir, source_metadata = _load_frozen_source_state(
        target_season,
        Path(frozen_artifacts_dir),
    )
    source_coefficients = pd.read_parquet(source_dir / "frozen_player_priors.parquet")
    stints = read_rapm_stints(target_season, analytical_dir=Path(analytical_dir))
    result = fit_forward_side_season(
        target_season,
        stints,
        source_coefficients,
    )
    bios_path = (
        Path(curated_dir)
        / "player_seasons"
        / target_season
        / "regular"
        / "part-00000.parquet"
    )
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    rankings = _side_player_rankings(
        stints,
        result.coefficients,
        player_bios=player_bios,
        minimum_possessions=minimum_ranking_possessions,
    )
    return _write_ranking_run(
        season=target_season,
        result=result,
        rankings=rankings,
        source_dir=source_dir,
        source_metadata=source_metadata,
        artifacts_dir=Path(artifacts_dir),
        minimum_ranking_possessions=minimum_ranking_possessions,
    )


def evaluate_frozen_offense_defense_rapm(
    *,
    season: str,
    coefficients: pd.DataFrame,
    league_offensive_rating: float,
    home_offense_shift: float,
    analytical_dir: Path,
    curated_dir: Path,
) -> dict[str, object]:
    """Score a fixed O/D player state on all target regular and playoff outcomes."""

    target_season = _validate_target_season(season)
    source_season = _previous_season(target_season)
    priors = _validate_coefficients(coefficients)
    regular = read_neural_possessions(target_season, analytical_dir=analytical_dir)
    playoffs, _ = _read_playoff_possessions(target_season, curated_dir)
    regular_predictions = score_frozen_offense_defense_possessions(
        regular,
        priors,
        league_offensive_rating=league_offensive_rating,
        home_offense_shift=home_offense_shift,
        cohort="regular_season",
    )
    playoff_predictions = score_frozen_offense_defense_possessions(
        playoffs,
        priors,
        league_offensive_rating=league_offensive_rating,
        home_offense_shift=home_offense_shift,
        cohort="playoffs",
    )
    possession_predictions = pd.concat(
        [regular_predictions, playoff_predictions],
        ignore_index=True,
    )
    source_mean = league_offensive_rating / 100.0
    cohort_metrics = pd.concat(
        [
            score_possession_cohort(
                regular_predictions,
                source_mean=source_mean,
                model=MODEL_NAME,
            ),
            score_possession_cohort(
                playoff_predictions,
                source_mean=source_mean,
                model=MODEL_NAME,
            ),
        ],
        ignore_index=True,
    )
    game_predictions = _game_prediction_frame(possession_predictions)
    target_stints = read_rapm_stints(target_season, analytical_dir=analytical_dir)
    regular_games, team_ratings = _regular_stint_predictions(
        target_stints,
        priors,
        league_offensive_rating=league_offensive_rating,
        home_offense_shift=home_offense_shift,
    )
    team_metrics = _team_net_rating_metrics(team_ratings, model=MODEL_NAME)
    calibration = _historical_team_seasons(
        analytical_dir=analytical_dir,
        through_season=source_season,
    )
    pythagorean = fit_pythagorean_win_model(calibration)
    team_wins, win_metrics = _team_win_evaluation(
        regular_games,
        team_ratings,
        pythagorean,
        model=MODEL_NAME,
    )
    return {
        "season": target_season,
        "source_season": source_season,
        "frozen_player_priors": priors,
        "cohort_metrics": cohort_metrics,
        "possession_predictions": possession_predictions,
        "game_predictions": game_predictions,
        "regular_game_predictions": regular_games,
        "team_net_rating_predictions": team_ratings,
        "team_net_rating_metrics": team_metrics,
        "team_win_predictions": team_wins,
        "team_win_metrics": win_metrics,
        "pythagorean_calibration_team_seasons": calibration,
        "source_state": {
            "target_season": target_season,
            "source_season": source_season,
            "player_prior_method": (
                "completed prior-season forward regular-only offense/defense RAPM"
            ),
            "league_offensive_rating": league_offensive_rating,
            "home_offense_shift": home_offense_shift,
            "target_season_refit": False,
            "target_regular_outcomes_used_for_fit": False,
            "target_playoff_outcomes_used_for_fit": False,
            "oracle_information": "realized target-season lineups and exposure only",
            "pythagorean_win_model": {
                "intercept": pythagorean.intercept,
                "net_rating_slope": pythagorean.net_rating_slope,
                "training_first_season": pythagorean.training_seasons[0],
                "training_last_season": pythagorean.training_seasons[-1],
                "training_team_season_count": pythagorean.training_team_season_count,
            },
        },
    }


def score_frozen_offense_defense_possessions(
    possessions: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    league_offensive_rating: float,
    home_offense_shift: float,
    cohort: str,
) -> pd.DataFrame:
    """Score target offensive possessions from a fixed O/D player state."""

    required = {
        "game_id",
        "possession_id",
        "home_offense_sign",
        "offense_player_ids",
        "defense_player_ids",
        "target_offense_margin",
    }
    missing = required - set(possessions)
    if missing:
        raise ValueError(f"O/D frozen possessions missing columns: {sorted(missing)}")
    priors = _validate_coefficients(coefficients)
    offense = dict(zip(priors["player_id"], priors["offense_rapm"], strict=True))
    defense = dict(zip(priors["player_id"], priors["defense_rapm"], strict=True))
    predicted_rate = np.empty(len(possessions), dtype=float)
    unknown = np.empty(len(possessions), dtype=int)
    for index, (offense_players, defense_players, sign) in enumerate(
        zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            possessions["home_offense_sign"],
            strict=True,
        )
    ):
        player_ids = [
            *(int(player) for player in offense_players),
            *(int(player) for player in defense_players),
        ]
        unknown[index] = sum(player not in offense for player in player_ids)
        predicted_rate[index] = (
            league_offensive_rating
            + sum(offense.get(int(player), 0.0) for player in offense_players)
            - sum(defense.get(int(player), 0.0) for player in defense_players)
            + float(sign) * home_offense_shift
        )
    output_columns = [
        "game_id",
        "possession_id",
        "home_offense_sign",
        "target_offense_margin",
    ]
    for column in ("game_date", "game_time_utc"):
        if column in possessions:
            output_columns.append(column)
    output = possessions.loc[:, output_columns].copy()
    output["cohort"] = cohort
    output["prediction_offense_margin"] = predicted_rate / 100.0
    output["target_home_margin"] = (
        output["target_offense_margin"] * output["home_offense_sign"]
    )
    output["prediction_home_margin"] = (
        output["prediction_offense_margin"] * output["home_offense_sign"]
    )
    output["residual_offense_margin"] = (
        output["target_offense_margin"] - output["prediction_offense_margin"]
    )
    output["unknown_player_exposures"] = unknown
    return output


def _regular_stint_predictions(
    stints: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    league_offensive_rating: float,
    home_offense_shift: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    priors = _validate_coefficients(coefficients)
    offense = dict(zip(priors["player_id"], priors["offense_rapm"], strict=True))
    defense = dict(zip(priors["player_id"], priors["defense_rapm"], strict=True))
    base = stints.loc[
        :,
        [
            "game_id",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
            "home_player_ids",
            "away_player_ids",
            "points_home",
            "points_away",
            "home_margin",
            "home_offensive_possessions",
            "away_offensive_possessions",
            "possessions",
        ],
    ].copy()
    home_rate = np.array(
        [
            league_offensive_rating
            + sum(offense.get(int(player), 0.0) for player in home)
            - sum(defense.get(int(player), 0.0) for player in away)
            + home_offense_shift
            for home, away in zip(base["home_player_ids"], base["away_player_ids"], strict=True)
        ],
        dtype=float,
    )
    away_rate = np.array(
        [
            league_offensive_rating
            + sum(offense.get(int(player), 0.0) for player in away)
            - sum(defense.get(int(player), 0.0) for player in home)
            - home_offense_shift
            for home, away in zip(base["home_player_ids"], base["away_player_ids"], strict=True)
        ],
        dtype=float,
    )
    base["predicted_home_margin"] = (
        home_rate * base["home_offensive_possessions"].to_numpy(dtype=float) / 100.0
        - away_rate * base["away_offensive_possessions"].to_numpy(dtype=float) / 100.0
    )
    games = base.groupby(
        ["game_id", "home_team_id", "away_team_id", "home_team_tricode", "away_team_tricode"],
        as_index=False,
        sort=False,
    ).agg(
        actual_home_margin=("home_margin", "sum"),
        predicted_home_margin=("predicted_home_margin", "sum"),
    )
    games["actual_home_win"] = games["actual_home_margin"].gt(0)
    games["predicted_home_win"] = games["predicted_home_margin"].gt(0)
    games["predicted_tie"] = games["predicted_home_margin"].eq(0.0)
    games["margin_error"] = games["predicted_home_margin"] - games["actual_home_margin"]
    home = base.loc[
        :,
        [
            "home_team_id",
            "home_team_tricode",
            "possessions",
            "home_margin",
            "predicted_home_margin",
        ],
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
        }
    )
    away = base.loc[
        :,
        [
            "away_team_id",
            "away_team_tricode",
            "possessions",
            "home_margin",
            "predicted_home_margin",
        ],
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_margin": "actual_margin",
            "predicted_home_margin": "predicted_margin",
        }
    )
    away["actual_margin"] *= -1.0
    away["predicted_margin"] *= -1.0
    teams = pd.concat([home, away], ignore_index=True).groupby(
        ["team_id", "team_tricode"], as_index=False
    ).agg(
        possessions=("possessions", "sum"),
        actual_total_margin=("actual_margin", "sum"),
        predicted_total_margin=("predicted_margin", "sum"),
    )
    teams["actual_net_rating"] = 100.0 * teams["actual_total_margin"] / teams["possessions"]
    teams["predicted_net_rating"] = (
        100.0 * teams["predicted_total_margin"] / teams["possessions"]
    )
    teams["net_rating_error"] = teams["predicted_net_rating"] - teams["actual_net_rating"]
    return (
        games.sort_values("game_id", kind="stable").reset_index(drop=True),
        teams.sort_values("team_id", kind="stable").reset_index(drop=True),
    )


def _cross_validate_side_model(
    design: SideDesign,
    split_plan: GameSplitPlan,
    prior: np.ndarray,
    lambda_grid: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold in split_plan.folds:
        train = np.isin(design.game_ids, fold.train_game_ids)
        validation = np.isin(design.game_ids, fold.validation_game_ids)
        for regularization in lambda_grid:
            model = PriorCenteredRidgeLineupModel(regularization).fit(
                design.features[train],
                design.target[train],
                design.weights[train],
                prior,
            )
            prediction = model.predict(design.features[validation])
            mse = mean_squared_error(
                design.target[validation],
                prediction,
                design.weights[validation],
            )
            rows.append(
                {
                    "fold": fold.fold,
                    "regularization": regularization,
                    "validation_game_count": len(fold.validation_game_ids),
                    "validation_offense_row_count": int(validation.sum()),
                    "validation_possessions": float(design.weights[validation].sum()),
                    "squared_error_sum": float(mse * design.weights[validation].sum()),
                    "weighted_mse": mse,
                    "weighted_rmse": rmse(
                        design.target[validation],
                        prediction,
                        design.weights[validation],
                    ),
                    "weighted_mae": mean_absolute_error(
                        design.target[validation],
                        prediction,
                        design.weights[validation],
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["regularization", "fold"], kind="stable").reset_index(
        drop=True
    )


def _select_lambda(cv_results: pd.DataFrame) -> float:
    summary = cv_results.groupby("regularization", as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    selected = summary.sort_values(["weighted_mse", "regularization"], kind="stable").iloc[0]
    return float(selected["regularization"])


def _prior_vector(
    player_ids: tuple[int, ...],
    prior_coefficients: pd.DataFrame | None,
) -> np.ndarray:
    player_count = len(player_ids)
    prior = np.zeros(2 * player_count + 1, dtype=float)
    if prior_coefficients is None:
        return prior
    checked = _validate_coefficients(prior_coefficients)
    offense = dict(zip(checked["player_id"], checked["offense_rapm"], strict=True))
    defense = dict(zip(checked["player_id"], checked["defense_rapm"], strict=True))
    for index, player_id in enumerate(player_ids):
        prior[index] = offense.get(player_id, 0.0)
        prior[player_count + index] = defense.get(player_id, 0.0)
    return prior


def _coefficient_frame(
    season: str,
    player_ids: tuple[int, ...],
    coefficients: np.ndarray,
    prior: np.ndarray,
    selected_lambda: float,
) -> pd.DataFrame:
    player_count = len(player_ids)
    output = pd.DataFrame(
        {
            "season": season,
            "player_id": player_ids,
            "offense_rapm": coefficients[:player_count],
            "defense_rapm": coefficients[player_count : 2 * player_count],
            "offense_prior": prior[:player_count],
            "defense_prior": prior[player_count : 2 * player_count],
        }
    )
    output["net_rapm"] = output["offense_rapm"] + output["defense_rapm"]
    output["selected_lambda"] = selected_lambda
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _validate_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "offense_rapm", "defense_rapm"}
    missing = required - set(coefficients)
    if missing:
        raise ValueError(f"O/D player coefficients missing columns: {sorted(missing)}")
    if coefficients["player_id"].duplicated().any():
        raise ValueError("O/D player coefficients contain duplicate players")
    output = coefficients.loc[:, ["player_id", "offense_rapm", "defense_rapm"]].copy()
    output["player_id"] = output["player_id"].astype(int)
    if not np.isfinite(output[["offense_rapm", "defense_rapm"]].to_numpy(dtype=float)).all():
        raise ValueError("O/D player coefficients must be finite")
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _validate_target_season(season: str) -> str:
    return validate_season(season)


def _load_frozen_source_state(
    target_season: str,
    frozen_artifacts_dir: Path,
) -> tuple[Path, dict[str, object]]:
    root = frozen_artifacts_dir / "frozen_offense_defense_rapm" / target_season
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Frozen O/D source pointer not found: {latest_path}")
    run_id = json.loads(latest_path.read_text()).get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(RUN_PREFIX):
        raise ValueError("Frozen O/D source pointer has no valid run id")
    source_dir = root / run_id
    validate_frozen_offense_defense_run(source_dir)
    metadata = json.loads((source_dir / "metadata.json").read_text())
    source_state = json.loads((source_dir / "source_state.json").read_text())
    if source_state.get("target_season") != target_season:
        raise ValueError("Frozen O/D source target season does not match ranking season")
    if source_state.get("source_season") != _previous_season(target_season):
        raise ValueError("Frozen O/D source season does not precede ranking season")
    return source_dir, {
        "run_id": run_id,
        "manifest_sha256": _sha256_file(source_dir / "manifest.json"),
        "frozen_priors_sha256": _sha256_file(source_dir / "frozen_player_priors.parquet"),
        "source_season": source_state["source_season"],
        "source_code_version": metadata["evaluation_code_version"],
    }


def _side_player_rankings(
    stints: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    player_bios: pd.DataFrame | None,
    minimum_possessions: float,
) -> pd.DataFrame:
    """Attach side-specific regular-season exposure and names to O/D values."""

    required = {
        "home_player_ids",
        "away_player_ids",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_offensive_possessions",
        "away_offensive_possessions",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"O/D ranking stints missing columns: {sorted(missing)}")
    home = stints.loc[
        :,
        [
            "home_player_ids",
            "home_team_id",
            "home_team_tricode",
            "home_offensive_possessions",
            "away_offensive_possessions",
        ],
    ].explode("home_player_ids", ignore_index=True).rename(
        columns={
            "home_player_ids": "player_id",
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "home_offensive_possessions": "offense_possessions",
            "away_offensive_possessions": "defense_possessions",
        }
    )
    away = stints.loc[
        :,
        [
            "away_player_ids",
            "away_team_id",
            "away_team_tricode",
            "home_offensive_possessions",
            "away_offensive_possessions",
        ],
    ].explode("away_player_ids", ignore_index=True).rename(
        columns={
            "away_player_ids": "player_id",
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "away_offensive_possessions": "offense_possessions",
            "home_offensive_possessions": "defense_possessions",
        }
    )
    exposure = pd.concat([home, away], ignore_index=True)
    exposure["player_id"] = exposure["player_id"].astype("int64")
    totals = exposure.groupby("player_id", as_index=False).agg(
        stint_count=("player_id", "size"),
        offense_possessions=("offense_possessions", "sum"),
        defense_possessions=("defense_possessions", "sum"),
    )
    totals["total_possessions"] = (
        totals["offense_possessions"] + totals["defense_possessions"]
    ) / 2.0
    primary_teams = (
        exposure.assign(total_possessions=(
            exposure["offense_possessions"] + exposure["defense_possessions"]
        ) / 2.0)
        .groupby(["player_id", "team_id", "team_tricode"], as_index=False)["total_possessions"]
        .sum()
        .sort_values(
            ["player_id", "total_possessions", "team_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("player_id")
        .rename(
            columns={
                "team_id": "primary_team_id",
                "team_tricode": "primary_team_tricode",
                "total_possessions": "primary_team_possessions",
            }
        )
    )
    checked_coefficients = _validate_coefficients(coefficients)
    for column in ("offense_prior", "defense_prior"):
        if column in coefficients:
            checked_coefficients = checked_coefficients.merge(
                coefficients.loc[:, ["player_id", column]],
                on="player_id",
                how="left",
                validate="one_to_one",
            )
        else:
            checked_coefficients[column] = 0.0
    checked_coefficients["net_rapm"] = (
        checked_coefficients["offense_rapm"] + checked_coefficients["defense_rapm"]
    )
    frame = checked_coefficients.merge(
        totals,
        on="player_id",
        how="left",
        validate="one_to_one",
    ).merge(
        primary_teams,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    name_map: dict[int, str] = {}
    if player_bios is not None and {"player_id", "player_name"} <= set(player_bios):
        names = player_bios.loc[:, ["player_id", "player_name"]].drop_duplicates("player_id")
        name_map = dict(
            zip(
                names["player_id"].astype(int),
                names["player_name"].astype(str),
                strict=True,
            )
        )
    frame["player_name"] = frame["player_id"].map(name_map).fillna(frame["player_id"].astype(str))
    rows: list[pd.DataFrame] = []
    for ranking_type, value_column, exposure_column in (
        ("offense", "offense_rapm", "offense_possessions"),
        ("defense", "defense_rapm", "defense_possessions"),
        ("overall", "net_rapm", "total_possessions"),
    ):
        side = frame.copy()
        side["ranking_type"] = ranking_type
        side["rating"] = side[value_column]
        side["ranking_possessions"] = side[exposure_column]
        side["exposure_eligible"] = side["ranking_possessions"].ge(minimum_possessions)
        side = side.sort_values(
            ["rating", "ranking_possessions", "player_id"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        side["rank"] = np.arange(1, len(side) + 1)
        eligible = side.loc[side["exposure_eligible"]]
        side["eligible_rank"] = pd.Series(
            np.arange(1, len(eligible) + 1), index=eligible.index, dtype="Int64"
        )
        rows.append(side)
    columns = [
        "ranking_type",
        "rank",
        "eligible_rank",
        "player_id",
        "player_name",
        "primary_team_id",
        "primary_team_tricode",
        "rating",
        "offense_rapm",
        "defense_rapm",
        "net_rapm",
        "offense_prior",
        "defense_prior",
        "stint_count",
        "ranking_possessions",
        "offense_possessions",
        "defense_possessions",
        "total_possessions",
        "primary_team_possessions",
        "exposure_eligible",
    ]
    return pd.concat(rows, ignore_index=True).loc[:, columns]


def _write_ranking_run(
    *,
    season: str,
    result: ForwardSeasonResult,
    rankings: pd.DataFrame,
    source_dir: Path,
    source_metadata: dict[str, object],
    artifacts_dir: Path,
    minimum_ranking_possessions: float,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"{RANKING_RUN_PREFIX}-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "offense_defense_rapm" / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        top_tables = {
            ranking_type: rankings.loc[
                (rankings["ranking_type"] == ranking_type)
                & rankings["exposure_eligible"]
            ].head(25)
            for ranking_type in ("offense", "defense", "overall")
        }
        tables = {
            "player_coefficients.parquet": result.coefficients,
            "cv_results.parquet": result.cv_results,
            "player_rankings.parquet": rankings,
            "top_25_offense.parquet": top_tables["offense"],
            "top_25_defense.parquet": top_tables["defense"],
            "top_25_overall.parquet": top_tables["overall"],
        }
        for filename, table in tables.items():
            table.to_parquet(temporary / filename, index=False)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": RANKING_MODEL_NAME,
            "season": season,
            "season_type": "regular",
            "ranking_scope": "retrospective_all_regular_season",
            "ranking_code_version": offense_defense_code_fingerprint(),
            "selected_lambda": result.selected_lambda,
            "league_offensive_rating": result.league_offensive_rating,
            "home_offense_shift": result.home_offense_shift,
            "minimum_ranking_possessions": minimum_ranking_possessions,
            "source_frozen_prior": source_metadata,
            "target_regular_outcomes_used_for_fit": True,
            "target_playoff_outcomes_used_for_fit": False,
            "forecast_artifact_updated": False,
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        artifacts = [
            {
                "filename": path.name,
                "row_count": len(tables[path.name]) if path.name in tables else None,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": artifacts}, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(output)
        validate_all_season_offense_defense_ranking_run(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_all_season_offense_defense_ranking_run(run_dir: Path | str) -> dict[str, object]:
    """Validate immutable retrospective O/D ranking artifacts."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if not str(manifest.get("run_id", "")).startswith(RANKING_RUN_PREFIX):
        raise ValueError("O/D ranking artifact has an invalid run id")
    if manifest.get("ranking_scope") != "retrospective_all_regular_season":
        raise ValueError("O/D ranking artifact does not declare a retrospective scope")
    if manifest.get("forecast_artifact_updated") is not False:
        raise ValueError("O/D ranking artifact modifies a forecast artifact")
    if manifest.get("target_regular_outcomes_used_for_fit") is not True:
        raise ValueError("O/D ranking artifact does not use completed regular outcomes")
    required = {
        "player_coefficients.parquet",
        "cv_results.parquet",
        "player_rankings.parquet",
        "top_25_offense.parquet",
        "top_25_defense.parquet",
        "top_25_overall.parquet",
        "metadata.json",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("O/D ranking artifact is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"O/D ranking artifact changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"O/D ranking artifact hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"O/D ranking artifact row count changed: {filename}")
    for ranking_type in ("offense", "defense", "overall"):
        top = pd.read_parquet(root / f"top_25_{ranking_type}.parquet")
        if len(top) > 25 or set(top["ranking_type"]) != {ranking_type}:
            raise ValueError(f"O/D ranking artifact has invalid {ranking_type} leaders")
        if not top["exposure_eligible"].all():
            raise ValueError(f"O/D ranking artifact includes ineligible {ranking_type} leaders")
    return manifest


def _write_run(
    evaluation: dict[str, object],
    *,
    source: ForwardSeasonResult,
    results: list[ForwardSeasonResult],
    artifacts_dir: Path,
    analytical_dir: Path,
    curated_dir: Path,
    lambda_grid: tuple[float, ...],
) -> Path:
    now = datetime.now(UTC)
    season = str(evaluation["season"])
    run_id = f"{RUN_PREFIX}-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / "frozen_offense_defense_rapm" / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "historical_player_coefficients.parquet": pd.concat(
                [result.coefficients for result in results], ignore_index=True
            ),
            "historical_cv_results.parquet": pd.concat(
                [result.cv_results.assign(season=result.season) for result in results],
                ignore_index=True,
            ),
            "frozen_player_priors.parquet": evaluation["frozen_player_priors"],
            "cohort_metrics.parquet": evaluation["cohort_metrics"],
            "possession_predictions.parquet": evaluation["possession_predictions"],
            "game_predictions.parquet": evaluation["game_predictions"],
            "regular_game_predictions.parquet": evaluation["regular_game_predictions"],
            "team_net_rating_predictions.parquet": evaluation["team_net_rating_predictions"],
            "team_net_rating_metrics.parquet": evaluation["team_net_rating_metrics"],
            "team_win_predictions.parquet": evaluation["team_win_predictions"],
            "team_win_metrics.parquet": evaluation["team_win_metrics"],
            "pythagorean_calibration_team_seasons.parquet": evaluation[
                "pythagorean_calibration_team_seasons"
            ],
        }
        for filename, table in tables.items():
            table.to_parquet(temporary / filename, index=False)
        (temporary / "source_state.json").write_text(
            json.dumps(evaluation["source_state"], indent=2, sort_keys=True) + "\n"
        )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "model": MODEL_NAME,
            "evaluation_code_version": offense_defense_code_fingerprint(),
            "season": season,
            "source_season": evaluation["source_season"],
            "historical_seasons": list(HISTORICAL_SEASONS),
            "historical_training_variant": "regular_only",
            "lambda_grid": list(lambda_grid),
            "historical_selected_lambdas": {
                result.season: result.selected_lambda for result in results
            },
            "source_league_offensive_rating": source.league_offensive_rating,
            "source_home_offense_shift": source.home_offense_shift,
            "information_boundary": "all fitted information ends with source regular season",
            "oracle_information": "target-season realized lineup exposure only",
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {
                "filename": path.name,
                "row_count": len(tables[path.name]) if path.name in tables else None,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary.replace(output)
        validate_frozen_offense_defense_run(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_frozen_offense_defense_run(run_dir: Path | str) -> dict[str, object]:
    """Validate the immutable O/D artifact and its no-refit declaration."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    code_version = manifest.get("evaluation_code_version")
    if not isinstance(code_version, str) or not code_version.startswith("sha256:"):
        raise ValueError("Frozen O/D artifact has no valid evaluation code fingerprint")
    required = {
        "historical_player_coefficients.parquet",
        "historical_cv_results.parquet",
        "frozen_player_priors.parquet",
        "cohort_metrics.parquet",
        "possession_predictions.parquet",
        "game_predictions.parquet",
        "regular_game_predictions.parquet",
        "team_net_rating_predictions.parquet",
        "team_net_rating_metrics.parquet",
        "team_win_predictions.parquet",
        "team_win_metrics.parquet",
        "pythagorean_calibration_team_seasons.parquet",
        "source_state.json",
        "metadata.json",
    }
    records = {record["filename"]: record for record in manifest["artifacts"]}
    if not required <= set(records):
        raise ValueError("Frozen O/D artifact is missing required files")
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["byte_count"]:
            raise ValueError(f"Frozen O/D artifact changed: {filename}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen O/D artifact hash changed: {filename}")
        if record["row_count"] is not None and len(pd.read_parquet(path)) != record["row_count"]:
            raise ValueError(f"Frozen O/D artifact row count changed: {filename}")
    source = json.loads((root / "source_state.json").read_text())
    if source.get("target_season_refit") is not False:
        raise ValueError("Frozen O/D artifact permits a target-season refit")
    if source.get("target_regular_outcomes_used_for_fit") is not False:
        raise ValueError("Frozen O/D artifact uses target regular outcomes")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate frozen forward O/D RAPM")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    args = parser.parse_args()
    run_dir = train_frozen_offense_defense_rapm(
        season=args.season,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"Frozen O/D RAPM evaluation: run={run_dir}{tracking_text}")


def rankings_main() -> None:
    """Build retrospective offense and defense rankings for one completed season."""

    parser = argparse.ArgumentParser(description="Build all-season O/D RAPM rankings")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--frozen-artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--minimum-ranking-possessions",
        type=float,
        default=DEFAULT_MINIMUM_RANKING_POSSESSIONS,
    )
    args = parser.parse_args()
    run_dir = build_all_season_offense_defense_rankings(
        season=args.season,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        frozen_artifacts_dir=args.frozen_artifacts_dir,
        minimum_ranking_possessions=args.minimum_ranking_possessions,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(f"All-season O/D RAPM rankings: run={run_dir}{tracking_text}")


if __name__ == "__main__":
    main()
