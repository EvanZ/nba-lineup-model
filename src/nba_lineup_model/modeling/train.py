from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.evaluation.metrics import (
    game_margin_rmse,
    mean_absolute_error,
    mean_squared_error,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    BaselineRunManifest,
    ChronologicalFold,
    ChronologicalSplitConfig,
    RapmStintManifest,
)
from nba_lineup_model.modeling.stints import (
    build_rapm_stint_dataset,
    modeling_code_fingerprint,
    read_rapm_stints,
)
from nba_lineup_model.models.baselines import (
    FittedMeanModel,
    RidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)

DEFAULT_LAMBDA_GRID = (
    0.0,
    0.00001,
    0.00003,
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
)


@dataclass(frozen=True)
class GameFold:
    """Concrete train and validation game IDs for one expanding fold."""

    fold: int
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]


@dataclass(frozen=True)
class GameSplitPlan:
    """All chronological folds plus the untouched final test."""

    ordered_games: pd.DataFrame
    folds: tuple[GameFold, ...]
    final_train_game_ids: tuple[str, ...]
    final_test_game_ids: tuple[str, ...]


@dataclass(frozen=True)
class BaselineExperiment:
    """In-memory outputs ready for atomic model-run persistence."""

    split_plan: GameSplitPlan
    player_ids: tuple[int, ...]
    team_ids: tuple[int, ...]
    selected_team_lambda: float
    selected_rapm_lambda: float
    cv_results: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    player_rankings: pd.DataFrame
    team_ratings: pd.DataFrame
    game_splits: pd.DataFrame
    player_columns: dict[int, int]
    team_columns: dict[int, int]
    model_parameters: dict[str, Any]


def chronological_game_splits(
    stints: pd.DataFrame,
    config: ChronologicalSplitConfig,
) -> GameSplitPlan:
    """Create expanding folds without dividing any NBA game date."""

    required = {"game_id", "game_date", "game_time_utc"}
    missing = required - set(stints.columns)
    if missing:
        raise ValueError(f"Stints missing split columns: {sorted(missing)}")
    games = (
        stints.loc[:, ["game_id", "game_date", "game_time_utc"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_time_utc", "game_id"], kind="stable")
        .reset_index(drop=True)
    )
    if games["game_id"].duplicated().any():
        raise ValueError("Each game must have one game time")
    game_count = len(games)
    target_test_count = max(1, int(np.ceil(game_count * config.test_fraction)))
    target_validation_count = max(
        1,
        int(np.floor(game_count * config.validation_fraction)),
    )
    target_initial_train_count = (
        game_count - target_test_count - config.cv_folds * target_validation_count
    )
    if target_initial_train_count < 1:
        raise ValueError("Not enough games for the requested chronological splits")

    identifiers = tuple(games["game_id"].astype(str))
    date_boundaries = (
        games.groupby("game_date", sort=False).size().cumsum().iloc[:-1].astype(int).tolist()
    )
    desired_boundaries = [
        target_initial_train_count + fold * target_validation_count
        for fold in range(config.cv_folds + 1)
    ]
    boundaries = _nearest_increasing_boundaries(
        desired_boundaries,
        date_boundaries,
    )
    folds: list[GameFold] = []
    for fold in range(config.cv_folds):
        validation_start = boundaries[fold]
        validation_end = boundaries[fold + 1]
        folds.append(
            GameFold(
                fold=fold,
                train_game_ids=identifiers[:validation_start],
                validation_game_ids=identifiers[validation_start:validation_end],
            )
        )
    final_train_end = boundaries[-1]
    return GameSplitPlan(
        ordered_games=games,
        folds=tuple(folds),
        final_train_game_ids=identifiers[:final_train_end],
        final_test_game_ids=identifiers[final_train_end:],
    )


def _nearest_increasing_boundaries(
    desired: list[int],
    available: list[int],
) -> list[int]:
    if len(available) < len(desired):
        raise ValueError("Not enough distinct game dates for chronological splits")
    selected: list[int] = []
    for index, target in enumerate(desired):
        minimum = selected[-1] + 1 if selected else 1
        remaining = len(desired) - index - 1
        candidates = [
            boundary
            for boundary in available
            if boundary >= minimum and sum(later > boundary for later in available) >= remaining
        ]
        if not candidates:
            raise ValueError("Cannot place chronological boundaries on game dates")
        selected.append(min(candidates, key=lambda boundary: (abs(boundary - target), boundary)))
    return selected


def fit_baseline_experiment(
    stints: pd.DataFrame,
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    split_config: ChronologicalSplitConfig | None = None,
    minimum_ranking_possessions: float = 500.0,
    player_bios: pd.DataFrame | None = None,
) -> BaselineExperiment:
    """Tune, test, and refit null, team, and canonical one-number RAPM models."""

    config = split_config or ChronologicalSplitConfig()
    _validate_experiment_inputs(
        stints,
        lambda_grid,
        minimum_ranking_possessions,
    )
    split_plan = chronological_game_splits(stints, config)
    player_ids = entity_vocabulary(
        stints,
        "home_player_ids",
        "away_player_ids",
        multiple=True,
    )
    team_ids = entity_vocabulary(
        stints,
        "home_team_id",
        "away_team_id",
        multiple=False,
    )
    player_columns = vocabulary_mapping(player_ids)
    team_columns = vocabulary_mapping(team_ids)
    player_matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    team_matrix = signed_entity_matrix(
        stints,
        "home_team_id",
        "away_team_id",
        team_columns,
        multiple=False,
    )
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    game_ids = stints["game_id"].astype(str).to_numpy()

    cv_results = _cross_validate_models(
        stints,
        target,
        weights,
        game_ids,
        team_matrix,
        player_matrix,
        split_plan,
        lambda_grid,
    )
    selected_team_lambda = _select_lambda(cv_results, "team")
    selected_rapm_lambda = _select_lambda(cv_results, "rapm")

    final_train_mask = np.isin(game_ids, split_plan.final_train_game_ids)
    final_test_mask = np.isin(game_ids, split_plan.final_test_game_ids)
    evaluation = _evaluate_final_test(
        stints,
        target,
        weights,
        team_matrix,
        player_matrix,
        final_train_mask,
        final_test_mask,
        selected_team_lambda,
        selected_rapm_lambda,
    )

    final_team_model = RidgeLineupModel(selected_team_lambda).fit(
        team_matrix,
        target,
        weights,
    )
    final_rapm_model = RidgeLineupModel(selected_rapm_lambda).fit(
        player_matrix,
        target,
        weights,
    )
    player_rankings = _player_rankings(
        stints,
        player_ids,
        final_rapm_model.coef_,
        minimum_ranking_possessions,
        player_bios,
    )
    team_ratings = _team_ratings(
        stints,
        team_ids,
        final_team_model.coef_,
    )
    game_splits = _game_splits_frame(split_plan)
    parameters = {
        "lambda_convention": (
            "mean weighted squared error plus lambda times squared coefficient norm"
        ),
        "possession_weight_normalization": "sample weights normalized to mean one",
        "team": {
            "selected_lambda": selected_team_lambda,
            "final_sklearn_alpha": final_team_model.sklearn_alpha,
            "all_season_intercept_home_court": final_team_model.intercept_,
        },
        "rapm": {
            "selected_lambda": selected_rapm_lambda,
            "final_sklearn_alpha": final_rapm_model.sklearn_alpha,
            "all_season_intercept_home_court": final_rapm_model.intercept_,
        },
    }
    return BaselineExperiment(
        split_plan=split_plan,
        player_ids=player_ids,
        team_ids=team_ids,
        selected_team_lambda=selected_team_lambda,
        selected_rapm_lambda=selected_rapm_lambda,
        cv_results=cv_results,
        test_metrics=evaluation["metrics"],
        test_predictions=evaluation["predictions"],
        player_rankings=player_rankings,
        team_ratings=team_ratings,
        game_splits=game_splits,
        player_columns=player_columns,
        team_columns=team_columns,
        model_parameters=parameters,
    )


def train_regular_season_baselines(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    split_config: ChronologicalSplitConfig | None = None,
    minimum_ranking_possessions: float = 500.0,
) -> tuple[BaselineRunManifest, Path]:
    """Build RAPM stints, run all baselines, and atomically persist artifacts."""

    config = split_config or ChronologicalSplitConfig()
    dataset_manifest = build_rapm_stint_dataset(
        season,
        curated_dir=curated_dir,
        analytical_dir=analytical_dir,
    )
    stints = read_rapm_stints(season, analytical_dir)
    bios_path = Path(curated_dir) / "player_seasons" / season / "regular" / "part-00000.parquet"
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    experiment = fit_baseline_experiment(
        stints,
        lambda_grid=lambda_grid,
        split_config=config,
        minimum_ranking_possessions=minimum_ranking_possessions,
        player_bios=player_bios,
    )
    return _write_experiment(
        season,
        dataset_manifest,
        experiment,
        lambda_grid,
        config,
        minimum_ranking_possessions,
        analytical_dir,
        artifacts_dir,
    )


def validate_baseline_run(run_dir: Path | str) -> BaselineRunManifest:
    """Require every recorded model artifact to match its manifest."""

    root = Path(run_dir)
    manifest = BaselineRunManifest.model_validate_json((root / "manifest.json").read_text())
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    expected = {artifact.filename for artifact in manifest.artifacts}
    if actual != expected:
        raise ValueError("Model run files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Model artifact byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Model artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None:
            if len(pd.read_parquet(path)) != artifact.row_count:
                raise ValueError(f"Model artifact rows changed: {artifact.filename}")
    return manifest


def _cross_validate_models(
    stints: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    game_ids: np.ndarray,
    team_matrix: sparse.csr_matrix,
    player_matrix: sparse.csr_matrix,
    split_plan: GameSplitPlan,
    lambda_grid: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in split_plan.folds:
        train_mask = np.isin(game_ids, fold.train_game_ids)
        validation_mask = np.isin(game_ids, fold.validation_game_ids)
        mean_model = FittedMeanModel.fit(target[train_mask], weights[train_mask])
        mean_predictions = mean_model.predict(int(validation_mask.sum()))
        rows.append(
            _metric_row(
                stints.loc[validation_mask],
                target[validation_mask],
                mean_predictions,
                weights[validation_mask],
                model="mean",
                fold=fold.fold,
                regularization=None,
                train_game_count=len(fold.train_game_ids),
            )
        )
        for model_name, matrix in (
            ("team", team_matrix),
            ("rapm", player_matrix),
        ):
            for regularization in lambda_grid:
                model = RidgeLineupModel(regularization).fit(
                    matrix[train_mask],
                    target[train_mask],
                    weights[train_mask],
                )
                predictions = model.predict(matrix[validation_mask])
                rows.append(
                    _metric_row(
                        stints.loc[validation_mask],
                        target[validation_mask],
                        predictions,
                        weights[validation_mask],
                        model=model_name,
                        fold=fold.fold,
                        regularization=regularization,
                        train_game_count=len(fold.train_game_ids),
                    )
                )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["model", "regularization", "fold"],
            kind="stable",
            na_position="first",
        )
        .reset_index(drop=True)
    )


def _select_lambda(cv_results: pd.DataFrame, model: str) -> float:
    candidates = cv_results.loc[cv_results["model"].eq(model)].copy()
    summary = (
        candidates.groupby("regularization", as_index=False)
        .agg(
            squared_error_sum=("squared_error_sum", "sum"),
            validation_possessions=("validation_possessions", "sum"),
        )
        .sort_values("regularization", kind="stable")
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    best = summary.sort_values(
        ["weighted_mse", "regularization"],
        kind="stable",
    ).iloc[0]
    return float(best["regularization"])


def _evaluate_final_test(
    stints: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    team_matrix: sparse.csr_matrix,
    player_matrix: sparse.csr_matrix,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    team_lambda: float,
    rapm_lambda: float,
) -> dict[str, pd.DataFrame]:
    mean_model = FittedMeanModel.fit(target[train_mask], weights[train_mask])
    team_model = RidgeLineupModel(team_lambda).fit(
        team_matrix[train_mask],
        target[train_mask],
        weights[train_mask],
    )
    rapm_model = RidgeLineupModel(rapm_lambda).fit(
        player_matrix[train_mask],
        target[train_mask],
        weights[train_mask],
    )
    predictions = {
        "mean": mean_model.predict(int(test_mask.sum())),
        "team": team_model.predict(team_matrix[test_mask]),
        "rapm": rapm_model.predict(player_matrix[test_mask]),
    }
    metric_rows = [
        _metric_row(
            stints.loc[test_mask],
            target[test_mask],
            prediction,
            weights[test_mask],
            model=model_name,
            fold=None,
            regularization={
                "mean": None,
                "team": team_lambda,
                "rapm": rapm_lambda,
            }[model_name],
            train_game_count=int(stints.loc[train_mask, "game_id"].nunique()),
        )
        for model_name, prediction in predictions.items()
    ]
    metrics = pd.DataFrame(metric_rows)
    mean_mse = float(metrics.loc[metrics["model"].eq("mean"), "weighted_mse"].iloc[0])
    team_mse = float(metrics.loc[metrics["model"].eq("team"), "weighted_mse"].iloc[0])
    metrics["skill_vs_mean"] = metrics["weighted_mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    metrics["skill_vs_team"] = metrics["weighted_mse"].map(
        lambda value: skill_score(float(value), team_mse)
    )

    test_rows = stints.loc[
        test_mask,
        [
            "game_id",
            "game_time_utc",
            "stint_index",
            "home_team_id",
            "away_team_id",
            "possessions",
            "home_margin",
            "target_home_net_rating",
        ],
    ].reset_index(drop=True)
    for model_name, prediction in predictions.items():
        test_rows[f"prediction_{model_name}"] = prediction
        test_rows[f"predicted_margin_{model_name}"] = prediction * test_rows["possessions"] / 100.0
    return {"metrics": metrics, "predictions": test_rows}


def _metric_row(
    stints: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    *,
    model: str,
    fold: int | None,
    regularization: float | None,
    train_game_count: int,
) -> dict[str, Any]:
    mse = mean_squared_error(actual, predicted, weights)
    return {
        "model": model,
        "fold": fold,
        "regularization": regularization,
        "train_game_count": train_game_count,
        "validation_game_count": int(stints["game_id"].nunique()),
        "validation_stint_count": len(stints),
        "validation_possessions": float(np.sum(weights)),
        "squared_error_sum": float(mse * np.sum(weights)),
        "weighted_mse": mse,
        "weighted_rmse": rmse(actual, predicted, weights),
        "weighted_mae": mean_absolute_error(actual, predicted, weights),
        "game_margin_rmse": game_margin_rmse(
            stints["game_id"],
            actual,
            predicted,
            weights,
        ),
    }


def _player_rankings(
    stints: pd.DataFrame,
    player_ids: tuple[int, ...],
    coefficients: np.ndarray,
    minimum_possessions: float,
    player_bios: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for side, sign in (("home", 1), ("away", -1)):
        columns = [
            f"{side}_player_ids",
            f"{side}_team_id",
            f"{side}_team_tricode",
            "possessions",
            "duration_seconds",
            "home_margin",
        ]
        side_frame = stints.loc[:, columns].explode(
            f"{side}_player_ids",
            ignore_index=True,
        )
        side_frame = side_frame.rename(
            columns={
                f"{side}_player_ids": "player_id",
                f"{side}_team_id": "team_id",
                f"{side}_team_tricode": "team_tricode",
            }
        )
        side_frame["player_margin"] = sign * side_frame["home_margin"]
        rows.append(side_frame)
    exposure = pd.concat(rows, ignore_index=True)
    exposure["player_id"] = exposure["player_id"].astype("int64")
    totals = exposure.groupby("player_id", as_index=False).agg(
        stint_count=("player_id", "size"),
        possessions=("possessions", "sum"),
        seconds=("duration_seconds", "sum"),
        point_margin=("player_margin", "sum"),
    )
    totals["raw_on_court_net_rating"] = 100.0 * totals["point_margin"] / totals["possessions"]
    team_exposure = (
        exposure.groupby(
            ["player_id", "team_id", "team_tricode"],
            as_index=False,
        )["possessions"]
        .sum()
        .sort_values(
            ["player_id", "possessions", "team_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("player_id")
        .rename(
            columns={
                "team_id": "primary_team_id",
                "team_tricode": "primary_team_tricode",
                "possessions": "primary_team_possessions",
            }
        )
    )
    rankings = pd.DataFrame(
        {
            "player_id": player_ids,
            "rapm": coefficients,
        }
    ).merge(totals, on="player_id", how="left", validate="one_to_one")
    rankings = rankings.merge(
        team_exposure,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    name_map: dict[int, str] = {}
    if player_bios is not None:
        required = {"player_id", "player_name"}
        if required <= set(player_bios.columns):
            name_map = {
                int(row.player_id): str(row.player_name)
                for row in player_bios.loc[:, sorted(required)].itertuples(index=False)
            }
    rankings["player_name"] = (
        rankings["player_id"].map(name_map).fillna(rankings["player_id"].astype(str))
    )
    rankings = rankings.sort_values(
        ["rapm", "possessions", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    rankings["rank"] = np.arange(1, len(rankings) + 1)
    if len(rankings) == 1:
        rankings["percentile"] = 100.0
    else:
        rankings["percentile"] = 100.0 * (len(rankings) - rankings["rank"]) / (len(rankings) - 1)
    rankings["exposure_eligible"] = rankings["possessions"].ge(minimum_possessions)
    eligible = rankings.loc[rankings["exposure_eligible"]].sort_values(
        ["rapm", "possessions", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    eligible_ranks = pd.Series(
        np.arange(1, len(eligible) + 1),
        index=eligible.index,
        dtype="Int64",
    )
    rankings["eligible_rank"] = eligible_ranks
    return rankings.loc[
        :,
        [
            "rank",
            "eligible_rank",
            "player_id",
            "player_name",
            "primary_team_id",
            "primary_team_tricode",
            "rapm",
            "percentile",
            "raw_on_court_net_rating",
            "stint_count",
            "possessions",
            "seconds",
            "point_margin",
            "primary_team_possessions",
            "exposure_eligible",
        ],
    ]


def _team_ratings(
    stints: pd.DataFrame,
    team_ids: tuple[int, ...],
    coefficients: np.ndarray,
) -> pd.DataFrame:
    tricode_map: dict[int, str] = {}
    for side in ("home", "away"):
        tricode_map.update(
            zip(
                stints[f"{side}_team_id"].astype(int),
                stints[f"{side}_team_tricode"].astype(str),
                strict=False,
            )
        )
    ratings = pd.DataFrame(
        {
            "team_id": team_ids,
            "team_rating": coefficients,
        }
    )
    ratings["team_tricode"] = ratings["team_id"].map(tricode_map)
    ratings = ratings.sort_values(
        ["team_rating", "team_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ratings["rank"] = np.arange(1, len(ratings) + 1)
    return ratings.loc[:, ["rank", "team_id", "team_tricode", "team_rating"]]


def _game_splits_frame(split_plan: GameSplitPlan) -> pd.DataFrame:
    game_times = dict(
        zip(
            split_plan.ordered_games["game_id"].astype(str),
            split_plan.ordered_games["game_time_utc"],
            strict=True,
        )
    )
    game_dates = dict(
        zip(
            split_plan.ordered_games["game_id"].astype(str),
            split_plan.ordered_games["game_date"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for fold in split_plan.folds:
        for role, identifiers in (
            ("train", fold.train_game_ids),
            ("validation", fold.validation_game_ids),
        ):
            rows.extend(
                {
                    "split": f"cv_{fold.fold}",
                    "role": role,
                    "game_id": game_id,
                    "game_date": game_dates[game_id],
                    "game_time_utc": game_times[game_id],
                }
                for game_id in identifiers
            )
    for role, identifiers in (
        ("train", split_plan.final_train_game_ids),
        ("test", split_plan.final_test_game_ids),
    ):
        rows.extend(
            {
                "split": "final",
                "role": role,
                "game_id": game_id,
                "game_date": game_dates[game_id],
                "game_time_utc": game_times[game_id],
            }
            for game_id in identifiers
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["split", "game_time_utc", "game_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _write_experiment(
    season: str,
    dataset_manifest: RapmStintManifest,
    experiment: BaselineExperiment,
    lambda_grid: tuple[float, ...],
    split_config: ChronologicalSplitConfig,
    minimum_ranking_possessions: float,
    analytical_dir: Path | str,
    artifacts_dir: Path | str,
) -> tuple[BaselineRunManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"baseline-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(artifacts_dir) / "rapm" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        parquet_outputs = {
            "cv_results.parquet": experiment.cv_results,
            "test_metrics.parquet": experiment.test_metrics,
            "test_predictions.parquet": experiment.test_predictions,
            "player_rankings.parquet": experiment.player_rankings,
            "team_ratings.parquet": experiment.team_ratings,
            "game_splits.parquet": experiment.game_splits,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        json_outputs = {
            "player_columns.json": {
                str(identifier): column for identifier, column in experiment.player_columns.items()
            },
            "team_columns.json": {
                str(identifier): column for identifier, column in experiment.team_columns.items()
            },
            "model_parameters.json": experiment.model_parameters,
        }
        for filename, payload in json_outputs.items():
            (temporary_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )

        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=(
                    len(parquet_outputs[path.name]) if path.name in parquet_outputs else None
                ),
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in sorted(temporary_dir.iterdir())
            if path.is_file()
        )
        dataset_dir = Path(analytical_dir) / "rapm_stints" / season / "regular"
        fold_records = tuple(
            ChronologicalFold(
                fold=fold.fold,
                train_game_count=len(fold.train_game_ids),
                validation_game_count=len(fold.validation_game_ids),
                train_first_game_id=fold.train_game_ids[0],
                train_last_game_id=fold.train_game_ids[-1],
                validation_first_game_id=fold.validation_game_ids[0],
                validation_last_game_id=fold.validation_game_ids[-1],
            )
            for fold in experiment.split_plan.folds
        )
        manifest = BaselineRunManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            modeling_code_version=modeling_code_fingerprint(),
            dataset_manifest_sha256=_sha256_file(dataset_dir / "_manifest.json"),
            dataset_part_sha256=dataset_manifest.part_sha256,
            stint_count=dataset_manifest.row_count,
            game_count=dataset_manifest.source_game_count,
            player_count=len(experiment.player_ids),
            team_count=len(experiment.team_ids),
            split_config=split_config,
            folds=fold_records,
            final_train_game_count=len(experiment.split_plan.final_train_game_ids),
            final_test_game_count=len(experiment.split_plan.final_test_game_ids),
            lambda_grid=lambda_grid,
            selected_team_lambda=experiment.selected_team_lambda,
            selected_rapm_lambda=experiment.selected_rapm_lambda,
            minimum_ranking_possessions=minimum_ranking_possessions,
            artifacts=artifacts,
        )
        (temporary_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        temporary_dir.replace(run_dir)
        validate_baseline_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _validate_experiment_inputs(
    stints: pd.DataFrame,
    lambda_grid: tuple[float, ...],
    minimum_ranking_possessions: float,
) -> None:
    required = {
        "game_id",
        "game_date",
        "game_time_utc",
        "home_team_id",
        "away_team_id",
        "home_team_tricode",
        "away_team_tricode",
        "home_player_ids",
        "away_player_ids",
        "duration_seconds",
        "home_margin",
        "possessions",
        "target_home_net_rating",
    }
    missing = required - set(stints.columns)
    if missing:
        raise ValueError(f"RAPM stints missing columns: {sorted(missing)}")
    if stints.empty:
        raise ValueError("RAPM stints cannot be empty")
    if not stints["possessions"].gt(0).all():
        raise ValueError("RAPM stint weights must be positive")
    if len(lambda_grid) < 2 or any(value < 0 for value in lambda_grid):
        raise ValueError("At least two non-negative lambda values are required")
    if len(set(lambda_grid)) != len(lambda_grid):
        raise ValueError("Lambda grid values must be unique")
    if minimum_ranking_possessions < 0:
        raise ValueError("Minimum ranking possessions cannot be negative")


def _parse_lambdas(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Lambdas must be comma-separated numbers") from exc
    if len(values) < 2 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Provide at least two non-negative lambdas")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Lambda values must be unique")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Train regular-season mean, team, and canonical one-number RAPM baselines.")
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--lambdas",
        type=_parse_lambdas,
        default=DEFAULT_LAMBDA_GRID,
        help="Comma-separated normalized ridge lambda grid",
    )
    parser.add_argument(
        "--minimum-ranking-possessions",
        type=float,
        default=500.0,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_config = ChronologicalSplitConfig(
        cv_folds=args.cv_folds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    manifest, run_dir = train_regular_season_baselines(
        args.season,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        lambda_grid=args.lambdas,
        split_config=split_config,
        minimum_ranking_possessions=args.minimum_ranking_possessions,
    )
    print(
        f"{manifest.season} regular baselines: "
        f"stints={manifest.stint_count}, games={manifest.game_count}, "
        f"players={manifest.player_count}, "
        f"team_lambda={manifest.selected_team_lambda:g}, "
        f"rapm_lambda={manifest.selected_rapm_lambda:g}; "
        f"run={run_dir}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
