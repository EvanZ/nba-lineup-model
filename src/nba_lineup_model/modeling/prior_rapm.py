"""Forward-prior-centered one-number RAPM experiments."""

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

from nba_lineup_model.evaluation.metrics import (
    game_margin_rmse,
    mean_absolute_error,
    mean_squared_error,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.modeling.stints import (
    build_rapm_stints_from_curated_games,
    build_rapm_stints_from_legacy_processed_games,
    read_rapm_stints,
)
from nba_lineup_model.modeling.train import (
    DEFAULT_LAMBDA_GRID,
    GameSplitPlan,
    _player_rankings,
    chronological_game_splits,
)
from nba_lineup_model.models.baselines import (
    FittedMeanModel,
    PriorCenteredRidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)

PRIOR_MEAN_COLUMN = "lagged_rapm_prior"
HISTORICAL_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)


@dataclass(frozen=True)
class PriorRapmExperiment:
    """Selection, holdout, and ranking outputs for one prior-centered RAPM."""

    split_plan: GameSplitPlan
    player_ids: tuple[int, ...]
    player_columns: dict[int, int]
    selected_lambda: float
    cv_results: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    player_rankings: pd.DataFrame
    player_priors: pd.DataFrame
    model_parameters: dict[str, object]


@dataclass(frozen=True)
class ForwardLaggedRapmSeason:
    """One completed forward season used to form the next year's prior."""

    season: str
    selected_lambda: float
    cv_results: pd.DataFrame
    player_estimates: pd.DataFrame
    player_priors: pd.DataFrame


def fit_prior_rapm_experiment(
    stints: pd.DataFrame,
    player_priors: pd.DataFrame,
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    split_config: ChronologicalSplitConfig | None = None,
    minimum_ranking_possessions: float = 500.0,
    player_bios: pd.DataFrame | None = None,
) -> PriorRapmExperiment:
    """Tune one-number RAPM centered on pre-season player prior means.

    ``player_priors`` must be frozen before the target season. Missing players
    receive the documented cold-start prior of zero.
    """

    _validate_inputs(stints, player_priors, lambda_grid)
    config = split_config or ChronologicalSplitConfig()
    split_plan = chronological_game_splits(stints, config)
    player_ids = entity_vocabulary(
        stints,
        "home_player_ids",
        "away_player_ids",
        multiple=True,
    )
    player_columns = vocabulary_mapping(player_ids)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    prior, prior_frame = _prior_vector(player_ids, player_priors)
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    game_ids = stints["game_id"].astype(str).to_numpy()

    cv_results = _cross_validate(
        stints,
        matrix,
        prior,
        target,
        weights,
        game_ids,
        split_plan,
        lambda_grid,
    )
    selected_lambda = _select_lambda(cv_results)
    train_mask = np.isin(game_ids, split_plan.final_train_game_ids)
    test_mask = np.isin(game_ids, split_plan.final_test_game_ids)
    prior_model = PriorCenteredRidgeLineupModel(selected_lambda).fit(
        matrix[train_mask], target[train_mask], weights[train_mask], prior
    )
    mean_model = FittedMeanModel.fit(target[train_mask], weights[train_mask])
    prior_prediction = prior_model.predict(matrix[test_mask])
    mean_prediction = mean_model.predict(int(test_mask.sum()))
    test_metrics = pd.DataFrame(
        [
            _metric_row(
                stints.loc[test_mask],
                target[test_mask],
                mean_prediction,
                weights[test_mask],
                "mean",
                None,
            ),
            _metric_row(
                stints.loc[test_mask],
                target[test_mask],
                prior_prediction,
                weights[test_mask],
                "prior_rapm",
                selected_lambda,
            ),
        ]
    )
    mean_mse = float(test_metrics.loc[test_metrics["model"].eq("mean"), "weighted_mse"].iloc[0])
    test_metrics["skill_vs_mean"] = test_metrics["weighted_mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    test_predictions = stints.loc[
        test_mask,
        [
            "game_id",
            "game_time_utc",
            "stint_index",
            "possessions",
            "home_margin",
            "target_home_net_rating",
        ],
    ].reset_index(drop=True)
    test_predictions["prediction_prior_rapm"] = prior_prediction
    test_predictions["predicted_margin_prior_rapm"] = (
        prior_prediction * test_predictions["possessions"] / 100.0
    )
    rankings = _player_rankings(
        stints,
        player_ids,
        prior_model.coef_,
        minimum_ranking_possessions,
        player_bios,
    ).merge(prior_frame, on="player_id", how="left", validate="one_to_one")
    adjustments = dict(zip(player_ids, prior_model.adjustment_, strict=True))
    rankings["rapm_adjustment_from_prior"] = rankings["player_id"].map(adjustments)
    rankings = rankings.sort_values(
        ["rapm", "possessions", "player_id"], ascending=[False, False, True], kind="stable"
    ).reset_index(drop=True)
    rankings["rank"] = np.arange(1, len(rankings) + 1)
    parameters: dict[str, object] = {
        "model": "prior_centered_ridge_rapm",
        "prior_mean_column": PRIOR_MEAN_COLUMN,
        "cold_start_prior": 0.0,
        "regularization_convention": (
            "mean weighted squared error plus lambda times squared coefficient "
            "deviation from prior"
        ),
        "selected_lambda": selected_lambda,
        "sklearn_alpha": prior_model.sklearn_alpha,
        "intercept_home_court": prior_model.intercept_,
    }
    return PriorRapmExperiment(
        split_plan=split_plan,
        player_ids=player_ids,
        player_columns=player_columns,
        selected_lambda=selected_lambda,
        cv_results=cv_results,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        player_rankings=rankings,
        player_priors=prior_frame,
        model_parameters=parameters,
    )


def fit_forward_lagged_rapm_season(
    season: str,
    stints: pd.DataFrame,
    player_priors: pd.DataFrame | None = None,
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    split_config: ChronologicalSplitConfig | None = None,
) -> ForwardLaggedRapmSeason:
    """Tune on chronological folds, then refit a completed season on all rows.

    This output is a prior source for a *later* season, never an evaluation of
    the current one. The first season receives an empty prior table and is
    therefore canonical zero-centered RAPM.
    """

    priors = (
        player_priors
        if player_priors is not None
        else pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN])
    )
    _validate_inputs(stints, priors, lambda_grid)
    config = split_config or ChronologicalSplitConfig()
    split_plan = chronological_game_splits(stints, config)
    player_ids = entity_vocabulary(
        stints, "home_player_ids", "away_player_ids", multiple=True
    )
    player_columns = vocabulary_mapping(player_ids)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    prior, prior_frame = _prior_vector(player_ids, priors)
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    game_ids = stints["game_id"].astype(str).to_numpy()
    cv_results = _cross_validate(
        stints,
        matrix,
        prior,
        target,
        weights,
        game_ids,
        split_plan,
        lambda_grid,
    )
    selected_lambda = _select_lambda(cv_results)
    fitted = PriorCenteredRidgeLineupModel(selected_lambda).fit(
        matrix,
        target,
        weights,
        prior,
    )
    estimates = pd.DataFrame(
        {
            "season": season,
            "player_id": player_ids,
            "rapm": fitted.coef_,
            "prior_rapm": prior,
            "rapm_adjustment_from_prior": fitted.adjustment_,
        }
    ).merge(prior_frame.loc[:, ["player_id", "prior_available"]], on="player_id")
    estimates["selected_lambda"] = selected_lambda
    estimates = estimates.sort_values("player_id", kind="stable").reset_index(drop=True)
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=selected_lambda,
        cv_results=cv_results,
        player_estimates=estimates,
        player_priors=prior_frame,
    )


def fit_forward_lagged_rapm_history(
    stints_by_season: dict[str, pd.DataFrame],
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    split_config: ChronologicalSplitConfig | None = None,
) -> tuple[ForwardLaggedRapmSeason, ...]:
    """Fit zero-centered then prior-centered RAPM in strict season order."""

    seasons = tuple(sorted(stints_by_season, key=lambda value: int(value[:4])))
    if not seasons:
        raise ValueError("Lagged RAPM history requires at least one season")
    results: list[ForwardLaggedRapmSeason] = []
    priors: pd.DataFrame | None = None
    for season in seasons:
        result = fit_forward_lagged_rapm_season(
            season,
            stints_by_season[season],
            priors,
            lambda_grid=lambda_grid,
            split_config=split_config,
        )
        results.append(result)
        priors = result.player_estimates.loc[:, ["player_id", "rapm"]].rename(
            columns={"rapm": PRIOR_MEAN_COLUMN}
        )
    return tuple(results)


def train_forward_lagged_rapm(
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    include_historical_playoffs: bool = False,
) -> Path:
    """Fit historical forward priors, then evaluate 2025-26 on its fixed holdout.

    The historical seasons are fitted in order on all usable regular-season
    stints.  Each completed season supplies the next season's coefficient
    prior.  The 2025-26 fit retains the project's standard chronological
    1,044-game train / 186-game holdout contract.
    """

    variant = "regular_plus_playoffs" if include_historical_playoffs else "regular_only"
    priors: pd.DataFrame | None = None
    estimates: list[pd.DataFrame] = []
    cv_results: list[pd.DataFrame] = []
    exclusions: dict[str, list[str]] = {}
    playoff_game_counts: dict[str, int] = {}
    for season in HISTORICAL_SEASONS:
        stints, excluded_game_ids = build_rapm_stints_from_curated_games(
            season,
            curated_dir=curated_dir,
        )
        season_exclusions = list(excluded_game_ids)
        if include_historical_playoffs:
            playoff_ids = _available_processed_playoff_game_ids(season)
            excluded_playoff_ids: tuple[str, ...] = ()
            if playoff_ids:
                playoff_stints, excluded_playoff_ids = (
                    build_rapm_stints_from_legacy_processed_games(playoff_ids)
                )
                stints = pd.concat([stints, playoff_stints], ignore_index=True).sort_values(
                    ["game_time_utc", "game_id", "stint_index"], kind="stable"
                )
                season_exclusions.extend(excluded_playoff_ids)
            playoff_game_counts[season] = len(playoff_ids) - len(excluded_playoff_ids)
        result = fit_forward_lagged_rapm_season(
            season,
            stints,
            priors,
            lambda_grid=lambda_grid,
        )
        estimates.append(result.player_estimates)
        cv_results.append(result.cv_results.assign(season=season))
        exclusions[season] = season_exclusions
        priors = result.player_estimates.loc[:, ["player_id", "rapm"]].rename(
            columns={"rapm": PRIOR_MEAN_COLUMN}
        )

    target_stints = read_rapm_stints("2025-26", analytical_dir=analytical_dir)
    bios_path = (
        Path(curated_dir) / "player_seasons" / "2025-26" / "regular" / "part-00000.parquet"
    )
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    target_result = fit_prior_rapm_experiment(
        target_stints,
        priors if priors is not None else pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN]),
        lambda_grid=lambda_grid,
        player_bios=player_bios,
    )
    target_ids = entity_vocabulary(
        target_stints,
        "home_player_ids",
        "away_player_ids",
        multiple=True,
    )
    target_columns = vocabulary_mapping(target_ids)
    target_matrix = signed_entity_matrix(
        target_stints,
        "home_player_ids",
        "away_player_ids",
        target_columns,
        multiple=True,
    )
    target_prior, _ = _prior_vector(
        target_ids,
        priors if priors is not None else pd.DataFrame(columns=["player_id", PRIOR_MEAN_COLUMN]),
    )
    full_season_model = PriorCenteredRidgeLineupModel(
        target_result.selected_lambda
    ).fit(
        target_matrix,
        target_stints["target_home_net_rating"].to_numpy(dtype=float),
        target_stints["possessions"].to_numpy(dtype=float),
        target_prior,
    )
    full_season_coefficients = pd.DataFrame(
        {"player_id": target_ids, "rapm": full_season_model.coef_}
    )
    final_train_mask = np.isin(
        target_stints["game_id"].astype(str),
        target_result.split_plan.final_train_game_ids,
    )
    final_training_model = PriorCenteredRidgeLineupModel(
        target_result.selected_lambda
    ).fit(
        target_matrix[final_train_mask],
        target_stints.loc[final_train_mask, "target_home_net_rating"].to_numpy(dtype=float),
        target_stints.loc[final_train_mask, "possessions"].to_numpy(dtype=float),
        target_prior,
    )
    final_training_coefficients = pd.DataFrame(
        {"player_id": target_ids, "rapm": final_training_model.coef_}
    )
    run_id = f"forward-lagged-rapm-2025-26-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root_name = "prior_rapm_playoffs" if include_historical_playoffs else "prior_rapm"
    root = Path(artifacts_dir) / root_name / "2025-26"
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / run_id
    temporary_dir = root / f".{run_id}.tmp"
    temporary_dir.mkdir()
    try:
        pd.concat(estimates, ignore_index=True).to_parquet(
            temporary_dir / "historical_player_coefficients.parquet", index=False
        )
        pd.concat(cv_results, ignore_index=True).to_parquet(
            temporary_dir / "historical_cv_results.parquet", index=False
        )
        target_result.cv_results.to_parquet(
            temporary_dir / "target_cv_results.parquet", index=False
        )
        target_result.test_metrics.to_parquet(
            temporary_dir / "holdout_metrics.parquet", index=False
        )
        target_result.test_predictions.to_parquet(
            temporary_dir / "holdout_predictions.parquet", index=False
        )
        target_result.player_rankings.to_parquet(
            temporary_dir / "player_rankings.parquet", index=False
        )
        target_result.player_priors.to_parquet(
            temporary_dir / "target_player_priors.parquet", index=False
        )
        full_season_coefficients.to_parquet(
            temporary_dir / "full_season_player_coefficients.parquet", index=False
        )
        final_training_coefficients.to_parquet(
            temporary_dir / "final_training_player_coefficients.parquet", index=False
        )
        (temporary_dir / "full_season_state.json").write_text(
            json.dumps(
                {
                    "intercept_home_net_rating": full_season_model.intercept_,
                    "selected_lambda": target_result.selected_lambda,
                    "training_game_count": int(target_stints["game_id"].nunique()),
                },
                indent=2,
            )
            + "\n"
        )
        (temporary_dir / "final_training_state.json").write_text(
            json.dumps(
                {
                    "intercept_home_net_rating": final_training_model.intercept_,
                    "selected_lambda": target_result.selected_lambda,
                    "training_game_count": len(target_result.split_plan.final_train_game_ids),
                },
                indent=2,
            )
            + "\n"
        )
        split = target_result.split_plan
        pd.DataFrame(
            {
                "game_id": (*split.final_train_game_ids, *split.final_test_game_ids),
                "split": ["train"] * len(split.final_train_game_ids)
                + ["test"] * len(split.final_test_game_ids),
            }
        ).to_parquet(temporary_dir / "target_game_splits.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "model": "forward_lagged_prior_centered_ridge_rapm",
            "historical_training_variant": variant,
            "target_season": "2025-26",
            "historical_seasons": list(HISTORICAL_SEASONS),
            "lambda_grid": list(lambda_grid),
            "historical_selected_lambdas": {
                season: float(frame["selected_lambda"].iloc[0])
                for season, frame in zip(HISTORICAL_SEASONS, estimates, strict=True)
            },
            "target_selected_lambda": target_result.selected_lambda,
            "target_final_train_games": len(split.final_train_game_ids),
            "target_final_test_games": len(split.final_test_game_ids),
            "historical_excluded_game_ids": exclusions,
            "historical_playoff_game_counts": playoff_game_counts,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (temporary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = []
        for path in sorted(temporary_dir.iterdir()):
            if path.is_file():
                artifacts.append(
                    {
                        "filename": path.name,
                        "byte_count": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        (temporary_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, **metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary_dir.replace(output_dir)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return output_dir


def _available_processed_playoff_game_ids(season: str) -> tuple[str, ...]:
    catalog = pd.read_parquet("data/catalog/games.parquet")
    candidates = catalog.loc[
        catalog["season"].eq(season) & catalog["season_type"].eq("playoffs"),
        "game_id",
    ].astype(str)
    processed_root = Path("data/processed")
    return tuple(
        game_id
        for game_id in candidates
        if (processed_root / "lineup_stints" / f"{game_id}.parquet").is_file()
        and (processed_root / "possession_segments" / f"{game_id}.parquet").is_file()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _prior_vector(
    player_ids: tuple[int, ...],
    player_priors: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    indexed = player_priors.set_index("player_id")[PRIOR_MEAN_COLUMN]
    values_by_player = {int(player_id): float(value) for player_id, value in indexed.items()}
    prior = np.array([values_by_player.get(player_id, 0.0) for player_id in player_ids])
    frame = pd.DataFrame({"player_id": player_ids, "prior_rapm_mean": prior})
    frame["prior_available"] = frame["player_id"].isin(indexed.index)
    return prior, frame


def _cross_validate(
    stints: pd.DataFrame,
    matrix: object,
    prior: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    game_ids: np.ndarray,
    split_plan: GameSplitPlan,
    lambda_grid: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for fold in split_plan.folds:
        train = np.isin(game_ids, fold.train_game_ids)
        validation = np.isin(game_ids, fold.validation_game_ids)
        for regularization in lambda_grid:
            model = PriorCenteredRidgeLineupModel(regularization).fit(
                matrix[train], target[train], weights[train], prior
            )
            rows.append(
                _metric_row(
                    stints.loc[validation],
                    target[validation],
                    model.predict(matrix[validation]),
                    weights[validation],
                    "prior_rapm",
                    regularization,
                    fold.fold,
                )
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["regularization", "fold"], kind="stable")
        .reset_index(drop=True)
    )


def _select_lambda(cv_results: pd.DataFrame) -> float:
    summary = cv_results.groupby("regularization", as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    return float(
        summary.sort_values(["weighted_mse", "regularization"], kind="stable").iloc[0][
            "regularization"
        ]
    )


def _metric_row(
    stints: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    model: str,
    regularization: float | None,
    fold: int | None = None,
) -> dict[str, float | int | str | None]:
    mse = mean_squared_error(actual, predicted, weights)
    return {
        "model": model,
        "fold": fold,
        "regularization": regularization,
        "validation_game_count": int(stints["game_id"].nunique()),
        "validation_stint_count": len(stints),
        "validation_possessions": float(weights.sum()),
        "squared_error_sum": float(mse * weights.sum()),
        "weighted_mse": mse,
        "weighted_rmse": rmse(actual, predicted, weights),
        "weighted_mae": mean_absolute_error(actual, predicted, weights),
        "game_margin_rmse": game_margin_rmse(stints["game_id"], actual, predicted, weights),
    }


def _validate_inputs(
    stints: pd.DataFrame, priors: pd.DataFrame, lambda_grid: tuple[float, ...]
) -> None:
    required_stints = {
        "game_id",
        "game_date",
        "game_time_utc",
        "home_player_ids",
        "away_player_ids",
        "possessions",
        "target_home_net_rating",
        "home_margin",
        "stint_index",
    }
    missing_stints = required_stints - set(stints)
    if missing_stints:
        raise ValueError(f"RAPM stints missing columns: {sorted(missing_stints)}")
    required_priors = {"player_id", PRIOR_MEAN_COLUMN}
    missing_priors = required_priors - set(priors)
    if missing_priors:
        raise ValueError(f"Player priors missing columns: {sorted(missing_priors)}")
    if priors["player_id"].duplicated().any():
        raise ValueError("Player priors must have unique player IDs")
    if not np.isfinite(priors[PRIOR_MEAN_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Player prior means must be finite")
    if (
        len(lambda_grid) < 2
        or any(value < 0 for value in lambda_grid)
        or len(set(lambda_grid)) != len(lambda_grid)
    ):
        raise ValueError("Prior RAPM requires unique non-negative lambda values")


def main() -> None:
    """Run the forward lagged-RAPM exemplar."""

    parser = argparse.ArgumentParser(description="Train forward lagged-prior RAPM")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument(
        "--include-historical-playoffs",
        action="store_true",
        help="Include available historical playoff games when forming season priors",
    )
    args = parser.parse_args()
    print(
        train_forward_lagged_rapm(
            curated_dir=args.curated_dir,
            analytical_dir=args.analytical_dir,
            artifacts_dir=args.artifacts_dir,
            include_historical_playoffs=args.include_historical_playoffs,
        )
    )


if __name__ == "__main__":
    main()
