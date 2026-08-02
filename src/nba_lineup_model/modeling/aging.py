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

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.player_history import (
    PlayerSeasonPanelManifest,
    validate_player_season_panel,
)
from nba_lineup_model.modeling.schema import (
    AgingModelRunManifest,
    AgingSeasonFold,
    ArtifactRecord,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_AGING_REGULARIZATION_GRID = (
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
)
AGING_FEATURE_COLUMNS = (
    "target_age",
    "target_nba_experience_years",
    "prior_rapm_filled",
    "log1p_prior_rapm_possessions",
    "has_prior_season",
    "is_rookie",
)
_TARGET_OUTCOME_COLUMNS = {
    "target_rapm",
    "target_rapm_possessions",
    "target_rapm_seconds",
    "target_rapm_exposure_eligible",
}
_REQUIRED_TRANSITION_COLUMNS = {
    "target_season",
    "prior_season",
    "player_id",
    "player_name",
    "target_age",
    "target_nba_experience_years",
    "is_rookie",
    "has_prior_season",
    "prior_rapm",
    "prior_rapm_possessions",
    *_TARGET_OUTCOME_COLUMNS,
}


@dataclass(frozen=True)
class TargetSeasonFold:
    """Concrete expanding train and validation target seasons."""

    fold: int
    train_target_seasons: tuple[str, ...]
    validation_target_season: str


@dataclass(frozen=True)
class AgingExperiment:
    """In-memory outputs for one forward-only aging experiment."""

    holdout_target_season: str
    training_target_seasons: tuple[str, ...]
    folds: tuple[TargetSeasonFold, ...]
    selected_regularization: float
    fold_metrics: pd.DataFrame
    hyperparameter_summary: pd.DataFrame
    cv_predictions: pd.DataFrame
    holdout_predictions: pd.DataFrame
    holdout_metrics: pd.DataFrame
    player_priors: pd.DataFrame
    feature_coefficients: pd.DataFrame
    uncertainty_scales: pd.DataFrame
    model_parameters: dict[str, Any]
    fitted_model: Pipeline


def expanding_target_season_folds(
    transitions: pd.DataFrame,
    *,
    holdout_target_season: str | None = None,
) -> tuple[tuple[str, ...], str, tuple[TargetSeasonFold, ...]]:
    """Create expanding target-season folds and one untouched latest holdout."""

    if "target_season" not in transitions:
        raise ValueError("Aging transitions require target_season")
    seasons = tuple(
        sorted(
            {
                validate_season(str(season))
                for season in transitions["target_season"].dropna().unique()
            },
            key=lambda value: int(value[:4]),
        )
    )
    if len(seasons) < 3:
        raise ValueError("Aging selection requires at least three target seasons")
    holdout = validate_season(holdout_target_season or seasons[-1])
    if holdout not in seasons:
        raise ValueError(f"Aging holdout season is absent from transitions: {holdout}")
    holdout_year = int(holdout[:4])
    training_seasons = tuple(season for season in seasons if int(season[:4]) < holdout_year)
    if len(training_seasons) < 2:
        raise ValueError("Aging selection requires two pre-holdout target seasons")
    folds = tuple(
        TargetSeasonFold(
            fold=index - 1,
            train_target_seasons=training_seasons[:index],
            validation_target_season=training_seasons[index],
        )
        for index in range(1, len(training_seasons))
    )
    return training_seasons, holdout, folds


def run_aging_experiment(
    transitions: pd.DataFrame,
    *,
    holdout_target_season: str | None = None,
    regularization_grid: tuple[float, ...] = DEFAULT_AGING_REGULARIZATION_GRID,
    age_spline_knots: int = 5,
    age_spline_degree: int = 2,
) -> AgingExperiment:
    """Select and evaluate an aging model without using holdout outcomes."""

    prepared = prepare_aging_transitions(transitions)
    _validate_configuration(
        regularization_grid,
        age_spline_knots,
        age_spline_degree,
    )
    training_seasons, holdout, folds = expanding_target_season_folds(
        prepared,
        holdout_target_season=holdout_target_season,
    )
    eligible = prepared.loc[
        prepared["target_season"].map(lambda value: int(value[:4])) <= int(holdout[:4])
    ].copy()

    fold_metric_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[float, list[pd.DataFrame]] = {
        regularization: [] for regularization in regularization_grid
    }
    for candidate_index, regularization in enumerate(regularization_grid):
        for fold in folds:
            train = eligible.loc[eligible["target_season"].isin(fold.train_target_seasons)]
            validation = eligible.loc[eligible["target_season"].eq(fold.validation_target_season)]
            model = fit_aging_pipeline(
                train,
                regularization=regularization,
                age_spline_knots=age_spline_knots,
                age_spline_degree=age_spline_degree,
            )
            prediction = model.predict(validation.loc[:, AGING_FEATURE_COLUMNS])
            weights = validation["target_rapm_possessions"].to_numpy(dtype=float)
            metrics = _regression_metrics(
                validation["target_rapm"].to_numpy(dtype=float),
                prediction,
                weights,
            )
            fold_metric_rows.append(
                {
                    "candidate_index": candidate_index,
                    "regularization": regularization,
                    "fold": fold.fold,
                    "train_target_seasons": list(fold.train_target_seasons),
                    "validation_target_season": fold.validation_target_season,
                    "train_player_count": len(train),
                    "validation_player_count": len(validation),
                    **metrics,
                }
            )
            candidate_predictions[regularization].append(
                _cv_prediction_frame(
                    validation,
                    prediction,
                    regularization=regularization,
                    candidate_index=candidate_index,
                    fold=fold.fold,
                )
            )
    fold_metrics = (
        pd.DataFrame(fold_metric_rows)
        .sort_values(
            ["candidate_index", "fold"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    hyperparameter_summary = _hyperparameter_summary(
        fold_metrics,
        len(regularization_grid),
    )
    selected_regularization = float(
        hyperparameter_summary.loc[
            hyperparameter_summary["selected"],
            "regularization",
        ].item()
    )
    cv_predictions = (
        pd.concat(
            candidate_predictions[selected_regularization],
            ignore_index=True,
        )
        .sort_values(
            ["target_season", "player_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    uncertainty_scales = _uncertainty_scales(cv_predictions)

    training = eligible.loc[eligible["target_season"].isin(training_seasons)]
    holdout_rows = eligible.loc[eligible["target_season"].eq(holdout)]
    fitted_model = fit_aging_pipeline(
        training,
        regularization=selected_regularization,
        age_spline_knots=age_spline_knots,
        age_spline_degree=age_spline_degree,
    )
    holdout_prediction = fitted_model.predict(holdout_rows.loc[:, AGING_FEATURE_COLUMNS])
    training_mean = float(
        np.average(
            training["target_rapm"].to_numpy(dtype=float),
            weights=training["target_rapm_possessions"].to_numpy(dtype=float),
        )
    )
    holdout_predictions = _holdout_prediction_frame(
        holdout_rows,
        holdout_prediction,
        training_mean,
    )
    holdout_metrics = _holdout_metrics(holdout_predictions)
    player_priors = _player_prior_frame(
        holdout_rows,
        holdout_prediction,
        uncertainty_scales,
        training_target_seasons=training_seasons,
    )
    feature_coefficients = _feature_coefficient_frame(fitted_model)
    sklearn_alpha = float(fitted_model.named_steps["ridge"].alpha)
    model_parameters = {
        "model": "forward_aging_ridge",
        "target": "target_rapm",
        "sample_weight": "target_rapm_possessions",
        "regularization_convention": "regularization * training_row_count",
        "selected_regularization": selected_regularization,
        "sklearn_alpha": sklearn_alpha,
        "age_spline_knots": age_spline_knots,
        "age_spline_degree": age_spline_degree,
        "age_spline_extrapolation": "linear",
        "feature_columns": list(AGING_FEATURE_COLUMNS),
        "training_target_seasons": list(training_seasons),
        "holdout_target_season": holdout,
        "training_player_season_count": len(training),
        "intercept": float(fitted_model.named_steps["ridge"].intercept_),
    }
    return AgingExperiment(
        holdout_target_season=holdout,
        training_target_seasons=training_seasons,
        folds=folds,
        selected_regularization=selected_regularization,
        fold_metrics=fold_metrics,
        hyperparameter_summary=hyperparameter_summary,
        cv_predictions=cv_predictions,
        holdout_predictions=holdout_predictions,
        holdout_metrics=holdout_metrics,
        player_priors=player_priors,
        feature_coefficients=feature_coefficients,
        uncertainty_scales=uncertainty_scales,
        model_parameters=model_parameters,
        fitted_model=fitted_model,
    )


def train_forward_aging_model(
    *,
    panel_dir: Path | str = Path("data/analytical/player_season_panel"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    holdout_target_season: str | None = None,
    regularization_grid: tuple[float, ...] = DEFAULT_AGING_REGULARIZATION_GRID,
    age_spline_knots: int = 5,
    age_spline_degree: int = 2,
) -> tuple[AgingModelRunManifest, Path]:
    """Train from a validated player-season panel and publish an immutable run."""

    panel_root = Path(panel_dir)
    panel_manifest = validate_player_season_panel(panel_root)
    transitions = pd.read_parquet(panel_root / "transitions.parquet")
    if transitions.empty:
        raise ValueError("Aging model requires a multi-season panel with transition rows")
    experiment = run_aging_experiment(
        transitions,
        holdout_target_season=holdout_target_season,
        regularization_grid=regularization_grid,
        age_spline_knots=age_spline_knots,
        age_spline_degree=age_spline_degree,
    )
    return _write_aging_run(
        panel_manifest,
        panel_root,
        experiment,
        regularization_grid,
        age_spline_knots,
        age_spline_degree,
        artifacts_dir,
    )


def validate_aging_model_run(
    run_dir: Path | str,
) -> AgingModelRunManifest:
    """Validate hashes, rows, holdout identity, and label-free player priors."""

    root = Path(run_dir)
    manifest = AgingModelRunManifest.model_validate_json((root / "manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if actual != expected:
        raise ValueError("Aging model files do not match the manifest")
    required = {
        "fold_metrics.parquet",
        "hyperparameter_summary.parquet",
        "cv_predictions.parquet",
        "holdout_predictions.parquet",
        "holdout_metrics.parquet",
        "player_priors.parquet",
        "feature_coefficients.parquet",
        "uncertainty_scales.parquet",
        "model_parameters.json",
        "model.joblib",
    }
    if not required <= expected:
        raise ValueError("Aging model run is missing required artifacts")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Aging artifact byte count changed: {path.name}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Aging artifact hash changed: {path.name}")
        if artifact.row_count is not None:
            if len(pd.read_parquet(path)) != artifact.row_count:
                raise ValueError(f"Aging artifact rows changed: {path.name}")
    priors = pd.read_parquet(root / "player_priors.parquet")
    if _TARGET_OUTCOME_COLUMNS & set(priors):
        raise ValueError("Aging player priors contain target-season outcomes")
    if priors.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Aging player priors contain duplicate player keys")
    if set(priors["target_season"].astype(str)) != {manifest.season}:
        raise ValueError("Aging player priors have the wrong target season")
    if len(priors) != manifest.holdout_player_count:
        raise ValueError("Aging player prior count changed")
    return manifest


def aging_code_fingerprint(
    source_paths: tuple[Path | str, ...] | None = None,
) -> str:
    """Hash the implementation sources that define the aging experiment."""

    paths = (
        (
            Path(__file__),
            Path(__file__).with_name("player_history.py"),
            Path(__file__).with_name("schema.py"),
        )
        if source_paths is None
        else tuple(Path(path) for path in source_paths)
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def prepare_aging_transitions(transitions: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_TRANSITION_COLUMNS - set(transitions)
    if missing:
        raise ValueError(f"Aging transitions missing columns: {sorted(missing)}")
    if transitions.empty:
        raise ValueError("Aging transitions cannot be empty")
    frame = transitions.copy()
    frame["target_season"] = frame["target_season"].map(lambda value: validate_season(str(value)))
    frame["prior_season"] = frame["prior_season"].map(lambda value: validate_season(str(value)))
    if any(
        int(target[:4]) != int(prior[:4]) + 1
        for prior, target in zip(
            frame["prior_season"],
            frame["target_season"],
            strict=True,
        )
    ):
        raise ValueError("Aging transitions must connect consecutive seasons")
    if frame.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Aging transitions contain duplicate player-season rows")
    numeric_columns = (
        "target_age",
        "target_nba_experience_years",
        "prior_rapm",
        "prior_rapm_possessions",
        "target_rapm",
        "target_rapm_possessions",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required_numeric = (
        "target_age",
        "target_nba_experience_years",
        "target_rapm",
        "target_rapm_possessions",
    )
    if frame.loc[:, required_numeric].isna().any().any():
        raise ValueError("Aging transitions contain missing target values")
    if not frame["target_rapm_possessions"].gt(0).all():
        raise ValueError("Aging target RAPM possessions must be positive")
    frame["has_prior_season"] = frame["has_prior_season"].astype(bool)
    frame["is_rookie"] = frame["is_rookie"].astype(bool)
    frame["target_rapm_exposure_eligible"] = frame["target_rapm_exposure_eligible"].astype(bool)
    returning = frame["has_prior_season"]
    if frame.loc[returning, ["prior_rapm", "prior_rapm_possessions"]].isna().any().any():
        raise ValueError("Returning players require prior RAPM and exposure")
    if frame.loc[returning, "prior_rapm_possessions"].lt(0).any():
        raise ValueError("Prior RAPM possessions cannot be negative")
    frame["prior_rapm_filled"] = frame["prior_rapm"].fillna(0.0)
    frame["prior_rapm_possessions"] = frame["prior_rapm_possessions"].fillna(0.0)
    frame["log1p_prior_rapm_possessions"] = np.log1p(frame["prior_rapm_possessions"].clip(lower=0))
    for column in AGING_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame.sort_values(
        ["target_season", "player_id"],
        kind="stable",
    ).reset_index(drop=True)


def _validate_configuration(
    regularization_grid: tuple[float, ...],
    age_spline_knots: int,
    age_spline_degree: int,
) -> None:
    if len(regularization_grid) < 2:
        raise ValueError("Aging model requires at least two regularization values")
    if any(value < 0 for value in regularization_grid):
        raise ValueError("Aging regularization values must be non-negative")
    if len(regularization_grid) != len(set(regularization_grid)):
        raise ValueError("Aging regularization values must be unique")
    if age_spline_knots < 3:
        raise ValueError("Aging spline requires at least three knots")
    if age_spline_degree not in {1, 2, 3}:
        raise ValueError("Aging spline degree must be 1, 2, or 3")


def fit_aging_pipeline(
    frame: pd.DataFrame,
    *,
    regularization: float,
    age_spline_knots: int,
    age_spline_degree: int,
) -> Pipeline:
    if frame.empty:
        raise ValueError("Aging model training frame cannot be empty")
    features = ColumnTransformer(
        transformers=[
            (
                "age",
                Pipeline(
                    steps=[
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=age_spline_knots,
                                degree=age_spline_degree,
                                include_bias=False,
                                knots="quantile",
                                extrapolation="linear",
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["target_age"],
            ),
            (
                "numeric",
                StandardScaler(),
                [
                    "target_nba_experience_years",
                    "prior_rapm_filled",
                    "log1p_prior_rapm_possessions",
                ],
            ),
            (
                "binary",
                "passthrough",
                ["has_prior_season", "is_rookie"],
            ),
        ],
        verbose_feature_names_out=True,
    )
    model = Pipeline(
        steps=[
            ("features", features),
            (
                "ridge",
                Ridge(
                    alpha=float(regularization) * len(frame),
                    fit_intercept=True,
                    solver="lsqr",
                    tol=1e-8,
                ),
            ),
        ]
    )
    weights = frame["target_rapm_possessions"].to_numpy(dtype=float)
    normalized_weights = weights / np.mean(weights)
    model.fit(
        frame.loc[:, AGING_FEATURE_COLUMNS],
        frame["target_rapm"].to_numpy(dtype=float),
        ridge__sample_weight=normalized_weights,
    )
    return model


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    weighted_squared_error_sum = float(np.dot(sample_weight, residual**2))
    weight_sum = float(sample_weight.sum())
    weighted_absolute_error_sum = float(np.dot(sample_weight, np.abs(residual)))
    return {
        "player_count": float(len(residual)),
        "weight_sum": weight_sum,
        "squared_error_sum": float(np.dot(residual, residual)),
        "absolute_error_sum": float(np.abs(residual).sum()),
        "weighted_squared_error_sum": weighted_squared_error_sum,
        "weighted_absolute_error_sum": weighted_absolute_error_sum,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "weighted_rmse": float(np.sqrt(weighted_squared_error_sum / weight_sum)),
        "weighted_mae": float(weighted_absolute_error_sum / weight_sum),
    }


def _cv_prediction_frame(
    validation: pd.DataFrame,
    prediction: np.ndarray,
    *,
    regularization: float,
    candidate_index: int,
    fold: int,
) -> pd.DataFrame:
    output = validation.loc[
        :,
        [
            "target_season",
            "prior_season",
            "player_id",
            "player_name",
            "has_prior_season",
            "target_rapm",
            "target_rapm_possessions",
        ],
    ].copy()
    output.insert(0, "candidate_index", candidate_index)
    output.insert(1, "regularization", regularization)
    output.insert(2, "fold", fold)
    output["prediction_aging"] = prediction
    output["residual_aging"] = output["target_rapm"] - output["prediction_aging"]
    return output


def _hyperparameter_summary(
    fold_metrics: pd.DataFrame,
    candidate_count: int,
) -> pd.DataFrame:
    summary = fold_metrics.groupby(
        ["candidate_index", "regularization"],
        as_index=False,
        sort=True,
    ).agg(
        fold_count=("fold", "nunique"),
        validation_player_count=("validation_player_count", "sum"),
        validation_weight_sum=("weight_sum", "sum"),
        squared_error_sum=("squared_error_sum", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
        weighted_squared_error_sum=("weighted_squared_error_sum", "sum"),
        weighted_absolute_error_sum=("weighted_absolute_error_sum", "sum"),
    )
    summary["validation_rmse"] = np.sqrt(
        summary["squared_error_sum"] / summary["validation_player_count"]
    )
    summary["validation_mae"] = summary["absolute_error_sum"] / summary["validation_player_count"]
    summary["weighted_validation_mse"] = (
        summary["weighted_squared_error_sum"] / summary["validation_weight_sum"]
    )
    summary["weighted_validation_rmse"] = np.sqrt(summary["weighted_validation_mse"])
    summary["weighted_validation_mae"] = (
        summary["weighted_absolute_error_sum"] / summary["validation_weight_sum"]
    )
    summary = summary.sort_values(
        ["weighted_validation_mse", "regularization"],
        kind="stable",
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1, dtype=int)
    summary["selected"] = summary["rank"].eq(1)
    if len(summary) != candidate_count:
        raise ValueError("Aging hyperparameter summary lost a candidate")
    return summary.sort_values("candidate_index", kind="stable").reset_index(drop=True)


def _uncertainty_scales(cv_predictions: pd.DataFrame) -> pd.DataFrame:
    all_metrics = _regression_metrics(
        cv_predictions["target_rapm"].to_numpy(dtype=float),
        cv_predictions["prediction_aging"].to_numpy(dtype=float),
        cv_predictions["target_rapm_possessions"].to_numpy(dtype=float),
    )
    fallback = max(all_metrics["weighted_rmse"], 1e-6)
    rows: list[dict[str, Any]] = []
    for cohort, mask in (
        ("returning", cv_predictions["has_prior_season"].astype(bool)),
        ("cold_start", ~cv_predictions["has_prior_season"].astype(bool)),
    ):
        subset = cv_predictions.loc[mask]
        if subset.empty:
            scale = fallback
            weight_sum = 0.0
        else:
            metrics = _regression_metrics(
                subset["target_rapm"].to_numpy(dtype=float),
                subset["prediction_aging"].to_numpy(dtype=float),
                subset["target_rapm_possessions"].to_numpy(dtype=float),
            )
            scale = max(metrics["weighted_rmse"], 1e-6)
            weight_sum = metrics["weight_sum"]
        rows.append(
            {
                "cohort": cohort,
                "player_count": len(subset),
                "weight_sum": weight_sum,
                "weighted_residual_rmse": scale,
            }
        )
    return pd.DataFrame(rows)


def _holdout_prediction_frame(
    holdout: pd.DataFrame,
    prediction: np.ndarray,
    training_mean: float,
) -> pd.DataFrame:
    output = holdout.loc[
        :,
        [
            "target_season",
            "prior_season",
            "player_id",
            "player_name",
            "has_prior_season",
            "target_rapm_exposure_eligible",
            "target_rapm",
            "target_rapm_possessions",
            "prior_rapm",
        ],
    ].copy()
    output["prediction_zero"] = 0.0
    output["prediction_training_mean"] = training_mean
    output["prediction_persistence"] = np.where(
        output["has_prior_season"].astype(bool),
        output["prior_rapm"].fillna(0.0),
        0.0,
    )
    output["prediction_aging"] = prediction
    for model in ("zero", "training_mean", "persistence", "aging"):
        output[f"residual_{model}"] = output["target_rapm"] - output[f"prediction_{model}"]
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _holdout_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, mask in (
        ("all", pd.Series(True, index=predictions.index)),
        (
            "exposure_eligible",
            predictions["target_rapm_exposure_eligible"].astype(bool),
        ),
        ("returning", predictions["has_prior_season"].astype(bool)),
        ("cold_start", ~predictions["has_prior_season"].astype(bool)),
    ):
        subset = predictions.loc[mask]
        if subset.empty:
            continue
        actual = subset["target_rapm"].to_numpy(dtype=float)
        weights = subset["target_rapm_possessions"].to_numpy(dtype=float)
        zero_metrics = _regression_metrics(
            actual,
            subset["prediction_zero"].to_numpy(dtype=float),
            weights,
        )
        persistence_metrics = _regression_metrics(
            actual,
            subset["prediction_persistence"].to_numpy(dtype=float),
            weights,
        )
        for model in ("zero", "training_mean", "persistence", "aging"):
            metrics = _regression_metrics(
                actual,
                subset[f"prediction_{model}"].to_numpy(dtype=float),
                weights,
            )
            rows.append(
                {
                    "model": model,
                    "cohort": cohort,
                    **metrics,
                    "weighted_skill_vs_zero": (
                        1.0
                        - metrics["weighted_squared_error_sum"]
                        / zero_metrics["weighted_squared_error_sum"]
                        if zero_metrics["weighted_squared_error_sum"] > 0
                        else 0.0
                    ),
                    "weighted_skill_vs_persistence": (
                        1.0
                        - metrics["weighted_squared_error_sum"]
                        / persistence_metrics["weighted_squared_error_sum"]
                        if persistence_metrics["weighted_squared_error_sum"] > 0
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _player_prior_frame(
    holdout: pd.DataFrame,
    prediction: np.ndarray,
    uncertainty_scales: pd.DataFrame,
    *,
    training_target_seasons: tuple[str, ...],
) -> pd.DataFrame:
    output_columns = [
        "target_season",
        "prior_season",
        "player_id",
        "player_name",
        "has_prior_season",
        "target_age",
        "target_nba_experience_years",
        "is_rookie",
        "prior_rapm",
        "prior_rapm_possessions",
    ]
    for optional in (
        "listed_position",
        "height_inches",
        "weight_pounds",
        "draft_year",
        "draft_round",
        "draft_number",
        "is_undrafted",
    ):
        if optional in holdout:
            output_columns.append(optional)
    output = holdout.loc[:, output_columns].copy()
    output["aging_prior_mean"] = prediction
    scale_by_cohort = uncertainty_scales.set_index("cohort")["weighted_residual_rmse"].to_dict()
    output["aging_prior_error_scale"] = np.where(
        output["has_prior_season"].astype(bool),
        scale_by_cohort["returning"],
        scale_by_cohort["cold_start"],
    )
    output["prior_method"] = "forward_aging_ridge"
    output["model_train_last_target_season"] = training_target_seasons[-1]
    if _TARGET_OUTCOME_COLUMNS & set(output):
        raise ValueError("Aging prior publication retained target outcomes")
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _feature_coefficient_frame(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = np.asarray(model.named_steps["ridge"].coef_, dtype=float)
    if len(names) != len(coefficients):
        raise ValueError("Aging feature names and coefficients differ")
    return pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
        }
    )


def _write_aging_run(
    panel_manifest: PlayerSeasonPanelManifest,
    panel_dir: Path,
    experiment: AgingExperiment,
    regularization_grid: tuple[float, ...],
    age_spline_knots: int,
    age_spline_degree: int,
    artifacts_dir: Path | str,
) -> tuple[AgingModelRunManifest, Path]:
    now = datetime.now(UTC)
    season = experiment.holdout_target_season
    run_id = f"aging-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(artifacts_dir) / "aging" / season
    run_dir = season_dir / run_id
    temporary = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        parquet_outputs = {
            "fold_metrics.parquet": experiment.fold_metrics,
            "hyperparameter_summary.parquet": experiment.hyperparameter_summary,
            "cv_predictions.parquet": experiment.cv_predictions,
            "holdout_predictions.parquet": experiment.holdout_predictions,
            "holdout_metrics.parquet": experiment.holdout_metrics,
            "player_priors.parquet": experiment.player_priors,
            "feature_coefficients.parquet": experiment.feature_coefficients,
            "uncertainty_scales.parquet": experiment.uncertainty_scales,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        (temporary / "model_parameters.json").write_text(
            json.dumps(
                experiment.model_parameters,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        joblib.dump(experiment.fitted_model, temporary / "model.joblib")
        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=(
                    len(parquet_outputs[path.name]) if path.name in parquet_outputs else None
                ),
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in sorted(temporary.iterdir())
            if path.is_file()
        )
        holdout = experiment.holdout_predictions
        fold_records = tuple(
            AgingSeasonFold(
                fold=fold.fold,
                train_target_seasons=fold.train_target_seasons,
                validation_target_season=fold.validation_target_season,
                train_player_count=int(
                    experiment.fold_metrics.loc[
                        experiment.fold_metrics["fold"].eq(fold.fold),
                        "train_player_count",
                    ].iloc[0]
                ),
                validation_player_count=int(
                    experiment.fold_metrics.loc[
                        experiment.fold_metrics["fold"].eq(fold.fold),
                        "validation_player_count",
                    ].iloc[0]
                ),
            )
            for fold in experiment.folds
        )
        manifest = AgingModelRunManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            aging_code_version=aging_code_fingerprint(),
            source_panel_manifest_sha256=_sha256_file(panel_dir / "_manifest.json"),
            source_seasons=panel_manifest.seasons,
            training_target_seasons=experiment.training_target_seasons,
            holdout_target_season=season,
            folds=fold_records,
            feature_columns=AGING_FEATURE_COLUMNS,
            regularization_grid=regularization_grid,
            selected_regularization=experiment.selected_regularization,
            age_spline_knots=age_spline_knots,
            age_spline_degree=age_spline_degree,
            training_player_season_count=int(
                experiment.model_parameters["training_player_season_count"]
            ),
            holdout_player_count=len(holdout),
            holdout_returning_player_count=int(holdout["has_prior_season"].astype(bool).sum()),
            holdout_cold_start_player_count=int((~holdout["has_prior_season"].astype(bool)).sum()),
            artifacts=artifacts,
        )
        (temporary / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        temporary.replace(run_dir)
        validate_aging_model_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_regularization_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Regularization values must be comma-separated numbers"
        ) from exc
    if len(values) < 2 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Provide at least two non-negative regularization values")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Regularization values must be unique")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a forward-only RAPM aging prior from the player-season transition panel."
        )
    )
    parser.add_argument(
        "--panel-dir",
        default="data/analytical/player_season_panel",
    )
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--holdout-season")
    parser.add_argument(
        "--regularizations",
        type=_parse_regularization_grid,
        default=DEFAULT_AGING_REGULARIZATION_GRID,
        help="Comma-separated normalized ridge regularization grid",
    )
    parser.add_argument("--age-spline-knots", type=int, default=5)
    parser.add_argument("--age-spline-degree", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, run_dir = train_forward_aging_model(
        panel_dir=args.panel_dir,
        artifacts_dir=args.artifacts_dir,
        holdout_target_season=args.holdout_season,
        regularization_grid=args.regularizations,
        age_spline_knots=args.age_spline_knots,
        age_spline_degree=args.age_spline_degree,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    print(
        f"{manifest.season} forward aging model: "
        f"training_seasons={len(manifest.training_target_seasons)}, "
        f"holdout_players={manifest.holdout_player_count}, "
        f"regularization={manifest.selected_regularization:g}; "
        f"run={run_dir}{tracking_text}"
    )


if __name__ == "__main__":
    main()
