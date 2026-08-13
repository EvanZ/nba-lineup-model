"""Forward-only box-score forecasts of next-season canonical RAPM.

This first model intentionally covers returning players only.  It measures the
incremental value of a possession-native box-score profile over simply carrying
forward the player's prior canonical RAPM.  A later cold-start component will
be required before this forecast can become a complete RAPM prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nba_lineup_model.modeling.aging import TargetSeasonFold, expanding_target_season_folds
from nba_lineup_model.modeling.box_score_prior import (
    BoxScorePriorPanelManifest,
    model_feature_columns,
    validate_box_score_prior_panel,
)
from nba_lineup_model.modeling.schema import CODE_VERSION_PATTERN, ArtifactRecord
from nba_lineup_model.season.schema import SEASON_PATTERN, SHA256_PATTERN, validate_season

DEFAULT_PANEL_DIR = Path("data/analytical/box_score_prior_panel")
DEFAULT_REGULARIZATION_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
_EXCLUDED_FEATURE_COLUMNS = {
    "has_prior_season",
    "prior_rapm_available",
    "prior_boxscore_features_available",
}
BOX_SCORE_FEATURE_COLUMNS = tuple(
    column for column in model_feature_columns() if column not in _EXCLUDED_FEATURE_COLUMNS
)
_CATEGORICAL_FEATURE_COLUMNS = ("listed_position",)
_NUMERIC_FEATURE_COLUMNS = tuple(
    column for column in BOX_SCORE_FEATURE_COLUMNS if column not in _CATEGORICAL_FEATURE_COLUMNS
)
_REQUIRED_COLUMNS = {
    "target_season",
    "prior_source_season",
    "player_id",
    "player_name",
    "target_rapm",
    "target_rapm_possessions",
    "prior_rapm",
    "prior_exposure_cohort",
    "has_prior_season",
    "prior_rapm_available",
    "prior_boxscore_features_available",
    *BOX_SCORE_FEATURE_COLUMNS,
}


class BoxScoreRapmRunManifest(BaseModel):
    """Reproducibility contract for a returning-player box-score forecast."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    created_at: datetime
    season: str = Field(pattern=SEASON_PATTERN)
    season_type: Literal["regular"] = "regular"
    box_score_rapm_code_version: str = Field(pattern=CODE_VERSION_PATTERN)
    source_panel_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_target_seasons: tuple[str, ...] = Field(min_length=3)
    training_target_seasons: tuple[str, ...] = Field(min_length=2)
    holdout_target_season: str = Field(pattern=SEASON_PATTERN)
    folds: tuple[TargetSeasonFoldRecord, ...] = Field(min_length=1)
    feature_columns: tuple[str, ...] = Field(min_length=1)
    categorical_feature_columns: tuple[str, ...]
    target_column: Literal["target_rapm"] = "target_rapm"
    sample_weight_column: Literal["target_rapm_possessions"] = "target_rapm_possessions"
    regularization_grid: tuple[float, ...] = Field(min_length=2)
    selected_regularization: float = Field(ge=0)
    training_player_season_count: int = Field(ge=1)
    holdout_returning_player_count: int = Field(ge=1)
    holdout_excluded_cold_start_player_count: int = Field(ge=0)
    artifacts: tuple[ArtifactRecord, ...] = Field(min_length=6)

    @field_validator("season", "holdout_target_season")
    @classmethod
    def validate_single_season(cls, value: str) -> str:
        return validate_season(value)

    @field_validator("source_target_seasons", "training_target_seasons")
    @classmethod
    def validate_seasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_season(value) for value in values)
        if normalized != tuple(sorted(normalized, key=lambda value: int(value[:4]))):
            raise ValueError("Box-score RAPM seasons must be chronological")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Box-score RAPM seasons must be unique")
        return normalized

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Box-score RAPM timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("regularization_grid")
    @classmethod
    def validate_regularization_grid(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value < 0 for value in values) or len(values) != len(set(values)):
            raise ValueError("Regularization values must be unique and non-negative")
        return values

    @model_validator(mode="after")
    def validate_contract(self) -> BoxScoreRapmRunManifest:
        if self.season != self.holdout_target_season:
            raise ValueError("Run season must equal its holdout season")
        if any(
            int(season[:4]) >= int(self.holdout_target_season[:4])
            for season in self.training_target_seasons
        ):
            raise ValueError("Training target seasons must precede holdout")
        if self.selected_regularization not in self.regularization_grid:
            raise ValueError("Selected regularization is outside the grid")
        validation_seasons = tuple(fold.validation_target_season for fold in self.folds)
        if validation_seasons != self.training_target_seasons[1:]:
            raise ValueError("Folds must validate every training season after the first")
        if not set(self.categorical_feature_columns) <= set(self.feature_columns):
            raise ValueError("Categorical feature columns must be model features")
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Box-score RAPM artifact names must be unique")
        return self


class TargetSeasonFoldRecord(BaseModel):
    """One expanding target-season validation fold."""

    model_config = ConfigDict(strict=True, extra="forbid")

    fold: int = Field(ge=0)
    train_target_seasons: tuple[str, ...] = Field(min_length=1)
    validation_target_season: str = Field(pattern=SEASON_PATTERN)
    train_player_count: int = Field(ge=1)
    validation_player_count: int = Field(ge=1)

    @field_validator("train_target_seasons")
    @classmethod
    def validate_train_seasons(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_season(value) for value in values)

    @field_validator("validation_target_season")
    @classmethod
    def validate_validation_season(cls, value: str) -> str:
        return validate_season(value)

    @model_validator(mode="after")
    def validate_chronology(self) -> TargetSeasonFoldRecord:
        if any(
            int(season[:4]) >= int(self.validation_target_season[:4])
            for season in self.train_target_seasons
        ):
            raise ValueError("Fold training seasons must precede validation")
        return self


@dataclass(frozen=True)
class BoxScoreRapmExperiment:
    """In-memory outputs for one box-score prior selection experiment."""

    holdout_target_season: str
    training_target_seasons: tuple[str, ...]
    folds: tuple[TargetSeasonFold, ...]
    selected_regularization: float
    fold_metrics: pd.DataFrame
    hyperparameter_summary: pd.DataFrame
    cv_predictions: pd.DataFrame
    holdout_predictions: pd.DataFrame
    holdout_metrics: pd.DataFrame
    feature_coefficients: pd.DataFrame
    model_parameters: dict[str, Any]
    fitted_model: Pipeline


def prepare_returning_player_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Validate panel rows and retain players with a full prior-season profile."""

    missing = _REQUIRED_COLUMNS - set(features)
    if missing:
        raise ValueError(f"Box-score prior features missing columns: {sorted(missing)}")
    frame = features.copy()
    frame["target_season"] = frame["target_season"].map(lambda value: validate_season(str(value)))
    if frame.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Box-score prior rows contain duplicate player-season keys")
    for column in (*_NUMERIC_FEATURE_COLUMNS, "target_rapm", "target_rapm_possessions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["target_rapm", "target_rapm_possessions"]].isna().any().any():
        raise ValueError("Box-score RAPM targets must be present")
    if not frame["target_rapm_possessions"].gt(0).all():
        raise ValueError("Box-score RAPM target possessions must be positive")
    eligible = frame.loc[
        frame["has_prior_season"].astype(bool)
        & frame["prior_rapm_available"].astype(bool)
        & frame["prior_boxscore_features_available"].astype(bool)
    ].copy()
    if eligible.empty:
        raise ValueError("Box-score RAPM requires at least one returning player")
    if eligible["prior_rapm"].isna().any():
        raise ValueError("Eligible returning players require prior RAPM")
    return (
        eligible.sort_values(["target_season", "player_id"], kind="stable")
        .reset_index(drop=True)
    )


def fit_box_score_pipeline(
    frame: pd.DataFrame,
    *,
    regularization: float,
    feature_columns: tuple[str, ...] = BOX_SCORE_FEATURE_COLUMNS,
) -> Pipeline:
    """Fit one possession-weighted ridge specification."""

    if frame.empty:
        raise ValueError("Box-score RAPM training frame cannot be empty")
    missing = set(feature_columns) - set(frame)
    if missing:
        raise ValueError(f"Box-score RAPM training frame missing features: {sorted(missing)}")
    categorical_columns = tuple(
        column for column in feature_columns if column in _CATEGORICAL_FEATURE_COLUMNS
    )
    numeric_columns = tuple(
        column for column in feature_columns if column not in categorical_columns
    )
    transformers = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]
            ),
            list(numeric_columns),
        )
    ]
    if categorical_columns:
        transformers.append(
            (
                "position",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_columns),
            )
        )
    features = ColumnTransformer(
        transformers=transformers,
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
    model.fit(
        frame.loc[:, feature_columns],
        frame["target_rapm"].to_numpy(dtype=float),
        ridge__sample_weight=weights / np.mean(weights),
    )
    return model


def run_box_score_rapm_experiment(
    features: pd.DataFrame,
    *,
    holdout_target_season: str | None = None,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
) -> BoxScoreRapmExperiment:
    """Select a returning-player box profile in chronological folds."""

    _validate_configuration(regularization_grid)
    eligible = prepare_returning_player_rows(features)
    training_seasons, holdout, folds = expanding_target_season_folds(
        eligible,
        holdout_target_season=holdout_target_season,
    )
    eligible = eligible.loc[
        eligible["target_season"].map(lambda value: int(value[:4])) <= int(holdout[:4])
    ].copy()
    fold_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[float, list[pd.DataFrame]] = {
        regularization: [] for regularization in regularization_grid
    }
    for candidate_index, regularization in enumerate(regularization_grid):
        for fold in folds:
            train = eligible.loc[eligible["target_season"].isin(fold.train_target_seasons)]
            validation = eligible.loc[
                eligible["target_season"].eq(fold.validation_target_season)
            ]
            model = fit_box_score_pipeline(train, regularization=regularization)
            prediction = model.predict(validation.loc[:, BOX_SCORE_FEATURE_COLUMNS])
            metrics = _regression_metrics(
                validation["target_rapm"].to_numpy(dtype=float),
                prediction,
                validation["target_rapm_possessions"].to_numpy(dtype=float),
            )
            fold_rows.append(
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
                _prediction_frame(
                    validation,
                    prediction,
                    regularization=regularization,
                    candidate_index=candidate_index,
                    fold=fold.fold,
                )
            )
    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["candidate_index", "fold"], kind="stable"
    ).reset_index(drop=True)
    hyperparameter_summary = _hyperparameter_summary(fold_metrics, len(regularization_grid))
    selected_regularization = float(
        hyperparameter_summary.loc[hyperparameter_summary["selected"], "regularization"].item()
    )
    cv_predictions = pd.concat(
        candidate_predictions[selected_regularization], ignore_index=True
    ).sort_values(["target_season", "player_id"], kind="stable").reset_index(drop=True)

    training = eligible.loc[eligible["target_season"].isin(training_seasons)]
    holdout_rows = eligible.loc[eligible["target_season"].eq(holdout)]
    fitted_model = fit_box_score_pipeline(training, regularization=selected_regularization)
    holdout_prediction = fitted_model.predict(holdout_rows.loc[:, BOX_SCORE_FEATURE_COLUMNS])
    training_mean = float(
        np.average(
            training["target_rapm"].to_numpy(dtype=float),
            weights=training["target_rapm_possessions"].to_numpy(dtype=float),
        )
    )
    holdout_predictions = _prediction_frame(holdout_rows, holdout_prediction)
    holdout_predictions["prediction_training_mean"] = training_mean
    holdout_predictions["residual_training_mean"] = (
        holdout_predictions["target_rapm"] - training_mean
    )
    holdout_metrics = _holdout_metrics(holdout_predictions)
    feature_coefficients = _feature_coefficient_frame(fitted_model)
    model_parameters = {
        "model": "forward_box_score_ridge_returning_players",
        "target": "target_rapm",
        "sample_weight": "target_rapm_possessions",
        "regularization_convention": "regularization * training_row_count",
        "selected_regularization": selected_regularization,
        "sklearn_alpha": float(fitted_model.named_steps["ridge"].alpha),
        "feature_columns": list(BOX_SCORE_FEATURE_COLUMNS),
        "categorical_feature_columns": list(_CATEGORICAL_FEATURE_COLUMNS),
        "returning_player_eligibility": (
            "has_prior_season and prior_rapm_available and prior_boxscore_features_available"
        ),
        "training_target_seasons": list(training_seasons),
        "holdout_target_season": holdout,
        "training_player_season_count": len(training),
        "intercept": float(fitted_model.named_steps["ridge"].intercept_),
    }
    return BoxScoreRapmExperiment(
        holdout_target_season=holdout,
        training_target_seasons=training_seasons,
        folds=folds,
        selected_regularization=selected_regularization,
        fold_metrics=fold_metrics,
        hyperparameter_summary=hyperparameter_summary,
        cv_predictions=cv_predictions,
        holdout_predictions=holdout_predictions,
        holdout_metrics=holdout_metrics,
        feature_coefficients=feature_coefficients,
        model_parameters=model_parameters,
        fitted_model=fitted_model,
    )


def train_box_score_rapm_model(
    *,
    panel_dir: Path | str = DEFAULT_PANEL_DIR,
    artifacts_dir: Path | str = Path("artifacts/models"),
    holdout_target_season: str | None = None,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
) -> tuple[BoxScoreRapmRunManifest, Path]:
    """Train and atomically publish the returning-player box-score experiment."""

    root = Path(panel_dir)
    panel_manifest = validate_box_score_prior_panel(root)
    features = pd.read_parquet(root / "player_prior_features.parquet")
    experiment = run_box_score_rapm_experiment(
        features,
        holdout_target_season=holdout_target_season,
        regularization_grid=regularization_grid,
    )
    return _write_run(
        panel_manifest,
        root,
        features,
        experiment,
        regularization_grid,
        artifacts_dir,
    )


def validate_box_score_rapm_run(run_dir: Path | str) -> BoxScoreRapmRunManifest:
    """Validate immutable output files and the intentionally limited scope."""

    root = Path(run_dir)
    manifest = BoxScoreRapmRunManifest.model_validate_json((root / "manifest.json").read_text())
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if actual != expected:
        raise ValueError("Box-score RAPM files do not match manifest")
    required = {
        "fold_metrics.parquet",
        "hyperparameter_summary.parquet",
        "cv_predictions.parquet",
        "holdout_predictions.parquet",
        "holdout_metrics.parquet",
        "feature_coefficients.parquet",
        "model_parameters.json",
        "model.joblib",
    }
    if not required <= expected:
        raise ValueError("Box-score RAPM run is missing required artifacts")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count or _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Box-score RAPM artifact integrity changed: {path.name}")
        if artifact.row_count is not None and len(pd.read_parquet(path)) != artifact.row_count:
            raise ValueError(f"Box-score RAPM artifact rows changed: {path.name}")
    holdout = pd.read_parquet(root / "holdout_predictions.parquet")
    if len(holdout) != manifest.holdout_returning_player_count:
        raise ValueError("Box-score RAPM holdout player count changed")
    if not holdout["has_prior_season"].astype(bool).all():
        raise ValueError("Returning-player run contains a cold start")
    return manifest


def _prediction_frame(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    regularization: float | None = None,
    candidate_index: int | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    columns = [
        "target_season",
        "prior_source_season",
        "player_id",
        "player_name",
        "prior_exposure_cohort",
        "has_prior_season",
        "target_rapm",
        "target_rapm_possessions",
        "prior_rapm",
    ]
    output = frame.loc[:, columns].copy()
    if candidate_index is not None:
        output.insert(0, "candidate_index", candidate_index)
    if regularization is not None:
        output.insert(1 if candidate_index is not None else 0, "regularization", regularization)
    if fold is not None:
        output.insert(2 if candidate_index is not None else 0, "fold", fold)
    output["prediction_persistence"] = output["prior_rapm"]
    output["prediction_box_score"] = prediction
    output["residual_persistence"] = output["target_rapm"] - output["prediction_persistence"]
    output["residual_box_score"] = output["target_rapm"] - output["prediction_box_score"]
    return output.sort_values("player_id", kind="stable").reset_index(drop=True)


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    weighted_squared_error_sum = float(np.dot(sample_weight, residual**2))
    weight_sum = float(sample_weight.sum())
    return {
        "player_count": float(len(residual)),
        "weight_sum": weight_sum,
        "squared_error_sum": float(np.dot(residual, residual)),
        "absolute_error_sum": float(np.abs(residual).sum()),
        "weighted_squared_error_sum": weighted_squared_error_sum,
        "weighted_absolute_error_sum": float(np.dot(sample_weight, np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "weighted_rmse": float(np.sqrt(weighted_squared_error_sum / weight_sum)),
        "weighted_mae": float(np.dot(sample_weight, np.abs(residual)) / weight_sum),
    }


def _hyperparameter_summary(fold_metrics: pd.DataFrame, candidate_count: int) -> pd.DataFrame:
    summary = fold_metrics.groupby(["candidate_index", "regularization"], as_index=False).agg(
        fold_count=("fold", "nunique"),
        validation_player_count=("validation_player_count", "sum"),
        validation_weight_sum=("weight_sum", "sum"),
        weighted_squared_error_sum=("weighted_squared_error_sum", "sum"),
        weighted_absolute_error_sum=("weighted_absolute_error_sum", "sum"),
    )
    summary["weighted_validation_mse"] = (
        summary["weighted_squared_error_sum"] / summary["validation_weight_sum"]
    )
    summary["weighted_validation_rmse"] = np.sqrt(summary["weighted_validation_mse"])
    summary["weighted_validation_mae"] = (
        summary["weighted_absolute_error_sum"] / summary["validation_weight_sum"]
    )
    ranked = summary.sort_values(["weighted_validation_mse", "regularization"], kind="stable")
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["selected"] = ranked["rank"].eq(1)
    if len(ranked) != candidate_count:
        raise ValueError("Box-score RAPM hyperparameter summary lost a candidate")
    return ranked.sort_values("candidate_index", kind="stable").reset_index(drop=True)


def _holdout_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, mask in [("returning_all", pd.Series(True, index=predictions.index))]:
        for profile_cohort in ("low_exposure", "developing", "established"):
            subset = predictions.loc[predictions["prior_exposure_cohort"].eq(profile_cohort)]
            if not subset.empty:
                rows.extend(_metric_rows(subset, profile_cohort))
        subset = predictions.loc[mask]
        rows.extend(_metric_rows(subset, cohort))
    return pd.DataFrame(rows)


def _metric_rows(subset: pd.DataFrame, cohort: str) -> list[dict[str, Any]]:
    actual = subset["target_rapm"].to_numpy(dtype=float)
    weights = subset["target_rapm_possessions"].to_numpy(dtype=float)
    persistence = _regression_metrics(actual, subset["prediction_persistence"], weights)
    rows: list[dict[str, Any]] = []
    for model in ("persistence", "box_score"):
        metrics = _regression_metrics(actual, subset[f"prediction_{model}"], weights)
        rows.append(
            {
                "model": model,
                "cohort": cohort,
                **metrics,
                "weighted_skill_vs_persistence": (
                    1.0
                    - metrics["weighted_squared_error_sum"]
                    / persistence["weighted_squared_error_sum"]
                    if persistence["weighted_squared_error_sum"] > 0
                    else 0.0
                ),
            }
        )
    return rows


def _feature_coefficient_frame(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = np.asarray(model.named_steps["ridge"].coef_, dtype=float)
    if len(names) != len(coefficients):
        raise ValueError("Box-score RAPM feature names and coefficients differ")
    return pd.DataFrame({"feature": names, "coefficient": coefficients})


def _write_run(
    panel_manifest: BoxScorePriorPanelManifest,
    panel_dir: Path,
    all_features: pd.DataFrame,
    experiment: BoxScoreRapmExperiment,
    regularization_grid: tuple[float, ...],
    artifacts_dir: Path | str,
) -> tuple[BoxScoreRapmRunManifest, Path]:
    now = datetime.now(UTC)
    season = experiment.holdout_target_season
    run_id = f"box-score-prior-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(artifacts_dir) / "box_score_prior" / season
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
            "feature_coefficients.parquet": experiment.feature_coefficients,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        (temporary / "model_parameters.json").write_text(
            json.dumps(experiment.model_parameters, indent=2, sort_keys=True) + "\n"
        )
        joblib.dump(experiment.fitted_model, temporary / "model.joblib")
        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=len(parquet_outputs[path.name]) if path.name in parquet_outputs else None,
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in sorted(temporary.iterdir())
            if path.is_file()
        )
        holdout = experiment.holdout_predictions
        manifest = BoxScoreRapmRunManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            box_score_rapm_code_version=box_score_rapm_code_fingerprint(),
            source_panel_manifest_sha256=_sha256_file(panel_dir / "_manifest.json"),
            source_target_seasons=panel_manifest.target_seasons,
            training_target_seasons=experiment.training_target_seasons,
            holdout_target_season=season,
            folds=tuple(
                TargetSeasonFoldRecord(
                    fold=fold.fold,
                    train_target_seasons=fold.train_target_seasons,
                    validation_target_season=fold.validation_target_season,
                    train_player_count=int(
                        experiment.fold_metrics.loc[
                            experiment.fold_metrics["fold"].eq(fold.fold), "train_player_count"
                        ].iloc[0]
                    ),
                    validation_player_count=int(
                        experiment.fold_metrics.loc[
                            experiment.fold_metrics["fold"].eq(fold.fold), "validation_player_count"
                        ].iloc[0]
                    ),
                )
                for fold in experiment.folds
            ),
            feature_columns=BOX_SCORE_FEATURE_COLUMNS,
            categorical_feature_columns=_CATEGORICAL_FEATURE_COLUMNS,
            regularization_grid=regularization_grid,
            selected_regularization=experiment.selected_regularization,
            training_player_season_count=int(experiment.model_parameters["training_player_season_count"]),
            holdout_returning_player_count=len(holdout),
            holdout_excluded_cold_start_player_count=int(
                all_features.loc[
                    all_features["target_season"].eq(season), "has_prior_season"
                ].astype(bool).eq(False).sum()
            ),
            artifacts=artifacts,
        )
        (temporary / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        temporary.replace(run_dir)
        validate_box_score_rapm_run(run_dir)
        latest = season_dir / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return manifest, run_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def box_score_rapm_code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("box_score_prior.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_configuration(regularization_grid: tuple[float, ...]) -> None:
    if len(regularization_grid) < 2 or any(value < 0 for value in regularization_grid):
        raise ValueError("Provide at least two non-negative regularization values")
    if len(regularization_grid) != len(set(regularization_grid)):
        raise ValueError("Regularization values must be unique")


def _parse_regularization_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Regularizations must be comma-separated numbers") from exc
    _validate_configuration(grid)
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a forward-only returning-player box-score RAPM forecast."
    )
    parser.add_argument("--panel-dir", default=str(DEFAULT_PANEL_DIR))
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--holdout-season")
    parser.add_argument(
        "--regularizations",
        type=_parse_regularization_grid,
        default=DEFAULT_REGULARIZATION_GRID,
        help="Comma-separated normalized ridge regularization grid",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, run_dir = train_box_score_rapm_model(
        panel_dir=args.panel_dir,
        artifacts_dir=args.artifacts_dir,
        holdout_target_season=args.holdout_season,
        regularization_grid=args.regularizations,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    print(
        f"{manifest.season} returning-player box-score RAPM: "
        f"training_seasons={len(manifest.training_target_seasons)}, "
        f"holdout_players={manifest.holdout_returning_player_count}, "
        f"regularization={manifest.selected_regularization:g}; run={run_dir}{tracking_text}"
    )


if __name__ == "__main__":
    main()
