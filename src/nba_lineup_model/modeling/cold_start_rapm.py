"""Forward-only preseason-profile RAPM forecasts for NBA cold starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nba_lineup_model.modeling.aging import expanding_target_season_folds
from nba_lineup_model.modeling.box_score_prior import (
    STATIC_FEATURE_COLUMNS,
    validate_box_score_prior_panel,
)
from nba_lineup_model.modeling.box_score_rapm import (
    DEFAULT_PANEL_DIR,
    DEFAULT_REGULARIZATION_GRID,
    _parse_regularization_grid,
    _regression_metrics,
)
from nba_lineup_model.modeling.schema import ArtifactRecord

COLD_START_FEATURE_COLUMNS = STATIC_FEATURE_COLUMNS
_CATEGORICAL_COLUMNS = ("listed_position",)
_NUMERIC_COLUMNS = tuple(
    column for column in COLD_START_FEATURE_COLUMNS if column not in _CATEGORICAL_COLUMNS
)


def prepare_cold_start_rows(features: pd.DataFrame) -> pd.DataFrame:
    """Return valid target rows that have no immediately prior NBA season."""

    required = {
        "target_season",
        "player_id",
        "player_name",
        "target_rapm",
        "target_rapm_possessions",
        "has_prior_season",
        *COLD_START_FEATURE_COLUMNS,
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Cold-start panel rows missing columns: {sorted(missing)}")
    frame = features.copy()
    frame["target_season"] = frame["target_season"].astype(str)
    if frame.duplicated(["target_season", "player_id"]).any():
        raise ValueError("Cold-start panel contains duplicate player-season keys")
    for column in (*_NUMERIC_COLUMNS, "target_rapm", "target_rapm_possessions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    cold = frame.loc[~frame["has_prior_season"].astype(bool)].copy()
    if cold.empty or cold[["target_rapm", "target_rapm_possessions"]].isna().any().any():
        raise ValueError("Cold-start model requires complete target labels")
    if not cold["target_rapm_possessions"].gt(0).all():
        raise ValueError("Cold-start target possessions must be positive")
    return cold.sort_values(["target_season", "player_id"], kind="stable").reset_index(drop=True)


def fit_cold_start_pipeline(frame: pd.DataFrame, *, regularization: float) -> Pipeline:
    """Fit one preseason-profile ridge forecast."""

    transform = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(_NUMERIC_COLUMNS),
            ),
            (
                "position",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(_CATEGORICAL_COLUMNS),
            ),
        ],
        verbose_feature_names_out=True,
    )
    model = Pipeline(
        [
            ("features", transform),
            (
                "ridge",
                Ridge(
                    alpha=float(regularization) * len(frame),
                    solver="lsqr",
                    tol=1e-8,
                ),
            ),
        ]
    )
    weights = frame["target_rapm_possessions"].to_numpy(dtype=float)
    model.fit(
        frame.loc[:, COLD_START_FEATURE_COLUMNS],
        frame["target_rapm"].to_numpy(dtype=float),
        ridge__sample_weight=weights / np.mean(weights),
    )
    return model


def run_cold_start_experiment(
    features: pd.DataFrame,
    *,
    holdout_target_season: str | None = None,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
) -> dict[str, object]:
    """Tune and evaluate the profile-only model in expanding seasons."""

    if len(regularization_grid) < 2 or any(value < 0 for value in regularization_grid):
        raise ValueError("Cold-start regularization grid needs two non-negative values")
    cold = prepare_cold_start_rows(features)
    training_seasons, holdout, folds = expanding_target_season_folds(
        cold, holdout_target_season=holdout_target_season
    )
    rows: list[dict[str, object]] = []
    for index, regularization in enumerate(regularization_grid):
        for fold in folds:
            train = cold.loc[cold["target_season"].isin(fold.train_target_seasons)]
            valid = cold.loc[cold["target_season"].eq(fold.validation_target_season)]
            prediction = fit_cold_start_pipeline(train, regularization=regularization).predict(
                valid.loc[:, COLD_START_FEATURE_COLUMNS]
            )
            rows.append(
                {
                    "candidate_index": index,
                    "regularization": regularization,
                    "fold": fold.fold,
                    "train_target_seasons": list(fold.train_target_seasons),
                    "validation_target_season": fold.validation_target_season,
                    "train_player_count": len(train),
                    "validation_player_count": len(valid),
                    **_regression_metrics(
                        valid["target_rapm"].to_numpy(float),
                        prediction,
                        valid["target_rapm_possessions"].to_numpy(float),
                    ),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    summary = fold_metrics.groupby(["candidate_index", "regularization"], as_index=False).agg(
        validation_weight_sum=("weight_sum", "sum"),
        weighted_squared_error_sum=("weighted_squared_error_sum", "sum"),
    )
    summary["weighted_validation_mse"] = (
        summary["weighted_squared_error_sum"] / summary["validation_weight_sum"]
    )
    summary = summary.sort_values(["weighted_validation_mse", "regularization"], kind="stable")
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["selected"] = summary["rank"].eq(1)
    selected = float(summary.loc[summary["selected"], "regularization"].item())
    train = cold.loc[cold["target_season"].isin(training_seasons)]
    holdout_rows = cold.loc[cold["target_season"].eq(holdout)]
    model = fit_cold_start_pipeline(train, regularization=selected)
    prediction = model.predict(holdout_rows.loc[:, COLD_START_FEATURE_COLUMNS])
    training_mean = float(
        np.average(
            train["target_rapm"].to_numpy(float),
            weights=train["target_rapm_possessions"],
        )
    )
    predictions = holdout_rows.loc[
        :, ["target_season", "player_id", "player_name", "target_rapm", "target_rapm_possessions"]
    ].copy()
    predictions["prediction_zero"] = 0.0
    predictions["prediction_training_mean"] = training_mean
    predictions["prediction_cold_start"] = prediction
    metric_rows = []
    zero = _regression_metrics(
        predictions["target_rapm"].to_numpy(float),
        predictions["prediction_zero"].to_numpy(float),
        predictions["target_rapm_possessions"].to_numpy(float),
    )
    for name in ("zero", "training_mean", "cold_start"):
        metrics = _regression_metrics(
            predictions["target_rapm"].to_numpy(float),
            predictions[f"prediction_{name}"].to_numpy(float),
            predictions["target_rapm_possessions"].to_numpy(float),
        )
        metric_rows.append(
            {
                "model": name,
                "cohort": "cold_start",
                **metrics,
                "weighted_skill_vs_zero": 1.0
                - metrics["weighted_squared_error_sum"] / zero["weighted_squared_error_sum"],
            }
        )
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {"feature": names, "coefficient": model.named_steps["ridge"].coef_}
    )
    return {
        "season": holdout,
        "training_seasons": training_seasons,
        "selected_regularization": selected,
        "fold_metrics": fold_metrics.sort_values(["candidate_index", "fold"]),
        "hyperparameter_summary": summary.sort_values("candidate_index"),
        "holdout_predictions": predictions,
        "holdout_metrics": pd.DataFrame(metric_rows),
        "feature_coefficients": coefficients,
        "model": model,
    }


def train_cold_start_rapm_model(
    *,
    panel_dir: Path | str = DEFAULT_PANEL_DIR,
    artifacts_dir: Path | str = Path("artifacts/models"),
    holdout_target_season: str | None = None,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
) -> Path:
    """Train and publish an immutable cold-start forecast run."""

    root = Path(panel_dir)
    panel = validate_box_score_prior_panel(root)
    result = run_cold_start_experiment(
        pd.read_parquet(root / "player_prior_features.parquet"),
        holdout_target_season=holdout_target_season,
        regularization_grid=regularization_grid,
    )
    now = datetime.now(UTC)
    season = str(result["season"])
    run_id = f"cold-start-prior-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = Path(artifacts_dir) / "cold_start_prior" / season / run_id
    temporary = run_dir.parent / f".{run_id}.tmp"
    temporary.mkdir(parents=True)
    try:
        outputs = {
            "fold_metrics.parquet": result["fold_metrics"],
            "hyperparameter_summary.parquet": result["hyperparameter_summary"],
            "holdout_predictions.parquet": result["holdout_predictions"],
            "holdout_metrics.parquet": result["holdout_metrics"],
            "feature_coefficients.parquet": result["feature_coefficients"],
        }
        for filename, frame in outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        parameters = {
            "model": "forward_cold_start_profile_ridge",
            "feature_columns": list(COLD_START_FEATURE_COLUMNS),
            "regularization_convention": "regularization * training_row_count",
            "selected_regularization": result["selected_regularization"],
            "training_target_seasons": list(result["training_seasons"]),
        }
        (temporary / "model_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
        joblib.dump(result["model"], temporary / "model.joblib")
        artifacts = [
            ArtifactRecord(
                filename=path.name,
                row_count=len(outputs[path.name]) if path.name in outputs else None,
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            ).model_dump()
            for path in sorted(temporary.iterdir())
        ]
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": now.isoformat(),
            "season": season,
            "model_kind": "cold_start_profile",
            "source_panel_manifest_sha256": _sha256_file(root / "_manifest.json"),
            "source_target_seasons": list(panel.target_seasons),
            "training_target_seasons": list(result["training_seasons"]),
            "selected_regularization": result["selected_regularization"],
            "holdout_player_count": len(result["holdout_predictions"]),
            "artifacts": artifacts,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(run_dir)
        latest = run_dir.parent / "latest.json"
        latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return run_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a preseason cold-start RAPM forecast.")
    parser.add_argument("--panel-dir", default=str(DEFAULT_PANEL_DIR))
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--holdout-season")
    parser.add_argument("--regularizations", type=_parse_regularization_grid,
                        default=DEFAULT_REGULARIZATION_GRID)
    args = parser.parse_args()
    run_dir = train_cold_start_rapm_model(
        panel_dir=args.panel_dir,
        artifacts_dir=args.artifacts_dir,
        holdout_target_season=args.holdout_season,
        regularization_grid=args.regularizations,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    print(f"Cold-start profile RAPM run={run_dir}; mlflow_run_id={tracking.mlflow_run_id}")


if __name__ == "__main__":
    main()
