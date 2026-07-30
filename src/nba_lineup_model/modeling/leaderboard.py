from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from torch.utils.data import DataLoader

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.bayesian import validate_bayesian_rapm_run
from nba_lineup_model.modeling.catboost import (
    catboost_predictions,
    validate_catboost_run,
)
from nba_lineup_model.modeling.deep_sets import validate_deep_sets_run
from nba_lineup_model.modeling.neural import validate_neural_rapm_run
from nba_lineup_model.modeling.neural_data import (
    PossessionTensorDataset,
    neural_possessions_frame,
    read_neural_possessions,
    validate_neural_possession_partition,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    BaselineRunManifest,
    BayesianRapmRunManifest,
    CatBoostRunManifest,
    ModelEvaluationManifest,
    NeuralRapmRunManifest,
    RapmTransformerRunManifest,
)
from nba_lineup_model.modeling.stints import (
    _assign_segments_to_stints,
    validate_rapm_stint_partition,
)
from nba_lineup_model.modeling.train import validate_baseline_run
from nba_lineup_model.modeling.transformer import (
    frozen_rapm_predictions,
    rapm_transformer_predictions,
    validate_rapm_transformer_run,
)
from nba_lineup_model.models.neural import (
    AdditiveRapmModule,
    DeepSetsRapmModule,
    RapmTransformerModule,
)
from nba_lineup_model.season.compact import (
    CuratedPartitionManifest,
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition

MODEL_ORDER = (
    "ridge_rapm",
    "bayesian_rapm",
    "additive_neural",
    "deep_sets",
    "catboost",
    "rapm_transformer",
)
MODEL_NAMES = {
    "ridge_rapm": "One-year ridge RAPM",
    "bayesian_rapm": "One-year Bayesian RAPM",
    "additive_neural": "One-year additive neural",
    "deep_sets": "One-year Deep Sets",
    "catboost": "One-year categorical CatBoost",
    "rapm_transformer": "One-year RAPM + Transformer",
}
COHORT_NAMES = {
    "regular_holdout": "Regular-season holdout",
    "playoffs": "Playoffs",
}


@dataclass(frozen=True)
class EvaluationSources:
    ridge_dir: Path
    ridge_manifest: BaselineRunManifest
    bayesian_dir: Path
    bayesian_manifest: BayesianRapmRunManifest
    neural_dir: Path
    neural_manifest: NeuralRapmRunManifest
    deep_sets_dir: Path
    deep_sets_manifest: NeuralRapmRunManifest
    catboost_dir: Path
    catboost_manifest: CatBoostRunManifest
    transformer_dir: Path
    transformer_manifest: RapmTransformerRunManifest


@dataclass(frozen=True)
class ModelEvaluation:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    cohorts: pd.DataFrame
    comparisons: pd.DataFrame
    model_sources: dict[str, Any]


def evaluation_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash evaluation-owned sources for reproducible comparison reports."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        paths = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "catboost.py",
            package_root / "modeling" / "leaderboard.py",
            package_root / "modeling" / "neural_data.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "transformer.py",
            package_root / "models" / "neural.py",
        )
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one evaluation source path is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"Evaluation source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def build_model_evaluation(
    season: str,
    *,
    ridge_run_id: str | None = None,
    bayesian_run_id: str | None = None,
    neural_run_id: str | None = None,
    deep_sets_run_id: str | None = None,
    catboost_run_id: str | None = None,
    rapm_transformer_run_id: str | None = None,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    model_artifacts_dir: Path | str = Path("artifacts/models"),
    reports_dir: Path | str = Path("artifacts/reports"),
    docs_path: Path | str | None = Path("docs/models/leaderboard.md"),
) -> tuple[ModelEvaluationManifest, Path]:
    """Evaluate fitted regular-season models on regular holdout and playoffs."""

    model_root = Path(model_artifacts_dir)
    sources = _resolve_sources(
        season,
        model_root,
        ridge_run_id,
        bayesian_run_id,
        neural_run_id,
        deep_sets_run_id,
        catboost_run_id,
        rapm_transformer_run_id,
    )
    _validate_source_relationships(sources, season, analytical_dir)
    regular_segments_manifest, regular_segments, regular_segments_dir = (
        _read_curated_partition(
            "possession_segments",
            season,
            "regular",
            curated_dir,
        )
    )
    regular_lineups_manifest, regular_lineups, regular_lineups_dir = (
        _read_curated_partition(
            "lineup_stints",
            season,
            "regular",
            curated_dir,
        )
    )
    playoff_segments_manifest, playoff_segments, playoff_segments_dir = (
        _read_curated_partition(
            "possession_segments",
            season,
            "playoffs",
            curated_dir,
        )
    )
    evaluation = evaluate_fitted_models(
        season,
        sources,
        regular_segments,
        regular_lineups,
        playoff_segments,
        analytical_dir=analytical_dir,
    )
    manifest, run_dir = _write_evaluation(
        season,
        sources,
        evaluation,
        regular_segments_manifest,
        regular_segments_dir,
        regular_lineups_manifest,
        regular_lineups_dir,
        playoff_segments_manifest,
        playoff_segments_dir,
        reports_dir,
    )
    if docs_path is not None:
        render_evaluation_page(
            manifest,
            evaluation.metrics,
            evaluation.cohorts,
            Path(docs_path),
            comparisons=evaluation.comparisons,
        )
    return manifest, run_dir


def evaluate_fitted_models(
    season: str,
    sources: EvaluationSources,
    regular_segments: pd.DataFrame,
    regular_lineups: pd.DataFrame,
    playoff_segments: pd.DataFrame,
    *,
    analytical_dir: Path | str = Path("data/analytical"),
) -> ModelEvaluation:
    """Build common cohorts and score each stored point-prediction model."""

    regular_possessions = read_neural_possessions(season, analytical_dir)
    ridge_splits = pd.read_parquet(sources.ridge_dir / "game_splits.parquet")
    neural_splits = pd.read_parquet(sources.neural_dir / "game_splits.parquet")
    deep_sets_splits = pd.read_parquet(
        sources.deep_sets_dir / "game_splits.parquet"
    )
    catboost_splits = pd.read_parquet(
        sources.catboost_dir / "game_splits.parquet"
    )
    transformer_splits = pd.read_parquet(
        sources.transformer_dir / "game_splits.parquet"
    )
    final_ridge = ridge_splits.loc[ridge_splits["split"].eq("final")]
    final_neural = neural_splits.loc[neural_splits["split"].eq("final")]
    final_deep_sets = deep_sets_splits.loc[
        deep_sets_splits["split"].eq("final")
    ]
    final_catboost = catboost_splits.loc[catboost_splits["split"].eq("final")]
    final_transformer = transformer_splits.loc[
        transformer_splits["split"].eq("final")
    ]
    train_game_ids = set(
        final_ridge.loc[final_ridge["role"].eq("train"), "game_id"].astype(str)
    )
    test_game_ids = set(
        final_ridge.loc[final_ridge["role"].eq("test"), "game_id"].astype(str)
    )
    neural_test_ids = set(
        final_neural.loc[final_neural["role"].eq("test"), "game_id"].astype(str)
    )
    deep_sets_test_ids = set(
        final_deep_sets.loc[
            final_deep_sets["role"].eq("test"),
            "game_id",
        ].astype(str)
    )
    catboost_test_ids = set(
        final_catboost.loc[
            final_catboost["role"].eq("test"),
            "game_id",
        ].astype(str)
    )
    transformer_test_ids = set(
        final_transformer.loc[
            final_transformer["role"].eq("test"),
            "game_id",
        ].astype(str)
    )
    if (
        test_game_ids != neural_test_ids
        or test_game_ids != deep_sets_test_ids
        or test_game_ids != catboost_test_ids
        or test_game_ids != transformer_test_ids
    ):
        raise ValueError("Leaderboard model final test games do not match")
    regular_holdout = regular_possessions.loc[
        regular_possessions["game_id"].astype(str).isin(test_game_ids)
    ].reset_index(drop=True)
    regular_train = regular_possessions.loc[
        regular_possessions["game_id"].astype(str).isin(train_game_ids)
    ]
    if regular_holdout["game_id"].nunique() != len(test_game_ids):
        raise ValueError("Regular holdout neural possessions do not cover every test game")

    regular_stint_lookup = _regular_stint_lookup(
        regular_holdout,
        regular_segments,
        regular_lineups,
    )
    ridge_regular = _mapped_stint_predictions(
        regular_holdout,
        regular_stint_lookup,
        pd.read_parquet(sources.ridge_dir / "test_predictions.parquet"),
        "prediction_rapm",
        float(regular_train["target_offense_margin"].mean()),
    )
    bayesian_regular = _mapped_stint_predictions(
        regular_holdout,
        regular_stint_lookup,
        pd.read_parquet(sources.bayesian_dir / "test_predictions.parquet"),
        "posterior_mean_prediction",
        float(regular_train["target_offense_margin"].mean()),
    )
    neural_regular = _mapped_possession_predictions(
        regular_holdout,
        pd.read_parquet(sources.neural_dir / "test_predictions.parquet"),
        "prediction_additive_neural",
    )
    deep_sets_regular = _mapped_possession_predictions(
        regular_holdout,
        pd.read_parquet(sources.deep_sets_dir / "test_predictions.parquet"),
        "prediction_deep_sets",
    )
    catboost_regular = _mapped_possession_predictions(
        regular_holdout,
        pd.read_parquet(sources.catboost_dir / "test_predictions.parquet"),
        "prediction_catboost",
    )
    transformer_regular = _mapped_possession_predictions(
        regular_holdout,
        pd.read_parquet(
            sources.transformer_dir / "test_predictions.parquet"
        ),
        "prediction_rapm_transformer",
    )

    playoff_possessions = neural_possessions_frame(playoff_segments)
    full_regular_mean = float(regular_possessions["target_offense_margin"].mean())
    ridge_playoffs, ridge_unknown = _ridge_playoff_predictions(
        playoff_possessions,
        sources.ridge_dir,
        full_regular_mean,
    )
    bayesian_playoffs, bayesian_unknown = _bayesian_playoff_predictions(
        playoff_possessions,
        sources.bayesian_dir,
        full_regular_mean,
    )
    neural_playoffs, neural_unknown = _neural_playoff_predictions(
        playoff_possessions,
        sources.neural_dir,
    )
    deep_sets_playoffs, deep_sets_unknown = _deep_sets_playoff_predictions(
        playoff_possessions,
        sources.deep_sets_dir,
    )
    catboost_playoffs, catboost_unknown = _catboost_playoff_predictions(
        playoff_possessions,
        sources.catboost_dir,
    )
    transformer_playoffs, transformer_unknown = (
        _transformer_playoff_predictions(
            playoff_possessions,
            sources.transformer_dir,
        )
    )

    regular_predictions = {
        "ridge_rapm": ridge_regular,
        "bayesian_rapm": bayesian_regular,
        "additive_neural": neural_regular,
        "deep_sets": deep_sets_regular,
        "catboost": catboost_regular,
        "rapm_transformer": transformer_regular,
    }
    playoff_predictions = {
        "ridge_rapm": ridge_playoffs,
        "bayesian_rapm": bayesian_playoffs,
        "additive_neural": neural_playoffs,
        "deep_sets": deep_sets_playoffs,
        "catboost": catboost_playoffs,
        "rapm_transformer": transformer_playoffs,
    }
    regular_metrics, regular_prediction_rows = score_prediction_cohort(
        regular_holdout,
        regular_predictions,
        cohort="regular_holdout",
        training_window="first_1044_regular_season_games",
        mean_prediction=float(regular_train["target_offense_margin"].mean()),
    )
    playoff_metrics, playoff_prediction_rows = score_prediction_cohort(
        playoff_possessions,
        playoff_predictions,
        cohort="playoffs",
        training_window="full_1230_game_regular_season",
        mean_prediction=full_regular_mean,
    )
    comparisons = pd.concat(
        [
            paired_game_cluster_bootstrap(
                regular_holdout,
                neural_regular,
                deep_sets_regular,
                cohort="regular_holdout",
                reference_model="additive_neural",
                candidate_model="deep_sets",
            ),
            paired_game_cluster_bootstrap(
                playoff_possessions,
                neural_playoffs,
                deep_sets_playoffs,
                cohort="playoffs",
                reference_model="additive_neural",
                candidate_model="deep_sets",
            ),
            paired_game_cluster_bootstrap(
                regular_holdout,
                neural_regular,
                catboost_regular,
                cohort="regular_holdout",
                reference_model="additive_neural",
                candidate_model="catboost",
            ),
            paired_game_cluster_bootstrap(
                playoff_possessions,
                neural_playoffs,
                catboost_playoffs,
                cohort="playoffs",
                reference_model="additive_neural",
                candidate_model="catboost",
            ),
            paired_game_cluster_bootstrap(
                regular_holdout,
                ridge_regular,
                transformer_regular,
                cohort="regular_holdout",
                reference_model="ridge_rapm",
                candidate_model="rapm_transformer",
            ),
            paired_game_cluster_bootstrap(
                playoff_possessions,
                ridge_playoffs,
                transformer_playoffs,
                cohort="playoffs",
                reference_model="ridge_rapm",
                candidate_model="rapm_transformer",
            ),
        ],
        ignore_index=True,
    )
    regular_source = regular_segments.loc[
        regular_segments["game_id"].astype(str).isin(test_game_ids)
    ]
    cohorts = pd.DataFrame(
        [
            _cohort_summary(
                regular_holdout,
                regular_source,
                cohort="regular_holdout",
                training_game_count=len(train_game_ids),
            ),
            _cohort_summary(
                playoff_possessions,
                playoff_segments,
                cohort="playoffs",
                training_game_count=int(regular_possessions["game_id"].nunique()),
            ),
        ]
    )
    model_sources = {
        "season": season,
        "ridge": {
            "run_id": sources.ridge_manifest.run_id,
            "regular_holdout_state": "stored final-test stint predictions",
            "playoff_state": "stored all-regular-season coefficients",
        },
        "bayesian": {
            "run_id": sources.bayesian_manifest.run_id,
            "source_ridge_run_id": sources.bayesian_manifest.source_model_run_id,
            "regular_holdout_state": "stored final-training posterior-mean predictions",
            "playoff_state": "stored all-regular-season posterior location",
        },
        "neural": {
            "run_id": sources.neural_manifest.run_id,
            "learning_rate": sources.neural_manifest.learning_rate,
            "weight_decay": sources.neural_manifest.weight_decay,
            "selected_epochs": sources.neural_manifest.selected_epochs,
            "learning_rate_candidates": list(
                sources.neural_manifest.learning_rate_candidates
            ),
            "weight_decay_candidates": list(
                sources.neural_manifest.weight_decay_candidates
            ),
            "regular_holdout_state": "stored test_model predictions",
            "playoff_state": "stored all-regular-season model checkpoint",
        },
        "deep_sets": {
            "run_id": sources.deep_sets_manifest.run_id,
            "learning_rate": sources.deep_sets_manifest.learning_rate,
            "weight_decay": sources.deep_sets_manifest.weight_decay,
            "selected_epochs": sources.deep_sets_manifest.selected_epochs,
            "leaderboard_seed": sources.deep_sets_manifest.leaderboard_seed,
            "refit_seeds": list(sources.deep_sets_manifest.refit_seeds),
            "regular_holdout_state": "stored canonical-seed test_model predictions",
            "playoff_state": "stored canonical-seed all-regular-season checkpoint",
        },
        "catboost": {
            "run_id": sources.catboost_manifest.run_id,
            "max_iterations": sources.catboost_manifest.max_iterations,
            "best_iteration": sources.catboost_manifest.best_iteration,
            "selected_tree_count": sources.catboost_manifest.selected_tree_count,
            "resolved_learning_rate": (
                sources.catboost_manifest.resolved_learning_rate
            ),
            "regular_holdout_state": "stored final-training model predictions",
            "playoff_state": "stored all-regular-season CatBoost model",
        },
        "rapm_transformer": {
            "run_id": sources.transformer_manifest.run_id,
            "source_rapm_run_id": (
                sources.transformer_manifest.source_rapm_run_id
            ),
            "learning_rate": sources.transformer_manifest.learning_rate,
            "weight_decay": sources.transformer_manifest.weight_decay,
            "selected_epochs": sources.transformer_manifest.selected_epochs,
            "leaderboard_seed": (
                sources.transformer_manifest.leaderboard_seed
            ),
            "refit_seeds": list(sources.transformer_manifest.refit_seeds),
            "regular_holdout_state": (
                "stored frozen-RAPM plus canonical-seed test predictions"
            ),
            "playoff_state": (
                "stored all-regular-season RAPM state and Transformer checkpoint"
            ),
        },
        "translation": (
            "RAPM home net rating becomes offense margin as regular-training mean "
            "+ home_offense_sign * predicted_home_net_rating / 200"
        ),
        "regular_training_mean_offense_margin": float(
            regular_train["target_offense_margin"].mean()
        ),
        "full_regular_mean_offense_margin": full_regular_mean,
        "playoff_bayesian_ridge_max_absolute_prediction_difference": float(
            np.max(np.abs(bayesian_playoffs - ridge_playoffs))
        ),
        "playoff_unknown_player_exposures": {
            "ridge_rapm": ridge_unknown,
            "bayesian_rapm": bayesian_unknown,
            "additive_neural": neural_unknown,
            "deep_sets": deep_sets_unknown,
            "catboost": catboost_unknown,
            "rapm_transformer": transformer_unknown,
        },
    }
    return ModelEvaluation(
        metrics=pd.concat([regular_metrics, playoff_metrics], ignore_index=True),
        predictions=pd.concat(
            [regular_prediction_rows, playoff_prediction_rows],
            ignore_index=True,
        ),
        cohorts=cohorts,
        comparisons=comparisons,
        model_sources=model_sources,
    )


def score_prediction_cohort(
    possessions: pd.DataFrame,
    model_predictions: Mapping[str, np.ndarray],
    *,
    cohort: str,
    training_window: str,
    mean_prediction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score multiple models on one identical possession cohort."""

    if tuple(model_predictions) != MODEL_ORDER:
        raise ValueError(f"Model predictions must follow canonical order: {MODEL_ORDER}")
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    mean_values = np.full(len(possessions), mean_prediction)
    mean_mse = mean_squared_error(actual, mean_values)
    mean_game_rmse = possession_game_margin_rmse(
        possessions["game_id"],
        actual,
        mean_values,
        signs,
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model, raw_predictions in model_predictions.items():
        predicted = np.asarray(raw_predictions, dtype=float)
        if predicted.shape != actual.shape or not np.isfinite(predicted).all():
            raise ValueError(f"{model} predictions must be finite and match cohort rows")
        mse = mean_squared_error(actual, predicted)
        game_rmse = possession_game_margin_rmse(
            possessions["game_id"],
            actual,
            predicted,
            signs,
        )
        metric_rows.append(
            {
                "cohort": cohort,
                "model": model,
                "training_window": training_window,
                "game_count": int(possessions["game_id"].nunique()),
                "possession_count": len(possessions),
                "possession_mse": mse,
                "possession_rmse": rmse(actual, predicted),
                "possession_mae": mean_absolute_error(actual, predicted),
                "eligible_possession_game_margin_rmse": game_rmse,
                "possession_skill_vs_mean": skill_score(mse, mean_mse),
                "game_margin_skill_vs_mean": skill_score(game_rmse**2, mean_game_rmse**2),
                "mean_reference_possession_rmse": float(np.sqrt(mean_mse)),
                "mean_reference_game_margin_rmse": mean_game_rmse,
            }
        )
        frame = possessions.loc[
            :,
            [
                "season",
                "season_type",
                "game_id",
                "game_date",
                "game_time_utc",
                "possession_id",
                "possession_index",
                "period",
                "offense_team_id",
                "defense_team_id",
                "home_offense_sign",
                "target_offense_margin",
                "target_home_margin",
            ],
        ].copy()
        frame.insert(0, "cohort", cohort)
        frame.insert(1, "model", model)
        frame["prediction_offense_margin"] = predicted
        frame["prediction_home_margin"] = predicted * signs
        frame["residual_offense_margin"] = actual - predicted
        prediction_frames.append(frame)
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def paired_game_cluster_bootstrap(
    possessions: pd.DataFrame,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    *,
    cohort: str,
    reference_model: str = "additive_neural",
    candidate_model: str = "deep_sets",
    draws: int = 2_000,
    random_seed: int = 20_260_729,
) -> pd.DataFrame:
    """Bootstrap paired candidate-minus-reference RMSE differences by game."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    if reference_model == candidate_model:
        raise ValueError("Bootstrap models must be distinct")
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    reference = np.asarray(reference_predictions, dtype=float)
    candidate = np.asarray(candidate_predictions, dtype=float)
    if reference.shape != actual.shape or candidate.shape != actual.shape:
        raise ValueError("Bootstrap predictions must match possession rows")
    game_codes, game_ids = pd.factorize(
        possessions["game_id"].astype(str),
        sort=False,
    )
    game_count = len(game_ids)
    if game_count < 2:
        raise ValueError("Bootstrap requires at least two games")
    possession_counts = np.bincount(game_codes, minlength=game_count).astype(float)
    reference_sse = np.bincount(
        game_codes,
        weights=np.square(actual - reference),
        minlength=game_count,
    )
    candidate_sse = np.bincount(
        game_codes,
        weights=np.square(actual - candidate),
        minlength=game_count,
    )
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    actual_game = np.bincount(
        game_codes,
        weights=actual * signs,
        minlength=game_count,
    )
    reference_game = np.bincount(
        game_codes,
        weights=reference * signs,
        minlength=game_count,
    )
    candidate_game = np.bincount(
        game_codes,
        weights=candidate * signs,
        minlength=game_count,
    )
    reference_game_se = np.square(actual_game - reference_game)
    candidate_game_se = np.square(actual_game - candidate_game)

    generator = np.random.default_rng(random_seed)
    samples = generator.integers(0, game_count, size=(draws, game_count))
    sampled_possessions = possession_counts[samples].sum(axis=1)
    reference_possession_rmse = np.sqrt(
        reference_sse[samples].sum(axis=1) / sampled_possessions
    )
    candidate_possession_rmse = np.sqrt(
        candidate_sse[samples].sum(axis=1) / sampled_possessions
    )
    reference_game_rmse = np.sqrt(reference_game_se[samples].mean(axis=1))
    candidate_game_rmse = np.sqrt(candidate_game_se[samples].mean(axis=1))

    rows = []
    for metric, point, differences in (
        (
            "possession_rmse",
            float(
                np.sqrt(candidate_sse.sum() / possession_counts.sum())
                - np.sqrt(reference_sse.sum() / possession_counts.sum())
            ),
            candidate_possession_rmse - reference_possession_rmse,
        ),
        (
            "eligible_possession_game_margin_rmse",
            float(
                np.sqrt(candidate_game_se.mean())
                - np.sqrt(reference_game_se.mean())
            ),
            candidate_game_rmse - reference_game_rmse,
        ),
    ):
        rows.append(
            {
                "cohort": cohort,
                "comparison": f"{candidate_model}_minus_{reference_model}",
                "reference_model": reference_model,
                "candidate_model": candidate_model,
                "metric": metric,
                "difference": point,
                "ci_lower": float(np.quantile(differences, 0.025)),
                "ci_upper": float(np.quantile(differences, 0.975)),
                "probability_candidate_better": float(np.mean(differences < 0)),
                "bootstrap_unit": "game",
                "bootstrap_draws": draws,
                "random_seed": random_seed,
            }
        )
    return pd.DataFrame(rows)


def render_evaluation_page(
    manifest: ModelEvaluationManifest,
    metrics: pd.DataFrame,
    cohorts: pd.DataFrame,
    output_path: Path | str,
    comparisons: pd.DataFrame | None = None,
) -> Path:
    """Render the canonical equations and current model comparison tables."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort_rows = []
    for row in cohorts.itertuples(index=False):
        cohort_rows.append(
            f"| {COHORT_NAMES[str(row.cohort)]} | {int(row.game_count):,} | "
            f"{int(row.source_possession_count):,} | "
            f"{int(row.eligible_possession_count):,} | "
            f"{int(row.excluded_multi_lineup_possession_count):,} | "
            f"{float(row.eligible_fraction):.3%} |"
        )
    regular_table = _render_metric_table(
        metrics.loc[metrics["cohort"].eq("regular_holdout")]
    )
    playoff_table = _render_metric_table(metrics.loc[metrics["cohort"].eq("playoffs")])
    comparison_table = (
        _render_comparison_table(comparisons)
        if comparisons is not None and not comparisons.empty
        else "No paired comparison is available for this report."
    )
    regular_reference = metrics.loc[metrics["cohort"].eq("regular_holdout")].iloc[0]
    playoff_reference = metrics.loc[metrics["cohort"].eq("playoffs")].iloc[0]
    created = manifest.created_at.strftime("%Y-%m-%d %H:%M UTC")
    neural_selection = "not recorded"
    if (
        manifest.neural_learning_rate is not None
        and manifest.neural_weight_decay is not None
        and manifest.neural_selected_epochs is not None
    ):
        neural_selection = (
            f"`learning_rate={manifest.neural_learning_rate:g}`, "
            f"`weight_decay={manifest.neural_weight_decay:g}`, "
            f"`epochs={manifest.neural_selected_epochs}`"
        )
    deep_sets_selection = "not recorded"
    if (
        manifest.deep_sets_learning_rate is not None
        and manifest.deep_sets_weight_decay is not None
        and manifest.deep_sets_selected_epochs is not None
        and manifest.deep_sets_leaderboard_seed is not None
    ):
        deep_sets_selection = (
            f"`learning_rate={manifest.deep_sets_learning_rate:g}`, "
            f"`weight_decay={manifest.deep_sets_weight_decay:g}`, "
            f"`epochs={manifest.deep_sets_selected_epochs}`, "
            f"`seed={manifest.deep_sets_leaderboard_seed}`"
        )
    catboost_selection = "not recorded"
    if (
        manifest.catboost_max_iterations is not None
        and manifest.catboost_best_iteration is not None
        and manifest.catboost_selected_tree_count is not None
        and manifest.catboost_resolved_learning_rate is not None
    ):
        catboost_selection = (
            f"`max_iterations={manifest.catboost_max_iterations}`, "
            f"`best_iteration={manifest.catboost_best_iteration}`, "
            f"`trees={manifest.catboost_selected_tree_count}`, "
            f"`learning_rate={manifest.catboost_resolved_learning_rate:g}`"
        )
    transformer_selection = "not recorded"
    if (
        manifest.rapm_transformer_learning_rate is not None
        and manifest.rapm_transformer_weight_decay is not None
        and manifest.rapm_transformer_selected_epochs is not None
        and manifest.rapm_transformer_leaderboard_seed is not None
    ):
        transformer_selection = (
            f"`learning_rate={manifest.rapm_transformer_learning_rate:g}`, "
            f"`weight_decay={manifest.rapm_transformer_weight_decay:g}`, "
            f"`epochs={manifest.rapm_transformer_selected_epochs}`, "
            f"`seed={manifest.rapm_transformer_leaderboard_seed}`"
        )
    cohort_header = (
        "| Cohort | Games | Source possessions | Eligible possessions | "
        "Excluded multi-lineup | Eligible share |"
    )
    content = rf"""# Leaderboard

This page is the cross-model scoreboard. Every row within a cohort uses the
same eligible possessions, target, and game boundaries, with no information
after the cohort's training cutoff. Training objectives remain model-specific.
**Bold values are best at the displayed precision.** Lower error is better;
higher skill is better.

Last generated: **{created}** from `{manifest.run_id}`.

## Evaluation cohorts

{cohort_header}
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(cohort_rows)}

The regular holdout is the final 186 regular-season games and is untouched by
model selection. The playoff cohort contains all games in the `playoffs`
partition and excludes play-in games. Every evaluated model is frozen before
its cohort begins:

- regular holdout predictions use models fit on the first 1,044 regular-season
  games;
- playoff predictions use models refit on all 1,230 regular-season games;
- no playoff outcomes are used for fitting, calibration, or model selection.

## Possession target

For eligible possession \(i\), define the offense-oriented point margin

\[
y_i = P_{{offense,i}} - P_{{defense,i}}.
\]

Most possessions have zero defense points, but retaining the second term
handles unusual opponent scoring without changing the target definition.

Only possessions with exactly one lineup segment are eligible. A possession
with a substitution boundary is excluded in full rather than assigned to a
starting or terminal lineup.

## Possession metrics

For \(N\) eligible possessions:

\[
\operatorname{{MSE}}_{{poss}}
= \frac{{1}}{{N}}\sum_{{i=1}}^N(y_i-\widehat{{y}}_i)^2,
\]

\[
\operatorname{{RMSE}}_{{poss}}
= \sqrt{{\operatorname{{MSE}}_{{poss}}}},
\]

\[
\operatorname{{MAE}}_{{poss}}
= \frac{{1}}{{N}}\sum_{{i=1}}^N
\left|y_i-\widehat{{y}}_i\right|.
\]

RMSE and MAE are measured in points per possession. RMSE penalizes large
errors more strongly; MAE gives the average absolute miss.

The mean reference predicts the offense-margin mean from the model's training
window. Possession skill is

\[
\operatorname{{Skill}}_{{poss}}
= 1 - \frac{{\operatorname{{MSE}}_{{model}}}}
{{\operatorname{{MSE}}_{{mean}}}}.
\]

Positive skill beats the training mean; zero ties it; negative skill is worse.

## Eligible-possession game margin

Let \(s_i=+1\) when the home team is on offense and \(s_i=-1\) when the away
team is on offense. For game \(g\), aggregate only its eligible possessions:

\[
M_g^{{eligible}} = \sum_{{i \in g}}s_i y_i,
\qquad
\widehat{{M}}_g^{{eligible}} = \sum_{{i \in g}}s_i\widehat{{y}}_i.
\]

Across \(G\) games:

\[
\operatorname{{RMSE}}_{{game}}
= \sqrt{{
\frac{{1}}{{G}}\sum_{{g=1}}^G
\left(
M_g^{{eligible}}-\widehat{{M}}_g^{{eligible}}
\right)^2
}}.
\]

This is deliberately named **eligible-possession game-margin RMSE**. It is not
the error against the official final margin because points from excluded
multi-lineup possessions are absent from both actual and predicted totals.

Using the corresponding mean-reference game predictions, game-margin skill is

\[
\operatorname{{Skill}}_{{game}}
= 1 - \frac{{\operatorname{{MSE}}_{{game,model}}}}
{{\operatorname{{MSE}}_{{game,mean}}}}
= 1 - \frac{{\operatorname{{RMSE}}_{{game,model}}^2}}
{{\operatorname{{RMSE}}_{{game,mean}}^2}}.
\]

Possession skill and game-margin skill therefore normalize improvement at
different aggregation levels. Each skill score must be interpreted alongside
the RMSE from the same level.

## RAPM conversion

Ridge and Bayesian RAPM predict home net rating, while the common target is
offense margin per possession. Their predictions are translated as

\[
\widehat{{y}}_i
= \overline{{y}}_{{train}}
+ s_i\frac{{\widehat{{r}}_i}}{{200}},
\]

where \(\widehat{{r}}_i\) is the predicted home net rating for the possession's
lineup and \(\overline{{y}}_{{train}}\) is the regular-training offense-margin
mean. The factor 200 follows from comparing the same two lineups over two
role-swapped possessions. If their signed lineup effect is \(\delta_i\), then

\[
\widehat{{r}}_i
= 100\left[
(\overline{{y}}_{{train}}+\delta_i)
-
(\overline{{y}}_{{train}}-\delta_i)
\right]
= 200\delta_i.
\]

## Regular-season holdout

Mean reference: possession RMSE
**{float(regular_reference.mean_reference_possession_rmse):.6f}** and
eligible-possession game-margin RMSE
**{float(regular_reference.mean_reference_game_margin_rmse):.4f}**.

{regular_table}

## Playoffs

Mean reference: possession RMSE
**{float(playoff_reference.mean_reference_possession_rmse):.6f}** and
eligible-possession game-margin RMSE
**{float(playoff_reference.mean_reference_game_margin_rmse):.4f}**.

{playoff_table}

## Interpretation

Bayesian RAPM uses the same Gaussian prior and lambda corresponding to ridge,
so its posterior mean is the ridge point estimate. Equal point-prediction
metrics are expected. Bayesian value appears in uncertainty, interval
calibration, and rank probabilities rather than lower posterior-mean RMSE.

The additive neural and Deep Sets exemplars select learning rate and AdamW
weight decay by validation-possession-weighted MSE across expanding
regular-season folds. CatBoost uses its resolved defaults and chooses its tree
count from the latest chronological validation fold. RAPM + Transformer keeps
the ridge prediction frozen and learns only a position-free attention
residual, using a RAPM fit that excludes every validation or test game it
predicts. Regular holdout and playoff outcomes remain outside every selection
process.

## Paired model comparisons

To preserve correlation among possessions from the same game, uncertainty is
estimated by resampling complete games with replacement. Each row identifies
its candidate and reference model. For bootstrap draw \(b\),

\[
\Delta_b =
\operatorname{{RMSE}}_{{candidate,b}}
-
\operatorname{{RMSE}}_{{reference,b}}.
\]

Negative differences favor the candidate. The interval is the 2.5th through
97.5th percentile of 2,000 paired game-cluster bootstrap draws. The final
column is the share of draws where \(\Delta_b < 0\).

{comparison_table}

## Correctness checks

`tests/test_model_evaluation.py` verifies offense-to-home aggregation,
identical row counts and keys across models, playoff possession construction,
metric calculations, and bolded-winner rendering. The evaluator additionally
requires:

- validated source model manifests and exact artifact hashes;
- Bayesian and Transformer runs derived from the selected ridge run;
- matching regular-holdout game IDs across every model;
- matching possession, game, and player counts across neural-model sources;
- exact held-out possession keys for every stored prediction set;
- Bayesian and ridge posterior-mean equivalence within tolerance.

Run the focused checks with:

```bash
uv run pytest -q tests/test_model_evaluation.py
```

## Reproduce

```bash
uv run nba-evaluate-models {manifest.season} \
  --ridge-run-id {manifest.ridge_run_id} \
  --bayesian-run-id {manifest.bayesian_run_id} \
  --neural-run-id {manifest.neural_run_id} \
  --deep-sets-run-id {manifest.deep_sets_run_id} \
  --catboost-run-id {manifest.catboost_run_id} \
  --rapm-transformer-run-id {manifest.rapm_transformer_run_id}
```

| Provenance | Value |
| --- | --- |
| Evaluation run | `{manifest.run_id}` |
| Ridge run | `{manifest.ridge_run_id}` |
| Bayesian run | `{manifest.bayesian_run_id}` |
| Neural run | `{manifest.neural_run_id}` |
| Neural selection | {neural_selection} |
| Deep Sets run | `{manifest.deep_sets_run_id}` |
| Deep Sets selection | {deep_sets_selection} |
| CatBoost run | `{manifest.catboost_run_id}` |
| CatBoost selection | {catboost_selection} |
| RAPM + Transformer run | `{manifest.rapm_transformer_run_id}` |
| RAPM + Transformer selection | {transformer_selection} |
| Evaluation code | `{manifest.evaluation_code_version}` |
| Evaluation manifest SHA-256 | `{_sha256_file_for_display(manifest)}` |

The underlying `metrics.parquet`, possession predictions, cohort summary, and
source metadata are stored under
`artifacts/reports/model_evaluation/{manifest.season}/{manifest.run_id}/`.

| Artifact | Contents |
| --- | --- |
| `metrics.parquet` | One row per cohort and model |
| `predictions.parquet` | Every model prediction on every eligible possession |
| `cohorts.parquet` | Inclusion counts, dates, and training cutoffs |
| `comparisons.parquet` | Paired game-cluster bootstrap intervals |
| `model_sources.json` | Model states, translation, means, and unknown exposures |
| `manifest.json` | Source hashes, code fingerprint, and artifact integrity |
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return path


def validate_model_evaluation_run(
    run_dir: Path | str,
) -> ModelEvaluationManifest:
    """Require every recorded evaluation artifact to match its manifest."""

    root = Path(run_dir)
    manifest = ModelEvaluationManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    expected = {artifact.filename for artifact in manifest.artifacts}
    if actual != expected:
        raise ValueError("Evaluation run files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Evaluation artifact byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Evaluation artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None and len(pd.read_parquet(path)) != artifact.row_count:
            raise ValueError(f"Evaluation artifact rows changed: {artifact.filename}")
    return manifest


def _resolve_sources(
    season: str,
    model_root: Path,
    ridge_run_id: str | None,
    bayesian_run_id: str | None,
    neural_run_id: str | None,
    deep_sets_run_id: str | None,
    catboost_run_id: str | None,
    rapm_transformer_run_id: str | None,
) -> EvaluationSources:
    ridge_dir = _resolve_run(model_root / "rapm" / season, ridge_run_id)
    bayesian_dir = _resolve_run(
        model_root / "bayesian_rapm" / season,
        bayesian_run_id,
    )
    neural_dir = _resolve_run(model_root / "neural_rapm" / season, neural_run_id)
    deep_sets_dir = _resolve_run(
        model_root / "deep_sets" / season,
        deep_sets_run_id,
    )
    catboost_dir = _resolve_run(
        model_root / "catboost" / season,
        catboost_run_id,
    )
    transformer_dir = _resolve_run(
        model_root / "rapm_transformer" / season,
        rapm_transformer_run_id,
    )
    return EvaluationSources(
        ridge_dir=ridge_dir,
        ridge_manifest=validate_baseline_run(ridge_dir),
        bayesian_dir=bayesian_dir,
        bayesian_manifest=validate_bayesian_rapm_run(bayesian_dir),
        neural_dir=neural_dir,
        neural_manifest=validate_neural_rapm_run(neural_dir),
        deep_sets_dir=deep_sets_dir,
        deep_sets_manifest=validate_deep_sets_run(deep_sets_dir),
        catboost_dir=catboost_dir,
        catboost_manifest=validate_catboost_run(catboost_dir),
        transformer_dir=transformer_dir,
        transformer_manifest=validate_rapm_transformer_run(transformer_dir),
    )


def _resolve_run(season_dir: Path, run_id: str | None) -> Path:
    if run_id is None:
        latest_path = season_dir / "latest.json"
        if not latest_path.is_file():
            raise ValueError(f"No latest model run is available under {season_dir}")
        run_id = str(json.loads(latest_path.read_text())["run_id"])
    run_dir = season_dir / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Model run does not exist: {run_dir}")
    return run_dir


def _validate_source_relationships(
    sources: EvaluationSources,
    season: str,
    analytical_dir: Path | str,
) -> None:
    manifests = (
        sources.ridge_manifest,
        sources.bayesian_manifest,
        sources.neural_manifest,
        sources.deep_sets_manifest,
        sources.catboost_manifest,
        sources.transformer_manifest,
    )
    if any(manifest.season != season for manifest in manifests):
        raise ValueError("Evaluation source seasons do not match")
    if sources.bayesian_manifest.source_model_run_id != sources.ridge_manifest.run_id:
        raise ValueError("Bayesian evaluation source does not derive from ridge source")
    if (
        sources.transformer_manifest.source_rapm_run_id
        != sources.ridge_manifest.run_id
    ):
        raise ValueError("Transformer evaluation source does not derive from ridge source")
    rapm_data = validate_rapm_stint_partition(
        Path(analytical_dir) / "rapm_stints" / season / "regular"
    )
    neural_data = validate_neural_possession_partition(
        Path(analytical_dir) / "neural_possessions" / season / "regular"
    )
    if (
        rapm_data.included_stint_count != sources.ridge_manifest.stint_count
        or rapm_data.source_game_count != sources.ridge_manifest.game_count
        or rapm_data.player_count != sources.ridge_manifest.player_count
    ):
        raise ValueError("Current RAPM stint structure does not match the ridge run")
    neural_manifests = (
        sources.neural_manifest,
        sources.deep_sets_manifest,
        sources.catboost_manifest,
        sources.transformer_manifest,
    )
    for manifest in neural_manifests:
        if (
            manifest.possession_count != neural_data.included_possession_count
            or manifest.game_count != neural_data.source_game_count
            or manifest.player_count != neural_data.player_count
        ):
            raise ValueError(
                f"Current neural possession structure does not match {manifest.run_id}"
            )
    comparison = pd.read_parquet(sources.bayesian_dir / "comparison_metrics.parquet").iloc[0]
    if float(comparison["max_absolute_test_prediction_difference"]) > 1e-4:
        raise ValueError("Bayesian posterior mean no longer matches ridge test predictions")


def _read_curated_partition(
    table: str,
    season: str,
    season_type: str,
    curated_dir: Path | str,
) -> tuple[CuratedPartitionManifest, pd.DataFrame, Path]:
    partition = CuratedPartition(
        table=table,
        season=season,
        season_type=season_type,
    )
    manifest = read_curated_partition_manifest(partition, curated_dir)
    validate_curated_partition(manifest, curated_dir)
    partition_dir = CuratedDatasetLayout(Path(curated_dir)).partition_dir(partition)
    return manifest, pd.read_parquet(partition_dir), partition_dir


def _regular_stint_lookup(
    possessions: pd.DataFrame,
    segments: pd.DataFrame,
    lineup_stints: pd.DataFrame,
) -> pd.DataFrame:
    game_ids = set(possessions["game_id"].astype(str))
    game_segments = segments.loc[segments["game_id"].astype(str).isin(game_ids)].copy()
    counts = game_segments.groupby(["game_id", "possession_id"], sort=False)[
        "possession_id"
    ].transform("size")
    single = game_segments.loc[counts.eq(1)].copy()
    game_stints = lineup_stints.loc[
        lineup_stints["game_id"].astype(str).isin(game_ids)
    ].copy()
    single["stint_index"] = _assign_segments_to_stints(game_stints, single).to_numpy()
    lookup = single.loc[:, ["game_id", "possession_id", "stint_index"]]
    if lookup.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Regular single-lineup possession lookup is not unique")
    if len(lookup) != len(possessions):
        raise ValueError("Regular possession-to-stint lookup does not match cohort")
    return lookup


def _mapped_stint_predictions(
    possessions: pd.DataFrame,
    stint_lookup: pd.DataFrame,
    stint_predictions: pd.DataFrame,
    prediction_column: str,
    training_mean: float,
) -> np.ndarray:
    source = stint_predictions.loc[:, ["game_id", "stint_index", prediction_column]]
    mapped = (
        possessions.loc[:, ["game_id", "possession_id", "home_offense_sign"]]
        .merge(
            stint_lookup,
            on=["game_id", "possession_id"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        .merge(
            source,
            on=["game_id", "stint_index"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )
    if mapped[prediction_column].isna().any():
        raise ValueError(f"Missing mapped stint predictions: {prediction_column}")
    return (
        training_mean
        + mapped["home_offense_sign"].to_numpy(dtype=float)
        * mapped[prediction_column].to_numpy(dtype=float)
        / 200.0
    )


def _mapped_possession_predictions(
    possessions: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    prediction_column: str,
) -> np.ndarray:
    mapped = possessions.loc[:, ["game_id", "possession_id"]].merge(
        prediction_rows.loc[:, ["game_id", "possession_id", prediction_column]],
        on=["game_id", "possession_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if mapped[prediction_column].isna().any():
        raise ValueError(f"Missing mapped possession predictions: {prediction_column}")
    return mapped[prediction_column].to_numpy(dtype=float)


def _ridge_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
    training_mean: float,
) -> tuple[np.ndarray, int]:
    rankings = pd.read_parquet(run_dir / "player_rankings.parquet")
    coefficients = dict(
        zip(
            rankings["player_id"].astype(int),
            rankings["rapm"].astype(float),
            strict=True,
        )
    )
    parameters = json.loads((run_dir / "model_parameters.json").read_text())
    intercept = float(parameters["rapm"]["all_season_intercept_home_court"])
    return _translated_lineup_predictions(
        possessions,
        coefficients,
        intercept,
        training_mean,
    )


def _bayesian_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
    training_mean: float,
) -> tuple[np.ndarray, int]:
    player_columns = {
        int(player_id): int(column)
        for player_id, column in json.loads(
            (run_dir / "player_columns.json").read_text()
        ).items()
    }
    state = np.load(run_dir / "posterior_state.npz")
    location = np.asarray(state["location"], dtype=float)
    coefficients = {
        player_id: float(location[column + 1])
        for player_id, column in player_columns.items()
    }
    return _translated_lineup_predictions(
        possessions,
        coefficients,
        float(location[0]),
        training_mean,
    )


def _translated_lineup_predictions(
    possessions: pd.DataFrame,
    coefficients: Mapping[int, float],
    home_net_rating_intercept: float,
    training_mean: float,
) -> tuple[np.ndarray, int]:
    signed_player_effect = np.empty(len(possessions), dtype=float)
    unknown_exposures = 0
    for index, (offense, defense) in enumerate(
        zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            strict=True,
        )
    ):
        unknown_exposures += sum(int(player_id) not in coefficients for player_id in offense)
        unknown_exposures += sum(int(player_id) not in coefficients for player_id in defense)
        signed_player_effect[index] = sum(
            coefficients.get(int(player_id), 0.0) for player_id in offense
        ) - sum(coefficients.get(int(player_id), 0.0) for player_id in defense)
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    prediction = (
        training_mean
        + signed_player_effect / 200.0
        + signs * home_net_rating_intercept / 200.0
    )
    return prediction, unknown_exposures


def _neural_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
) -> tuple[np.ndarray, int]:
    return _torch_model_playoff_predictions(
        possessions,
        run_dir,
        AdditiveRapmModule,
    )


def _deep_sets_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
) -> tuple[np.ndarray, int]:
    return _torch_model_playoff_predictions(
        possessions,
        run_dir,
        DeepSetsRapmModule,
    )


def _catboost_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
) -> tuple[np.ndarray, int]:
    player_columns = {
        int(player_id): int(column)
        for player_id, column in json.loads(
            (run_dir / "player_columns.json").read_text()
        ).items()
    }
    model = CatBoostRegressor()
    model.load_model(run_dir / "model.cbm")
    return catboost_predictions(model, possessions, player_columns)


def _transformer_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
) -> tuple[np.ndarray, int]:
    player_columns = {
        int(player_id): int(column)
        for player_id, column in json.loads(
            (run_dir / "player_columns.json").read_text()
        ).items()
    }
    coefficient_rows = pd.read_parquet(
        run_dir / "rapm_player_coefficients.parquet"
    )
    coefficients = dict(
        zip(
            coefficient_rows["player_id"].astype(int),
            coefficient_rows["rapm"].astype(float),
            strict=True,
        )
    )
    state = json.loads((run_dir / "rapm_state.json").read_text())
    base, base_unknown = frozen_rapm_predictions(
        possessions,
        coefficients,
        intercept_home_net_rating=float(
            state["intercept_home_net_rating"]
        ),
        mean_offense_margin=float(state["mean_offense_margin"]),
    )
    module = RapmTransformerModule.load_from_checkpoint(
        run_dir / "model.ckpt",
        map_location="cpu",
    )
    module.eval()
    total, _ = rapm_transformer_predictions(
        module,
        possessions,
        player_columns,
        base,
    )
    player_exposures = pd.concat(
        [
            possessions["offense_player_ids"].explode(),
            possessions["defense_player_ids"].explode(),
        ],
        ignore_index=True,
    ).astype(int)
    transformer_unknown = int((~player_exposures.isin(player_columns)).sum())
    if transformer_unknown != base_unknown:
        raise ValueError("Transformer and frozen RAPM unknown exposures differ")
    return total, transformer_unknown


def _torch_model_playoff_predictions(
    possessions: pd.DataFrame,
    run_dir: Path,
    module_class: type[AdditiveRapmModule] | type[DeepSetsRapmModule],
) -> tuple[np.ndarray, int]:
    player_columns = {
        int(player_id): int(column)
        for player_id, column in json.loads(
            (run_dir / "player_columns.json").read_text()
        ).items()
    }
    dataset = PossessionTensorDataset(possessions, player_columns)
    loader = DataLoader(dataset, batch_size=2_048, shuffle=False, num_workers=0)
    module = module_class.load_from_checkpoint(
        run_dir / "model.ckpt",
        map_location="cpu",
    )
    module.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            prediction = module(
                batch["offense_player_indices"],
                batch["defense_player_indices"],
                batch["home_offense_sign"],
            )
            predictions.append(prediction.numpy())
    player_exposures = pd.concat(
        [
            possessions["offense_player_ids"].explode(),
            possessions["defense_player_ids"].explode(),
        ],
        ignore_index=True,
    ).astype(int)
    unknown_exposures = int((~player_exposures.isin(player_columns)).sum())
    return np.concatenate(predictions), unknown_exposures


def _cohort_summary(
    eligible: pd.DataFrame,
    source_segments: pd.DataFrame,
    *,
    cohort: str,
    training_game_count: int,
) -> dict[str, Any]:
    source_counts = source_segments.groupby(["game_id", "possession_id"], sort=False).size()
    source_possessions = len(source_counts)
    excluded = int(source_counts.gt(1).sum())
    if source_possessions != len(eligible) + excluded:
        raise ValueError(f"{cohort} source possessions do not conserve eligibility")
    return {
        "cohort": cohort,
        "season": str(eligible["season"].iloc[0]),
        "season_type": str(eligible["season_type"].iloc[0]),
        "game_count": int(eligible["game_id"].nunique()),
        "source_possession_count": source_possessions,
        "eligible_possession_count": len(eligible),
        "excluded_multi_lineup_possession_count": excluded,
        "eligible_fraction": len(eligible) / source_possessions,
        "first_game_date": eligible["game_date"].min(),
        "last_game_date": eligible["game_date"].max(),
        "training_game_count": training_game_count,
    }


def _write_evaluation(
    season: str,
    sources: EvaluationSources,
    evaluation: ModelEvaluation,
    regular_segments_manifest: CuratedPartitionManifest,
    regular_segments_dir: Path,
    regular_lineups_manifest: CuratedPartitionManifest,
    regular_lineups_dir: Path,
    playoff_segments_manifest: CuratedPartitionManifest,
    playoff_segments_dir: Path,
    reports_dir: Path | str,
) -> tuple[ModelEvaluationManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"evaluation-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(reports_dir) / "model_evaluation" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        parquet_outputs = {
            "metrics.parquet": evaluation.metrics,
            "predictions.parquet": evaluation.predictions,
            "cohorts.parquet": evaluation.cohorts,
            "comparisons.parquet": evaluation.comparisons,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        (temporary_dir / "model_sources.json").write_text(
            json.dumps(evaluation.model_sources, indent=2, sort_keys=True) + "\n"
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
        regular = evaluation.cohorts.set_index("cohort").loc["regular_holdout"]
        playoffs = evaluation.cohorts.set_index("cohort").loc["playoffs"]
        manifest = ModelEvaluationManifest(
            schema_version=3,
            run_id=run_id,
            created_at=now,
            season=season,
            evaluation_code_version=evaluation_code_fingerprint(),
            ridge_run_id=sources.ridge_manifest.run_id,
            ridge_manifest_sha256=_sha256_file(sources.ridge_dir / "manifest.json"),
            bayesian_run_id=sources.bayesian_manifest.run_id,
            bayesian_manifest_sha256=_sha256_file(
                sources.bayesian_dir / "manifest.json"
            ),
            neural_run_id=sources.neural_manifest.run_id,
            neural_manifest_sha256=_sha256_file(sources.neural_dir / "manifest.json"),
            neural_learning_rate=sources.neural_manifest.learning_rate,
            neural_weight_decay=sources.neural_manifest.weight_decay,
            neural_selected_epochs=sources.neural_manifest.selected_epochs,
            deep_sets_run_id=sources.deep_sets_manifest.run_id,
            deep_sets_manifest_sha256=_sha256_file(
                sources.deep_sets_dir / "manifest.json"
            ),
            deep_sets_learning_rate=sources.deep_sets_manifest.learning_rate,
            deep_sets_weight_decay=sources.deep_sets_manifest.weight_decay,
            deep_sets_selected_epochs=sources.deep_sets_manifest.selected_epochs,
            deep_sets_leaderboard_seed=(
                sources.deep_sets_manifest.leaderboard_seed
            ),
            catboost_run_id=sources.catboost_manifest.run_id,
            catboost_manifest_sha256=_sha256_file(
                sources.catboost_dir / "manifest.json"
            ),
            catboost_max_iterations=sources.catboost_manifest.max_iterations,
            catboost_best_iteration=sources.catboost_manifest.best_iteration,
            catboost_selected_tree_count=(
                sources.catboost_manifest.selected_tree_count
            ),
            catboost_resolved_learning_rate=(
                sources.catboost_manifest.resolved_learning_rate
            ),
            rapm_transformer_run_id=sources.transformer_manifest.run_id,
            rapm_transformer_manifest_sha256=_sha256_file(
                sources.transformer_dir / "manifest.json"
            ),
            rapm_transformer_source_rapm_run_id=(
                sources.transformer_manifest.source_rapm_run_id
            ),
            rapm_transformer_learning_rate=(
                sources.transformer_manifest.learning_rate
            ),
            rapm_transformer_weight_decay=(
                sources.transformer_manifest.weight_decay
            ),
            rapm_transformer_selected_epochs=(
                sources.transformer_manifest.selected_epochs
            ),
            rapm_transformer_leaderboard_seed=(
                sources.transformer_manifest.leaderboard_seed
            ),
            regular_segments_manifest_sha256=_sha256_file(
                regular_segments_dir / "_manifest.json"
            ),
            regular_lineup_stints_manifest_sha256=_sha256_file(
                regular_lineups_dir / "_manifest.json"
            ),
            playoff_segments_manifest_sha256=_sha256_file(
                playoff_segments_dir / "_manifest.json"
            ),
            models=MODEL_ORDER,
            regular_holdout_game_count=int(regular["game_count"]),
            regular_holdout_possession_count=int(
                regular["eligible_possession_count"]
            ),
            playoff_game_count=int(playoffs["game_count"]),
            playoff_possession_count=int(playoffs["eligible_possession_count"]),
            artifacts=artifacts,
        )
        _validate_curated_manifest_arguments(
            regular_segments_manifest,
            regular_lineups_manifest,
            playoff_segments_manifest,
        )
        (temporary_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        temporary_dir.replace(run_dir)
        validate_model_evaluation_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _validate_curated_manifest_arguments(
    regular_segments: CuratedPartitionManifest,
    regular_lineups: CuratedPartitionManifest,
    playoff_segments: CuratedPartitionManifest,
) -> None:
    if regular_segments.game_ids != regular_lineups.game_ids:
        raise ValueError("Regular segment and lineup manifests have different games")
    if playoff_segments.partition.season_type != "playoffs":
        raise ValueError("Playoff evaluation source is not a playoff partition")


def _render_comparison_table(comparisons: pd.DataFrame) -> str:
    metric_names = {
        "possession_rmse": "Possession RMSE",
        "eligible_possession_game_margin_rmse": (
            "Eligible-possession game-margin RMSE"
        ),
    }
    lines = [
        "| Cohort | Candidate | Reference | Metric | Difference | "
        "95% interval | P(candidate better) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {COHORT_NAMES[str(row.cohort)]} | "
            f"{MODEL_NAMES[str(row.candidate_model)]} | "
            f"{MODEL_NAMES[str(row.reference_model)]} | "
            f"{metric_names[str(row.metric)]} | {float(row.difference):.6f} | "
            f"[{float(row.ci_lower):.6f}, {float(row.ci_upper):.6f}] | "
            f"{float(row.probability_candidate_better):.1%} |"
        )
    return "\n".join(lines)


def _render_metric_table(metrics: pd.DataFrame) -> str:
    ordered = metrics.set_index("model").loc[list(MODEL_ORDER)].reset_index()
    columns = (
        ("possession_rmse", 6, "min"),
        ("possession_mae", 6, "min"),
        ("possession_skill_vs_mean", 4, "max"),
        ("eligible_possession_game_margin_rmse", 4, "min"),
        ("game_margin_skill_vs_mean", 4, "max"),
    )
    lines = [
        "| Model | Possession RMSE | Possession MAE | "
        "Possession skill vs mean | Eligible-possession game-margin RMSE | "
        "Game-margin skill vs mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered.itertuples(index=False):
        values = []
        for column, decimals, direction in columns:
            value = float(getattr(row, column))
            comparison = ordered[column].to_numpy(dtype=float)
            is_winner = _is_display_winner(value, comparison, decimals, direction)
            rendered = (
                f"{100.0 * value:.{decimals}f}%"
                if column.endswith("skill_vs_mean")
                else f"{value:.{decimals}f}"
            )
            values.append(f"**{rendered}**" if is_winner else rendered)
        lines.append(
            f"| {MODEL_NAMES[str(row.model)]} | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines)


def _is_display_winner(
    value: float,
    comparison: np.ndarray,
    decimals: int,
    direction: str,
) -> bool:
    scaled = comparison * 100.0 if direction == "max" else comparison
    candidate = value * 100.0 if direction == "max" else value
    rounded = np.round(scaled, decimals)
    target = np.max(rounded) if direction == "max" else np.min(rounded)
    return bool(np.round(candidate, decimals) == target)


def _sha256_file_for_display(manifest: ModelEvaluationManifest) -> str:
    return hashlib.sha256(
        (manifest.model_dump_json(indent=2) + "\n").encode()
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen ridge, Bayesian, neural, CatBoost, and Transformer "
            "models on the regular holdout and playoffs."
        )
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--ridge-run-id")
    parser.add_argument("--bayesian-run-id")
    parser.add_argument("--neural-run-id")
    parser.add_argument("--deep-sets-run-id")
    parser.add_argument("--catboost-run-id")
    parser.add_argument("--rapm-transformer-run-id")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--model-artifacts-dir", default="artifacts/models")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--docs-path", default="docs/models/leaderboard.md")
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Do not regenerate the documentation scoreboard",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, run_dir = build_model_evaluation(
        args.season,
        ridge_run_id=args.ridge_run_id,
        bayesian_run_id=args.bayesian_run_id,
        neural_run_id=args.neural_run_id,
        deep_sets_run_id=args.deep_sets_run_id,
        catboost_run_id=args.catboost_run_id,
        rapm_transformer_run_id=args.rapm_transformer_run_id,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        model_artifacts_dir=args.model_artifacts_dir,
        reports_dir=args.reports_dir,
        docs_path=None if args.no_docs else args.docs_path,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = (
        f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    )
    print(
        f"{manifest.season} model evaluation: "
        f"regular_games={manifest.regular_holdout_game_count}, "
        f"regular_possessions={manifest.regular_holdout_possession_count}, "
        f"playoff_games={manifest.playoff_game_count}, "
        f"playoff_possessions={manifest.playoff_possession_count}; "
        f"run={run_dir}{tracking_text}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
