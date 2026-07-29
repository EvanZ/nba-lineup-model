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
from scipy import sparse

from nba_lineup_model.evaluation.metrics import (
    game_margin_rmse,
    mean_absolute_error,
    mean_squared_error,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    BaselineRunManifest,
    BayesianRapmRunManifest,
)
from nba_lineup_model.modeling.stints import (
    read_rapm_stints,
    validate_rapm_stint_partition,
)
from nba_lineup_model.modeling.train import validate_baseline_run
from nba_lineup_model.models.baselines import signed_entity_matrix
from nba_lineup_model.models.bayesian import ConjugateBayesianRidge

DEFAULT_POSTERIOR_DRAWS = 4_000
DEFAULT_POSTERIOR_SEED = 17
DEFAULT_CREDIBLE_INTERVAL = 0.90
_CALIBRATION_INTERVALS = (0.50, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class BayesianRapmExperiment:
    """In-memory outputs for one exact Bayesian RAPM analysis."""

    posterior: ConjugateBayesianRidge
    posterior_rankings: pd.DataFrame
    comparison_metrics: pd.DataFrame
    predictive_calibration: pd.DataFrame
    test_predictions: pd.DataFrame
    player_columns: dict[int, int]
    model_parameters: dict[str, Any]


def bayesian_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash Bayesian RAPM source files for reproducible model runs."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        paths = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "bayesian.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "stints.py",
            package_root / "modeling" / "train.py",
            package_root / "models" / "baselines.py",
            package_root / "models" / "bayesian.py",
        )
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one Bayesian RAPM source path is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"Bayesian RAPM source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def fit_bayesian_rapm_experiment(
    stints: pd.DataFrame,
    *,
    player_columns: Mapping[int, int],
    ridge_rankings: pd.DataFrame,
    game_splits: pd.DataFrame,
    ridge_test_predictions: pd.DataFrame,
    ridge_intercept: float,
    selected_lambda: float,
    posterior_draws: int = DEFAULT_POSTERIOR_DRAWS,
    posterior_seed: int = DEFAULT_POSTERIOR_SEED,
    credible_interval_probability: float = DEFAULT_CREDIBLE_INTERVAL,
) -> BayesianRapmExperiment:
    """Fit, evaluate, and summarize the conjugate counterpart to one ridge RAPM."""

    _validate_experiment_inputs(
        stints,
        player_columns,
        ridge_rankings,
        game_splits,
        selected_lambda,
        posterior_draws,
        posterior_seed,
        credible_interval_probability,
    )
    normalized_columns = {int(key): int(value) for key, value in player_columns.items()}
    player_ids = _player_ids_in_column_order(normalized_columns)
    player_matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        normalized_columns,
        multiple=True,
    )
    target = stints["target_home_net_rating"].to_numpy(dtype=np.float64)
    weights = stints["possessions"].to_numpy(dtype=np.float64)
    posterior = ConjugateBayesianRidge.fit(
        player_matrix,
        target,
        weights,
        selected_lambda,
    )
    parameter_draws = posterior.draw_parameters(
        posterior_draws,
        seed=posterior_seed,
    )
    posterior_rankings = _posterior_rankings(
        ridge_rankings,
        player_ids,
        posterior,
        parameter_draws[:, 1:],
        credible_interval_probability,
    )

    final_split = game_splits.loc[game_splits["split"].eq("final")]
    final_train_ids = set(
        final_split.loc[final_split["role"].eq("train"), "game_id"].astype(str)
    )
    final_test_ids = set(
        final_split.loc[final_split["role"].eq("test"), "game_id"].astype(str)
    )
    game_ids = stints["game_id"].astype(str)
    train_mask = game_ids.isin(final_train_ids).to_numpy()
    test_mask = game_ids.isin(final_test_ids).to_numpy()
    test_posterior = ConjugateBayesianRidge.fit(
        player_matrix[train_mask],
        target[train_mask],
        weights[train_mask],
        selected_lambda,
    )
    test_predictions, predictive_calibration = _evaluate_test_posterior(
        stints.loc[test_mask].reset_index(drop=True),
        player_matrix[test_mask],
        target[test_mask],
        weights[test_mask],
        test_posterior,
        ridge_test_predictions,
    )
    comparison_metrics = _comparison_metrics(
        posterior_rankings,
        ridge_intercept,
        posterior,
        test_predictions,
        stints.loc[test_mask].reset_index(drop=True),
        weights[test_mask],
    )
    marginal = posterior.marginal_summary(
        interval_probability=credible_interval_probability
    )
    test_marginal = test_posterior.marginal_summary(
        interval_probability=credible_interval_probability
    )
    model_parameters = {
        "inference": "exact conjugate Gaussian linear regression",
        "likelihood": (
            "target_i ~ Normal(intercept + signed_player_effects, "
            "sigma_squared / normalized_possessions_i)"
        ),
        "player_prior": (
            "beta | sigma_squared ~ Normal(0, sigma_squared / ridge_alpha)"
        ),
        "intercept_prior": "flat",
        "residual_variance_prior": "p(sigma_squared) proportional to 1 / sigma_squared",
        "credible_interval_probability": credible_interval_probability,
        "posterior_draws": posterior_draws,
        "posterior_seed": posterior_seed,
        "all_season": {
            "observation_count": len(stints),
            "regularization": posterior.regularization,
            "ridge_alpha": posterior.ridge_alpha,
            "degrees_of_freedom": posterior.degrees_of_freedom,
            "training_possession_weight_mean": posterior.training_weight_mean,
            "residual_quadratic": posterior.residual_quadratic,
            "residual_variance_posterior_mean": posterior.residual_variance_mean,
            "intercept_mean": posterior.intercept_mean,
            "intercept_standard_deviation": float(marginal.standard_deviation[0]),
            "intercept_lower": float(marginal.lower[0]),
            "intercept_upper": float(marginal.upper[0]),
        },
        "final_training_window": {
            "observation_count": int(train_mask.sum()),
            "regularization": test_posterior.regularization,
            "ridge_alpha": test_posterior.ridge_alpha,
            "degrees_of_freedom": test_posterior.degrees_of_freedom,
            "training_possession_weight_mean": test_posterior.training_weight_mean,
            "residual_quadratic": test_posterior.residual_quadratic,
            "residual_variance_posterior_mean": test_posterior.residual_variance_mean,
            "intercept_mean": test_posterior.intercept_mean,
            "intercept_standard_deviation": float(test_marginal.standard_deviation[0]),
        },
    }
    return BayesianRapmExperiment(
        posterior=posterior,
        posterior_rankings=posterior_rankings,
        comparison_metrics=comparison_metrics,
        predictive_calibration=predictive_calibration,
        test_predictions=test_predictions,
        player_columns=normalized_columns,
        model_parameters=model_parameters,
    )


def train_bayesian_rapm(
    season: str,
    *,
    source_run_id: str | None = None,
    analytical_dir: Path | str = Path("data/analytical"),
    model_artifacts_dir: Path | str = Path("artifacts/models"),
    posterior_draws: int = DEFAULT_POSTERIOR_DRAWS,
    posterior_seed: int = DEFAULT_POSTERIOR_SEED,
    credible_interval_probability: float = DEFAULT_CREDIBLE_INTERVAL,
) -> tuple[BayesianRapmRunManifest, Path]:
    """Fit exact Bayesian RAPM from one validated canonical ridge run."""

    model_root = Path(model_artifacts_dir)
    source_dir = _resolve_source_run(season, model_root, source_run_id)
    source_manifest = validate_baseline_run(source_dir)
    dataset_dir = Path(analytical_dir) / "rapm_stints" / season / "regular"
    dataset_manifest = validate_rapm_stint_partition(dataset_dir)
    if source_manifest.dataset_part_sha256 != dataset_manifest.part_sha256:
        raise ValueError("Current RAPM stints do not match the source ridge run")
    stints = read_rapm_stints(season, analytical_dir)
    player_columns = {
        int(identifier): int(column)
        for identifier, column in json.loads(
            (source_dir / "player_columns.json").read_text()
        ).items()
    }
    ridge_rankings = pd.read_parquet(source_dir / "player_rankings.parquet")
    game_splits = pd.read_parquet(source_dir / "game_splits.parquet")
    ridge_test_predictions = pd.read_parquet(source_dir / "test_predictions.parquet")
    parameters = json.loads((source_dir / "model_parameters.json").read_text())
    ridge_intercept = float(parameters["rapm"]["all_season_intercept_home_court"])
    experiment = fit_bayesian_rapm_experiment(
        stints,
        player_columns=player_columns,
        ridge_rankings=ridge_rankings,
        game_splits=game_splits,
        ridge_test_predictions=ridge_test_predictions,
        ridge_intercept=ridge_intercept,
        selected_lambda=source_manifest.selected_rapm_lambda,
        posterior_draws=posterior_draws,
        posterior_seed=posterior_seed,
        credible_interval_probability=credible_interval_probability,
    )
    return _write_experiment(
        season,
        source_dir,
        source_manifest,
        experiment,
        posterior_draws,
        posterior_seed,
        credible_interval_probability,
        model_root,
    )


def validate_bayesian_rapm_run(
    run_dir: Path | str,
) -> BayesianRapmRunManifest:
    """Require every recorded Bayesian RAPM artifact to match its manifest."""

    root = Path(run_dir)
    manifest = BayesianRapmRunManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    expected = {artifact.filename for artifact in manifest.artifacts}
    if actual != expected:
        raise ValueError("Bayesian RAPM run files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Bayesian artifact byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Bayesian artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None:
            if len(pd.read_parquet(path)) != artifact.row_count:
                raise ValueError(f"Bayesian artifact rows changed: {artifact.filename}")
    return manifest


def _posterior_rankings(
    ridge_rankings: pd.DataFrame,
    player_ids: tuple[int, ...],
    posterior: ConjugateBayesianRidge,
    coefficient_draws: np.ndarray,
    interval_probability: float,
) -> pd.DataFrame:
    marginal = posterior.marginal_summary(interval_probability=interval_probability)
    coefficient_summary = pd.DataFrame(
        {
            "player_id": player_ids,
            "posterior_mean": marginal.mean[1:],
            "posterior_standard_deviation": marginal.standard_deviation[1:],
            "posterior_lower": marginal.lower[1:],
            "posterior_upper": marginal.upper[1:],
            "probability_positive": marginal.probability_positive[1:],
        }
    )
    rankings = ridge_rankings.rename(columns={"rapm": "ridge_rapm"}).merge(
        coefficient_summary,
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    if len(rankings) != len(player_ids):
        raise ValueError("Ridge rankings do not match the player column vocabulary")
    rankings["posterior_minus_ridge"] = rankings["posterior_mean"] - rankings["ridge_rapm"]

    player_to_column = {player_id: column for column, player_id in enumerate(player_ids)}
    eligible = rankings.loc[rankings["exposure_eligible"], ["player_id"]].copy()
    eligible_columns = np.array(
        [player_to_column[int(player_id)] for player_id in eligible["player_id"]],
        dtype=np.int64,
    )
    eligible_draws = coefficient_draws[:, eligible_columns]
    order = np.argsort(-eligible_draws, axis=1, kind="stable")
    draw_ranks = np.empty_like(order)
    np.put_along_axis(
        draw_ranks,
        order,
        np.arange(1, len(eligible_columns) + 1, dtype=np.int64)[None, :],
        axis=1,
    )
    rank_summary = pd.DataFrame(
        {
            "player_id": eligible["player_id"].to_numpy(dtype=np.int64),
            "posterior_rank_mean": draw_ranks.mean(axis=0),
            "posterior_rank_p05": np.quantile(draw_ranks, 0.05, axis=0),
            "posterior_rank_median": np.median(draw_ranks, axis=0),
            "posterior_rank_p95": np.quantile(draw_ranks, 0.95, axis=0),
            "posterior_top_25_probability": (draw_ranks <= 25).mean(axis=0),
            "posterior_top_50_probability": (draw_ranks <= 50).mean(axis=0),
        }
    )
    rankings = rankings.merge(
        rank_summary,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    rankings = rankings.sort_values(
        ["eligible_rank", "rank"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    return rankings.loc[
        :,
        [
            "rank",
            "eligible_rank",
            "player_id",
            "player_name",
            "primary_team_id",
            "primary_team_tricode",
            "ridge_rapm",
            "posterior_mean",
            "posterior_minus_ridge",
            "posterior_standard_deviation",
            "posterior_lower",
            "posterior_upper",
            "probability_positive",
            "posterior_rank_mean",
            "posterior_rank_p05",
            "posterior_rank_median",
            "posterior_rank_p95",
            "posterior_top_25_probability",
            "posterior_top_50_probability",
            "raw_on_court_net_rating",
            "stint_count",
            "possessions",
            "seconds",
            "point_margin",
            "primary_team_possessions",
            "exposure_eligible",
        ],
    ]


def _evaluate_test_posterior(
    test_stints: pd.DataFrame,
    test_matrix: sparse.csr_matrix,
    target: np.ndarray,
    weights: np.ndarray,
    posterior: ConjugateBayesianRidge,
    ridge_test_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = {
        probability: posterior.predictive_summary(
            test_matrix,
            weights,
            interval_probability=probability,
        )
        for probability in _CALIBRATION_INTERVALS
    }
    primary = summaries[DEFAULT_CREDIBLE_INTERVAL]
    predictions = test_stints.loc[
        :,
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
    ].copy()
    ridge = ridge_test_predictions.loc[
        :,
        ["game_id", "stint_index", "prediction_rapm"],
    ].rename(columns={"prediction_rapm": "ridge_prediction"})
    predictions = predictions.merge(
        ridge,
        on=["game_id", "stint_index"],
        how="left",
        validate="one_to_one",
    )
    if predictions["ridge_prediction"].isna().any():
        raise ValueError("Source ridge test predictions do not match final test stints")
    predictions["posterior_mean_prediction"] = primary.mean
    predictions["posterior_predictive_standard_deviation"] = primary.standard_deviation
    predictions["posterior_predictive_lower"] = primary.lower
    predictions["posterior_predictive_upper"] = primary.upper
    predictions["posterior_mean_minus_ridge"] = (
        predictions["posterior_mean_prediction"] - predictions["ridge_prediction"]
    )

    calibration_rows = []
    for probability, summary in summaries.items():
        covered = (target >= summary.lower) & (target <= summary.upper)
        width = summary.upper - summary.lower
        calibration_rows.append(
            {
                "nominal_coverage": probability,
                "unweighted_coverage": float(np.mean(covered)),
                "possession_weighted_coverage": float(np.average(covered, weights=weights)),
                "mean_interval_width": float(np.mean(width)),
                "possession_weighted_mean_interval_width": float(
                    np.average(width, weights=weights)
                ),
                "test_stint_count": len(target),
                "test_possessions": float(np.sum(weights)),
            }
        )
    return predictions, pd.DataFrame(calibration_rows)


def _comparison_metrics(
    posterior_rankings: pd.DataFrame,
    ridge_intercept: float,
    posterior: ConjugateBayesianRidge,
    test_predictions: pd.DataFrame,
    test_stints: pd.DataFrame,
    test_weights: np.ndarray,
) -> pd.DataFrame:
    eligible = posterior_rankings.loc[posterior_rankings["exposure_eligible"]]
    ridge_top_25 = set(eligible.nsmallest(25, "eligible_rank")["player_id"])
    posterior_top_25 = set(eligible.nlargest(25, "posterior_mean")["player_id"])
    actual = test_predictions["target_home_net_rating"].to_numpy(dtype=float)
    ridge_prediction = test_predictions["ridge_prediction"].to_numpy(dtype=float)
    posterior_prediction = test_predictions["posterior_mean_prediction"].to_numpy(dtype=float)
    ridge_mse = mean_squared_error(actual, ridge_prediction, test_weights)
    posterior_mse = mean_squared_error(actual, posterior_prediction, test_weights)
    return pd.DataFrame(
        [
            {
                "selected_lambda": posterior.regularization,
                "player_count": len(posterior_rankings),
                "eligible_player_count": len(eligible),
                "max_absolute_coefficient_difference": float(
                    posterior_rankings["posterior_minus_ridge"].abs().max()
                ),
                "coefficient_rmse_difference": float(
                    np.sqrt(np.mean(posterior_rankings["posterior_minus_ridge"] ** 2))
                ),
                "coefficient_correlation": float(
                    posterior_rankings["posterior_mean"].corr(
                        posterior_rankings["ridge_rapm"]
                    )
                ),
                "eligible_rank_spearman": float(
                    eligible["posterior_mean"].rank(ascending=False).corr(
                        eligible["eligible_rank"].astype(float),
                        method="spearman",
                    )
                ),
                "top_25_overlap": len(ridge_top_25 & posterior_top_25),
                "ridge_intercept": ridge_intercept,
                "posterior_intercept_mean": posterior.intercept_mean,
                "absolute_intercept_difference": abs(
                    posterior.intercept_mean - ridge_intercept
                ),
                "max_absolute_test_prediction_difference": float(
                    np.max(np.abs(posterior_prediction - ridge_prediction))
                ),
                "ridge_weighted_rmse": float(np.sqrt(ridge_mse)),
                "posterior_mean_weighted_rmse": float(np.sqrt(posterior_mse)),
                "ridge_weighted_mae": mean_absolute_error(
                    actual,
                    ridge_prediction,
                    test_weights,
                ),
                "posterior_mean_weighted_mae": mean_absolute_error(
                    actual,
                    posterior_prediction,
                    test_weights,
                ),
                "ridge_game_margin_rmse": game_margin_rmse(
                    test_stints["game_id"],
                    actual,
                    ridge_prediction,
                    test_weights,
                ),
                "posterior_mean_game_margin_rmse": game_margin_rmse(
                    test_stints["game_id"],
                    actual,
                    posterior_prediction,
                    test_weights,
                ),
                "posterior_residual_standard_deviation": float(
                    np.sqrt(posterior.residual_variance_mean)
                ),
            }
        ]
    )


def _write_experiment(
    season: str,
    source_dir: Path,
    source_manifest: BaselineRunManifest,
    experiment: BayesianRapmExperiment,
    posterior_draws: int,
    posterior_seed: int,
    credible_interval_probability: float,
    model_artifacts_dir: Path,
) -> tuple[BayesianRapmRunManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"bayesian-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = model_artifacts_dir / "bayesian_rapm" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        parquet_outputs = {
            "posterior_rankings.parquet": experiment.posterior_rankings,
            "comparison_metrics.parquet": experiment.comparison_metrics,
            "predictive_calibration.parquet": experiment.predictive_calibration,
            "test_predictions.parquet": experiment.test_predictions,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        json_outputs = {
            "player_columns.json": {
                str(identifier): column
                for identifier, column in experiment.player_columns.items()
            },
            "model_parameters.json": experiment.model_parameters,
        }
        for filename, payload in json_outputs.items():
            (temporary_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        np.savez_compressed(
            temporary_dir / "posterior_state.npz",
            location=experiment.posterior.location,
            precision_cholesky=experiment.posterior.precision_cholesky,
            residual_quadratic=np.array(experiment.posterior.residual_quadratic),
            degrees_of_freedom=np.array(experiment.posterior.degrees_of_freedom),
            regularization=np.array(experiment.posterior.regularization),
            ridge_alpha=np.array(experiment.posterior.ridge_alpha),
            training_weight_mean=np.array(experiment.posterior.training_weight_mean),
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
        manifest = BayesianRapmRunManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            bayesian_code_version=bayesian_code_fingerprint(),
            source_model_run_id=source_manifest.run_id,
            source_model_manifest_sha256=_sha256_file(source_dir / "manifest.json"),
            dataset_part_sha256=source_manifest.dataset_part_sha256,
            selected_rapm_lambda=source_manifest.selected_rapm_lambda,
            posterior_draws=posterior_draws,
            posterior_seed=posterior_seed,
            credible_interval_probability=credible_interval_probability,
            minimum_ranking_possessions=source_manifest.minimum_ranking_possessions,
            stint_count=source_manifest.stint_count,
            game_count=source_manifest.game_count,
            player_count=source_manifest.player_count,
            final_train_game_count=source_manifest.final_train_game_count,
            final_test_game_count=source_manifest.final_test_game_count,
            artifacts=artifacts,
        )
        (temporary_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        temporary_dir.replace(run_dir)
        validate_bayesian_rapm_run(run_dir)
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
    player_columns: Mapping[int, int],
    ridge_rankings: pd.DataFrame,
    game_splits: pd.DataFrame,
    selected_lambda: float,
    posterior_draws: int,
    posterior_seed: int,
    credible_interval_probability: float,
) -> None:
    required_stints = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        "possessions",
        "target_home_net_rating",
    }
    missing_stints = required_stints - set(stints.columns)
    if missing_stints:
        raise ValueError(f"RAPM stints missing Bayesian columns: {sorted(missing_stints)}")
    required_rankings = {
        "player_id",
        "rapm",
        "eligible_rank",
        "exposure_eligible",
    }
    missing_rankings = required_rankings - set(ridge_rankings.columns)
    if missing_rankings:
        raise ValueError(
            f"Ridge rankings missing Bayesian columns: {sorted(missing_rankings)}"
        )
    if not {"split", "role", "game_id"} <= set(game_splits.columns):
        raise ValueError("Game splits are missing required final-split columns")
    columns = sorted(int(value) for value in player_columns.values())
    if columns != list(range(len(columns))):
        raise ValueError("Player columns must be unique and contiguous")
    if selected_lambda <= 0:
        raise ValueError("Bayesian RAPM requires a positive source ridge lambda")
    if posterior_draws < 1:
        raise ValueError("Posterior draws must be positive")
    if posterior_seed < 0:
        raise ValueError("Posterior seed must be non-negative")
    if not 0 < credible_interval_probability < 1:
        raise ValueError("Credible interval probability must be between zero and one")


def _player_ids_in_column_order(player_columns: Mapping[int, int]) -> tuple[int, ...]:
    return tuple(
        identifier
        for identifier, _ in sorted(
            ((int(identifier), int(column)) for identifier, column in player_columns.items()),
            key=lambda item: item[1],
        )
    )


def _resolve_source_run(
    season: str,
    model_artifacts_dir: Path,
    source_run_id: str | None,
) -> Path:
    season_dir = model_artifacts_dir / "rapm" / season
    if source_run_id is None:
        source_run_id = json.loads((season_dir / "latest.json").read_text())["run_id"]
    source_dir = season_dir / source_run_id
    if not source_dir.is_dir():
        raise ValueError(f"RAPM source run does not exist: {source_dir}")
    return source_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit exact conjugate Bayesian RAPM from a validated ridge run."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--source-run-id")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--model-artifacts-dir", default="artifacts/models")
    parser.add_argument("--posterior-draws", type=int, default=DEFAULT_POSTERIOR_DRAWS)
    parser.add_argument("--posterior-seed", type=int, default=DEFAULT_POSTERIOR_SEED)
    parser.add_argument(
        "--credible-interval",
        type=float,
        default=DEFAULT_CREDIBLE_INTERVAL,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, run_dir = train_bayesian_rapm(
        args.season,
        source_run_id=args.source_run_id,
        analytical_dir=args.analytical_dir,
        model_artifacts_dir=args.model_artifacts_dir,
        posterior_draws=args.posterior_draws,
        posterior_seed=args.posterior_seed,
        credible_interval_probability=args.credible_interval,
    )
    print(
        f"{manifest.season} Bayesian RAPM: "
        f"stints={manifest.stint_count}, players={manifest.player_count}, "
        f"lambda={manifest.selected_rapm_lambda:g}, draws={manifest.posterior_draws}; "
        f"run={run_dir}"
    )


if __name__ == "__main__":
    main()
