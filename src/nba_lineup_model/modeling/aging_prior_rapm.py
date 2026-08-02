"""Prior-centered RAPM using a frozen forward aging-model prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.aging import validate_aging_model_run
from nba_lineup_model.modeling.neural_data import neural_possessions_frame, read_neural_possessions
from nba_lineup_model.modeling.prior_rapm import (
    PRIOR_MEAN_COLUMN,
    _prior_vector,
    fit_prior_rapm_experiment,
)
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.train import DEFAULT_LAMBDA_GRID
from nba_lineup_model.models.baselines import (
    PriorCenteredRidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)
from nba_lineup_model.season.compact import (
    read_curated_partition_manifest,
    validate_curated_partition,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition

AGING_PRIOR_MEAN_COLUMN = "aging_prior_mean"
_FORBIDDEN_AGING_PRIOR_COLUMNS = {
    "target_rapm",
    "target_rapm_possessions",
    "target_rapm_seconds",
    "target_rapm_exposure_eligible",
}


def aging_prior_frame(
    player_priors: pd.DataFrame,
    *,
    target_season: str,
) -> pd.DataFrame:
    """Validate a label-free aging prior table and project its RAPM mean."""

    required = {
        "target_season",
        "player_id",
        AGING_PRIOR_MEAN_COLUMN,
        "model_train_last_target_season",
    }
    missing = required - set(player_priors)
    if missing:
        raise ValueError(f"Aging prior table missing columns: {sorted(missing)}")
    if _FORBIDDEN_AGING_PRIOR_COLUMNS & set(player_priors):
        raise ValueError("Aging prior table contains target-season outcomes")
    if set(player_priors["target_season"].astype(str)) != {target_season}:
        raise ValueError("Aging prior table target season does not match RAPM target")
    if player_priors["player_id"].duplicated().any():
        raise ValueError("Aging prior table has duplicate player IDs")
    if not np.isfinite(player_priors[AGING_PRIOR_MEAN_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Aging prior means must be finite")
    if not player_priors["model_train_last_target_season"].astype(str).lt(target_season).all():
        raise ValueError("Aging prior uses target-season or future training outcomes")
    return (
        player_priors.loc[:, ["player_id", AGING_PRIOR_MEAN_COLUMN]]
        .rename(columns={AGING_PRIOR_MEAN_COLUMN: PRIOR_MEAN_COLUMN})
        .sort_values("player_id", kind="stable")
        .reset_index(drop=True)
    )


def train_aging_prior_rapm(
    *,
    season: str = "2025-26",
    aging_run_id: str | None = None,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> Path:
    """Fit a target-season RAPM centered on a pre-season aging-model forecast."""

    root = Path(artifacts_dir)
    aging_dir = _resolve_run(root / "aging" / season, aging_run_id)
    aging_manifest = validate_aging_model_run(aging_dir)
    published_priors = pd.read_parquet(aging_dir / "player_priors.parquet")
    priors = aging_prior_frame(published_priors, target_season=season)
    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    bios_path = Path(curated_dir) / "player_seasons" / season / "regular" / "part-00000.parquet"
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    experiment = fit_prior_rapm_experiment(
        stints,
        priors,
        lambda_grid=lambda_grid,
        player_bios=player_bios,
    )
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    player_columns = vocabulary_mapping(player_ids)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    prior, _ = _prior_vector(player_ids, priors)
    train_mask = np.isin(
        stints["game_id"].astype(str),
        experiment.split_plan.final_train_game_ids,
    )
    frozen_model = PriorCenteredRidgeLineupModel(experiment.selected_lambda).fit(
        matrix[train_mask],
        stints.loc[train_mask, "target_home_net_rating"].to_numpy(dtype=float),
        stints.loc[train_mask, "possessions"].to_numpy(dtype=float),
        prior,
    )
    playoff_metrics, playoff_predictions, playoff_source_hash = _frozen_playoff_evaluation(
        season=season,
        coefficients=frozen_model.coef_,
        player_ids=player_ids,
        intercept=frozen_model.intercept_,
        training_game_ids=experiment.split_plan.final_train_game_ids,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    return _write_run(
        season=season,
        stints=stints,
        experiment=experiment,
        published_priors=published_priors,
        aging_dir=aging_dir,
        aging_run_id=aging_manifest.run_id,
        lambda_grid=lambda_grid,
        frozen_coefficients=pd.DataFrame({"player_id": player_ids, "rapm": frozen_model.coef_}),
        frozen_intercept=frozen_model.intercept_,
        playoff_metrics=playoff_metrics,
        playoff_predictions=playoff_predictions,
        playoff_source_hash=playoff_source_hash,
        artifacts_dir=root,
    )


def validate_aging_prior_rapm_run(run_dir: Path | str) -> dict[str, object]:
    """Validate the immutable age-informed prior RAPM artifact contract."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    expected = {str(item["filename"]): item for item in manifest["artifacts"]}
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if set(expected) != actual:
        raise ValueError("Aging-prior RAPM files do not match its manifest")
    for filename, record in expected.items():
        path = root / filename
        if path.stat().st_size != int(record["byte_count"]):
            raise ValueError(f"Aging-prior RAPM artifact size changed: {filename}")
        if _sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Aging-prior RAPM artifact hash changed: {filename}")
    required = {
        "run_id",
        "model",
        "target_season",
        "aging_run_id",
        "aging_manifest_sha256",
        "target_selected_lambda",
        "target_final_train_games",
        "target_final_test_games",
    }
    if required - set(metadata):
        raise ValueError("Aging-prior RAPM metadata is incomplete")
    if metadata["model"] != "aging_prior_centered_ridge_rapm":
        raise ValueError("Aging-prior RAPM model identifier is invalid")
    if metadata["run_id"] != manifest["run_id"]:
        raise ValueError("Aging-prior RAPM run identifiers do not match")
    return metadata


def _write_run(
    *,
    season: str,
    stints: pd.DataFrame,
    experiment: object,
    published_priors: pd.DataFrame,
    aging_dir: Path,
    aging_run_id: str,
    lambda_grid: tuple[float, ...],
    frozen_coefficients: pd.DataFrame,
    frozen_intercept: float,
    playoff_metrics: pd.DataFrame,
    playoff_predictions: pd.DataFrame,
    playoff_source_hash: str,
    artifacts_dir: Path,
) -> Path:
    now = datetime.now(UTC)
    run_id = f"aging-prior-rapm-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    season_dir = artifacts_dir / "aging_prior_rapm" / season
    output_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        experiment.cv_results.to_parquet(temporary_dir / "target_cv_results.parquet", index=False)
        experiment.test_metrics.to_parquet(temporary_dir / "holdout_metrics.parquet", index=False)
        experiment.test_predictions.to_parquet(
            temporary_dir / "holdout_predictions.parquet",
            index=False,
        )
        experiment.player_rankings.to_parquet(
            temporary_dir / "player_rankings.parquet",
            index=False,
        )
        experiment.player_priors.to_parquet(
            temporary_dir / "target_player_priors.parquet",
            index=False,
        )
        published_priors.to_parquet(temporary_dir / "aging_player_priors.parquet", index=False)
        frozen_coefficients.to_parquet(
            temporary_dir / "final_training_player_coefficients.parquet", index=False
        )
        playoff_metrics.to_parquet(temporary_dir / "frozen_playoff_metrics.parquet", index=False)
        playoff_predictions.to_parquet(
            temporary_dir / "frozen_playoff_predictions.parquet", index=False
        )
        (temporary_dir / "final_training_state.json").write_text(
            json.dumps(
                {
                    "intercept_home_net_rating": frozen_intercept,
                    "selected_lambda": experiment.selected_lambda,
                    "training_game_count": len(experiment.split_plan.final_train_game_ids),
                },
                indent=2,
            )
            + "\n"
        )
        split = experiment.split_plan
        pd.DataFrame(
            {
                "game_id": (*split.final_train_game_ids, *split.final_test_game_ids),
                "split": ["train"] * len(split.final_train_game_ids)
                + ["test"] * len(split.final_test_game_ids),
            }
        ).to_parquet(temporary_dir / "target_game_splits.parquet", index=False)
        metadata = {
            "run_id": run_id,
            "model": "aging_prior_centered_ridge_rapm",
            "season": season,
            "target_season": season,
            "aging_run_id": aging_run_id,
            "aging_manifest_sha256": _sha256_file(aging_dir / "manifest.json"),
            "prior_mean_column": AGING_PRIOR_MEAN_COLUMN,
            "lambda_grid": list(lambda_grid),
            "target_selected_lambda": experiment.selected_lambda,
            "target_final_train_games": len(split.final_train_game_ids),
            "target_final_test_games": len(split.final_test_game_ids),
            "frozen_playoff_segments_manifest_sha256": playoff_source_hash,
            "created_at": now.isoformat(),
        }
        (temporary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary_dir.iterdir())
            if path.is_file()
        ]
        (temporary_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, **metadata, "artifacts": artifacts}, indent=2)
            + "\n"
        )
        temporary_dir.replace(output_dir)
        validate_aging_prior_rapm_run(output_dir)
        (season_dir / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return output_dir


def _resolve_run(season_dir: Path, run_id: str | None) -> Path:
    if run_id is None:
        run_id = str(json.loads((season_dir / "latest.json").read_text())["run_id"])
    run_dir = season_dir / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Model run does not exist: {run_dir}")
    return run_dir


def _frozen_playoff_evaluation(
    *,
    season: str,
    coefficients: np.ndarray,
    player_ids: tuple[int, ...],
    intercept: float,
    training_game_ids: tuple[str, ...],
    analytical_dir: Path | str,
    curated_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    regular = read_neural_possessions(season, analytical_dir=analytical_dir)
    regular_train = regular.loc[regular["game_id"].astype(str).isin(training_game_ids)]
    if regular_train["game_id"].nunique() != len(training_game_ids):
        raise ValueError("Frozen training possessions do not cover every training game")
    partition = CuratedPartition(
        table="possession_segments",
        season=season,
        season_type="playoffs",
    )
    playoff_manifest = read_curated_partition_manifest(partition, curated_dir)
    validate_curated_partition(playoff_manifest, curated_dir)
    playoff_segments = pd.read_parquet(
        CuratedDatasetLayout(Path(curated_dir)).partition_dir(partition)
    )
    possessions = neural_possessions_frame(playoff_segments)
    coefficient_map = dict(zip(player_ids, coefficients, strict=True))
    predictions, unknown_exposures = _translated_predictions(
        possessions,
        coefficient_map,
        intercept,
        float(regular_train["target_offense_margin"].mean()),
    )
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    training_mean = float(regular_train["target_offense_margin"].mean())
    mean_prediction = np.full(len(possessions), training_mean)
    mse = mean_squared_error(actual, predictions)
    game_rmse = possession_game_margin_rmse(
        possessions["game_id"],
        actual,
        predictions,
        possessions["home_offense_sign"],
    )
    mean_mse = mean_squared_error(actual, mean_prediction)
    mean_game_rmse = possession_game_margin_rmse(
        possessions["game_id"],
        actual,
        mean_prediction,
        possessions["home_offense_sign"],
    )
    metrics = pd.DataFrame(
        [
            {
                "cohort": "playoffs",
                "training_window": "first_1044_regular_season_games",
                "game_count": int(possessions["game_id"].nunique()),
                "possession_count": len(possessions),
                "possession_mse": mse,
                "possession_rmse": rmse(actual, predictions),
                "possession_mae": mean_absolute_error(actual, predictions),
                "eligible_possession_game_margin_rmse": game_rmse,
                "possession_skill_vs_mean": skill_score(mse, mean_mse),
                "game_margin_skill_vs_mean": skill_score(game_rmse**2, mean_game_rmse**2),
                "mean_reference_possession_rmse": float(np.sqrt(mean_mse)),
                "mean_reference_game_margin_rmse": mean_game_rmse,
                "unknown_player_exposures": unknown_exposures,
            }
        ]
    )
    output = possessions.loc[
        :,
        ["game_id", "possession_id", "home_offense_sign", "target_offense_margin"],
    ].copy()
    output["prediction_offense_margin"] = predictions
    output["prediction_home_margin"] = predictions * output["home_offense_sign"]
    return metrics, output, _sha256_file(
        CuratedDatasetLayout(Path(curated_dir)).partition_dir(partition) / "_manifest.json"
    )


def _translated_predictions(
    possessions: pd.DataFrame,
    coefficients: dict[int, float],
    intercept: float,
    training_mean: float,
) -> tuple[np.ndarray, int]:
    effects = np.empty(len(possessions), dtype=float)
    unknown_exposures = 0
    for index, (offense, defense) in enumerate(
        zip(possessions["offense_player_ids"], possessions["defense_player_ids"], strict=True)
    ):
        unknown_exposures += sum(int(player_id) not in coefficients for player_id in offense)
        unknown_exposures += sum(int(player_id) not in coefficients for player_id in defense)
        effects[index] = sum(coefficients.get(int(player_id), 0.0) for player_id in offense) - sum(
            coefficients.get(int(player_id), 0.0) for player_id in defense
        )
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    return training_mean + effects / 200.0 + signs * intercept / 200.0, unknown_exposures


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RAPM with a frozen aging-model prior")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--aging-run-id")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = train_aging_prior_rapm(
        season=args.season,
        aging_run_id=args.aging_run_id,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    suffix = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    print(f"Aging-prior RAPM: run={run_dir}{suffix}")


if __name__ == "__main__":
    main()
