"""Evaluate a complete hard-switch box-score and cold-start RAPM prior."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.aging_prior_rapm import _frozen_playoff_evaluation
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

DEFAULT_BOX_SCORE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def combined_prior_frame(
    box_predictions: pd.DataFrame,
    cold_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one complete, label-free prior table from disjoint forecasts."""

    box_required = {"player_id", "prediction_box_score"}
    cold_required = {"player_id", "prediction_cold_start"}
    if box_required - set(box_predictions) or cold_required - set(cold_predictions):
        raise ValueError("Component forecasts do not contain their prediction columns")
    box = box_predictions.loc[:, ["player_id", "prediction_box_score"]].rename(
        columns={"prediction_box_score": PRIOR_MEAN_COLUMN}
    )
    cold = cold_predictions.loc[:, ["player_id", "prediction_cold_start"]].rename(
        columns={"prediction_cold_start": PRIOR_MEAN_COLUMN}
    )
    if box["player_id"].duplicated().any() or cold["player_id"].duplicated().any():
        raise ValueError("Component forecasts contain duplicate players")
    if set(box["player_id"]) & set(cold["player_id"]):
        raise ValueError("Returning and cold-start forecasts must be disjoint")
    combined = pd.concat(
        [box.assign(prior_branch="box_score"), cold.assign(prior_branch="cold_start")],
        ignore_index=True,
    )
    if not np.isfinite(combined[PRIOR_MEAN_COLUMN].to_numpy(float)).all():
        raise ValueError("Combined prior means must be finite")
    return combined.sort_values("player_id", kind="stable").reset_index(drop=True)


def blend_with_lagged_prior(
    box_cold_prior: pd.DataFrame, lagged_prior: pd.DataFrame, *, box_score_weight: float
) -> pd.DataFrame:
    """Blend the complete box/cold table with the frozen lagged RAPM prior."""
    if not 0.0 <= box_score_weight <= 1.0:
        raise ValueError("Box-score weight must lie in [0, 1]")
    lagged = lagged_prior.loc[:, ["player_id", "prior_rapm_mean"]].rename(
        columns={"prior_rapm_mean": "lagged_prior"}
    )
    joined = box_cold_prior.merge(lagged, on="player_id", how="outer", validate="one_to_one")
    joined[PRIOR_MEAN_COLUMN] = (
        box_score_weight * joined[PRIOR_MEAN_COLUMN].fillna(0.0)
        + (1.0 - box_score_weight) * joined["lagged_prior"].fillna(0.0)
    )
    return joined.drop(columns="lagged_prior").sort_values("player_id", kind="stable")


def train_combined_box_score_prior_rapm(
    *,
    season: str = "2025-26",
    artifacts_dir: Path | str = Path("artifacts/models"),
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    box_score_weights: tuple[float, ...] = DEFAULT_BOX_SCORE_WEIGHTS,
) -> Path:
    """Fit prior-centered RAPM using frozen component forecasts."""

    root = Path(artifacts_dir)
    box_dir = _latest_run(root / "box_score_prior" / season)
    cold_dir = _latest_run(root / "cold_start_prior" / season)
    box_cold = combined_prior_frame(
        pd.read_parquet(box_dir / "holdout_predictions.parquet"),
        pd.read_parquet(cold_dir / "holdout_predictions.parquet"),
    )
    lagged_dir = _latest_run(root / "prior_rapm" / season)
    lagged = pd.read_parquet(lagged_dir / "target_player_priors.parquet")
    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    bios_path = Path(curated_dir) / "player_seasons" / season / "regular" / "part-00000.parquet"
    bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    candidates = []
    selection = []
    for weight in box_score_weights:
        priors = blend_with_lagged_prior(box_cold, lagged, box_score_weight=weight)
        result = fit_prior_rapm_experiment(
            stints, priors.loc[:, ["player_id", PRIOR_MEAN_COLUMN]],
            lambda_grid=lambda_grid, player_bios=bios,
        )
        candidates.append((weight, result, priors))
        selection.append(result.cv_results.assign(box_score_weight=weight))
    selection_frame = pd.concat(selection, ignore_index=True)
    summary = selection_frame.groupby(["box_score_weight", "regularization"], as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    selected = summary.sort_values(
        ["weighted_mse", "box_score_weight", "regularization"], kind="stable"
    ).iloc[0]
    weight = float(selected["box_score_weight"])
    result, priors = next(
        (result, priors) for candidate_weight, result, priors in candidates
        if candidate_weight == weight
        and result.selected_lambda == float(selected["regularization"])
    )
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    matrix = signed_entity_matrix(stints, "home_player_ids", "away_player_ids",
                                  vocabulary_mapping(player_ids), multiple=True)
    prior, _ = _prior_vector(player_ids, priors)
    train_mask = np.isin(stints["game_id"].astype(str), result.split_plan.final_train_game_ids)
    frozen = PriorCenteredRidgeLineupModel(result.selected_lambda).fit(
        matrix[train_mask], stints.loc[train_mask, "target_home_net_rating"].to_numpy(float),
        stints.loc[train_mask, "possessions"].to_numpy(float), prior,
    )
    playoff_metrics, playoff_predictions, playoff_hash = _frozen_playoff_evaluation(
        season=season,
        coefficients=frozen.coef_,
        player_ids=player_ids,
        intercept=frozen.intercept_,
        training_game_ids=result.split_plan.final_train_game_ids,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    return _write_run(
        root, season, result, priors, box_dir, cold_dir, lagged_dir, weight,
        selection_frame, summary, frozen, playoff_metrics, playoff_predictions, playoff_hash,
    )


def _write_run(root, season, result, priors, box_dir, cold_dir, lagged_dir, weight, selection,
               summary, frozen, playoff_metrics, playoff_predictions, playoff_hash) -> Path:
    now = datetime.now(UTC)
    run_id = f"combined-box-score-prior-rapm-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    parent = root / "combined_box_score_prior_rapm" / season
    output, temporary = parent / run_id, parent / f".{run_id}.tmp"
    temporary.mkdir(parents=True)
    try:
        files = {
            "target_cv_results.parquet": result.cv_results,
            "holdout_metrics.parquet": result.test_metrics,
            "holdout_predictions.parquet": result.test_predictions,
            "player_rankings.parquet": result.player_rankings,
            "combined_player_priors.parquet": priors,
            "blend_selection.parquet": selection,
            "blend_summary.parquet": summary,
            "final_training_player_coefficients.parquet": pd.DataFrame(
                {"player_id": result.player_ids, "rapm": frozen.coef_}
            ),
            "frozen_playoff_metrics.parquet": playoff_metrics,
            "frozen_playoff_predictions.parquet": playoff_predictions,
        }
        for filename, frame in files.items():
            frame.to_parquet(temporary / filename, index=False)
        metadata = {
            "run_id": run_id, "season": season,
            "model": "combined_box_score_and_cold_start_prior_centered_ridge_rapm",
            "box_score_run_id": json.loads((box_dir / "manifest.json").read_text())["run_id"],
            "cold_start_run_id": json.loads((cold_dir / "manifest.json").read_text())["run_id"],
            "lagged_run_id": json.loads((lagged_dir / "manifest.json").read_text())["run_id"],
            "selected_box_score_weight": weight,
            "prior_branch_counts": priors.groupby("prior_branch").size().to_dict(),
            "target_selected_lambda": result.selected_lambda,
            "target_final_train_games": len(result.split_plan.final_train_game_ids),
            "target_final_test_games": len(result.split_plan.final_test_game_ids),
            "frozen_playoff_segments_manifest_sha256": playoff_hash,
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {"filename": path.name, "byte_count": path.stat().st_size}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        manifest = {**metadata, "artifacts": artifacts}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
        (parent / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        return output
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _latest_run(parent: Path) -> Path:
    return parent / json.loads((parent / "latest.json").read_text())["run_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train combined box-score/cold-start prior RAPM.")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--box-score-weights", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()
    weights = tuple(float(value) for value in args.box_score_weights.split(","))
    run_dir = train_combined_box_score_prior_rapm(
        season=args.season, artifacts_dir=args.artifacts_dir, analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        box_score_weights=weights,
    )
    from nba_lineup_model.tracking import track_completed_run
    tracking = track_completed_run(run_dir)
    print(f"Combined box-score prior RAPM run={run_dir}; mlflow_run_id={tracking.mlflow_run_id}")


if __name__ == "__main__":
    main()
