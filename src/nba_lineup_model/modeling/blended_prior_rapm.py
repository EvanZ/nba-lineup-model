"""RAPM centered on a chronologically selected blend of two frozen priors."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.aging import validate_aging_model_run
from nba_lineup_model.modeling.aging_prior_rapm import (
    _frozen_playoff_evaluation,
    _resolve_run,
    _sha256_file,
    aging_prior_frame,
)
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

DEFAULT_LAGGED_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def blended_prior_frame(
    aging_priors: pd.DataFrame,
    lagged_priors: pd.DataFrame,
    *,
    lagged_weight: float,
) -> pd.DataFrame:
    """Blend pre-season aging and lagged RAPM means, using zero for gaps."""

    if not 0.0 <= lagged_weight <= 1.0:
        raise ValueError("Lagged prior weight must be within [0, 1]")
    required = {"player_id", PRIOR_MEAN_COLUMN}
    for name, frame in (("aging", aging_priors), ("lagged", lagged_priors)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} prior table missing columns: {sorted(missing)}")
        if frame["player_id"].duplicated().any():
            raise ValueError(f"{name} prior table has duplicate player IDs")
    joined = aging_priors.rename(columns={PRIOR_MEAN_COLUMN: "aging_prior_mean"}).merge(
        lagged_priors.rename(columns={PRIOR_MEAN_COLUMN: "lagged_prior_mean"}),
        on="player_id",
        how="outer",
        validate="one_to_one",
    )
    joined["aging_prior_mean"] = joined["aging_prior_mean"].fillna(0.0)
    joined["lagged_prior_mean"] = joined["lagged_prior_mean"].fillna(0.0)
    joined[PRIOR_MEAN_COLUMN] = (
        lagged_weight * joined["lagged_prior_mean"]
        + (1.0 - lagged_weight) * joined["aging_prior_mean"]
    )
    return joined.sort_values("player_id", kind="stable").reset_index(drop=True)


def train_blended_prior_rapm(
    *,
    season: str = "2025-26",
    aging_run_id: str | None = None,
    lagged_run_id: str | None = None,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    lagged_weights: tuple[float, ...] = DEFAULT_LAGGED_WEIGHTS,
) -> Path:
    """Select a prior blend and ridge penalty on chronological target folds."""

    if len(lagged_weights) < 2 or len(lagged_weights) != len(set(lagged_weights)):
        raise ValueError("Provide at least two unique lagged weights")
    if any(not 0.0 <= weight <= 1.0 for weight in lagged_weights):
        raise ValueError("Lagged weights must be within [0, 1]")
    root = Path(artifacts_dir)
    aging_dir = _resolve_run(root / "aging" / season, aging_run_id)
    aging_manifest = validate_aging_model_run(aging_dir)
    lagged_dir = _resolve_run(root / "prior_rapm" / season, lagged_run_id)
    lagged_metadata = _validate_lagged_run(lagged_dir)
    aging_raw = pd.read_parquet(aging_dir / "player_priors.parquet")
    aging = aging_prior_frame(aging_raw, target_season=season)
    lagged = (
        pd.read_parquet(lagged_dir / "target_player_priors.parquet")
        .loc[:, ["player_id", "prior_rapm_mean"]]
        .rename(columns={"prior_rapm_mean": PRIOR_MEAN_COLUMN})
    )
    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    bios_path = Path(curated_dir) / "player_seasons" / season / "regular" / "part-00000.parquet"
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    candidates: list[tuple[float, object, pd.DataFrame]] = []
    selection_rows: list[pd.DataFrame] = []
    for weight in lagged_weights:
        prior = blended_prior_frame(aging, lagged, lagged_weight=weight)
        result = fit_prior_rapm_experiment(
            stints,
            prior.loc[:, ["player_id", PRIOR_MEAN_COLUMN]],
            lambda_grid=lambda_grid,
            player_bios=player_bios,
        )
        cv = result.cv_results.assign(lagged_weight=weight)
        candidates.append((weight, result, prior))
        selection_rows.append(cv)
    selection = pd.concat(selection_rows, ignore_index=True)
    summary = selection.groupby(["lagged_weight", "regularization"], as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = summary["squared_error_sum"] / summary["validation_possessions"]
    winner = summary.sort_values(
        ["weighted_mse", "lagged_weight", "regularization"], kind="stable"
    ).iloc[0]
    selected_weight = float(winner["lagged_weight"])
    selected_lambda = float(winner["regularization"])
    result, combined_prior = next(
        (result, prior)
        for weight, result, prior in candidates
        if weight == selected_weight and result.selected_lambda == selected_lambda
    )
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        vocabulary_mapping(player_ids),
        multiple=True,
    )
    prior_vector, _ = _prior_vector(player_ids, combined_prior)
    train_mask = np.isin(stints["game_id"].astype(str), result.split_plan.final_train_game_ids)
    frozen_model = PriorCenteredRidgeLineupModel(selected_lambda).fit(
        matrix[train_mask],
        stints.loc[train_mask, "target_home_net_rating"].to_numpy(dtype=float),
        stints.loc[train_mask, "possessions"].to_numpy(dtype=float),
        prior_vector,
    )
    playoff_metrics, playoff_predictions, playoff_hash = _frozen_playoff_evaluation(
        season=season,
        coefficients=frozen_model.coef_,
        player_ids=player_ids,
        intercept=frozen_model.intercept_,
        training_game_ids=result.split_plan.final_train_game_ids,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    return _write_run(
        season=season,
        result=result,
        selection=selection,
        combined_prior=combined_prior,
        aging_dir=aging_dir,
        aging_run_id=aging_manifest.run_id,
        lagged_dir=lagged_dir,
        lagged_run_id=str(lagged_metadata["run_id"]),
        lagged_weight=selected_weight,
        lambda_grid=lambda_grid,
        lagged_weights=lagged_weights,
        coefficients=pd.DataFrame({"player_id": player_ids, "rapm": frozen_model.coef_}),
        intercept=frozen_model.intercept_,
        playoff_metrics=playoff_metrics,
        playoff_predictions=playoff_predictions,
        playoff_hash=playoff_hash,
        artifacts_dir=root,
    )


def _write_run(**kwargs: object) -> Path:
    now = datetime.now(UTC)
    season = str(kwargs["season"])
    run_id = f"blended-prior-rapm-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(kwargs["artifacts_dir"]) / "blended_prior_rapm" / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    result = kwargs["result"]
    try:
        files = {
            "target_cv_results.parquet": result.cv_results,
            "blend_selection.parquet": kwargs["selection"],
            "holdout_metrics.parquet": result.test_metrics,
            "holdout_predictions.parquet": result.test_predictions,
            "player_rankings.parquet": result.player_rankings,
            "blended_player_priors.parquet": kwargs["combined_prior"],
            "final_training_player_coefficients.parquet": kwargs["coefficients"],
            "frozen_playoff_metrics.parquet": kwargs["playoff_metrics"],
            "frozen_playoff_predictions.parquet": kwargs["playoff_predictions"],
        }
        for filename, frame in files.items():
            frame.to_parquet(temporary / filename, index=False)
        split = result.split_plan
        pd.DataFrame(
            {
                "game_id": (*split.final_train_game_ids, *split.final_test_game_ids),
                "split": ["train"] * len(split.final_train_game_ids)
                + ["test"] * len(split.final_test_game_ids),
            }
        ).to_parquet(temporary / "target_game_splits.parquet", index=False)
        (temporary / "final_training_state.json").write_text(
            json.dumps(
                {
                    "intercept_home_net_rating": kwargs["intercept"],
                    "selected_lambda": result.selected_lambda,
                    "training_game_count": len(split.final_train_game_ids),
                },
                indent=2,
            )
            + "\n"
        )
        metadata = {
            "run_id": run_id,
            "model": "blended_aging_lagged_prior_centered_ridge_rapm",
            "season": season,
            "target_season": season,
            "aging_run_id": kwargs["aging_run_id"],
            "aging_manifest_sha256": _sha256_file(Path(kwargs["aging_dir"]) / "manifest.json"),
            "lagged_run_id": kwargs["lagged_run_id"],
            "lagged_manifest_sha256": _sha256_file(Path(kwargs["lagged_dir"]) / "manifest.json"),
            "selected_lagged_weight": kwargs["lagged_weight"],
            "selected_aging_weight": 1.0 - float(kwargs["lagged_weight"]),
            "lagged_weight_grid": list(kwargs["lagged_weights"]),
            "lambda_grid": list(kwargs["lambda_grid"]),
            "target_selected_lambda": result.selected_lambda,
            "target_final_train_games": len(split.final_train_game_ids),
            "target_final_test_games": len(split.final_test_game_ids),
            "frozen_playoff_segments_manifest_sha256": kwargs["playoff_hash"],
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts = [
            {"filename": path.name, "byte_count": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({"schema_version": 1, **metadata, "artifacts": artifacts}, indent=2) + "\n"
        )
        temporary.replace(output)
        validate_blended_prior_rapm_run(output)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def validate_blended_prior_rapm_run(run_dir: Path | str) -> dict[str, object]:
    """Validate a complete immutable blended-prior RAPM artifact."""

    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    expected = {str(item["filename"]): item for item in manifest["artifacts"]}
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    if set(expected) != actual:
        raise ValueError("Blended-prior RAPM files do not match its manifest")
    for filename, artifact in expected.items():
        path = root / filename
        if path.stat().st_size != int(artifact["byte_count"]):
            raise ValueError(f"Blended-prior RAPM artifact size changed: {filename}")
        if _sha256_file(path) != str(artifact["sha256"]):
            raise ValueError(f"Blended-prior RAPM artifact hash changed: {filename}")
    required = {
        "run_id",
        "model",
        "season",
        "target_season",
        "aging_run_id",
        "lagged_run_id",
        "selected_lagged_weight",
        "target_selected_lambda",
    }
    if required - set(metadata):
        raise ValueError("Blended-prior RAPM metadata is incomplete")
    if metadata["model"] != "blended_aging_lagged_prior_centered_ridge_rapm":
        raise ValueError("Blended-prior RAPM model identifier is invalid")
    if metadata["run_id"] != manifest["run_id"]:
        raise ValueError("Blended-prior RAPM run identifiers do not match")
    return metadata


def _validate_lagged_run(run_dir: Path) -> dict[str, object]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("model") != "forward_lagged_prior_centered_ridge_rapm":
        raise ValueError("Lagged-prior artifact has the wrong model type")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train blended aging and lagged prior RAPM")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--aging-run-id")
    parser.add_argument("--lagged-run-id")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    args = parser.parse_args()
    run_dir = train_blended_prior_rapm(
        season=args.season,
        aging_run_id=args.aging_run_id,
        lagged_run_id=args.lagged_run_id,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    suffix = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    print(f"Blended-prior RAPM: run={run_dir}{suffix}")


if __name__ == "__main__":
    main()
