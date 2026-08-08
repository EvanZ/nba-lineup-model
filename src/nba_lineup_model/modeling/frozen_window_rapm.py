"""Frozen no-prior RAPM baselines fit on one or three completed seasons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.evaluation.metrics import mean_squared_error
from nba_lineup_model.modeling.frozen_prior_evaluation import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_SEASON,
    FrozenEvaluation,
    _game_prediction_frame,
    _historical_team_seasons,
    _read_playoff_possessions,
    _read_regular_possessions,
    _regular_stint_predictions,
    _team_net_rating_metrics,
    _team_win_evaluation,
    _write_run,
    fit_pythagorean_win_model,
    score_frozen_possessions,
    score_possession_cohort,
)
from nba_lineup_model.modeling.neural_data import read_neural_possessions
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.train import (
    DEFAULT_LAMBDA_GRID,
    GameSplitPlan,
    chronological_game_splits,
)
from nba_lineup_model.models.baselines import (
    RidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)
from nba_lineup_model.season.schema import validate_season


@dataclass(frozen=True)
class FrozenWindowRapmFit:
    """Completed-season no-prior RAPM fit used as a frozen player vector."""

    training_seasons: tuple[str, ...]
    selected_lambda: float
    home_court_intercept: float
    player_coefficients: pd.DataFrame
    cv_results: pd.DataFrame
    training_summary: pd.DataFrame


def fit_frozen_window_rapm(
    stints_by_season: dict[str, pd.DataFrame],
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> FrozenWindowRapmFit:
    """Select lambda chronologically and refit one coefficient across all windows.

    The fit deliberately supplies no player-prior vector. A player has one
    coefficient for the entire completed-season window; players absent from the
    window are handled as zero-valued cold starts only when scoring the target.
    """

    if not stints_by_season:
        raise ValueError("Frozen window RAPM requires at least one training season")
    if not lambda_grid or len(lambda_grid) != len(set(lambda_grid)):
        raise ValueError("Lambda grid must contain unique values")
    if any(value < 0 for value in lambda_grid):
        raise ValueError("Lambda values must be non-negative")

    training_seasons = tuple(sorted(stints_by_season, key=lambda value: int(value[:4])))
    stints = pd.concat(
        [stints_by_season[season].assign(training_season=season) for season in training_seasons],
        ignore_index=True,
    )
    _validate_stints(stints)
    split_plan = chronological_game_splits(stints, _window_split_config())
    player_ids = entity_vocabulary(stints, "home_player_ids", "away_player_ids", multiple=True)
    player_columns = vocabulary_mapping(player_ids)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    game_ids = stints["game_id"].astype(str).to_numpy()
    cv_results = _cross_validate_window(
        matrix,
        target,
        weights,
        game_ids,
        split_plan,
        lambda_grid,
    )
    selected_lambda = _select_lambda(cv_results)
    fitted = RidgeLineupModel(selected_lambda).fit(matrix, target, weights)
    coefficients = pd.DataFrame(
        {"player_id": player_ids, "prior_rapm_mean": fitted.coef_}
    ).sort_values("player_id", kind="stable").reset_index(drop=True)
    coefficients["prior_available"] = True
    coefficients["training_season_count"] = len(training_seasons)
    summary = (
        stints.groupby("training_season", as_index=False)
        .agg(
            game_count=("game_id", "nunique"),
            stint_count=("game_id", "size"),
            possessions=("possessions", "sum"),
        )
        .sort_values("training_season", kind="stable")
        .reset_index(drop=True)
    )
    return FrozenWindowRapmFit(
        training_seasons=training_seasons,
        selected_lambda=selected_lambda,
        home_court_intercept=float(fitted.intercept_),
        player_coefficients=coefficients,
        cv_results=cv_results,
        training_summary=summary,
    )


def train_frozen_window_rapm(
    *,
    season: str = DEFAULT_SEASON,
    training_season_count: int,
    analytical_dir: Path | str = Path("data/analytical"),
    curated_dir: Path | str = Path("data/curated"),
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> tuple[dict[str, object], Path]:
    """Train and evaluate a frozen one- or three-year no-prior RAPM baseline."""

    target_season = validate_season(season)
    if training_season_count not in {1, 3}:
        raise ValueError("Frozen window RAPM currently supports one or three seasons")
    training_seasons = _training_seasons(target_season, training_season_count)
    analytical_root = Path(analytical_dir)
    curated_root = Path(curated_dir)
    artifact_root = Path(artifacts_dir)
    stints_by_season = {
        season_name: read_rapm_stints(season_name, analytical_root)
        for season_name in training_seasons
    }
    fit = fit_frozen_window_rapm(stints_by_season, lambda_grid=lambda_grid)
    source_root, source_metadata = _write_window_fit(
        fit,
        target_season=target_season,
        analytical_dir=analytical_root,
        artifacts_dir=artifact_root,
    )
    evaluation = _evaluate_frozen_window(
        fit,
        source_metadata=source_metadata,
        target_season=target_season,
        analytical_dir=analytical_root,
        curated_dir=curated_root,
    )
    model = (
        "frozen_one_year_rapm_no_priors"
        if training_season_count == 1
        else "frozen_three_year_rapm_no_priors"
    )
    prefix = "frozen-one-year-rapm" if training_season_count == 1 else "frozen-three-year-rapm"
    manifest, evaluation_root = _write_run(
        evaluation,
        prior_root=source_root,
        analytical_dir=analytical_root,
        curated_dir=curated_root,
        artifacts_dir=artifact_root,
        run_prefix=prefix,
        manifest_model=model,
    )
    return manifest.model_dump(mode="json"), evaluation_root


def _evaluate_frozen_window(
    fit: FrozenWindowRapmFit,
    *,
    source_metadata: dict[str, object],
    target_season: str,
    analytical_dir: Path,
    curated_dir: Path,
) -> FrozenEvaluation:
    source_season = _previous_season(target_season)
    source_possessions, source_manifest = _read_regular_possessions(
        source_season,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    regular_possessions = read_neural_possessions(target_season, analytical_dir=analytical_dir)
    playoff_possessions, _ = _read_playoff_possessions(target_season, curated_dir)
    priors = _target_player_priors(
        fit.player_coefficients,
        _player_ids_from_possessions([regular_possessions, playoff_possessions]),
        target_season=target_season,
        source_season=source_season,
    )
    source_mean = float(source_possessions["target_offense_margin"].mean())
    source_state: dict[str, object] = {
        "target_season": target_season,
        "source_season": source_season,
        "prior_run_id": source_metadata["run_id"],
        "player_prior_method": (
            f"pooled {len(fit.training_seasons)}-season regular-only ridge RAPM "
            "with no player prior"
        ),
        "cold_start_prior": 0.0,
        "training_seasons": list(fit.training_seasons),
        "training_season_count": len(fit.training_seasons),
        "selected_lambda": fit.selected_lambda,
        "source_offense_margin_mean": source_mean,
        "source_home_intercept_net_rating": fit.home_court_intercept,
        "source_possessions_manifest_sha256": _sha256_file(source_manifest),
        "window_fit_manifest_sha256": source_metadata["manifest_sha256"],
        "possession_translation": (
            "completed-source-season offense mean + pooled no-prior RAPM effect/200 "
            "+ pooled-window home-court intercept/200"
        ),
        "target_season_refit": False,
        "target_regular_outcomes_used_for_fit": False,
        "target_playoff_outcomes_used_for_fit": False,
        "oracle_information": "realized target-season lineups and exposure only",
    }
    regular_predictions = score_frozen_possessions(
        regular_possessions,
        priors,
        source_mean=source_mean,
        source_home_intercept=fit.home_court_intercept,
        cohort="regular_season",
    )
    playoff_predictions = score_frozen_possessions(
        playoff_possessions,
        priors,
        source_mean=source_mean,
        source_home_intercept=fit.home_court_intercept,
        cohort="playoffs",
    )
    possession_predictions = pd.concat(
        [regular_predictions, playoff_predictions],
        ignore_index=True,
    )
    model_name = f"frozen_{len(fit.training_seasons)}_year_rapm_no_priors"
    cohort_metrics = pd.concat(
        [
            score_possession_cohort(regular_predictions, source_mean=source_mean, model=model_name),
            score_possession_cohort(playoff_predictions, source_mean=source_mean, model=model_name),
        ],
        ignore_index=True,
    )
    game_predictions = _game_prediction_frame(possession_predictions)
    target_stints = read_rapm_stints(target_season, analytical_dir=analytical_dir)
    regular_game_predictions, team_net_rating_predictions = _regular_stint_predictions(
        target_stints,
        priors,
        source_home_intercept=fit.home_court_intercept,
    )
    team_net_rating_metrics = _team_net_rating_metrics(
        team_net_rating_predictions,
        model=model_name,
    )
    calibration = _historical_team_seasons(
        analytical_dir=analytical_dir,
        through_season=source_season,
    )
    pythagorean = fit_pythagorean_win_model(calibration)
    source_state["pythagorean_win_model"] = {
        "form": "weighted_linear_win_percentage_from_net_rating",
        "intercept": pythagorean.intercept,
        "net_rating_slope": pythagorean.net_rating_slope,
        "wins_per_net_rating_point_82_games": 82.0 * pythagorean.net_rating_slope,
        "training_first_season": pythagorean.training_seasons[0],
        "training_last_season": pythagorean.training_seasons[-1],
        "training_season_count": len(pythagorean.training_seasons),
        "training_team_season_count": pythagorean.training_team_season_count,
        "historical_win_total_rmse": pythagorean.historical_win_total_rmse,
    }
    team_win_predictions, team_win_metrics = _team_win_evaluation(
        regular_game_predictions,
        team_net_rating_predictions,
        pythagorean,
        model=model_name,
    )
    return FrozenEvaluation(
        source_state=source_state,
        frozen_player_priors=priors,
        cohort_metrics=cohort_metrics,
        possession_predictions=possession_predictions,
        game_predictions=game_predictions,
        regular_game_predictions=regular_game_predictions,
        team_net_rating_predictions=team_net_rating_predictions,
        team_net_rating_metrics=team_net_rating_metrics,
        team_win_predictions=team_win_predictions,
        team_win_metrics=team_win_metrics,
        pythagorean_calibration_team_seasons=calibration,
    )


def _cross_validate_window(
    matrix: object,
    target: np.ndarray,
    weights: np.ndarray,
    game_ids: np.ndarray,
    split_plan: GameSplitPlan,
    lambda_grid: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold in split_plan.folds:
        train = np.isin(game_ids, fold.train_game_ids)
        validation = np.isin(game_ids, fold.validation_game_ids)
        for regularization in lambda_grid:
            model = RidgeLineupModel(regularization).fit(
                matrix[train],
                target[train],
                weights[train],
            )
            prediction = model.predict(matrix[validation])
            mse = mean_squared_error(target[validation], prediction, weights[validation])
            rows.append(
                {
                    "fold": fold.fold,
                    "regularization": float(regularization),
                    "validation_game_count": len(fold.validation_game_ids),
                    "validation_possessions": float(weights[validation].sum()),
                    "squared_error_sum": float(mse * weights[validation].sum()),
                    "weighted_mse": mse,
                }
            )
    return pd.DataFrame(rows).sort_values(["regularization", "fold"], kind="stable").reset_index(
        drop=True
    )


def _select_lambda(cv_results: pd.DataFrame) -> float:
    summary = cv_results.groupby("regularization", as_index=False).agg(
        squared_error_sum=("squared_error_sum", "sum"),
        validation_possessions=("validation_possessions", "sum"),
    )
    summary["weighted_mse"] = (
        summary["squared_error_sum"] / summary["validation_possessions"]
    )
    selected = summary.sort_values(
        ["weighted_mse", "regularization"],
        kind="stable",
    ).iloc[0]
    return float(selected["regularization"])


def _target_player_priors(
    coefficients: pd.DataFrame,
    target_player_ids: Iterable[int],
    *,
    target_season: str,
    source_season: str,
) -> pd.DataFrame:
    coefficient_map = coefficients.set_index("player_id")["prior_rapm_mean"]
    ids = sorted({int(player_id) for player_id in target_player_ids})
    output = pd.DataFrame({"player_id": ids})
    output["prior_rapm_mean"] = output["player_id"].map(coefficient_map).fillna(0.0)
    output["prior_available"] = output["player_id"].isin(coefficient_map.index)
    output["prior_branch"] = np.where(
        output["prior_available"], "window_rapm", "zero_cold_start"
    )
    output["target_season"] = target_season
    output["source_season"] = source_season
    return output


def _player_ids_from_possessions(frames: Iterable[pd.DataFrame]) -> set[int]:
    player_ids: set[int] = set()
    for frame in frames:
        for column in ("offense_player_ids", "defense_player_ids"):
            for lineup in frame[column]:
                player_ids.update(int(player_id) for player_id in lineup)
    return player_ids


def _training_seasons(target_season: str, count: int) -> tuple[str, ...]:
    target_start = int(target_season[:4])
    return tuple(
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(target_start - count, target_start)
    )


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _window_split_config():
    from nba_lineup_model.modeling.schema import ChronologicalSplitConfig

    return ChronologicalSplitConfig()


def _validate_stints(stints: pd.DataFrame) -> None:
    required = {
        "game_id",
        "game_date",
        "game_time_utc",
        "home_player_ids",
        "away_player_ids",
        "possessions",
        "target_home_net_rating",
    }
    missing = required - set(stints)
    if missing:
        raise ValueError(f"Frozen window RAPM stints missing columns: {sorted(missing)}")


def _write_window_fit(
    fit: FrozenWindowRapmFit,
    *,
    target_season: str,
    analytical_dir: Path,
    artifacts_dir: Path,
) -> tuple[Path, dict[str, object]]:
    window_name = "one_year" if len(fit.training_seasons) == 1 else "three_year"
    now = datetime.now(UTC)
    run_id = f"frozen-{window_name}-rapm-{target_season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = artifacts_dir / f"frozen_{window_name}_rapm" / target_season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tables = {
            "player_coefficients.parquet": fit.player_coefficients,
            "cv_results.parquet": fit.cv_results,
            "training_summary.parquet": fit.training_summary,
        }
        for filename, frame in tables.items():
            frame.to_parquet(temporary / filename, index=False)
        metadata: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "frozen_window_ridge_rapm_no_priors",
            "target_season": target_season,
            "training_seasons": list(fit.training_seasons),
            "training_season_count": len(fit.training_seasons),
            "season_type": "regular",
            "selected_lambda": fit.selected_lambda,
            "home_court_intercept": fit.home_court_intercept,
            "player_prior": "none; zero-centered ridge coefficients",
            "cold_start_prior": 0.0,
            "target_regular_outcomes_used_for_fit": False,
            "target_playoff_outcomes_used_for_fit": False,
            "created_at": now.isoformat(),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        artifacts = [
            {
                "filename": path.name,
                "row_count": len(tables[path.name]) if path.name in tables else None,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        manifest = {**metadata, "artifacts": artifacts}
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(output)
        latest = root / "latest.json"
        latest_tmp = latest.with_suffix(".json.tmp")
        latest_tmp.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        latest_tmp.replace(latest)
        return output, {**metadata, "manifest_sha256": _sha256_file(output / "manifest.json")}
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


def _build_parser(training_season_count: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate frozen {training_season_count}-year no-prior RAPM"
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser


def _main(training_season_count: int) -> None:
    args = _build_parser(training_season_count).parse_args()
    manifest, run_dir = train_frozen_window_rapm(
        season=args.season,
        training_season_count=training_season_count,
        analytical_dir=args.analytical_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking else ""
    print(
        f"Frozen {training_season_count}-year no-prior RAPM: season={manifest['season']}, "
        f"regular_games={manifest['regular_game_count']}, "
        f"playoff_games={manifest['playoff_game_count']}; run={run_dir}{tracking_text}"
    )


def one_year_main() -> None:
    """Run the 2024-25-only frozen RAPM baseline."""

    _main(1)


def three_year_main() -> None:
    """Run the 2022-23 through 2024-25 frozen RAPM baseline."""

    _main(3)
