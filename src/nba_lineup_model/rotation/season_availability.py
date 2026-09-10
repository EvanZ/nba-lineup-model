"""Strictly preseason binomial forecasts of full-season player availability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_PANEL_PATH,
)
from nba_lineup_model.rotation.l20_source_aware_preseason_minute_share import (
    SourceAwareConfig,
    SourceAwareSeasonInputs,
    build_l20_source_aware_season_inputs,
    build_source_aware_features,
)

DEFAULT_SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2026))
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/season_availability")
DEFAULT_ROLE_SHRINKAGE_GAMES_GRID = (1.0, 5.0, 15.0)
DEFAULT_GAP_DECAY_GRID = (0.25, 0.5, 0.75)
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0)

_FEATURE_COLUMNS = (
    "log_total_minutes",
    "shrunken_mpg",
    "availability_rate",
    "shrunken_start_rate",
    "preseason_nail",
    "is_gap_returner",
    "gap_seasons",
    "is_cold_start",
    "cold_draft_capital",
    "cold_undrafted",
    "cold_age_offset",
    "cold_guard",
    "cold_forward",
    "cold_center",
)


@dataclass(frozen=True)
class AvailabilityConfig:
    """Preseason evidence attenuation and ridge strength."""

    role_shrinkage_games: float = 1.0
    gap_decay: float = 0.75
    alpha: float = 0.1


@dataclass(frozen=True)
class FittedAvailabilityModel:
    """A ridge-binomial model for a player's full-season game appearances."""

    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    config: AvailabilityConfig


@dataclass(frozen=True)
class SeasonAvailabilityRun:
    """Artifact identity for a frozen full-season availability replay."""

    run_dir: Path
    run_id: str


def build_season_availability_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
) -> SourceAwareSeasonInputs:
    """Build the existing strictly preseason role state for an availability target."""

    return build_l20_source_aware_season_inputs(season, panel=panel)


def fit_availability_model(
    inputs: list[SourceAwareSeasonInputs],
    *,
    config: AvailabilityConfig | None = None,
) -> FittedAvailabilityModel:
    """Fit player-season appearance counts with a binomial likelihood.

    A player's target remains their total regular-season GP across every team.
    Consequently, a trade changes neither the target definition nor the number
    of observed player appearances.
    """

    if not inputs:
        raise ValueError("Availability fit requires at least one source season")
    config = config or AvailabilityConfig()
    x_raw, appearances, scheduled_games = _training_rows(inputs, config=config)
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (x_raw - mean) / scale
    penalty = float(config.alpha)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, intercept = parameters[:-1], float(parameters[-1])
        logits = x @ coefficients + intercept
        probabilities = expit(logits)
        value = float(
            -np.sum(appearances * logits - scheduled_games * np.logaddexp(0.0, logits))
            + 0.5 * penalty * np.dot(coefficients, coefficients)
        )
        residual = scheduled_games * probabilities - appearances
        gradient = np.empty_like(parameters)
        gradient[:-1] = x.T @ residual + penalty * coefficients
        gradient[-1] = residual.sum()
        return value, gradient

    result = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=np.zeros(x.shape[1] + 1, dtype=float),
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Availability fit failed: {result.message}")
    return FittedAvailabilityModel(
        feature_columns=_FEATURE_COLUMNS,
        feature_mean=mean,
        feature_scale=scale,
        coefficients=np.asarray(result.x[:-1], dtype=float),
        intercept=float(result.x[-1]),
        config=config,
    )


def predict_availability(
    role_state: pd.DataFrame,
    *,
    model: FittedAvailabilityModel,
) -> pd.DataFrame:
    """Return per-player appearance probabilities from preseason evidence."""

    features = build_source_aware_features(
        role_state,
        config=SourceAwareConfig(
            model.config.role_shrinkage_games,
            model.config.gap_decay,
            model.config.alpha,
        ),
    )
    values = features.loc[:, list(model.feature_columns)].to_numpy(dtype=float)
    standardized = (values - model.feature_mean) / model.feature_scale
    output = features.loc[:, ["team_id", "team", "player_id", "player_name", "player_state"]].copy()
    output["predicted_availability"] = expit(standardized @ model.coefficients + model.intercept)
    return output


def evaluate_availability(
    inputs: SourceAwareSeasonInputs,
    *,
    model: FittedAvailabilityModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate full-season GP predictions for the preseason opening roster."""

    predictions = predict_availability(inputs.role_state, model=model)
    appearances, scheduled_games = _season_appearance_counts(inputs.target_game_minutes)
    output = predictions.copy()
    output["actual_games"] = output["player_id"].map(appearances).fillna(0.0)
    output["scheduled_games"] = float(scheduled_games)
    output["actual_availability"] = output["actual_games"] / float(scheduled_games)
    output["absolute_error"] = np.abs(
        output["actual_availability"] - output["predicted_availability"]
    )
    output["squared_error"] = np.square(
        output["actual_availability"] - output["predicted_availability"]
    )
    probability = output["predicted_availability"].clip(1e-12, 1.0 - 1e-12)
    games = output["actual_games"]
    n = float(scheduled_games)
    log_loss_per_game = -float(
        (games * np.log(probability) + (n - games) * np.log(1.0 - probability)).sum()
    ) / (n * len(output))
    metrics = pd.DataFrame(
        [
            {
                "season": inputs.season,
                "scheduled_games": scheduled_games,
                "player_count": len(output),
                "availability_mae": float(output["absolute_error"].mean()),
                "availability_rmse": float(np.sqrt(output["squared_error"].mean())),
                "binomial_log_loss_per_game": log_loss_per_game,
            }
        ]
    )
    return output, metrics


def select_availability_parameters(
    source_inputs: list[SourceAwareSeasonInputs],
    *,
    role_shrinkage_games_grid: tuple[float, ...] = DEFAULT_ROLE_SHRINKAGE_GAMES_GRID,
    gap_decay_grid: tuple[float, ...] = DEFAULT_GAP_DECAY_GRID,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> tuple[AvailabilityConfig, pd.DataFrame]:
    """Select availability parameters with strictly earlier season outcomes."""

    if not source_inputs:
        raise ValueError("Availability selection requires source seasons")
    rows: list[dict[str, float | int]] = []
    for q in sorted(set(role_shrinkage_games_grid)):
        for decay in sorted(set(gap_decay_grid)):
            for alpha in sorted(set(alpha_grid)):
                config = AvailabilityConfig(q, decay, alpha)
                validation: list[pd.DataFrame] = []
                if len(source_inputs) == 1:
                    model = fit_availability_model(source_inputs, config=config)
                    validation.append(evaluate_availability(source_inputs[0], model=model)[1])
                else:
                    for index in range(1, len(source_inputs)):
                        model = fit_availability_model(source_inputs[:index], config=config)
                        validation.append(
                            evaluate_availability(source_inputs[index], model=model)[1]
                        )
                metrics = pd.concat(validation, ignore_index=True)
                rows.append(
                    {
                        "role_shrinkage_games": q,
                        "gap_decay": decay,
                        "alpha": alpha,
                        "validation_season_count": len(validation),
                        "availability_mae": float(metrics["availability_mae"].mean()),
                        "availability_rmse": float(metrics["availability_rmse"].mean()),
                        "binomial_log_loss_per_game": float(
                            metrics["binomial_log_loss_per_game"].mean()
                        ),
                    }
                )
    grid = pd.DataFrame(rows).sort_values(
        ["binomial_log_loss_per_game", "availability_rmse", "availability_mae"],
        kind="stable",
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    return (
        AvailabilityConfig(
            float(winner["role_shrinkage_games"]),
            float(winner["gap_decay"]),
            float(winner["alpha"]),
        ),
        grid,
    )


def run_season_availability_frozen_evaluation(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> SeasonAvailabilityRun:
    """Run a rolling full-season GP forecast for every eligible holdout."""

    ordered_seasons = tuple(sorted(set(seasons)))
    if len(ordered_seasons) < 2:
        raise ValueError("Frozen availability evaluation requires at least two seasons")
    panel = pd.read_parquet(panel_path)
    print(f"[availability-v0.1] loading {len(ordered_seasons)} preseason states", flush=True)
    inputs = [build_season_availability_inputs(season, panel=panel) for season in ordered_seasons]
    run_id = f"season-availability-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    output_dir = Path(artifacts_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    grids: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for index, holdout in enumerate(inputs[1:], start=1):
        source_inputs = inputs[:index]
        print(
            f"[availability-v0.1] tuning {holdout.season} from {source_inputs[0].season} "
            f"through {source_inputs[-1].season}",
            flush=True,
        )
        config, grid = select_availability_parameters(source_inputs)
        model = fit_availability_model(source_inputs, config=config)
        fold_predictions, fold_metrics = evaluate_availability(holdout, model=model)
        fold_predictions["holdout_season"] = holdout.season
        fold_metrics["holdout_season"] = holdout.season
        grid["holdout_season"] = holdout.season
        grids.append(grid)
        predictions.append(fold_predictions)
        metrics.append(fold_metrics)
        selections.append(
            {
                "holdout_season": holdout.season,
                "source_first_season": source_inputs[0].season,
                "source_last_season": source_inputs[-1].season,
                "source_season_count": len(source_inputs),
                "role_shrinkage_games": config.role_shrinkage_games,
                "gap_decay": config.gap_decay,
                "alpha": config.alpha,
            }
        )
        print(
            f"[availability-v0.1] selected q={config.role_shrinkage_games:.0f}, "
            f"gap={config.gap_decay:.2f}, alpha={config.alpha:.2f} for {holdout.season}",
            flush=True,
        )
    pd.concat(grids, ignore_index=True).to_parquet(
        output_dir / "parameter_grid.parquet", index=False
    )
    pd.DataFrame(selections).to_parquet(output_dir / "selected_parameters.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        output_dir / "predictions.parquet", index=False
    )
    pd.concat(metrics, ignore_index=True).to_parquet(output_dir / "metrics.parquet", index=False)
    return SeasonAvailabilityRun(run_dir=output_dir, run_id=run_id)


def _training_rows(
    inputs: list[SourceAwareSeasonInputs], *, config: AvailabilityConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    appearance_rows: list[np.ndarray] = []
    scheduled_rows: list[np.ndarray] = []
    source_config = SourceAwareConfig(
        config.role_shrinkage_games, config.gap_decay, config.alpha
    )
    for season_inputs in inputs:
        features = build_source_aware_features(season_inputs.role_state, config=source_config)
        appearances, scheduled_games = _season_appearance_counts(season_inputs.target_game_minutes)
        feature_rows.append(features.loc[:, list(_FEATURE_COLUMNS)].to_numpy(dtype=float))
        appearance_rows.append(
            features["player_id"].map(appearances).fillna(0.0).to_numpy(dtype=float)
        )
        scheduled_rows.append(np.full(len(features), float(scheduled_games)))
    return (
        np.vstack(feature_rows),
        np.concatenate(appearance_rows),
        np.concatenate(scheduled_rows),
    )


def _season_appearance_counts(game_minutes: pd.DataFrame) -> tuple[pd.Series, int]:
    required = {"team_id", "game_id", "player_id"}
    missing = sorted(required - set(game_minutes))
    if missing:
        raise ValueError(f"Game minutes lacks availability columns: {missing}")
    scheduled_by_team = game_minutes.groupby("team_id")["game_id"].nunique()
    scheduled_games = int(scheduled_by_team.max())
    if scheduled_games <= 0:
        raise ValueError("Availability target has no scheduled games")
    appearances = game_minutes.groupby("player_id")["game_id"].nunique().astype(float)
    return appearances, scheduled_games
