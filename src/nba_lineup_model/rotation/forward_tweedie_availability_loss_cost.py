"""Forward Tweedie model for annual player availability loss cost.

The candidate predicts unavailable rostered player-games directly. It uses an
exposure-shrunk, strictly forward player-history loss state as its sole player
feature; age and workload are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    DEFAULT_FROZEN_SEASONS,
    DEFAULT_INITIAL_PRIOR_STRENGTH_GRID,
    DEFAULT_PERSISTENCE_GRID,
    DEFAULT_PLAYER_PANEL_PATH,
    DEFAULT_PRIOR_STRENGTH_GRID,
    DEFAULT_TUNING_SEASONS,
    ForwardAvailabilityConfig,
    build_availability_season_summary,
    predict_availability_season,
    summarize_availability_metrics,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/forward_tweedie_availability_loss_cost")
DEFAULT_TWEEDIE_POWER_GRID = (1.1, 1.3, 1.5, 1.7, 1.9, 1.99)
PRODUCTION_AVAILABILITY_CONFIG = ForwardAvailabilityConfig(0.5, 60.0, 15.0, -0.25)
_EPSILON = 1e-8


@dataclass(frozen=True)
class LossStateConfig:
    """Hyperparameters for the forward, no-age/no-workload loss state."""

    persistence: float
    prior_strength: float
    initial_prior_strength: float


@dataclass(frozen=True)
class TweedieLossCostModel:
    """Fitted log-link Tweedie mean model with an exposure offset."""

    intercept: float
    state_weight: float
    power: float


@dataclass(frozen=True)
class TweedieAvailabilityLossCostRun:
    """Immutable artifact location for a Tweedie loss-cost candidate replay."""

    run_dir: Path
    run_id: str


def build_forward_loss_state_panel(
    summary: pd.DataFrame,
    *,
    config: LossStateConfig,
) -> pd.DataFrame:
    """Return strict-forward loss-state features for every predictable season."""

    seasons = (
        summary.loc[:, ["season", "season_start_year"]]
        .drop_duplicates()
        .sort_values("season_start_year", kind="stable")
    )
    first_year = int(seasons["season_start_year"].min())
    rows: list[pd.DataFrame] = []
    availability_config = _as_availability_config(config)
    for season, year in seasons.itertuples(index=False):
        if int(year) == first_year:
            continue
        prediction, _baseline, _metadata = predict_availability_season(
            summary,
            target_season=str(season),
            config=availability_config,
        )
        output = prediction.copy()
        output["unavailable_games"] = (
            output["known_player_games"] - output["available_games"]
        )
        output["unavailable_share"] = 1.0 - output["available_share"]
        output["loss_state_logit"] = logit(
            np.clip(1.0 - output["predicted_available_share"], _EPSILON, 1.0 - _EPSILON)
        )
        rows.append(output)
    if not rows:
        raise ValueError("Loss-state panel requires at least two observed seasons")
    return pd.concat(rows, ignore_index=True)


def fit_tweedie_loss_cost_model(
    training: pd.DataFrame,
    *,
    power: float,
) -> TweedieLossCostModel:
    """Fit a log-link Tweedie mean with log rostered-game exposure offset."""

    if not 1.0 < power < 2.0:
        raise ValueError("Tweedie power must be strictly between 1 and 2")
    required = {"unavailable_games", "known_player_games", "loss_state_logit"}
    missing = sorted(required - set(training))
    if missing:
        raise ValueError(f"Tweedie training rows missing columns: {missing}")
    if training.empty:
        raise ValueError("Tweedie training requires at least one player-season")

    losses = training["unavailable_games"].to_numpy(dtype=float)
    exposure = training["known_player_games"].to_numpy(dtype=float)
    state = training["loss_state_logit"].to_numpy(dtype=float)
    if np.any(exposure <= 0.0):
        raise ValueError("Tweedie exposure must be positive")
    design = np.column_stack((np.ones(len(training)), state))
    offset = np.log(exposure)
    population_rate = np.clip(losses.sum() / exposure.sum(), _EPSILON, 1.0)
    initial = np.array((np.log(population_rate), 0.0))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(offset + design @ coefficients, -30.0, 30.0)
        mean = np.exp(linear)
        deviance = _tweedie_deviance(losses, mean, power=power)
        gradient_linear = 2.0 * (
            np.power(mean, 2.0 - power) - losses * np.power(mean, 1.0 - power)
        )
        return float(deviance.sum()), design.T @ gradient_linear

    result = minimize(
        fun=lambda values: objective(values)[0],
        x0=initial,
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Tweedie loss-cost fit failed: {result.message}")
    return TweedieLossCostModel(
        intercept=float(result.x[0]),
        state_weight=float(result.x[1]),
        power=float(power),
    )


def predict_tweedie_loss_cost(
    state_panel: pd.DataFrame,
    *,
    target_season: str,
    power: float,
) -> tuple[pd.DataFrame, TweedieLossCostModel]:
    """Fit on previous state rows and score one completed target season."""

    target_year = int(target_season[:4])
    training = state_panel.loc[state_panel["season_start_year"].lt(target_year)].copy()
    target = state_panel.loc[state_panel["season"].eq(target_season)].copy()
    if target.empty:
        raise ValueError(f"Loss-state panel has no target rows for {target_season}")
    model = fit_tweedie_loss_cost_model(training, power=power)
    exposure = target["known_player_games"].to_numpy(dtype=float)
    state = target["loss_state_logit"].to_numpy(dtype=float)
    raw_cost = np.exp(
        np.clip(
            np.log(exposure) + model.intercept + model.state_weight * state,
            -30.0,
            30.0,
        )
    )
    output = target.copy()
    output["predicted_unavailable_games_raw"] = raw_cost
    output["predicted_unavailable_games"] = np.clip(raw_cost, 0.0, exposure)
    output["loss_cost_capped_at_schedule"] = raw_cost > exposure
    output["predicted_unavailable_share"] = output["predicted_unavailable_games"] / exposure
    output["predicted_available_share"] = 1.0 - output["predicted_unavailable_share"]
    return output, model


def tune_loss_state_config(
    summary: pd.DataFrame,
    *,
    target_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    persistence_grid: tuple[float, ...] = DEFAULT_PERSISTENCE_GRID,
    prior_strength_grid: tuple[float, ...] = DEFAULT_PRIOR_STRENGTH_GRID,
    initial_prior_strength_grid: tuple[float, ...] = DEFAULT_INITIAL_PRIOR_STRENGTH_GRID,
) -> tuple[LossStateConfig, pd.DataFrame]:
    """Select player-history state parameters before fitting the Tweedie layer."""

    rows: list[dict[str, float]] = []
    for persistence, prior_strength, initial_prior_strength in product(
        sorted(set(persistence_grid)),
        sorted(set(prior_strength_grid)),
        sorted(set(initial_prior_strength_grid)),
    ):
        config = LossStateConfig(
            persistence=float(persistence),
            prior_strength=float(prior_strength),
            initial_prior_strength=float(initial_prior_strength),
        )
        availability_config = _as_availability_config(config)
        predictions = [
            predict_availability_season(
                summary,
                target_season=season,
                config=availability_config,
            )[0]
            for season in target_seasons
        ]
        rows.append(
            {
                **asdict(config),
                **summarize_availability_metrics(pd.concat(predictions, ignore_index=True)),
            }
        )
    grid = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "weighted_brier",
                "binomial_log_loss",
                "persistence",
                "prior_strength",
                "initial_prior_strength",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    winner = grid.iloc[0]
    return (
        LossStateConfig(
            persistence=float(winner["persistence"]),
            prior_strength=float(winner["prior_strength"]),
            initial_prior_strength=float(winner["initial_prior_strength"]),
        ),
        grid,
    )


def tune_tweedie_power(
    state_panel: pd.DataFrame,
    *,
    target_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    power_grid: tuple[float, ...] = DEFAULT_TWEEDIE_POWER_GRID,
) -> tuple[float, pd.DataFrame]:
    """Select Tweedie power on the same pre-frozen target seasons."""

    rows: list[dict[str, float]] = []
    for power in sorted(set(power_grid)):
        predictions = [
            predict_tweedie_loss_cost(
                state_panel,
                target_season=season,
                power=float(power),
            )[0]
            for season in target_seasons
        ]
        rows.append(
            {
                "power": float(power),
                **summarize_tweedie_loss_cost_metrics(pd.concat(predictions, ignore_index=True)),
            }
        )
    grid = (
        pd.DataFrame(rows)
        .sort_values(["weighted_brier", "binomial_log_loss", "power"], kind="stable")
        .reset_index(drop=True)
    )
    return float(grid.iloc[0]["power"]), grid


def summarize_tweedie_loss_cost_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Report common availability metrics and direct annual loss-cost error."""

    required = {"unavailable_games", "predicted_unavailable_games"}
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"Tweedie predictions missing columns: {missing}")
    output = summarize_availability_metrics(predictions)
    actual = predictions["unavailable_games"].to_numpy(dtype=float)
    predicted = predictions["predicted_unavailable_games"].to_numpy(dtype=float)
    output.update(
        {
            "loss_game_mae": float(np.abs(actual - predicted).mean()),
            "loss_game_rmse": float(np.sqrt(np.square(actual - predicted).mean())),
            "mean_actual_unavailable_games": float(actual.mean()),
            "mean_predicted_unavailable_games": float(predicted.mean()),
            "schedule_cap_rate": float(predictions["loss_cost_capped_at_schedule"].mean()),
        }
    )
    return output


def run_tweedie_availability_loss_cost(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
) -> TweedieAvailabilityLossCostRun:
    """Tune and evaluate the direct annual availability loss-cost candidate."""

    final_year = max(int(season[:4]) for season in (*tuning_seasons, *frozen_seasons))
    seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year + 1))
    summary = build_availability_season_summary(
        seasons,
        curated_dir=curated_dir,
        player_panel_path=player_panel_path,
    )
    state_config, state_grid = tune_loss_state_config(
        summary,
        target_seasons=tuning_seasons,
    )
    state_panel = build_forward_loss_state_panel(summary, config=state_config)
    power, power_grid = tune_tweedie_power(state_panel, target_seasons=tuning_seasons)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"tweedie-availability-loss-cost-{timestamp}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(run_dir / "player_season_summary.parquet", index=False)
    state_panel.to_parquet(run_dir / "forward_loss_state_panel.parquet", index=False)
    state_grid.to_parquet(run_dir / "state_tuning_grid.parquet", index=False)
    power_grid.to_parquet(run_dir / "tweedie_power_tuning_grid.parquet", index=False)

    comparison_rows: list[dict[str, float | str]] = []
    candidate_predictions: list[pd.DataFrame] = []
    production_predictions: list[pd.DataFrame] = []
    for season in frozen_seasons:
        candidate, model = predict_tweedie_loss_cost(
            state_panel,
            target_season=season,
            power=power,
        )
        candidate.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        candidate_predictions.append(candidate)
        comparison_rows.append(
            {
                "season": season,
                "model": "Tweedie availability loss cost v0.1",
                "tweedie_intercept": model.intercept,
                "tweedie_state_weight": model.state_weight,
                "tweedie_power": model.power,
                **summarize_tweedie_loss_cost_metrics(candidate),
            }
        )

        production, _baseline, _metadata = predict_availability_season(
            summary,
            target_season=season,
            config=PRODUCTION_AVAILABILITY_CONFIG,
        )
        production = _attach_loss_columns(production)
        production.to_parquet(run_dir / f"{season}_production_v02_predictions.parquet", index=False)
        production_predictions.append(production)
        comparison_rows.append(
            {
                "season": season,
                "model": "Forward availability v0.2",
                **summarize_tweedie_loss_cost_metrics(production),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["season", "weighted_brier"], kind="stable"
    )
    comparison.to_parquet(run_dir / "frozen_model_comparison.parquet", index=False)
    comparison.to_csv(run_dir / "frozen_model_comparison.csv", index=False)
    pooled = pd.DataFrame(
        [
            {
                "model": "Tweedie availability loss cost v0.1",
                **summarize_tweedie_loss_cost_metrics(
                    pd.concat(candidate_predictions, ignore_index=True)
                ),
            },
            {
                "model": "Forward availability v0.2",
                **summarize_tweedie_loss_cost_metrics(
                    pd.concat(production_predictions, ignore_index=True)
                ),
            },
        ]
    ).sort_values("weighted_brier", kind="stable")
    pooled.to_parquet(run_dir / "pooled_model_comparison.parquet", index=False)
    pooled.to_csv(run_dir / "pooled_model_comparison.csv", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "forward_tweedie_availability_loss_cost",
                "version": "v0.1",
                "run_id": run_id,
                "selected_loss_state_config": asdict(state_config),
                "selected_tweedie_power": power,
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "target": "annual unavailable rostered player-games",
                "contract": (
                    "forward player-history loss state enters a log-link Tweedie mean; "
                    "log known rostered games is the exposure offset; age and workload excluded"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return TweedieAvailabilityLossCostRun(run_dir=run_dir, run_id=run_id)


def _as_availability_config(config: LossStateConfig) -> ForwardAvailabilityConfig:
    return ForwardAvailabilityConfig(
        persistence=config.persistence,
        prior_strength=config.prior_strength,
        initial_prior_strength=config.initial_prior_strength,
        workload_weight=0.0,
        use_age_baseline=False,
    )


def _attach_loss_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["unavailable_games"] = output["known_player_games"] - output["available_games"]
    output["unavailable_share"] = 1.0 - output["available_share"]
    output["predicted_unavailable_games_raw"] = (
        output["known_player_games"] * (1.0 - output["predicted_available_share"])
    )
    output["predicted_unavailable_games"] = output["predicted_unavailable_games_raw"]
    output["loss_cost_capped_at_schedule"] = False
    return output


def _tweedie_deviance(
    losses: np.ndarray,
    mean: np.ndarray,
    *,
    power: float,
) -> np.ndarray:
    return 2.0 * (
        np.power(losses, 2.0 - power) / ((1.0 - power) * (2.0 - power))
        - losses * np.power(mean, 1.0 - power) / (1.0 - power)
        + np.power(mean, 2.0 - power) / (2.0 - power)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tweedie availability loss-cost candidate")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_tweedie_availability_loss_cost(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
