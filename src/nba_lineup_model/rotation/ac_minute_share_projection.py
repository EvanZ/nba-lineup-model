"""Availability-conditioned, game-level regulation-minute allocations.

AC-MSP v0.1 answers a deliberately narrow question: given the players who were
medically available for a historical team-game, how should 240 regulation
minutes be distributed among them? It does not forecast the availability mask;
that remains the separate Forward Availability model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/availability_conditioned_minutes")
DEFAULT_HISTORY_START_SEASON = "2015-16"
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_ALPHA_GRID = (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
REGULATION_TEAM_MINUTES = 240.0
MODEL_NAME = "ac_msp"
MODEL_VERSION = "v0.1"
FEATURE_COLUMNS = (
    "log_lag_1_minutes",
    "log_lag_5_mean_minutes",
    "log_season_to_date_mean_minutes",
)


@dataclass(frozen=True)
class ACMSPConfig:
    """Ridge penalty for roster-relative fractional-logit regression."""

    alpha: float


@dataclass(frozen=True)
class FittedACMSPModel:
    """Standardized player score used inside each available-roster softmax."""

    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    config: ACMSPConfig


@dataclass(frozen=True)
class ACMSPRun:
    """Immutable location and identity for one AC-MSP frozen replay."""

    run_dir: Path
    run_id: str


def load_regulation_available_minutes(
    season: str,
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> pd.DataFrame:
    """Load known, medically available player rows for exact-240 team-games.

    Overtime games and malformed/non-regulation records are excluded rather
    than rescaled. Available coach DNPs and G-League assignments remain in the
    candidate roster with zero actual minutes.
    """

    path = Path(curated_dir) / "player_availability" / season / "part-00000.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing availability mart for {season}: {path}")
    source = pd.read_parquet(path)
    required = {
        "season",
        "game_id",
        "game_date",
        "team_id",
        "team",
        "player_id",
        "player_name",
        "minutes",
        "available",
        "availability_state_known",
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"Availability mart lacks required columns for {season}: {missing}")
    if source.duplicated(["game_id", "team_id", "player_id"]).any():
        raise ValueError(f"Duplicate player-game rows in {season}")

    source = source.copy()
    source["minutes"] = pd.to_numeric(source["minutes"], errors="raise")
    totals = source.groupby(["game_id", "team_id"], as_index=False, sort=False).agg(
        team_minutes=("minutes", "sum")
    )
    known_rosters = source.groupby(["game_id", "team_id"], as_index=False, sort=False).agg(
        complete_availability_roster=("availability_state_known", "all")
    )
    regulation = totals.loc[
        np.isclose(totals["team_minutes"], REGULATION_TEAM_MINUTES, atol=0.01),
        ["game_id", "team_id"],
    ]
    eligible = regulation.merge(
        known_rosters.loc[known_rosters["complete_availability_roster"]],
        on=["game_id", "team_id"],
        how="inner",
    )
    output = source.merge(eligible, on=["game_id", "team_id"], how="inner")
    output = output.loc[output["available"].astype(bool)].copy()
    if output.empty:
        raise ValueError(f"No regulation available-player rows in {season}")
    candidate_totals = output.groupby(["game_id", "team_id"], as_index=False).agg(
        allocated_minutes=("minutes", "sum")
    )
    if not np.isclose(
        candidate_totals["allocated_minutes"], REGULATION_TEAM_MINUTES, atol=0.01
    ).all():
        raise ValueError(f"Available-player minutes do not sum to 240 in {season}")
    output["actual_minute_share"] = output["minutes"] / REGULATION_TEAM_MINUTES
    return output.sort_values(
        ["team_id", "game_date", "game_id", "player_id"], kind="stable"
    ).reset_index(drop=True)


def build_conditional_minute_panel(available_minutes: pd.DataFrame) -> pd.DataFrame:
    """Create strict-pregame team-history features for every available candidate."""

    required = {
        "season",
        "game_id",
        "game_date",
        "team_id",
        "team",
        "player_id",
        "player_name",
        "minutes",
        "actual_minute_share",
    }
    missing = sorted(required - set(available_minutes))
    if missing:
        raise ValueError(f"Available minutes lacks required columns: {missing}")
    if not available_minutes["actual_minute_share"].between(0.0, 1.0).all():
        raise ValueError("Minute-share targets must lie in [0, 1]")

    rows: list[dict[str, object]] = []
    ordered = available_minutes.sort_values(
        ["team_id", "game_date", "game_id", "player_id"], kind="stable"
    )
    for team_id, team_rows in ordered.groupby("team_id", sort=True):
        history: list[dict[int, float]] = []
        cumulative_minutes: dict[int, float] = {}
        for game_id, game in team_rows.groupby("game_id", sort=False):
            latest = history[-1] if history else {}
            recent = history[-5:]
            game_count = len(history)
            candidates = game.sort_values("player_id", kind="stable")
            for row in candidates.itertuples(index=False):
                player_id = int(row.player_id)
                lag_1 = float(latest.get(player_id, 0.0))
                lag_5 = (
                    float(sum(previous.get(player_id, 0.0) for previous in recent) / len(recent))
                    if recent
                    else 0.0
                )
                season_mean = (
                    float(cumulative_minutes.get(player_id, 0.0) / game_count)
                    if game_count
                    else 0.0
                )
                rows.append(
                    {
                        "season": str(row.season),
                        "game_id": str(game_id),
                        "game_date": row.game_date,
                        "team_id": int(team_id),
                        "team": str(row.team),
                        "player_id": player_id,
                        "player_name": str(row.player_name),
                        "minutes": float(row.minutes),
                        "actual_minute_share": float(row.actual_minute_share),
                        "candidate_count": len(candidates),
                        "team_games_completed": game_count,
                        "lag_1_minutes": lag_1,
                        "lag_5_mean_minutes": lag_5,
                        "season_to_date_mean_minutes": season_mean,
                    }
                )
            minute_map = {
                int(row.player_id): float(row.minutes) for row in candidates.itertuples(index=False)
            }
            history.append(minute_map)
            for player_id, minutes in minute_map.items():
                cumulative_minutes[player_id] = cumulative_minutes.get(player_id, 0.0) + minutes
    output = pd.DataFrame.from_records(rows)
    output["log_lag_1_minutes"] = np.log1p(output["lag_1_minutes"])
    output["log_lag_5_mean_minutes"] = np.log1p(output["lag_5_mean_minutes"])
    output["log_season_to_date_mean_minutes"] = np.log1p(output["season_to_date_mean_minutes"])
    _validate_panel(output)
    return output


def fit_ac_msp(panel: pd.DataFrame, *, config: ACMSPConfig) -> FittedACMSPModel:
    """Fit ridge-regularized softmax allocation over known available rosters."""

    _validate_panel(panel)
    feature_values = panel.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    feature_mean = feature_values.mean(axis=0)
    feature_scale = feature_values.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    x = (feature_values - feature_mean) / feature_scale
    target = panel["actual_minute_share"].to_numpy(dtype=float)
    groups = _team_game_codes(panel)
    group_count = int(groups.max()) + 1

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ coefficients
        probabilities, log_normalizer = _group_softmax(logits, groups, group_count)
        value = float(
            -(target * (logits - log_normalizer[groups])).sum()
            + 0.5 * config.alpha * np.dot(coefficients, coefficients)
        )
        return value, x.T @ (probabilities - target) + config.alpha * coefficients

    result = minimize(
        fun=lambda values: objective(values)[0],
        x0=np.zeros(x.shape[1], dtype=float),
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"AC-MSP fit failed: {result.message}")
    return FittedACMSPModel(
        feature_columns=FEATURE_COLUMNS,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=np.asarray(result.x, dtype=float),
        config=config,
    )


def predict_ac_msp(panel: pd.DataFrame, *, model: FittedACMSPModel) -> pd.DataFrame:
    """Predict a positive 240-minute allocation for every available candidate."""

    _validate_panel(panel)
    values = panel.loc[:, list(model.feature_columns)].to_numpy(dtype=float)
    standardized = (values - model.feature_mean) / model.feature_scale
    groups = _team_game_codes(panel)
    predicted, _ = _group_softmax(standardized @ model.coefficients, groups, int(groups.max()) + 1)
    output = panel.copy()
    output["predicted_minute_share"] = predicted
    output["predicted_minutes"] = REGULATION_TEAM_MINUTES * predicted
    output["absolute_error"] = np.abs(output["actual_minute_share"] - predicted)
    return output


def predict_conditional_l5_control(panel: pd.DataFrame) -> pd.DataFrame:
    """Allocate current available-roster shares from lag-five minutes plus one."""

    _validate_panel(panel)
    output = panel.copy()
    weights = output["lag_5_mean_minutes"].to_numpy(dtype=float) + 1.0
    groups = _team_game_codes(output)
    denominators = np.bincount(groups, weights=weights)
    output["predicted_minute_share"] = weights / denominators[groups]
    output["predicted_minutes"] = REGULATION_TEAM_MINUTES * output["predicted_minute_share"]
    output["absolute_error"] = np.abs(
        output["actual_minute_share"] - output["predicted_minute_share"]
    )
    return output


def summarize_allocation_metrics(
    predictions: pd.DataFrame, *, model: str, season: str
) -> pd.DataFrame:
    """Summarize allocation metrics on an identical observed-availability support."""

    required = {"game_id", "team_id", "player_id", "actual_minute_share", "predicted_minute_share"}
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"Predictions lack required columns: {missing}")
    work = predictions.copy()
    work["absolute_error"] = np.abs(work["actual_minute_share"] - work["predicted_minute_share"])
    work["squared_error"] = np.square(work["actual_minute_share"] - work["predicted_minute_share"])
    work["cross_entropy_term"] = -work["actual_minute_share"] * np.log(
        work["predicted_minute_share"].clip(1e-12, None)
    )
    team_games = work.groupby(["game_id", "team_id"], as_index=False).agg(
        allocation_total_variation=("absolute_error", lambda values: 0.5 * float(values.sum())),
        brier_score=("squared_error", "sum"),
        cross_entropy=("cross_entropy_term", "sum"),
        candidate_count=("player_id", "size"),
    )
    return pd.DataFrame(
        [
            {
                "model": model,
                "season": season,
                "evaluated_team_games": int(len(team_games)),
                "mean_allocation_total_variation": float(
                    team_games["allocation_total_variation"].mean()
                ),
                "median_allocation_total_variation": float(
                    team_games["allocation_total_variation"].median()
                ),
                "mean_brier_score": float(team_games["brier_score"].mean()),
                "mean_cross_entropy": float(team_games["cross_entropy"].mean()),
                "player_share_mae": float(work["absolute_error"].mean()),
                "player_share_rmse": float(np.sqrt(work["squared_error"].mean())),
                "mean_available_candidates": float(team_games["candidate_count"].mean()),
            }
        ]
    )


def tune_ac_msp(
    panels: dict[str, pd.DataFrame],
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> tuple[ACMSPConfig, pd.DataFrame]:
    """Choose ridge strength only from completed, pre-frozen target seasons."""

    rows: list[dict[str, float | int]] = []
    for alpha in sorted(set(alpha_grid)):
        season_metrics: list[pd.DataFrame] = []
        for target in tuning_seasons:
            model = fit_ac_msp(
                _history_panel(panels, target), config=ACMSPConfig(alpha=float(alpha))
            )
            predictions = predict_ac_msp(panels[target], model=model)
            season_metrics.append(
                summarize_allocation_metrics(predictions, model=MODEL_NAME, season=target)
            )
        metrics = pd.concat(season_metrics, ignore_index=True)
        rows.append(
            {
                "alpha": float(alpha),
                "validation_season_count": len(season_metrics),
                "mean_allocation_total_variation": float(
                    metrics["mean_allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(metrics["mean_brier_score"].mean()),
                "mean_cross_entropy": float(metrics["mean_cross_entropy"].mean()),
                "player_share_mae": float(metrics["player_share_mae"].mean()),
                "player_share_rmse": float(metrics["player_share_rmse"].mean()),
            }
        )
    grid = (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_allocation_total_variation", "mean_brier_score", "mean_cross_entropy", "alpha"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return ACMSPConfig(alpha=float(grid.loc[0, "alpha"])), grid


def run_ac_msp(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    history_start_season: str = DEFAULT_HISTORY_START_SEASON,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> ACMSPRun:
    """Tune AC-MSP and run its full-regular-season frozen replay."""

    target_seasons = (*tuning_seasons, *frozen_seasons)
    all_seasons = _season_range(history_start_season, max(target_seasons, key=_season_year))
    panels: dict[str, pd.DataFrame] = {}
    print(f"AC-MSP: building conditional panels for {len(all_seasons)} seasons", flush=True)
    for index, season in enumerate(all_seasons, start=1):
        available_minutes = load_regulation_available_minutes(season, curated_dir=curated_dir)
        panels[season] = build_conditional_minute_panel(available_minutes)
        team_games = available_minutes.loc[:, ["game_id", "team_id"]].drop_duplicates()
        print(
            f"AC-MSP: panel {index}/{len(all_seasons)} {season} "
            f"({len(team_games)} eligible team-games)",
            flush=True,
        )
    print("AC-MSP: tuning ridge penalty on pre-frozen seasons", flush=True)
    config, tuning_grid = tune_ac_msp(panels, tuning_seasons=tuning_seasons, alpha_grid=alpha_grid)
    print(f"AC-MSP: selected alpha={config.alpha:g}", flush=True)
    run_id = f"ac-msp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tuning_grid.to_parquet(run_dir / "tuning_grid.parquet", index=False)

    frozen_metrics: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []
    for index, target in enumerate(frozen_seasons, start=1):
        print(f"AC-MSP: frozen season {index}/{len(frozen_seasons)} {target}", flush=True)
        model = fit_ac_msp(_history_panel(panels, target), config=config)
        fitted = predict_ac_msp(panels[target], model=model)
        control = predict_conditional_l5_control(panels[target])
        fitted.to_parquet(run_dir / f"{target}_predictions.parquet", index=False)
        control.to_parquet(run_dir / f"{target}_l5_control_predictions.parquet", index=False)
        frozen_metrics.extend(
            [
                summarize_allocation_metrics(fitted, model="AC-MSP v0.1", season=target),
                summarize_allocation_metrics(
                    control, model="Availability-conditioned L5 control", season=target
                ),
            ]
        )
        coefficient_rows.append(
            pd.DataFrame(
                {
                    "season": target,
                    "feature": model.feature_columns,
                    "standardized_coefficient": model.coefficients,
                    "alpha": model.config.alpha,
                }
            )
        )
    metrics = pd.concat(frozen_metrics, ignore_index=True)
    metrics.to_parquet(run_dir / "frozen_metrics.parquet", index=False)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            frozen_seasons=("season", "size"),
            mean_allocation_total_variation=("mean_allocation_total_variation", "mean"),
            mean_brier_score=("mean_brier_score", "mean"),
            mean_cross_entropy=("mean_cross_entropy", "mean"),
            player_share_mae=("player_share_mae", "mean"),
            player_share_rmse=("player_share_rmse", "mean"),
            evaluated_team_games=("evaluated_team_games", "sum"),
        )
        .sort_values(["mean_allocation_total_variation", "mean_brier_score"], kind="stable")
        .reset_index(drop=True)
    )
    summary.to_parquet(run_dir / "frozen_summary.parquet", index=False)
    pd.concat(coefficient_rows, ignore_index=True).to_parquet(
        run_dir / "frozen_coefficients.parquet", index=False
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "selected_config": asdict(config),
                "history_start_season": history_start_season,
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "target": (
                    "player regulation minutes / 240 among observed medically available roster"
                ),
                "overtime": "excluded; no rescaling",
                "availability": "observed historical medical-availability mask; not yet forecasted",
                "features": list(FEATURE_COLUMNS),
                "control": "lag-five team minutes plus one-minute floor, normalized on same mask",
            },
            indent=2,
        )
        + "\n"
    )
    return ACMSPRun(run_dir=run_dir, run_id=run_id)


def _group_softmax(
    logits: np.ndarray, groups: np.ndarray, group_count: int
) -> tuple[np.ndarray, np.ndarray]:
    maxima = np.full(group_count, -np.inf, dtype=float)
    np.maximum.at(maxima, groups, logits)
    exponentiated = np.exp(logits - maxima[groups])
    denominators = np.bincount(groups, weights=exponentiated, minlength=group_count)
    return exponentiated / denominators[groups], maxima + np.log(denominators)


def _team_game_codes(panel: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(panel.loc[:, ["season", "game_id", "team_id"]])
    groups, _ = pd.factorize(keys, sort=False)
    return groups


def _history_panel(panels: dict[str, pd.DataFrame], target_season: str) -> pd.DataFrame:
    target_year = _season_year(target_season)
    history = [panel for season, panel in panels.items() if _season_year(season) < target_year]
    if not history:
        raise ValueError(f"No source seasons before {target_season}")
    return pd.concat(history, ignore_index=True)


def _season_range(first: str, last: str) -> tuple[str, ...]:
    return tuple(
        f"{year}-{str(year + 1)[-2:]}"
        for year in range(_season_year(first), _season_year(last) + 1)
    )


def _season_year(season: str) -> int:
    return int(season[:4])


def _validate_panel(panel: pd.DataFrame) -> None:
    required = {
        "season",
        "game_id",
        "team_id",
        "player_id",
        "actual_minute_share",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required - set(panel))
    if missing:
        raise ValueError(f"Conditional minute panel lacks required columns: {missing}")
    group_totals = panel.groupby(["season", "game_id", "team_id"])["actual_minute_share"].sum()
    if not np.isclose(group_totals, 1.0, atol=1e-8).all():
        raise ValueError("Every available team-game target must sum to one")
    if not np.isfinite(panel.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Conditional minute features must be finite")


def main() -> None:
    """Run the availability-conditioned full-season frozen replay."""

    parser = argparse.ArgumentParser(
        description="Evaluate availability-conditioned game-level minute allocations"
    )
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--history-start-season", default=DEFAULT_HISTORY_START_SEASON)
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=list(DEFAULT_ALPHA_GRID))
    args = parser.parse_args()
    run = run_ac_msp(
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        history_start_season=args.history_start_season,
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        alpha_grid=tuple(args.alpha_grid),
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
