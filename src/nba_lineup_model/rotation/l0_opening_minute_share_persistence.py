"""Roster-gated prior-season minute-share baseline for season-opening games."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)

MODEL_NAME = "l0_opening_minute_share_persistence"
MODEL_VERSION = "v0.0"
DEFAULT_TUNE_SEASON = "2024-25"
DEFAULT_HOLDOUT_SEASON = "2025-26"
DEFAULT_EPSILON_GRID = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.8, 1.0)


@dataclass(frozen=True)
class L0OpeningMinuteShareRun:
    """Immutable location and identity for one L0-MSP evaluation."""

    run_dir: Path
    run_id: str


def read_preseason_roster_candidates(
    season: str,
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> pd.DataFrame:
    """Read only membership known from transactions before the opening date.

    Opening-game reconciliations are intentionally excluded. They validate the
    mart after the fact, but cannot be treated as a preseason input.
    """

    path = Path(curated_dir) / "opening_rosters" / season / "part-00000.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Opening roster mart is missing for {season}: {path}")
    source = pd.read_parquet(path)
    required = {"team_id", "team", "player_id", "player_name", "membership_source"}
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"Opening roster mart lacks required columns: {missing}")
    candidates = source.loc[
        source["membership_source"].eq("transaction_reconstruction"),
        ["team_id", "team", "player_id", "player_name"],
    ].copy()
    candidates["team_id"] = pd.to_numeric(candidates["team_id"], errors="raise").astype(
        "int64"
    )
    candidates["player_id"] = pd.to_numeric(candidates["player_id"], errors="raise").astype(
        "int64"
    )
    if candidates.duplicated(["team_id", "player_id"]).any():
        raise ValueError(f"Opening roster mart has duplicate candidate rows for {season}")
    return candidates.sort_values(["team_id", "player_id"], kind="stable").reset_index(
        drop=True
    )


def evaluate_l0_opening_minute_share_persistence(
    prior_game_minutes: pd.DataFrame,
    target_game_minutes: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score opening-game shares from final prior-season shares and a roster gate."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("L0-MSP epsilon must be between zero and one")
    _validate_game_minutes(prior_game_minutes, label="prior")
    _validate_game_minutes(target_game_minutes, label="target")
    required_candidates = {"team_id", "team", "player_id", "player_name"}
    missing_candidates = sorted(required_candidates - set(candidates))
    if missing_candidates:
        raise ValueError(f"Candidate roster lacks required columns: {missing_candidates}")

    final_prior_games = _team_boundary_games(prior_game_minutes, first=False)
    opening_target_games = _team_boundary_games(target_game_minutes, first=True)
    candidate_groups = {
        int(team_id): group.copy()
        for team_id, group in candidates.groupby("team_id", sort=True)
    }
    records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for team_id, target in opening_target_games.items():
        if team_id not in final_prior_games:
            raise ValueError(f"Prior-season minutes have no final game for team {team_id}")
        candidate_frame = candidate_groups.get(team_id)
        if candidate_frame is None or candidate_frame.empty:
            raise ValueError(f"Opening roster candidates have no rows for team {team_id}")
        previous = final_prior_games[team_id]
        predicted_map = _smoothed_candidate_allocation(previous, candidate_frame, epsilon)
        actual_map = target.set_index("player_id")["minute_share"].to_dict()
        player_ids = sorted(set(predicted_map) | set(actual_map))
        actual = np.asarray(
            [float(actual_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        predicted = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        if not np.isclose(actual.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Actual minute shares do not sum to one for team {team_id}")
        if not np.isclose(predicted.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Predicted minute shares do not sum to one for team {team_id}")
        names = _player_names(previous, target, candidate_frame)
        absolute_error = np.abs(actual - predicted)
        squared_error = np.square(actual - predicted)
        zero_probability_active = (actual > 0.0) & (predicted == 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            positive_actual = actual > 0.0
            cross_entropy = float(
                -(actual[positive_actual] * np.log(predicted[positive_actual])).sum()
            )
        target_game_id = str(target["game_id"].iloc[0])
        prior_game_id = str(previous["game_id"].iloc[0])
        team = str(target["team_tricode"].iloc[0])
        candidate_ids = set(predicted_map)
        metrics.append(
            {
                "team_id": team_id,
                "team": team,
                "game_id": target_game_id,
                "prior_final_game_id": prior_game_id,
                "game_time_utc": target["game_time_utc"].iloc[0],
                "epsilon": epsilon,
                "candidate_player_count": len(candidate_ids),
                "actual_player_count": len(actual_map),
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "outside_candidate_actual_share": float(
                    actual[[player_id not in candidate_ids for player_id in player_ids]].sum()
                ),
            }
        )
        for player_id, actual_share, predicted_share in zip(
            player_ids, actual, predicted, strict=True
        ):
            records.append(
                {
                    "team_id": team_id,
                    "team": team,
                    "game_id": target_game_id,
                    "prior_final_game_id": prior_game_id,
                    "game_time_utc": target["game_time_utc"].iloc[0],
                    "player_id": player_id,
                    "player_name": names.get(player_id, str(player_id)),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "was_preseason_candidate": player_id in candidate_ids,
                    "was_in_prior_final_game": player_id
                    in set(previous["player_id"].astype(int)),
                    "was_in_target_opening_game": player_id in actual_map,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metrics)


def select_epsilon(
    prior_game_minutes: pd.DataFrame,
    target_game_minutes: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
) -> tuple[float, pd.DataFrame]:
    """Select the smallest epsilon minimizing source-season mean TV."""

    if not epsilon_grid:
        raise ValueError("L0-MSP epsilon grid cannot be empty")
    rows: list[dict[str, float]] = []
    for epsilon in sorted(set(epsilon_grid)):
        _predictions, metrics = evaluate_l0_opening_minute_share_persistence(
            prior_game_minutes,
            target_game_minutes,
            candidates,
            epsilon=epsilon,
        )
        rows.append(
            {
                "epsilon": epsilon,
                "mean_allocation_total_variation": float(
                    metrics["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(metrics["brier_score"].mean()),
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["mean_allocation_total_variation", "epsilon"], kind="stable"
    )
    return float(grid["epsilon"].iloc[0]), grid.reset_index(drop=True)


def summarize_l0_metrics(metrics: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """Summarize the 30 opening-game allocation forecasts for one season."""

    finite_cross_entropy = metrics.loc[np.isfinite(metrics["cross_entropy"]), "cross_entropy"]
    return pd.DataFrame(
        [
            {
                "season": season,
                "evaluated_team_games": int(len(metrics)),
                "mean_allocation_total_variation": float(
                    metrics["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(metrics["brier_score"].mean()),
                "player_share_mae": float(metrics["player_share_mae"].mean()),
                "player_share_rmse": float(np.sqrt(metrics["player_share_mse"].mean())),
                "strict_cross_entropy": float(metrics["cross_entropy"].mean()),
                "share_team_games_infinite_cross_entropy": float(
                    metrics["has_zero_probability_active_player"].mean()
                ),
                "mean_cross_entropy_when_finite": float(finite_cross_entropy.mean()),
                "mean_outside_candidate_actual_share": float(
                    metrics["outside_candidate_actual_share"].mean()
                ),
            }
        ]
    )


def run_l0_opening_minute_share_persistence(
    *,
    tune_season: str = DEFAULT_TUNE_SEASON,
    holdout_season: str = DEFAULT_HOLDOUT_SEASON,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
) -> L0OpeningMinuteShareRun:
    """Fit opening cold-start mass on one season and freeze it for the next."""

    tune_prior = read_regular_game_minutes(_previous_season(tune_season), curated_dir=curated_dir)
    tune_target = read_regular_game_minutes(tune_season, curated_dir=curated_dir)
    tune_candidates = read_preseason_roster_candidates(tune_season, curated_dir=curated_dir)
    selected_epsilon, grid = select_epsilon(
        tune_prior,
        tune_target,
        tune_candidates,
        epsilon_grid=epsilon_grid,
    )
    holdout_prior = read_regular_game_minutes(
        _previous_season(holdout_season), curated_dir=curated_dir
    )
    holdout_target = read_regular_game_minutes(holdout_season, curated_dir=curated_dir)
    holdout_candidates = read_preseason_roster_candidates(
        holdout_season,
        curated_dir=curated_dir,
    )
    tune_predictions, tune_metrics = evaluate_l0_opening_minute_share_persistence(
        tune_prior,
        tune_target,
        tune_candidates,
        epsilon=selected_epsilon,
    )
    holdout_predictions, holdout_metrics = evaluate_l0_opening_minute_share_persistence(
        holdout_prior,
        holdout_target,
        holdout_candidates,
        epsilon=selected_epsilon,
    )
    run_id = f"l0-msp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    grid.to_parquet(run_dir / "epsilon_grid.parquet", index=False)
    tune_predictions.to_parquet(run_dir / f"{tune_season}_opening_predictions.parquet", index=False)
    tune_metrics.to_parquet(run_dir / f"{tune_season}_opening_metrics.parquet", index=False)
    holdout_predictions.to_parquet(
        run_dir / f"{holdout_season}_opening_predictions.parquet", index=False
    )
    holdout_metrics.to_parquet(run_dir / f"{holdout_season}_opening_metrics.parquet", index=False)
    pd.concat(
        [
            summarize_l0_metrics(tune_metrics, season=tune_season),
            summarize_l0_metrics(holdout_metrics, season=holdout_season),
        ],
        ignore_index=True,
    ).to_parquet(run_dir / "metrics.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tune_season": tune_season,
                "holdout_season": holdout_season,
                "selected_epsilon": selected_epsilon,
                "epsilon_grid": list(epsilon_grid),
                "contract": (
                    "prior final-game minute shares restricted to transaction-derived "
                    "opening roster candidates, mixed with uniform cold-start mass"
                ),
                "reconciliation_input": "excluded to preserve a preseason-only input boundary",
            },
            indent=2,
        )
        + "\n"
    )
    return L0OpeningMinuteShareRun(run_dir=run_dir, run_id=run_id)


def _team_boundary_games(game_minutes: pd.DataFrame, *, first: bool) -> dict[int, pd.DataFrame]:
    ordered = game_minutes.sort_values(
        ["team_id", "game_time_utc", "game_id", "player_id"], kind="stable"
    )
    boundary: dict[int, pd.DataFrame] = {}
    for team_id, team_rows in ordered.groupby("team_id", sort=True):
        game_ids = team_rows.loc[:, ["game_id", "game_time_utc"]].drop_duplicates()
        selected_game_id = str(game_ids.iloc[0 if first else -1]["game_id"])
        boundary[int(team_id)] = team_rows.loc[team_rows["game_id"].eq(selected_game_id)].copy()
    return boundary


def _smoothed_candidate_allocation(
    previous: pd.DataFrame,
    candidates: pd.DataFrame,
    epsilon: float,
) -> dict[int, float]:
    candidate_ids = candidates["player_id"].astype(int).tolist()
    prior_shares = previous.set_index("player_id")["minute_share"].to_dict()
    restricted = np.asarray(
        [float(prior_shares.get(player_id, 0.0)) for player_id in candidate_ids]
    )
    if restricted.sum() <= 0.0:
        restricted = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
    else:
        restricted /= restricted.sum()
    smoothed = (1.0 - epsilon) * restricted + epsilon / len(candidate_ids)
    return dict(zip(candidate_ids, smoothed, strict=True))


def _player_names(*frames: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for frame in frames:
        for row in frame.loc[:, ["player_id", "player_name"]].itertuples(index=False):
            names[int(row.player_id)] = str(row.player_name)
    return names


def _validate_game_minutes(game_minutes: pd.DataFrame, *, label: str) -> None:
    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "player_id",
        "player_name",
        "game_time_utc",
        "minute_share",
    }
    missing = sorted(required - set(game_minutes))
    if missing:
        raise ValueError(f"{label} game minutes lacks required columns: {missing}")


def _previous_season(season: str) -> str:
    start_year = int(season[:4]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def main() -> None:
    """Fit the opening baseline on one season and evaluate the next season."""

    parser = argparse.ArgumentParser(description="Evaluate L0 opening minute-share persistence")
    parser.add_argument("--tune-season", default=DEFAULT_TUNE_SEASON)
    parser.add_argument("--holdout-season", default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_l0_opening_minute_share_persistence(
        tune_season=args.tune_season,
        holdout_season=args.holdout_season,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)
