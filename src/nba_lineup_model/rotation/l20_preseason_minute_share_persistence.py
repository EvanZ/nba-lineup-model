"""Static preseason prediction of each team's first-20-game minute allocation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    read_regular_game_minutes,
)

MODEL_NAME = "l20_preseason_minute_share_persistence"
MODEL_VERSION = "v0.0"
DEFAULT_TUNE_SEASON = "2024-25"
DEFAULT_HOLDOUT_SEASON = "2025-26"
DEFAULT_TEAM_GAMES = 20
DEFAULT_EPSILON_GRID = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.64, 0.8, 1.0)


@dataclass(frozen=True)
class L20PreseasonMinuteShareRun:
    """Immutable location and identity for one L20-MSP evaluation."""

    run_dir: Path
    run_id: str


def evaluate_l20_preseason_minute_share_persistence(
    prior_game_minutes: pd.DataFrame,
    target_game_minutes: pd.DataFrame,
    opening_candidates: pd.DataFrame,
    *,
    epsilon: float,
    team_games: int = DEFAULT_TEAM_GAMES,
    prior_minutes: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score static preseason allocations against the first ``team_games`` games.

    ``prior_minutes`` permits an experimental preseason prior to replace the
    observed immediate-prior-season totals while preserving the same allocation
    target, roster gate, and scoring contract.
    """

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("L20-MSP epsilon must be between zero and one")
    if team_games <= 0:
        raise ValueError("L20-MSP team_games must be positive")
    _validate_game_minutes(prior_game_minutes, label="prior")
    _validate_game_minutes(target_game_minutes, label="target")
    _validate_candidates(opening_candidates)

    allocation_scores = (
        prior_game_minutes.groupby("player_id", as_index=True)["minutes"].sum()
        if prior_minutes is None
        else prior_minutes.astype(float)
    )
    targets = _first_n_team_game_allocations(target_game_minutes, team_games=team_games)
    candidate_groups = {
        int(team_id): group.copy()
        for team_id, group in opening_candidates.groupby("team_id", sort=True)
    }
    records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for target in targets:
        candidates = candidate_groups.get(int(target.team_id))
        if candidates is None or candidates.empty:
            raise ValueError(f"Opening candidates have no rows for team {target.team_id}")
        candidate_ids = candidates["player_id"].astype(int).tolist()
        predicted_map = _smoothed_prior_minutes(
            candidate_ids,
            allocation_scores,
            epsilon=epsilon,
        )
        actual = target.player_shares
        player_ids = sorted(set(predicted_map) | set(actual))
        actual_values = np.asarray([float(actual.get(player_id, 0.0)) for player_id in player_ids])
        predicted_values = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        _assert_distribution(actual_values, label=f"actual for {target.team}")
        _assert_distribution(predicted_values, label=f"prediction for {target.team}")
        names = candidates.set_index("player_id")["player_name"].to_dict()
        absolute_error = np.abs(actual_values - predicted_values)
        squared_error = np.square(actual_values - predicted_values)
        zero_probability_active = (actual_values > 0.0) & (predicted_values == 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            positive_actual = actual_values > 0.0
            cross_entropy = float(
                -(actual_values[positive_actual] * np.log(predicted_values[positive_actual])).sum()
            )
        candidate_set = set(candidate_ids)
        metrics.append(
            {
                "team_id": int(target.team_id),
                "team": target.team,
                "first_game_id": target.first_game_id,
                "last_game_id": target.last_game_id,
                "target_game_count": target.target_game_count,
                "epsilon": epsilon,
                "candidate_player_count": len(candidate_set),
                "actual_player_count": len(actual),
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "outside_candidate_actual_share": float(
                    actual_values[
                        [player_id not in candidate_set for player_id in player_ids]
                    ].sum()
                ),
            }
        )
        for player_id, actual_share, predicted_share in zip(
            player_ids,
            actual_values,
            predicted_values,
            strict=True,
        ):
            records.append(
                {
                    "team_id": int(target.team_id),
                    "team": target.team,
                    "first_game_id": target.first_game_id,
                    "last_game_id": target.last_game_id,
                    "target_game_count": target.target_game_count,
                    "player_id": player_id,
                    "player_name": names.get(player_id, str(player_id)),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "was_opening_candidate": player_id in candidate_set,
                    "was_in_first_n_games": player_id in actual,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metrics)


def select_epsilon(
    prior_game_minutes: pd.DataFrame,
    target_game_minutes: pd.DataFrame,
    opening_candidates: pd.DataFrame,
    *,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> tuple[float, pd.DataFrame]:
    """Select the smallest source-season epsilon minimizing mean team TV."""

    if not epsilon_grid:
        raise ValueError("L20-MSP epsilon grid cannot be empty")
    rows: list[dict[str, float]] = []
    for epsilon in sorted(set(epsilon_grid)):
        _predictions, metrics = evaluate_l20_preseason_minute_share_persistence(
            prior_game_minutes,
            target_game_minutes,
            opening_candidates,
            epsilon=epsilon,
            team_games=team_games,
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
    ).reset_index(drop=True)
    return float(grid.loc[0, "epsilon"]), grid


def run_l20_preseason_minute_share_persistence(
    *,
    tune_season: str = DEFAULT_TUNE_SEASON,
    holdout_season: str = DEFAULT_HOLDOUT_SEASON,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID,
    team_games: int = DEFAULT_TEAM_GAMES,
) -> L20PreseasonMinuteShareRun:
    """Fit one static preseason allocation on source and freeze it for holdout."""

    tune_prior = read_regular_game_minutes(previous_season(tune_season), curated_dir=curated_dir)
    tune_target = read_regular_game_minutes(tune_season, curated_dir=curated_dir)
    tune_candidates = read_preseason_roster_candidates(tune_season, curated_dir=curated_dir)
    selected_epsilon, grid = select_epsilon(
        tune_prior,
        tune_target,
        tune_candidates,
        epsilon_grid=epsilon_grid,
        team_games=team_games,
    )
    outputs = {}
    for label, season in (("tune", tune_season), ("holdout", holdout_season)):
        prior = tune_prior if label == "tune" else read_regular_game_minutes(
            previous_season(season), curated_dir=curated_dir
        )
        target = tune_target if label == "tune" else read_regular_game_minutes(
            season, curated_dir=curated_dir
        )
        candidates = tune_candidates if label == "tune" else read_preseason_roster_candidates(
            season, curated_dir=curated_dir
        )
        outputs[label] = evaluate_l20_preseason_minute_share_persistence(
            prior,
            target,
            candidates,
            epsilon=selected_epsilon,
            team_games=team_games,
        )
    run_id = f"l20-msp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    grid.to_parquet(run_dir / "epsilon_grid.parquet", index=False)
    summaries: list[pd.DataFrame] = []
    for season, label in ((tune_season, "tune"), (holdout_season, "holdout")):
        predictions, metrics = outputs[label]
        predictions.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        metrics.to_parquet(run_dir / f"{season}_team_metrics.parquet", index=False)
        summaries.append(summarize_l20_metrics(metrics, season=season))
    pd.concat(summaries, ignore_index=True).to_parquet(run_dir / "metrics.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tune_season": tune_season,
                "holdout_season": holdout_season,
                "target": (
                    "cumulative minute-share allocation across "
                    f"first {team_games} team games"
                ),
                "team_games": team_games,
                "selected_epsilon": selected_epsilon,
                "epsilon_grid": list(epsilon_grid),
                "contract": (
                    "static prior-season total minutes restricted to transaction-derived "
                    "opening candidates with uniform cold-start smoothing"
                ),
                "reconciliation_input": "excluded from opening candidates",
            },
            indent=2,
        )
        + "\n"
    )
    return L20PreseasonMinuteShareRun(run_dir=run_dir, run_id=run_id)


@dataclass(frozen=True)
class _TeamTarget:
    team_id: int
    team: str
    first_game_id: str
    last_game_id: str
    target_game_count: int
    player_shares: dict[int, float]


def _first_n_team_game_allocations(
    game_minutes: pd.DataFrame,
    *,
    team_games: int,
) -> list[_TeamTarget]:
    ordered = game_minutes.sort_values(
        ["team_id", "game_time_utc", "game_id", "player_id"], kind="stable"
    )
    targets: list[_TeamTarget] = []
    for team_id, team_rows in ordered.groupby("team_id", sort=True):
        game_ids = team_rows.loc[:, ["game_id", "game_time_utc"]].drop_duplicates().head(team_games)
        if len(game_ids) != team_games:
            raise ValueError(f"Team {team_id} has fewer than {team_games} games")
        selected = team_rows.loc[team_rows["game_id"].isin(game_ids["game_id"])].copy()
        player_minutes = selected.groupby("player_id", as_index=True)["minutes"].sum()
        total_minutes = float(player_minutes.sum())
        if total_minutes <= 0.0:
            raise ValueError(f"First {team_games} games have no minutes for team {team_id}")
        targets.append(
            _TeamTarget(
                team_id=int(team_id),
                team=str(selected["team_tricode"].iloc[0]),
                first_game_id=str(game_ids["game_id"].iloc[0]),
                last_game_id=str(game_ids["game_id"].iloc[-1]),
                target_game_count=team_games,
                player_shares={
                    int(player_id): float(minutes / total_minutes)
                    for player_id, minutes in player_minutes.items()
                },
            )
        )
    return targets


def _smoothed_prior_minutes(
    candidate_ids: list[int],
    prior_minutes: pd.Series,
    *,
    epsilon: float,
) -> dict[int, float]:
    scores = np.asarray(
        [float(prior_minutes.get(player_id, 0.0)) for player_id in candidate_ids]
    )
    if scores.sum() <= 0.0:
        scores = np.full(len(candidate_ids), 1.0 / len(candidate_ids))
    else:
        scores /= scores.sum()
    values = (1.0 - epsilon) * scores + epsilon / len(candidate_ids)
    return dict(zip(candidate_ids, values, strict=True))


def summarize_l20_metrics(metrics: pd.DataFrame, *, season: str) -> pd.DataFrame:
    finite_cross_entropy = metrics.loc[np.isfinite(metrics["cross_entropy"]), "cross_entropy"]
    return pd.DataFrame(
        [
            {
                "season": season,
                "evaluated_teams": int(len(metrics)),
                "mean_allocation_total_variation": float(
                    metrics["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(metrics["brier_score"].mean()),
                "player_share_mae": float(metrics["player_share_mae"].mean()),
                "player_share_rmse": float(np.sqrt(metrics["player_share_mse"].mean())),
                "strict_cross_entropy": float(metrics["cross_entropy"].mean()),
                "share_teams_infinite_cross_entropy": float(
                    metrics["has_zero_probability_active_player"].mean()
                ),
                "mean_cross_entropy_when_finite": float(finite_cross_entropy.mean()),
                "mean_outside_candidate_actual_share": float(
                    metrics["outside_candidate_actual_share"].mean()
                ),
            }
        ]
    )


def _validate_game_minutes(game_minutes: pd.DataFrame, *, label: str) -> None:
    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "player_id",
        "minutes",
        "game_time_utc",
    }
    missing = sorted(required - set(game_minutes))
    if missing:
        raise ValueError(f"{label} game minutes lacks required columns: {missing}")


def _validate_candidates(candidates: pd.DataFrame) -> None:
    required = {"team_id", "team", "player_id", "player_name"}
    missing = sorted(required - set(candidates))
    if missing:
        raise ValueError(f"Opening candidates lack required columns: {missing}")
    if candidates.duplicated(["team_id", "player_id"]).any():
        raise ValueError("Opening candidates contain duplicate player-team rows")


def _assert_distribution(values: np.ndarray, *, label: str) -> None:
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError(f"{label} does not sum to one")
    if (values < 0.0).any():
        raise ValueError(f"{label} contains a negative minute share")


def previous_season(season: str) -> str:
    start_year = int(season[:4]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def main() -> None:
    """Fit static preseason first-20-game minutes on source and freeze holdout."""

    parser = argparse.ArgumentParser(description="Evaluate static preseason first-20 minutes")
    parser.add_argument("--tune-season", default=DEFAULT_TUNE_SEASON)
    parser.add_argument("--holdout-season", default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--team-games", type=int, default=DEFAULT_TEAM_GAMES)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_l20_preseason_minute_share_persistence(
        tune_season=args.tune_season,
        holdout_season=args.holdout_season,
        team_games=args.team_games,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)
