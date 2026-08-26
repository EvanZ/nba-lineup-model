"""Parameter-free last-five median-minutes persistence baseline."""

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
    DEFAULT_SEASONS,
    read_regular_game_minutes,
)

LOOKBACK_GAMES = 5
MODEL_NAME = "l5_median_minutes_persistence"
MODEL_VERSION = "v0.1"


@dataclass(frozen=True)
class L5MedianMinutesRun:
    """Immutable location and identity for one L5-MMP evaluation."""

    run_dir: Path
    run_id: str


def median_minute_share_forecast(history: list[pd.DataFrame]) -> dict[int, float]:
    """Return normalized median-minute shares from five completed team-games.

    A player absent from any historical game receives zero minutes for that
    game. This makes the median a five-game allocation rule, not a conditional
    median among only games in which the player appeared.
    """

    if len(history) != LOOKBACK_GAMES:
        raise ValueError(f"L5-MMP requires exactly {LOOKBACK_GAMES} history games")
    player_ids = sorted(set().union(*(set(game["player_id"]) for game in history)))
    median_minutes: dict[int, float] = {}
    for player_id in player_ids:
        minutes = [
            float(game.set_index("player_id")["minutes"].to_dict().get(player_id, 0.0))
            for game in history
        ]
        median_minutes[player_id] = float(np.median(minutes))
    total_median_minutes = sum(median_minutes.values())
    if total_median_minutes <= 0.0:
        raise ValueError("L5-MMP history has no positive median minutes")
    return {
        player_id: minutes / total_median_minutes
        for player_id, minutes in median_minutes.items()
    }


def evaluate_l5_median_minutes_persistence(
    game_minutes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score L5-MMP next-team-game minute-share forecasts."""

    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "player_id",
        "player_name",
        "game_time_utc",
        "minutes",
        "minute_share",
    }
    missing = sorted(required - set(game_minutes))
    if missing:
        raise ValueError(f"Game minutes lacks required columns: {missing}")

    records: list[dict[str, object]] = []
    team_game_metrics: list[dict[str, object]] = []
    ordered = game_minutes.sort_values(
        ["team_id", "game_time_utc", "game_id", "player_id"], kind="stable"
    )
    for team_id, team_rows in ordered.groupby("team_id", sort=True):
        games = list(team_rows.groupby("game_id", sort=False))
        for sequence, (game_id, current) in enumerate(games):
            if sequence < LOOKBACK_GAMES:
                continue
            history_pairs = games[sequence - LOOKBACK_GAMES : sequence]
            history_game_ids = [str(history_game_id) for history_game_id, _ in history_pairs]
            history = [history_game for _, history_game in history_pairs]
            predicted_map = median_minute_share_forecast(history)
            current_map = current.set_index("player_id")["minute_share"].to_dict()
            history_player_ids = set().union(*(set(game["player_id"]) for game in history))
            history_names = pd.concat(
                [game.loc[:, ["player_id", "player_name"]] for game in history]
                + [current.loc[:, ["player_id", "player_name"]]],
                ignore_index=True,
            )
            names = history_names.drop_duplicates("player_id", keep="last").set_index(
                "player_id"
            )["player_name"]
            player_ids = sorted(history_player_ids | set(current_map))
            actual = np.asarray(
                [float(current_map.get(player_id, 0.0)) for player_id in player_ids]
            )
            predicted = np.asarray(
                [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
            )
            if not np.isclose(actual.sum(), 1.0, atol=1e-8):
                raise ValueError(
                    f"Actual minute shares do not sum to one for {game_id}/{team_id}"
                )
            if not np.isclose(predicted.sum(), 1.0, atol=1e-8):
                raise ValueError(
                    f"Predicted minute shares do not sum to one for {game_id}/{team_id}"
                )
            absolute_error = np.abs(actual - predicted)
            squared_error = np.square(actual - predicted)
            zero_probability_active_player = (actual > 0.0) & (predicted == 0.0)
            cross_entropy = float("inf")
            if not zero_probability_active_player.any():
                positive_actual = actual > 0.0
                cross_entropy = float(
                    -(actual[positive_actual] * np.log(predicted[positive_actual])).sum()
                )
            team_game_metrics.append(
                {
                    "team_id": int(team_id),
                    "team": str(current["team_tricode"].iloc[0]),
                    "game_id": str(game_id),
                    "history_start_game_id": history_game_ids[0],
                    "previous_game_id": history_game_ids[-1],
                    "game_time_utc": current["game_time_utc"].iloc[0],
                    "allocation_total_variation": float(0.5 * absolute_error.sum()),
                    "brier_score": float(squared_error.sum()),
                    "player_share_mae": float(absolute_error.mean()),
                    "player_share_mse": float(squared_error.mean()),
                    "cross_entropy": cross_entropy,
                    "has_zero_probability_active_player": bool(
                        zero_probability_active_player.any()
                    ),
                    "zero_history_actual_share": float(
                        actual[
                            [player_id not in history_player_ids for player_id in player_ids]
                        ].sum()
                    ),
                    "zero_median_actual_share": float(
                        actual[[predicted_share == 0.0 for predicted_share in predicted]].sum()
                    ),
                    "departed_player_predicted_share": float(
                        predicted[[player_id not in current_map for player_id in player_ids]].sum()
                    ),
                }
            )
            for player_id, actual_share, predicted_share in zip(
                player_ids, actual, predicted, strict=True
            ):
                records.append(
                    {
                        "team_id": int(team_id),
                        "team": str(current["team_tricode"].iloc[0]),
                        "game_id": str(game_id),
                        "history_start_game_id": history_game_ids[0],
                        "previous_game_id": history_game_ids[-1],
                        "game_time_utc": current["game_time_utc"].iloc[0],
                        "player_id": int(player_id),
                        "player_name": str(names.loc[player_id]),
                        "actual_minute_share": float(actual_share),
                        "predicted_minute_share": float(predicted_share),
                        "absolute_error": float(abs(actual_share - predicted_share)),
                        "was_in_five_game_history": player_id in history_player_ids,
                        "was_in_current_game": player_id in current_map,
                    }
                )
    predictions = pd.DataFrame.from_records(records)
    metrics = pd.DataFrame.from_records(team_game_metrics)
    if predictions.empty or metrics.empty:
        raise ValueError("L5-MMP requires at least six completed team-games")
    return predictions, metrics


def summarize_l5_metrics(metrics: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """Return one transparent aggregate metric row for a season."""

    finite_cross_entropy = metrics.loc[np.isfinite(metrics["cross_entropy"]), "cross_entropy"]
    return pd.DataFrame(
        [
            {
                "season": season,
                "evaluated_team_games": int(len(metrics)),
                "mean_allocation_total_variation": float(
                    metrics["allocation_total_variation"].mean()
                ),
                "median_allocation_total_variation": float(
                    metrics["allocation_total_variation"].median()
                ),
                "mean_brier_score": float(metrics["brier_score"].mean()),
                "player_share_mae": float(metrics["player_share_mae"].mean()),
                "player_share_rmse": float(np.sqrt(metrics["player_share_mse"].mean())),
                "strict_cross_entropy": float(metrics["cross_entropy"].mean()),
                "share_team_games_infinite_cross_entropy": float(
                    metrics["has_zero_probability_active_player"].mean()
                ),
                "mean_cross_entropy_when_finite": float(finite_cross_entropy.mean()),
                "mean_zero_history_actual_share": float(
                    metrics["zero_history_actual_share"].mean()
                ),
                "mean_zero_median_actual_share": float(
                    metrics["zero_median_actual_share"].mean()
                ),
                "mean_departed_player_predicted_share": float(
                    metrics["departed_player_predicted_share"].mean()
                ),
            }
        ]
    )


def run_l5_median_minutes_persistence(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> L5MedianMinutesRun:
    """Materialize parameter-free L5-MMP predictions and holdout metrics."""

    if not seasons:
        raise ValueError("At least one season is required")
    run_id = f"l5-mmp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[pd.DataFrame] = []
    for season in seasons:
        game_minutes = read_regular_game_minutes(season, curated_dir=curated_dir)
        predictions, metrics = evaluate_l5_median_minutes_persistence(game_minutes)
        predictions.assign(season=season).to_parquet(
            run_dir / f"{season}_game_predictions.parquet", index=False
        )
        metrics.assign(season=season).to_parquet(
            run_dir / f"{season}_team_game_metrics.parquet", index=False
        )
        summaries.append(summarize_l5_metrics(metrics, season=season))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_parquet(run_dir / "metrics.parquet", index=False)
    metadata = {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seasons": list(seasons),
        "lookback_games": LOOKBACK_GAMES,
        "contract": "median player minutes over five completed team-games, normalized to shares",
        "target": "player minutes divided by actual total team minutes, including overtime",
        "cold_start": "zero predicted share for a player with zero five-game median minutes",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return L5MedianMinutesRun(run_dir=run_dir, run_id=run_id)


def main() -> None:
    """Run L5-MMP over one or more regular seasons."""

    parser = argparse.ArgumentParser(description="Evaluate Last-5 Median Minutes Persistence")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_l5_median_minutes_persistence(
        seasons=tuple(args.seasons),
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
