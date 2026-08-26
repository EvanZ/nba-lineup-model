"""Parameter-free lag-one team minute-share persistence baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

MODEL_NAME = "l1_minute_share_persistence"
MODEL_VERSION = "v0.0"
DEFAULT_SEASONS = ("2024-25", "2025-26")
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation")


@dataclass(frozen=True)
class L1MinuteShareRun:
    """Immutable location and identity for one L1-MSP evaluation."""

    run_dir: Path
    run_id: str


def read_regular_game_minutes(
    season: str,
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> pd.DataFrame:
    """Return one player row per regular-season team-game, including DNP zeros."""

    root = Path(curated_dir) / "players" / season / "regular"
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No regular-season player box scores for {season}: {root}")
    source = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "personId",
        "nameI",
        "statistics_minutes",
        "game_time_utc",
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"Game-player source lacks required columns: {missing}")
    if source.duplicated(["game_id", "team_id", "personId"]).any():
        raise ValueError(f"Duplicate game-player rows in {season}")

    output = source.loc[
        :,
        [
            "game_id",
            "team_id",
            "team_tricode",
            "personId",
            "nameI",
            "statistics_minutes",
            "game_time_utc",
        ],
    ].copy()
    output = output.rename(columns={"personId": "player_id", "nameI": "player_name"})
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype("int64")
    output["minutes"] = (
        pd.to_timedelta(output["statistics_minutes"], errors="coerce").dt.total_seconds() / 60.0
    )
    # CDN box scores may encode DNP rows with a blank minutes field.
    output["minutes"] = output["minutes"].fillna(0.0)
    if output["minutes"].lt(0).any():
        raise ValueError(f"Negative player minutes in {season}")
    team_minutes = output.groupby(["game_id", "team_id"], as_index=False, sort=False).agg(
        team_minutes=("minutes", "sum")
    )
    if team_minutes["team_minutes"].le(0).any():
        raise ValueError(f"A team-game has no recorded minutes in {season}")
    output = output.merge(team_minutes, on=["game_id", "team_id"], validate="many_to_one")
    output["minute_share"] = output["minutes"] / output["team_minutes"]
    return output.sort_values(
        ["team_id", "game_time_utc", "game_id", "player_id"], kind="stable"
    ).reset_index(drop=True)


def evaluate_l1_minute_share_persistence(
    game_minutes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score next-team-game shares by copying the prior team-game distribution.

    The support for a prediction is the union of players in the previous and
    current team-game.  A player without previous-game state receives zero;
    players no longer present retain their previous share and receive an actual
    share of zero.  This preserves the exact parameter-free L1 contract.
    """

    required = {
        "game_id",
        "team_id",
        "team_tricode",
        "player_id",
        "player_name",
        "game_time_utc",
        "minutes",
        "team_minutes",
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
            if sequence == 0:
                continue
            previous_game_id, previous = games[sequence - 1]
            previous_map = previous.set_index("player_id")["minute_share"].to_dict()
            current_map = current.set_index("player_id")["minute_share"].to_dict()
            names = pd.concat(
                [
                    previous.loc[:, ["player_id", "player_name"]],
                    current.loc[:, ["player_id", "player_name"]],
                ],
                ignore_index=True,
            ).drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
            player_ids = sorted(set(previous_map) | set(current_map))
            actual = np.asarray(
                [float(current_map.get(player_id, 0.0)) for player_id in player_ids]
            )
            predicted = np.asarray(
                [float(previous_map.get(player_id, 0.0)) for player_id in player_ids]
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
                    "previous_game_id": str(previous_game_id),
                    "game_time_utc": current["game_time_utc"].iloc[0],
                    "allocation_total_variation": float(0.5 * absolute_error.sum()),
                    "brier_score": float(squared_error.sum()),
                    "player_share_mae": float(absolute_error.mean()),
                    "player_share_mse": float(squared_error.mean()),
                    "cross_entropy": cross_entropy,
                    "has_zero_probability_active_player": bool(
                        zero_probability_active_player.any()
                    ),
                    "new_player_actual_share": float(
                        actual[[player_id not in previous_map for player_id in player_ids]].sum()
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
                        "previous_game_id": str(previous_game_id),
                        "game_time_utc": current["game_time_utc"].iloc[0],
                        "player_id": int(player_id),
                        "player_name": str(names.loc[player_id]),
                        "actual_minute_share": float(actual_share),
                        "predicted_minute_share": float(predicted_share),
                        "absolute_error": float(abs(actual_share - predicted_share)),
                        "was_in_previous_game": player_id in previous_map,
                        "was_in_current_game": player_id in current_map,
                    }
                )
    predictions = pd.DataFrame.from_records(records)
    metrics = pd.DataFrame.from_records(team_game_metrics)
    if predictions.empty or metrics.empty:
        raise ValueError("L1-MSP requires at least two completed team-games")
    return predictions, metrics


def summarize_l1_metrics(metrics: pd.DataFrame, *, season: str) -> pd.DataFrame:
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
                "mean_new_player_actual_share": float(metrics["new_player_actual_share"].mean()),
                "mean_departed_player_predicted_share": float(
                    metrics["departed_player_predicted_share"].mean()
                ),
            }
        ]
    )


def run_l1_minute_share_persistence(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> L1MinuteShareRun:
    """Materialize parameter-free L1-MSP predictions and holdout metrics."""

    if not seasons:
        raise ValueError("At least one season is required")
    run_id = f"l1-msp-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[pd.DataFrame] = []
    for season in seasons:
        game_minutes = read_regular_game_minutes(season, curated_dir=curated_dir)
        predictions, metrics = evaluate_l1_minute_share_persistence(game_minutes)
        predictions.assign(season=season).to_parquet(
            run_dir / f"{season}_game_predictions.parquet", index=False
        )
        metrics.assign(season=season).to_parquet(
            run_dir / f"{season}_team_game_metrics.parquet", index=False
        )
        summaries.append(summarize_l1_metrics(metrics, season=season))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_parquet(run_dir / "metrics.parquet", index=False)
    metadata = {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seasons": list(seasons),
        "contract": "previous completed team-game minute-share copied without fitted parameters",
        "target": "player minutes divided by actual total team minutes, including overtime",
        "cold_start": "zero predicted share for a player absent from the previous team-game",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return L1MinuteShareRun(run_dir=run_dir, run_id=run_id)


def main() -> None:
    """Run L1-MSP over one or more regular seasons."""

    parser = argparse.ArgumentParser(description="Evaluate Lag-1 Minute-Share Persistence")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_l1_minute_share_persistence(
        seasons=tuple(args.seasons),
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
