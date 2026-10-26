#!/usr/bin/env python3
"""Calibrate game-script-aware role shrinkage from local curated marts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.modeling.game_script_minutes import (
    DEFAULT_GAME_SCRIPT_CONFIG,
    reconstruct_competitive_role_targets,
)
from nba_lineup_model.modeling.in_season_minutes import (
    REGULATION_TEAM_MINUTES,
    calibrate_conditional_prior_games,
)


def _read_parts(root: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(root.glob("part-*.parquet"))
    if not parts:
        raise ValueError(f"No parquet parts found under {root}")
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in parts],
        ignore_index=True,
    )


def _input_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    event_parts = sorted(Path(args.events_dir).glob("part-*.parquet"))
    stint_parts = sorted(Path(args.stints_dir).glob("part-*.parquet"))
    events = _read_parts(
        Path(args.events_dir),
        [
            "game_id",
            "event_index",
            "period",
            "elapsed_game_seconds",
            "score_home",
            "score_away",
        ],
    )
    stints = _read_parts(
        Path(args.stints_dir),
        [
            "game_id",
            "start_elapsed_game_seconds",
            "end_elapsed_game_seconds",
            "home_player_ids",
            "away_player_ids",
        ],
    )
    targets, exclusions = reconstruct_competitive_role_targets(events, stints)
    availability = pd.read_parquet(args.availability_path)
    target_columns = [
        "game_id",
        "player_id",
        "competitive_role_minutes",
        "competitive_boundary_seconds",
        "game_duration_seconds",
        "garbage_time_detected",
    ]
    calibration = availability.merge(
        targets[target_columns],
        on=["game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    minutes = pd.to_numeric(calibration["minutes"], errors="raise").astype(float)
    totals = calibration.groupby(["game_id", "team"], sort=False)["minutes"].transform(
        "sum"
    )
    fallback = minutes * (REGULATION_TEAM_MINUTES / totals)
    missing_played = minutes.gt(0.0) & calibration["competitive_role_minutes"].isna()
    calibration["competitive_role_minutes"] = calibration[
        "competitive_role_minutes"
    ].fillna(fallback).fillna(0.0)
    priors = pd.read_parquet(args.preseason_priors_path)
    result = calibrate_conditional_prior_games(
        priors,
        calibration,
        conditional_grid=tuple(float(value) for value in args.conditional_grid),
    )
    game_targets = targets.drop_duplicates("game_id")
    final_scores = (
        events.sort_values(["game_id", "elapsed_game_seconds", "event_index"], kind="stable")
        .groupby("game_id", as_index=False)
        .tail(1)
        .assign(
            final_margin=lambda frame: (
                frame["score_home"].astype(float) - frame["score_away"].astype(float)
            ).abs()
        )
    )
    duration_calibration = game_targets.merge(
        final_scores[["game_id", "final_margin"]], on="game_id", validate="one_to_one"
    )
    duration_calibration["garbage_minutes"] = (
        duration_calibration["game_duration_seconds"]
        - duration_calibration["competitive_boundary_seconds"]
    ) / 60.0
    blowouts = duration_calibration.loc[
        duration_calibration["final_margin"].ge(
            DEFAULT_GAME_SCRIPT_CONFIG.blowout_final_margin
        )
    ]
    result.update(
        {
            "schema_version": 1,
            "calibration_season": str(args.season),
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "updater_version": "game_script_role_shrinkage_v4",
            "game_script_config": vars(DEFAULT_GAME_SCRIPT_CONFIG),
            "input_lineage": {
                "events_sha256": _input_digest(event_parts),
                "lineup_stints_sha256": _input_digest(stint_parts),
                "availability_sha256": hashlib.sha256(
                    Path(args.availability_path).read_bytes()
                ).hexdigest(),
                "preseason_priors_sha256": hashlib.sha256(
                    Path(args.preseason_priors_path).read_bytes()
                ).hexdigest(),
            },
            "games_reconstructed": int(game_targets["game_id"].nunique()),
            "garbage_time_games": int(game_targets["garbage_time_detected"].sum()),
            "blowout_duration_calibration": {
                "final_margin_cutoff": DEFAULT_GAME_SCRIPT_CONFIG.blowout_final_margin,
                "games": int(len(blowouts)),
                "mean_garbage_minutes": float(blowouts["garbage_minutes"].mean()),
                "median_garbage_minutes": float(blowouts["garbage_minutes"].median()),
            },
            "reconstruction_exclusions": exclusions,
            "played_rows_using_regulation_fallback": int(missing_played.sum()),
        }
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--stints-dir", type=Path, required=True)
    parser.add_argument("--availability-path", type=Path, required=True)
    parser.add_argument("--preseason-priors-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--conditional-grid",
        nargs="+",
        type=float,
        default=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0),
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
