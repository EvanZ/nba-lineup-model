"""Paired game-block bootstrap for NAIL profile-padding candidates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty import (
    MODEL_NAME as CONTROL_MODEL,
)
from nba_lineup_model.modeling.forward_nail_profile_padding import (
    LEARNED_MODEL_NAME,
    PUBLISHED_MODEL_NAME,
    UNIFORM_SEASON_MODEL_NAME,
)
from nba_lineup_model.modeling.frozen_model_tournament import _paired_metrics

DEFAULT_BACKTEST_ROOT = Path(
    "artifacts/models/nail_profile_padding_frozen_backtest/"
    "frozen_multiseason_backtest/2023-24_to_2025-26"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/models/nail_profile_padding_bootstrap/2023-24_to_2025-26"
)
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20_260_818


def run_nail_profile_padding_bootstrap(
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    backtest_root: Path | str = DEFAULT_BACKTEST_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Persist paired regular-season uncertainty for both padding candidates."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    source_run = _latest_directory(Path(backtest_root))
    games = pd.read_parquet(source_run / "regular_game_predictions.parquet")
    possessions = pd.read_parquet(source_run / "possession_predictions.parquet")

    def source(model: str) -> dict[str, pd.DataFrame]:
        return {
            "games": games.loc[games["model"].eq(model)].copy(),
            "possessions": possessions.loc[
                possessions["model"].eq(model)
                & possessions["cohort"].eq("regular_season")
            ].copy(),
        }

    rows: list[pd.DataFrame] = []
    comparisons = (
        (CONTROL_MODEL, UNIFORM_SEASON_MODEL_NAME),
        (UNIFORM_SEASON_MODEL_NAME, PUBLISHED_MODEL_NAME),
        (UNIFORM_SEASON_MODEL_NAME, LEARNED_MODEL_NAME),
        (CONTROL_MODEL, PUBLISHED_MODEL_NAME),
        (CONTROL_MODEL, LEARNED_MODEL_NAME),
        (LEARNED_MODEL_NAME, PUBLISHED_MODEL_NAME),
    )
    for index, (incumbent, challenger) in enumerate(comparisons):
        result = _paired_metrics(
            source(incumbent),
            source(challenger),
            draws=draws,
            seed=seed + 10 * index,
        )
        result.insert(0, "incumbent_model", incumbent)
        result.insert(1, "challenger_model", challenger)
        rows.append(result)
    metrics = pd.concat(rows, ignore_index=True)

    root = Path(output_root)
    run_id = f"nail-profile-padding-bootstrap-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(run_dir / "paired_bootstrap_metrics.parquet", index=False)
    metadata = {
        "run_id": run_id,
        "source_run_dir": str(source_run),
        "draws": draws,
        "seed": seed,
        "resampling_unit": "games stratified by season",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def _latest_directory(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No frozen profile-padding runs under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap NAIL profile-padding candidates")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run_dir = run_nail_profile_padding_bootstrap(draws=args.draws, seed=args.seed)
    print(f"NAIL profile-padding bootstrap: run={run_dir}")


if __name__ == "__main__":
    main()
