"""Materialize player-season team exposure for the NBA GESTALT profile pages."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    build_player_team_splits,
    player_team_splits_path,
)


def main() -> None:
    """Build the current web artifact from regular-season RAPM stints."""

    parser = argparse.ArgumentParser(
        description="Build NBA GESTALT player-season team exposure splits"
    )
    parser.add_argument("--analytical-dir", default="data/analytical")
    args = parser.parse_args()
    root = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / DISPLAY_SEASON
    run_id = str(json.loads((root / "latest.json").read_text())["run_id"])
    ratings = pd.read_parquet(root / run_id / "player_season_ratings.parquet")
    output = player_team_splits_path(MODEL_ARTIFACT, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_player_team_splits(ratings, analytical_dir=args.analytical_dir).to_parquet(
        output, index=False
    )
    print(f"Materialized player-team splits: {output}")


if __name__ == "__main__":
    main()
