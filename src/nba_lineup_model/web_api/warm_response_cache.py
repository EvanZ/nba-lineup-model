"""Materialize the NBA GESTALT response-curve cache outside API startup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    _warm_response_cache,
    player_context_exposure_path,
    response_cache_path,
    warm_player_context_exposure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the NBA GESTALT response cache")
    parser.add_argument("--season", default=DISPLAY_SEASON)
    args = parser.parse_args()
    root = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / args.season
    run_id = str(json.loads((root / "latest.json").read_text())["run_id"])
    model = joblib.load(root / run_id / "season_context_models.joblib")[args.season]
    if not isinstance(model, MatchupContextualModel):
        raise ValueError("Selected artifact has an incompatible contextual model")
    output = response_cache_path(MODEL_ARTIFACT, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_warm_response_cache(model), output)
    ratings = pd.read_parquet(root / run_id / "player_season_ratings.parquet")
    contexts = joblib.load(root / run_id / "season_context_models.joblib")
    context_output = player_context_exposure_path(MODEL_ARTIFACT, run_id)
    warm_player_context_exposure(
        contexts,
        ratings,
        panel_path=Path("data/analytical/player_season_panel/player_seasons.parquet"),
    ).to_parquet(context_output, index=False)
    print(f"Materialized response cache: {output}")
    print(f"Materialized player context exposure: {context_output}")


if __name__ == "__main__":
    main()
