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
    parser.add_argument("--all-seasons", action="store_true")
    args = parser.parse_args()
    root = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / DISPLAY_SEASON
    run_id = str(json.loads((root / "latest.json").read_text())["run_id"])
    contexts = joblib.load(root / run_id / "season_context_models.joblib")
    seasons = sorted(contexts) if args.all_seasons else [args.season]
    for season in seasons:
        model = contexts[season]
        if not isinstance(model, MatchupContextualModel):
            raise ValueError("Selected artifact has an incompatible contextual model")
        output = response_cache_path(MODEL_ARTIFACT, run_id, season)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(_warm_response_cache(model), output)
        print(f"Materialized response cache: {output}")
    # The default-run path preserves the existing single-season API startup contract.
    if args.season == DISPLAY_SEASON:
        output = response_cache_path(MODEL_ARTIFACT, run_id)
        joblib.dump(_warm_response_cache(contexts[DISPLAY_SEASON]), output)
    context_output = player_context_exposure_path(MODEL_ARTIFACT, run_id)
    if not context_output.is_file():
        ratings = pd.read_parquet(root / run_id / "player_season_ratings.parquet")
        warm_player_context_exposure(
            contexts,
            ratings,
            panel_path=Path("data/analytical/player_season_panel/player_seasons.parquet"),
        ).to_parquet(context_output, index=False)
        print(f"Materialized player context exposure: {context_output}")


if __name__ == "__main__":
    main()
