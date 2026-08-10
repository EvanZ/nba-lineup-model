"""Materialize the NBA GESTALT response-curve cache outside API startup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from nba_lineup_model.web_api.inference import (
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    DEFAULT_ARTIFACTS_DIR,
    _warm_response_cache,
    response_cache_path,
)
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel


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
    print(f"Materialized response cache: {output}")


if __name__ == "__main__":
    main()
