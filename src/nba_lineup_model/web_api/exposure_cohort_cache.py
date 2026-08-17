"""Materialize the compact exposure cohort required by historical Lab queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.web_api.inference import (
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    exposure_cohort_path,
)


def main() -> None:
    """Build the historical exposure cache from local RAPM stints once per model run."""

    parser = argparse.ArgumentParser(description="Build NBA GESTALT exposure cohort cache")
    parser.add_argument("--season", default=DISPLAY_SEASON)
    parser.add_argument(
        "--panel-path",
        default="data/analytical/player_season_panel/player_seasons.parquet",
    )
    args = parser.parse_args()

    season = str(args.season)
    latest_path = Path("artifacts/models") / MODEL_ARTIFACT / season / "latest.json"
    run_id = str(json.loads(latest_path.read_text())["run_id"])
    panel = pd.read_parquet(args.panel_path)
    cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(season)],
        through_season=season,
    )
    output = exposure_cohort_path(MODEL_ARTIFACT, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output, index=False)
    print(f"Wrote {len(cohort):,} exposure rows to {output}")


if __name__ == "__main__":
    main()
