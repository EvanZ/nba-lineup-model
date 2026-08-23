"""Materialize realized player profiles for retrospective NBA GESTALT seasons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.web_api.inference import (
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    LineupEvaluator,
    build_contextual_player_profiles,
    historical_realized_profiles_path,
)


def main() -> None:
    """Build one completed-season profile table for every available Lab season."""

    parser = argparse.ArgumentParser(
        description="Build realized historical profiles for the NBA GESTALT Lineup Lab"
    )
    parser.add_argument("--season", default=DISPLAY_SEASON)
    parser.add_argument(
        "--panel-path",
        default="data/analytical/player_season_panel/player_seasons.parquet",
    )
    args = parser.parse_args()

    evaluator = LineupEvaluator.from_latest_artifact(
        season=str(args.season), panel_path=Path(args.panel_path)
    )
    frames: list[pd.DataFrame] = []
    seasons = sorted(evaluator.available_lab_seasons(), key=lambda value: int(value[:4]))
    for index, season in enumerate(seasons, start=1):
        print(
            f"Materializing realized profiles for {season} ({index}/{len(seasons)})",
            flush=True,
        )
        player_ids = evaluator.historical_coefficients.loc[
            evaluator.historical_coefficients["season"].eq(season), "player_id"
        ].astype(int)
        profiles = build_contextual_player_profiles(
            evaluator.player_season_panel,
            target_season=season,
            target_player_ids=player_ids,
            exposure_cohort=evaluator.exposure_cohort.loc[
                evaluator.exposure_cohort["season"].astype(str).le(season)
            ],
            padding_contract=evaluator.profile_padding_contract,
            profile_timing="realized",
        )
        profiles.insert(0, "season", season)
        frames.append(profiles)

    output = historical_realized_profiles_path(MODEL_ARTIFACT, evaluator.run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.concat(frames, ignore_index=True)
    if cached.duplicated(["season", "player_id"]).any():
        raise ValueError("Realized profile cache must be unique by season and player")
    cached.to_parquet(output, index=False)
    print(f"Wrote {len(cached):,} realized profile rows to {output}")


if __name__ == "__main__":
    main()
