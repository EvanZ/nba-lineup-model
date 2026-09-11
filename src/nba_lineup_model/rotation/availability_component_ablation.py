"""Compare the promoted availability filter with a state-only ablation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    DEFAULT_PLAYER_PANEL_PATH,
    DEFAULT_ROSTER_DIR,
    ForwardAvailabilityConfig,
    build_availability_season_summary,
    predict_availability_season,
    summarize_availability_metrics,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/rotation/forward_availability/component_ablation")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
# Current promoted v0.2 hyperparameters. The challenger changes only the
# baseline and workload terms; filtered-state persistence and strengths match.
PRODUCTION_CONFIG = ForwardAvailabilityConfig(0.5, 60.0, 15.0, -0.25)


def run_component_ablation(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    roster_dir: Path | str = DEFAULT_ROSTER_DIR,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pooled and seasonal frozen metrics for production versus state-only."""

    final_year = max(int(season[:4]) for season in frozen_seasons)
    seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year + 1))
    summary = build_availability_season_summary(
        seasons,
        curated_dir=curated_dir,
        player_panel_path=player_panel_path,
        roster_dir=roster_dir,
    )
    variants = {
        "Production v0.2": PRODUCTION_CONFIG,
        "State only": replace(PRODUCTION_CONFIG, workload_weight=0.0, use_age_baseline=False),
    }
    pooled_rows: list[dict[str, float | str]] = []
    seasonal_rows: list[dict[str, float | str]] = []
    for model_name, config in variants.items():
        predictions: list[pd.DataFrame] = []
        for season in frozen_seasons:
            prediction, _baseline, _metadata = predict_availability_season(
                summary,
                target_season=season,
                config=config,
            )
            predictions.append(prediction)
            seasonal_rows.append(
                {
                    "model": model_name,
                    "season": season,
                    **summarize_availability_metrics(prediction),
                }
            )
        pooled_rows.append(
            {
                "model": model_name,
                **summarize_availability_metrics(pd.concat(predictions, ignore_index=True)),
            }
        )
    pooled = pd.DataFrame(pooled_rows).sort_values("weighted_brier", kind="stable")
    seasonal = pd.DataFrame(seasonal_rows).sort_values(["season", "weighted_brier"], kind="stable")
    return pooled.reset_index(drop=True), seasonal.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablate age and workload from the promoted availability filter"
    )
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--roster-dir", type=Path, default=DEFAULT_ROSTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    pooled, seasonal = run_component_ablation(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel,
        roster_dir=args.roster_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.output_dir / "pooled_frozen_metrics.csv", index=False)
    seasonal.to_csv(args.output_dir / "seasonal_frozen_metrics.csv", index=False)
    print(pooled.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
