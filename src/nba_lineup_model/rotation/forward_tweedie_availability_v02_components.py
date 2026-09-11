"""Test a Tweedie availability loss-cost model with the v0.2 input blocks.

The candidate uses the same strict-forward information blocks as Forward
Availability v0.2: pooled age baseline, carried player residual, and lagged
workload adjustment.  It differs only in the annual-loss likelihood and link.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    DEFAULT_FROZEN_SEASONS,
    DEFAULT_PLAYER_PANEL_PATH,
    DEFAULT_TUNING_SEASONS,
    build_availability_season_summary,
    predict_availability_season,
)
from nba_lineup_model.rotation.forward_tweedie_availability_loss_cost import (
    DEFAULT_TWEEDIE_POWER_GRID,
    PRODUCTION_AVAILABILITY_CONFIG,
    TweedieAvailabilityLossCostRun,
    _attach_loss_columns,
    predict_tweedie_loss_cost,
    summarize_tweedie_loss_cost_metrics,
    tune_tweedie_power,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/rotation/forward_tweedie_availability_v02_components")
COMPONENT_COLUMNS = (
    "age_baseline_logit",
    "carried_state_residual_logit",
    "carried_workload_adjustment_logit",
)


def build_v02_component_panel(summary: pd.DataFrame) -> pd.DataFrame:
    """Return strict-forward v0.2 component features for each predictable season."""

    seasons = (
        summary.loc[:, ["season", "season_start_year"]]
        .drop_duplicates()
        .sort_values("season_start_year", kind="stable")
    )
    first_year = int(seasons["season_start_year"].min())
    rows: list[pd.DataFrame] = []
    for season, year in seasons.itertuples(index=False):
        if int(year) == first_year:
            continue
        prediction, _baseline, _metadata = predict_availability_season(
            summary,
            target_season=str(season),
            config=PRODUCTION_AVAILABILITY_CONFIG,
        )
        rows.append(_attach_loss_columns(prediction))
    if not rows:
        raise ValueError("v0.2 component panel requires at least two observed seasons")
    return pd.concat(rows, ignore_index=True)


def run_tweedie_availability_v02_components(
    *,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
) -> TweedieAvailabilityLossCostRun:
    """Tune and compare Tweedie loss cost using all three v0.2 information blocks."""

    final_year = max(int(season[:4]) for season in (*tuning_seasons, *frozen_seasons))
    seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year + 1))
    summary = build_availability_season_summary(
        seasons,
        curated_dir=curated_dir,
        player_panel_path=player_panel_path,
    )
    component_panel = build_v02_component_panel(summary)
    power, power_grid = tune_tweedie_power(
        component_panel,
        target_seasons=tuning_seasons,
        power_grid=DEFAULT_TWEEDIE_POWER_GRID,
        feature_columns=COMPONENT_COLUMNS,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"tweedie-availability-v02-components-{timestamp}-{uuid4().hex[:7]}"
    run_dir = Path(artifacts_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(run_dir / "player_season_summary.parquet", index=False)
    component_panel.to_parquet(run_dir / "forward_component_panel.parquet", index=False)
    power_grid.to_parquet(run_dir / "tweedie_power_tuning_grid.parquet", index=False)

    comparison_rows: list[dict[str, float | str]] = []
    candidate_predictions: list[pd.DataFrame] = []
    production_predictions: list[pd.DataFrame] = []
    for season in frozen_seasons:
        candidate, model = predict_tweedie_loss_cost(
            component_panel,
            target_season=season,
            power=power,
            feature_columns=COMPONENT_COLUMNS,
        )
        candidate.to_parquet(run_dir / f"{season}_predictions.parquet", index=False)
        candidate_predictions.append(candidate)
        coefficients = dict(zip(model.feature_names, model.feature_weights, strict=True))
        comparison_rows.append(
            {
                "season": season,
                "model": "Tweedie loss cost with v0.2 inputs",
                "tweedie_intercept": model.intercept,
                "tweedie_age_baseline_weight": coefficients["age_baseline_logit"],
                "tweedie_state_residual_weight": coefficients[
                    "carried_state_residual_logit"
                ],
                "tweedie_workload_adjustment_weight": coefficients[
                    "carried_workload_adjustment_logit"
                ],
                "tweedie_power": model.power,
                **summarize_tweedie_loss_cost_metrics(candidate),
            }
        )

        production, _baseline, _metadata = predict_availability_season(
            summary,
            target_season=season,
            config=PRODUCTION_AVAILABILITY_CONFIG,
        )
        production = _attach_loss_columns(production)
        production.to_parquet(run_dir / f"{season}_production_v02_predictions.parquet", index=False)
        production_predictions.append(production)
        comparison_rows.append(
            {
                "season": season,
                "model": "Forward availability v0.2",
                **summarize_tweedie_loss_cost_metrics(production),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["season", "weighted_brier"], kind="stable"
    )
    comparison.to_parquet(run_dir / "frozen_model_comparison.parquet", index=False)
    comparison.to_csv(run_dir / "frozen_model_comparison.csv", index=False)
    pooled = pd.DataFrame(
        [
            {
                "model": "Tweedie loss cost with v0.2 inputs",
                **summarize_tweedie_loss_cost_metrics(
                    pd.concat(candidate_predictions, ignore_index=True)
                ),
            },
            {
                "model": "Forward availability v0.2",
                **summarize_tweedie_loss_cost_metrics(
                    pd.concat(production_predictions, ignore_index=True)
                ),
            },
        ]
    ).sort_values("weighted_brier", kind="stable")
    pooled.to_parquet(run_dir / "pooled_model_comparison.parquet", index=False)
    pooled.to_csv(run_dir / "pooled_model_comparison.csv", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "forward_tweedie_availability_v02_components",
                "version": "v0.1",
                "run_id": run_id,
                "source_availability_config": asdict(PRODUCTION_AVAILABILITY_CONFIG),
                "feature_columns": list(COMPONENT_COLUMNS),
                "selected_tweedie_power": power,
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "target": "annual unavailable rostered player-games",
                "contract": (
                    "log-link Tweedie annual loss cost with a log rostered-game offset; "
                    "the exact age, carried-state, and workload input blocks from v0.2"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return TweedieAvailabilityLossCostRun(run_dir=run_dir, run_id=run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Tweedie availability loss cost with v0.2 input components"
    )
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    run = run_tweedie_availability_v02_components(
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel_path,
        artifacts_dir=args.artifacts_dir,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
