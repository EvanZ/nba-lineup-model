"""Frozen post-training test for a conditional-role active-15 gate.

The production roster squash ranks the active-15 pool by expected season
minutes, ``P(available) * E[MPG | available]``. This candidate separates that
season-volume weight from roster membership: it selects players by their
conditional minutes score, then preserves the production availability-adjusted
weight for the 240-minute team allocation and win projection inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    DEFAULT_PLAYER_PANEL_PATH,
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    ColdStartConditionalMinutesConfig,
    attach_cold_start_biographies,
)
from nba_lineup_model.rotation.forward_conditional_team_strength import (
    DEFAULT_FROZEN_SEASONS,
    build_incumbent_team_strength_summary,
)
from nba_lineup_model.rotation.forward_conditional_team_strength_roster_squash import (
    TeamStrengthRosterInputs,
    attach_incumbent_team_strength_to_roster,
    build_team_strength_roster_inputs,
    evaluate_team_strength_roster_squash,
    paired_team_bootstrap,
    summarize_roster_squash_metrics,
)
from nba_lineup_model.rotation.forward_nail_roster_squash import (
    DEFAULT_CATALOG_PATH,
    assert_full_regular_season_minute_coverage,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_MODEL_RUN_DIR,
)

MODEL_NAME = "forward_conditional_role_gate_roster_squash"
MODEL_VERSION = "v0.1"
CONDITIONAL_ROLE_SCORE = "predicted_minutes_per_available_game"
PRODUCTION_SCORE = "raw_expected_total_minutes"


@dataclass(frozen=True)
class ConditionalRoleGateRun:
    """Immutable output location for one post-training frozen replay."""

    run_dir: Path
    run_id: str


def _history_seasons(frozen_seasons: tuple[str, ...]) -> tuple[str, ...]:
    final_year = max(int(season[:4]) for season in frozen_seasons)
    return tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year + 1))


def _production_cold_start_config() -> ColdStartConditionalMinutesConfig:
    """Return the released FCM v0.3 configuration without fitting a new model."""

    return ColdStartConditionalMinutesConfig(
        alpha=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG.alpha,
        draft_pick_half_life=PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG.draft_pick_half_life,
        include_incumbent_team_strength=True,
    )


def run_forward_conditional_role_gate_roster_squash(
    *,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    nail_run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
) -> ConditionalRoleGateRun:
    """Compare the conditional-role gate against the released W0 gate.

    Both arms reuse the same frozen availability and FCM v0.3 forecasts for
    each target season. No parameter is selected, fit, or tuned in this replay.
    """

    if not frozen_seasons:
        raise ValueError("At least one frozen season is required")
    history_seasons = _history_seasons(frozen_seasons)
    print(
        f"[Conditional role gate] loading history through {history_seasons[-1]}", flush=True
    )
    base_summary = attach_cold_start_biographies(
        build_availability_season_summary(history_seasons, curated_dir=curated_dir)
    )
    panel = pd.read_parquet(player_panel_path)
    print("[Conditional role gate] building frozen incumbent team strength", flush=True)
    augmented_summary, diagnostics = build_incumbent_team_strength_summary(
        base_summary,
        panel=panel,
        seasons=history_seasons,
        curated_dir=curated_dir,
        run_dir=nail_run_dir,
    )
    cold_start_config = _production_cold_start_config()
    inputs_by_season: dict[str, TeamStrengthRosterInputs] = {}
    for season in frozen_seasons:
        print(f"[Conditional role gate] preparing {season}", flush=True)
        opening = build_team_strength_roster_inputs(season, panel=panel, curated_dir=curated_dir)
        assert_full_regular_season_minute_coverage(
            season, opening.target_game_minutes, catalog_path=catalog_path
        )
        inputs_by_season[season] = attach_incumbent_team_strength_to_roster(
            opening, team_strength_diagnostics=diagnostics
        )

    candidate_results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    control_results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for season in frozen_seasons:
        common = {
            "availability_summary": augmented_summary,
            "cold_start_config": cold_start_config,
        }
        print(f"[Conditional role gate] freezing {season} (conditional-role gate)", flush=True)
        candidate_results.append(
            evaluate_team_strength_roster_squash(
                inputs_by_season[season],
                **common,
                selection_score_column=CONDITIONAL_ROLE_SCORE,
            )
        )
        print(f"[Conditional role gate] freezing {season} (production gate)", flush=True)
        control_results.append(
            evaluate_team_strength_roster_squash(
                inputs_by_season[season],
                **common,
                selection_score_column=PRODUCTION_SCORE,
            )
        )

    candidate_predictions = pd.concat(
        [result[0] for result in candidate_results], ignore_index=True
    )
    candidate_metrics = pd.concat([result[1] for result in candidate_results], ignore_index=True)
    control_predictions = pd.concat([result[0] for result in control_results], ignore_index=True)
    control_metrics = pd.concat([result[1] for result in control_results], ignore_index=True)
    metric_columns = (
        "allocation_total_variation",
        "brier_score",
        "player_share_mae",
        "player_share_mse",
        "actual_top_rotation_overlap",
    )
    comparison = candidate_metrics.loc[:, ["season", "team_id", *metric_columns]].merge(
        control_metrics.loc[:, ["season", "team_id", *metric_columns]],
        on=["season", "team_id"],
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    for column in metric_columns:
        comparison[f"{column}_delta"] = (
            comparison[f"{column}_candidate"] - comparison[f"{column}_control"]
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"forward-conditional-role-gate-roster-squash-{timestamp}-{uuid4().hex[:7]}"
    output_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    diagnostics.to_parquet(output_dir / "incumbent_team_strength.parquet", index=False)
    candidate_predictions.to_parquet(
        output_dir / "frozen_candidate_predictions.parquet", index=False
    )
    candidate_metrics.to_parquet(output_dir / "frozen_candidate_team_metrics.parquet", index=False)
    control_predictions.to_parquet(output_dir / "frozen_control_predictions.parquet", index=False)
    control_metrics.to_parquet(output_dir / "frozen_control_team_metrics.parquet", index=False)
    comparison.to_parquet(output_dir / "frozen_comparison.parquet", index=False)
    paired_team_bootstrap(comparison).to_parquet(
        output_dir / "paired_team_bootstrap.parquet", index=False
    )
    summarize_roster_squash_metrics(candidate_metrics, frozen_seasons=frozen_seasons).to_parquet(
        output_dir / "frozen_candidate_summary.parquet", index=False
    )
    summarize_roster_squash_metrics(control_metrics, frozen_seasons=frozen_seasons).to_parquet(
        output_dir / "frozen_control_summary.parquet", index=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "frozen_seasons": list(frozen_seasons),
                "evaluation_target": "all regular-season games for each team",
                "candidate_selection_score": CONDITIONAL_ROLE_SCORE,
                "control_selection_score": PRODUCTION_SCORE,
                "availability_config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                "conditional_minutes_config": asdict(PROMOTED_CONDITIONAL_MINUTES_CONFIG),
                "cold_start_config": asdict(cold_start_config),
                "contract": (
                    "Post-training gate-only comparison. Both arms use identical frozen "
                    "availability and FCM v0.3 forecasts. The candidate selects the active-15 "
                    "pool by conditional MPG, with zero raw expected-minute players excluded; "
                    "both arms retain availability-adjusted expected minutes for the 240-minute "
                    "normalization and downstream win inputs."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ConditionalRoleGateRun(run_dir=output_dir, run_id=run_id)


def main() -> None:
    """Run the frozen post-training conditional-role gate comparison."""

    parser = argparse.ArgumentParser(
        description="Evaluate a conditional-role active-15 gate against production W0"
    )
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--nail-run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    args = parser.parse_args()
    run = run_forward_conditional_role_gate_roster_squash(
        frozen_seasons=tuple(args.frozen_seasons),
        player_panel_path=args.player_panel_path,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        catalog_path=args.catalog_path,
        nail_run_dir=args.nail_run_dir,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
