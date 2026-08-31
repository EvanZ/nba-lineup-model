"""Production-identical NAIL candidate with lead-secondary usage allocation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_LEAD_SECONDARY_USAGE_GAP,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import (
    train_nail_v1212_back_to_back,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    RESIDUALIZED_LAMBDA_GRID,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    ForwardPortableMatchupContextualRapmRun,
)

MODEL_NAME = "forward_nail_rapm_lead_secondary_usage_gap"
RUN_PREFIX = "forward-nail-rapm-lead-secondary-usage-gap"


def train_nail_lead_secondary_usage_gap(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    schedule_alpha: float | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Add one frozen USG%-gap coordinate to production NAIL-RAPM v1.2.1.3."""

    return train_nail_v1212_back_to_back(
        through_season=through_season,
        context_alpha=context_alpha,
        schedule_alpha=schedule_alpha,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        context_feature_set=CONTEXT_FEATURE_SET_NAIL_LEAD_SECONDARY_USAGE_GAP,
        profile_contract_metadata_updates={
            "lead_secondary_usage_gap_contract": {
                "source": "immediately preceding completed regular season",
                "input": "shrinkage-adjusted conventional USG%",
                "lineup_formula": "max(USG%) - second_max(USG%)",
                "side_edge": "home lineup value - away lineup value",
                "interpretation": "lead-handler allocation, not a secondary-handler deficiency",
                "screen_contract": (
                    "frozen residual relationship is positive in all three target seasons"
                ),
                "superstar_control": (
                    "relationship remains positive after conditioning on the home-minus-away "
                    "maximum frozen player prior"
                ),
            }
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAIL lead-secondary usage-gap candidate")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--schedule-alpha", type=float)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_lead_secondary_usage_gap(
                    through_season=args.through_season,
                    context_alpha=args.context_alpha,
                    schedule_alpha=args.schedule_alpha,
                )
                print(f"NAIL lead-secondary usage-gap candidate: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_lead_secondary_usage_gap(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
        schedule_alpha=args.schedule_alpha,
    )
    print(f"NAIL lead-secondary usage-gap candidate: run={run.run_dir}")


if __name__ == "__main__":
    main()
