"""NAIL-RAPM v1.2.1.5: clean-offseason team-change prior precision candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from nba_lineup_model.modeling.team_change_precision import (
    TEAM_CHANGE_VARIANCE_RATIO_GRID,
    build_team_change_precision_candidates,
)

MODEL_NAME = "forward_nail_rapm_v1215_team_change_precision"
RUN_PREFIX = "forward-nail-rapm-v1215-team-change-precision"


def train_nail_v1215_team_change_precision(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    schedule_alpha: float | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train the production contract with only clean-move precision selection added."""

    return train_nail_v1212_back_to_back(
        through_season=through_season,
        context_alpha=context_alpha,
        schedule_alpha=schedule_alpha,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        player_precision_candidates_builder=build_team_change_precision_candidates,
        player_prior_description=(
            "NAIL-RAPM v1.2.1.5: the v1.2.1.3 value-conditioned aging, "
            "exposure-gated cold-start, additive-profile, non-additive context, and "
            "back-to-back contract, plus clean-offseason-move prior precision jointly "
            "selected with player lambda on chronological source-season folds"
        ),
        profile_contract_metadata_updates={
            "team_change_precision_contract": {
                "eligible_transition": (
                    "one primary team in each of two consecutive completed seasons, "
                    "with different team identifiers"
                ),
                "excluded": "rookies, gaps, multi-team seasons, and in-season trades",
                "relative_precision": "1 / (1 + q * normalized_log_prior_possessions)",
                "variance_ratio_grid": list(TEAM_CHANGE_VARIANCE_RATIO_GRID),
                "selection": "joint chronological CV with the player lambda grid",
            }
        },
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NAIL-RAPM v1.2.1.5 clean-offseason team-change precision"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--schedule-alpha", type=float)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    kwargs = {
        "through_season": args.through_season,
        "context_alpha": args.context_alpha,
        "schedule_alpha": args.schedule_alpha,
    }
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_v1215_team_change_precision(**kwargs)
                print(f"NAIL-RAPM v1.2.1.5 team-change precision: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_v1215_team_change_precision(**kwargs)
    print(f"NAIL-RAPM v1.2.1.5 team-change precision: run={run.run_dir}")


if __name__ == "__main__":
    main()
