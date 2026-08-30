"""Controlled NAIL candidate replacing top-two assists with teammate continuity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_TEAMMATE_CONTINUITY_REPLACEMENT,
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

MODEL_NAME = "forward_nail_rapm_teammate_continuity_replacement"
RUN_PREFIX = "forward-nail-rapm-teammate-continuity-replacement"


def train_nail_teammate_continuity_replacement(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    schedule_alpha: float | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Replace top-two assists with frozen prior-season pair continuity."""

    return train_nail_v1212_back_to_back(
        through_season=through_season,
        context_alpha=context_alpha,
        schedule_alpha=schedule_alpha,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        context_feature_set=CONTEXT_FEATURE_SET_NAIL_TEAMMATE_CONTINUITY_REPLACEMENT,
        use_prior_teammate_continuity=True,
        profile_contract_metadata_updates={
            "nonadditive_replacement_contract": {
                "retained": ["usage_concentration", "prior_teammate_continuity"],
                "removed": ["top_two_assists"],
            },
            "prior_teammate_continuity_contract": {
                "source": "immediately preceding regular season",
                "pair_exposure": "same-unit shared stint possessions",
                "pair_transform": "log(1 + shared_possessions)",
                "lineup_aggregation": "unweighted mean over all ten teammate pairs",
                "missing_pair_value": 0.0,
                "portability": "prediction-only relationship feature",
            },
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the NAIL teammate-continuity replacement candidate"
    )
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
                run = train_nail_teammate_continuity_replacement(
                    through_season=args.through_season,
                    context_alpha=args.context_alpha,
                    schedule_alpha=args.schedule_alpha,
                )
                print(f"NAIL continuity-replacement candidate: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_teammate_continuity_replacement(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
        schedule_alpha=args.schedule_alpha,
    )
    print(f"NAIL continuity-replacement candidate: run={run.run_dir}")


if __name__ == "__main__":
    main()
