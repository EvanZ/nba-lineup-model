"""NAIL-RAPM v1.2.1: retain only resolved non-additive lineup features."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
)
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    GAP_RETURNER_METHOD,
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_linear_ridge_matchup_contextual_model,
    model_metadata,
)

MODEL_NAME = "forward_nail_rapm_v121_pruned_nonadditive"
RUN_PREFIX = "forward-nail-rapm-v121-pruned-nonadditive"


def train_nail_v121_pruned_nonadditive(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train the v1.2 controlled non-additive feature ablation."""
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_gap_returner_priors,
        player_prior_description=(
            "NAIL-RAPM v1.2 value-conditioned aging and exposure-gated cold starts, "
            f"with {GAP_RETURNER_METHOD}; retain only usage concentration and "
            "top-two assists from v1.2's six non-additive lineup features"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
        profile_builder=partial(
            build_contextual_player_profiles,
            padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
            use_last_observed_profile=True,
        ),
        profile_contract_metadata={
            **MEDVEDOVSKY_2020_PROFILE_PADDING.metadata(),
            "gap_returner_profile_method": "last_observed_padded_profile",
            "nonadditive_contract": {
                "retained": ["top_two_assists", "usage_concentration"],
                "removed": [
                    "bottom_two_three_pm",
                    "credible_shooter_count",
                    "shooting_usage_interaction",
                    "shooter_passing_interaction",
                ],
            },
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAIL-RAPM v1.2.1 pruned non-additive model")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_v121_pruned_nonadditive(
                    through_season=args.through_season,
                    context_alpha=args.context_alpha,
                )
                print(f"NAIL-RAPM v1.2.1 pruned non-additive: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_v121_pruned_nonadditive(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"NAIL-RAPM v1.2.1 pruned non-additive: run={run.run_dir}")


if __name__ == "__main__":
    main()
