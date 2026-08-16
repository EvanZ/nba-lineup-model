"""HPM x3: replace the HPM v1 rebounding bundle with summed ORB% claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
)
from nba_lineup_model.modeling.forward_aging_player_prior import (
    build_centered_value_conditioned_aging_exposure_gated_priors,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_hierarchical_pspline_contextual_rapm import (
    DEFAULT_CONTEXT_CURVATURE_ALPHA,
    DEFAULT_CONTEXT_TEMPORAL_ALPHA,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.matchup_contextual import (
    bounded_model_metadata,
    fit_bounded_hierarchical_matchup_contextual_model,
)

MODEL_NAME = "forward_hpm_x3_v1_orb_claim_replacement"
RUN_PREFIX = "forward-hpm-x3-v1-orb-claim-replacement"


def train_hpm_x3(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    context_curvature_alpha: float = DEFAULT_CONTEXT_CURVATURE_ALPHA,
    context_temporal_alpha: float = DEFAULT_CONTEXT_TEMPORAL_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Roll the conditional ORB-claim replacement experiment forward."""

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts and "
            "HPM v1 context with its rebounding bundle replaced by summed player ORB% claims"
        ),
        context_fit=fit_bounded_hierarchical_matchup_contextual_model,
        context_metadata=bounded_model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HPM x3 ORB claim rebound replacement")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_hpm_x3(through_season=args.through_season)
    print(f"HPM x3 ORB claim rebound replacement: run={run.run_dir}")


if __name__ == "__main__":
    main()
