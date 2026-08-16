"""Compiled-linear HPM x3 ablation without profile-imputation context terms."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY
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
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_linear_ridge_matchup_contextual_model,
    model_metadata,
)

MODEL_NAME = "forward_hpm_x3_linear_ridge_without_uncertainty"
RUN_PREFIX = "forward-hpm-x3-linear-ridge-without-uncertainty"


def train_hpm_x3_linear_ridge_without_uncertainty(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train HPM x3 with only its two profile-uncertainty terms removed."""

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts and the "
            "compiled-linear HPM x3 feature contract without imputation-count or "
            "replacement-weight context"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train compiled-linear HPM x3 without profile-uncertainty context"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_hpm_x3_linear_ridge_without_uncertainty(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Linear HPM x3 without uncertainty context: run={run.run_dir}")


if __name__ == "__main__":
    main()
