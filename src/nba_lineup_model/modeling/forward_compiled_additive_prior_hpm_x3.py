"""Forward HPM x3 with learned additive profile terms carried as player prior state."""

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

MODEL_NAME = "forward_compiled_additive_prior_hpm_x3"
RUN_PREFIX = "forward-compiled-additive-prior-hpm-x3"


def train_forward_compiled_additive_prior_hpm_x3(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train the exact additive-prior-transfer HPM x3 candidate.

    Each completed season fits all fourteen x3 terms. The eight player-additive
    profile terms become next season's prior adjustment; the six unit-shape
    terms remain the carried contextual residual.
    """

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        compiled_additive_prior=True,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts, augmented by "
            "the prior season's learned additive HPM x3 basketball-profile coefficients"
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
        description="Train HPM x3 with additive profile context transferred into player priors"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_forward_compiled_additive_prior_hpm_x3(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Forward compiled-additive-prior HPM x3: run={run.run_dir}")


if __name__ == "__main__":
    main()
