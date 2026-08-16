"""Linear additive HPM x3 plus explicit quadratic per-side context residuals."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE,
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
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_linear_ridge_matchup_contextual_model,
    model_metadata,
)

MODEL_NAME = "forward_hpm_x3_linear_quadratic_side_context"
RUN_PREFIX = "forward-hpm-x3-linear-quadratic-side-context"


def train_hpm_x3_linear_quadratic_side_context(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit linear player-compilable terms plus quadratic unit residuals.

    The raw additive half is algebraically transferable to player priors.  The
    squared unit-total half is retained as explicit side context:
    ``sum_f gamma_f * (X_f(home)^2 - X_f(away)^2)``.
    """

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts, with linear "
            "additive HPM x3 terms compilable into player priors and per-side quadratic "
            "residual context"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_LINEAR_X3_ADDITIVE_QUADRATIC_SIDE,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear HPM x3 plus quadratic side-residual context"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_hpm_x3_linear_quadratic_side_context(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Linear HPM x3 quadratic side context: run={run.run_dir}")


if __name__ == "__main__":
    main()
