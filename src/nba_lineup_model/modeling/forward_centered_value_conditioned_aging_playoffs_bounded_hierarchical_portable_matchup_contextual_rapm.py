"""Value-conditioned aging HPM trained with completed historical playoffs."""

from __future__ import annotations

import argparse
from pathlib import Path

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

MODEL_NAME = (
    "forward_centered_value_conditioned_aging_playoffs_"
    "bounded_hierarchical_portable_matchup_contextual_rapm"
)
RUN_PREFIX = (
    "forward-centered-value-conditioned-aging-playoffs-bounded-hierarchical-"
    "portable-matchup-contextual-rapm"
)


def train_value_conditioned_aging_playoffs_hpm(
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
    """Train HPM with each completed season's regular season and playoffs."""

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        include_historical_playoffs=True,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "prior-season possession-centered value-conditioned aging-adjusted "
            "returning RAPM plus exposure-gated cold-start prior and bounded "
            "portable-matchup context; completed historical playoffs included"
        ),
        context_fit=fit_bounded_hierarchical_matchup_contextual_model,
        context_metadata=bounded_model_metadata,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train value-conditioned aging HPM with completed historical playoffs"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_value_conditioned_aging_playoffs_hpm(through_season=args.through_season)
    print(f"Forward value-conditioned aging HPM with playoffs: run={run.run_dir}")


if __name__ == "__main__":
    main()
