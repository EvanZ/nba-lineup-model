"""Forward portable-matchup contextual RAPM with P-spline temporal pooling."""

from __future__ import annotations

import argparse
from pathlib import Path

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

MODEL_NAME = "forward_hierarchical_pspline_contextual_rapm"
RUN_PREFIX = "forward-hierarchical-pspline-contextual-rapm"
DEFAULT_CONTEXT_CURVATURE_ALPHA = 1_000.0
DEFAULT_CONTEXT_TEMPORAL_ALPHA = DEFAULT_CONTEXT_ALPHA


def train_forward_hierarchical_pspline_contextual_rapm(
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
    """Fit recursive RAPM with P-spline shape and season-to-season priors.

    The underlying total context remains the portable-matchup contract:
    ``C(A, B) = h(A) - h(B) + q(A, B)``. The first contextual season has no
    temporal predecessor; every later season projects the immediately prior
    completed response functions onto its own spline basis before fitting.
    """

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=context_curvature_alpha,
        context_temporal_alpha=context_temporal_alpha,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    """Train the P-spline hierarchical contextual RAPM exemplar."""

    parser = argparse.ArgumentParser(
        description="Train forward portable-matchup contextual RAPM with P-spline pooling"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument(
        "--context-curvature-alpha",
        type=float,
        default=DEFAULT_CONTEXT_CURVATURE_ALPHA,
    )
    parser.add_argument(
        "--context-temporal-alpha",
        type=float,
        default=DEFAULT_CONTEXT_TEMPORAL_ALPHA,
    )
    args = parser.parse_args()
    run = train_forward_hierarchical_pspline_contextual_rapm(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
        context_curvature_alpha=args.context_curvature_alpha,
        context_temporal_alpha=args.context_temporal_alpha,
    )
    print(f"Forward hierarchical P-spline contextual RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
