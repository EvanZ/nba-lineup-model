"""Additive player-profile prior plus linear non-additive lineup-shape context."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.box_score_rapm import DEFAULT_REGULARIZATION_GRID
from nba_lineup_model.modeling.contextual_features import CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE
from nba_lineup_model.modeling.forward_additive_profile_prior_rapm import (
    ADDITIVE_PROFILE_PRIOR_COLUMNS,
    build_additive_profile_prior_features,
)
from nba_lineup_model.modeling.forward_box_score_hpm import ForwardBoxScoreResidualPriorBuilder
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

MODEL_NAME = "forward_additive_profile_linear_shape_context_rapm"
RUN_PREFIX = "forward-additive-profile-linear-shape-context-rapm"


def train_forward_additive_profile_linear_shape_context(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit additive player credit plus only non-additive linear unit context."""

    panel = pd.read_parquet(player_season_panel_path)
    features = build_additive_profile_prior_features(
        panel,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
    )
    builder = ForwardBoxScoreResidualPriorBuilder(
        features=features,
        regularization_grid=regularization_grid,
        feature_columns=ADDITIVE_PROFILE_PRIOR_COLUMNS,
        prior_method_suffix="lagged_hpm_additive_profile_residual",
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=builder,
        player_prior_description=(
            "strictly lagged additive HPM x3 player-profile residual prior plus "
            "linear Ridge context using only six non-additive lineup-shape features"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_X3_NONADDITIVE_SHAPE,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train additive player-profile RAPM plus linear lineup-shape context"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_forward_additive_profile_linear_shape_context(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"Additive-profile linear-shape context RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
