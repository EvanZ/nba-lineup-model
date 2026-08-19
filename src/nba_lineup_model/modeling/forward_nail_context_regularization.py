"""NAIL-RAPM v1.1 candidates with season-normalized context regularization."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
)
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_aging_player_prior import (
    build_centered_value_conditioned_aging_exposure_gated_priors,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
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
    fit_normalized_linear_ridge_matchup_contextual_model,
    model_metadata,
)

BASE_CONTEXT_LAMBDA = 0.044
DEFAULT_CONTEXT_LAMBDA_MULTIPLIERS = (
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
DEFAULT_CONTEXT_LAMBDAS = tuple(
    BASE_CONTEXT_LAMBDA * multiplier
    for multiplier in DEFAULT_CONTEXT_LAMBDA_MULTIPLIERS
)


def context_lambda_slug(context_lambda: float) -> str:
    """Return a stable filesystem-safe representation of one context lambda."""

    if context_lambda <= 0:
        raise ValueError("Context lambda must be positive")
    return f"{context_lambda:.8g}".replace(".", "p").replace("-", "m")


def model_name_for_context_lambda(context_lambda: float) -> str:
    return f"forward_nail_rapm_v11_normalized_context_lambda_{context_lambda_slug(context_lambda)}"


def run_prefix_for_context_lambda(context_lambda: float) -> str:
    return f"forward-nail-rapm-v11-normalized-context-lambda-{context_lambda_slug(context_lambda)}"


def raw_context_alpha_slug(context_alpha: float) -> str:
    """Return a stable filesystem-safe raw-alpha representation."""

    if context_alpha <= 0:
        raise ValueError("Context alpha must be positive")
    return f"{context_alpha:.8g}".replace(".", "p").replace("-", "m")


def model_name_for_raw_context_alpha(context_alpha: float) -> str:
    return f"forward_nail_rapm_v11_fixed_context_alpha_{raw_context_alpha_slug(context_alpha)}"


def run_prefix_for_raw_context_alpha(context_alpha: float) -> str:
    return f"forward-nail-rapm-v11-fixed-context-alpha-{raw_context_alpha_slug(context_alpha)}"


def train_nail_normalized_context_regularization(
    *,
    context_lambda: float,
    through_season: str = DEFAULT_TARGET_SEASON,
    evaluate_target: bool = True,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train NAIL v1.1 with context loss normalized by seasonal weight."""

    if context_lambda <= 0:
        raise ValueError("Context lambda must be positive")
    profile_builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        evaluate_target=evaluate_target,
        context_alpha=context_lambda,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=model_name_for_context_lambda(context_lambda),
        run_prefix=run_prefix_for_context_lambda(context_lambda),
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts, "
            "NAIL v1.1 stat-specific profile padding, and season-normalized "
            f"linear context regularization lambda_C={context_lambda:.8g}"
        ),
        context_fit=fit_normalized_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        profile_builder=profile_builder,
        profile_contract_metadata={
            **MEDVEDOVSKY_2020_PROFILE_PADDING.metadata(),
            "context_regularization_contract": "mean_weighted_loss",
            "context_lambda": context_lambda,
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def train_nail_fixed_context_regularization(
    *,
    context_alpha: float,
    through_season: str = DEFAULT_TARGET_SEASON,
    evaluate_target: bool = True,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train NAIL v1.1 while changing only the fixed raw context alpha."""

    if context_alpha <= 0:
        raise ValueError("Context alpha must be positive")
    profile_builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        evaluate_target=evaluate_target,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=model_name_for_raw_context_alpha(context_alpha),
        run_prefix=run_prefix_for_raw_context_alpha(context_alpha),
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts, "
            "NAIL v1.1 stat-specific profile padding, and fixed raw linear "
            f"context alpha={context_alpha:.8g}"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        profile_builder=profile_builder,
        profile_contract_metadata={
            **MEDVEDOVSKY_2020_PROFILE_PADDING.metadata(),
            "context_regularization_contract": "weighted_sum_loss",
            "context_raw_alpha": context_alpha,
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NAIL v1.1 season-normalized context-penalty candidates"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-lambda", type=float, action="append")
    parser.add_argument("--raw-alpha", type=float, action="append")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--skip-target-evaluation", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.grid, args.context_lambda, args.raw_alpha)) > 1:
        parser.error("Use exactly one of --grid, --context-lambda, or --raw-alpha")
    if args.raw_alpha:
        for context_alpha in args.raw_alpha:
            run = train_nail_fixed_context_regularization(
                context_alpha=context_alpha,
                through_season=args.through_season,
                evaluate_target=not args.skip_target_evaluation,
            )
            print(
                f"NAIL fixed context alpha={context_alpha:.8g}: run={run.run_dir}",
                flush=True,
            )
        return
    values = DEFAULT_CONTEXT_LAMBDAS if args.grid else tuple(args.context_lambda or ())
    if not values:
        parser.error("Provide --grid or at least one --context-lambda")
    for context_lambda in values:
        run = train_nail_normalized_context_regularization(
            context_lambda=context_lambda,
            through_season=args.through_season,
            evaluate_target=not args.skip_target_evaluation,
        )
        print(
            f"NAIL normalized context lambda={context_lambda:.8g}: run={run.run_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
