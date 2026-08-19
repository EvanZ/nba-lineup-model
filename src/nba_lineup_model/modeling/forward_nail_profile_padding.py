"""Controlled NAIL-RAPM profile-padding candidates."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
)
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    UNIFORM_300_SEASON_PROFILE_PADDING,
    ProfilePaddingContract,
    build_contextual_player_profiles,
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
from nba_lineup_model.modeling.profile_padding_study import load_latest_padding_contract

PUBLISHED_MODEL_NAME = "forward_nail_rapm_v1_medvedovsky_padding"
PUBLISHED_RUN_PREFIX = "forward-nail-rapm-v1-medvedovsky-padding"
LEARNED_MODEL_NAME = "forward_nail_rapm_v1_cross_season_padding"
LEARNED_RUN_PREFIX = "forward-nail-rapm-v1-cross-season-padding"
UNIFORM_SEASON_MODEL_NAME = "forward_nail_rapm_v1_uniform_300_season_reference"
UNIFORM_SEASON_RUN_PREFIX = "forward-nail-rapm-v1-uniform-300-season-reference"


def train_nail_profile_padding(
    *,
    contract: str,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Change only the player-profile padding contract from canonical NAIL v1.0."""

    selected, source_metadata, model_name, run_prefix = _resolve_contract(
        contract,
        artifacts_dir=artifacts_dir,
    )
    builder = partial(build_contextual_player_profiles, padding_contract=selected)
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=model_name,
        run_prefix=run_prefix,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "value-conditioned aging RAPM plus exposure-gated cold starts and canonical "
            f"NAIL v1.0 context; profile padding={selected.name}"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        profile_builder=builder,
        profile_contract_metadata={**selected.metadata(), **source_metadata},
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def profile_builder_for_contract(
    contract: str,
    *,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> tuple[Callable[..., pd.DataFrame], dict[str, object], str, str]:
    selected, metadata, model_name, label = _resolve_contract(
        contract,
        artifacts_dir=artifacts_dir,
        labels=True,
    )
    return (
        partial(build_contextual_player_profiles, padding_contract=selected),
        metadata,
        model_name,
        label,
    )


def _resolve_contract(
    contract: str,
    *,
    artifacts_dir: Path | str,
    labels: bool = False,
) -> tuple[ProfilePaddingContract, dict[str, object], str, str]:
    if contract == "published":
        suffix = "NAIL v1.0 published stat-specific padding" if labels else PUBLISHED_RUN_PREFIX
        return (
            MEDVEDOVSKY_2020_PROFILE_PADDING,
            {"study_run_id": None, "study_run_dir": None},
            PUBLISHED_MODEL_NAME,
            suffix,
        )
    if contract == "uniform-season":
        suffix = (
            "NAIL v1.0 uniform-300 source-season anchor"
            if labels
            else UNIFORM_SEASON_RUN_PREFIX
        )
        return (
            UNIFORM_300_SEASON_PROFILE_PADDING,
            {"study_run_id": None, "study_run_dir": None},
            UNIFORM_SEASON_MODEL_NAME,
            suffix,
        )
    if contract == "cross-season":
        selected, metadata = load_latest_padding_contract(artifacts_dir)
        suffix = "NAIL v1.0 cross-season padding" if labels else LEARNED_RUN_PREFIX
        return selected, metadata, LEARNED_MODEL_NAME, suffix
    raise ValueError(
        "Profile-padding contract must be 'uniform-season', 'published', or 'cross-season'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled NAIL profile-padding candidate")
    parser.add_argument(
        "--contract",
        choices=("uniform-season", "published", "cross-season"),
        required=True,
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    args = parser.parse_args()
    run = train_nail_profile_padding(
        contract=args.contract,
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"NAIL profile-padding candidate: run={run.run_dir}")


if __name__ == "__main__":
    main()
