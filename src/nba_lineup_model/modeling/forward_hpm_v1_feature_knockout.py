"""Frozen HPM v1 contextual feature-family knockout candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import V1_KNOCKOUT_EXCLUSIONS
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


def train_hpm_v1_feature_knockout(
    family: str,
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train HPM v1 with one named contextual feature family removed."""

    feature_set = _feature_set_for_family(family)
    slug = family.replace("_", "-")
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=DEFAULT_CONTEXT_ALPHA,
        context_curvature_alpha=DEFAULT_CONTEXT_CURVATURE_ALPHA,
        context_temporal_alpha=DEFAULT_CONTEXT_TEMPORAL_ALPHA,
        model_name=f"forward_hpm_v1_without_{family}",
        run_prefix=f"forward-hpm-v1-without-{slug}",
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            f"HPM v1 player prior with one frozen contextual feature-family knockout: {family}"
        ),
        context_fit=fit_bounded_hierarchical_matchup_contextual_model,
        context_metadata=bounded_model_metadata,
        context_feature_set=feature_set,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def _feature_set_for_family(family: str) -> str:
    feature_set = f"v1_without_{family}"
    if feature_set not in V1_KNOCKOUT_EXCLUSIONS:
        choices = ", ".join(_families())
        raise ValueError(f"Unknown HPM v1 feature family {family!r}; choose one of: {choices}")
    return feature_set


def _families() -> list[str]:
    return sorted(value.removeprefix("v1_without_") for value in V1_KNOCKOUT_EXCLUSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an HPM v1 feature-family knockout")
    parser.add_argument("family", choices=_families())
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_hpm_v1_feature_knockout(args.family, through_season=args.through_season)
    print(f"HPM v1 feature knockout ({args.family}): run={run.run_dir}")


if __name__ == "__main__":
    main()
