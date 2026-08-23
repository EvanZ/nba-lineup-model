"""NAIL-RAPM v1.3.2 with forward dynamic additive-profile coefficients."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
    LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.contextual_profiles import (
    ASSISTED_SHOT_PROFILE_PADDING,
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    GAP_RETURNER_METHOD,
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_mean_reverting_linear_ridge_matchup_contextual_model,
    model_metadata,
)

MODEL_NAME = "forward_nail_rapm_v132_dynamic_additive_profiles"
RUN_PREFIX = "forward-nail-rapm-v132-dynamic-additive-profiles"
MEAN_REVERSION_PRIOR_STRENGTH = 6.0
PROCESS_VARIANCE_FLOOR_RATIO = 0.10

ADDITIVE_FEATURES = tuple(LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES)
STABLE_MATERIAL_FEATURES = frozenset(
    set(ADDITIVE_FEATURES) - {"unassisted_three_makes_per_100"}
)
REGIME_FEATURES = frozenset({"unassisted_three_makes_per_100"})
# v1.3.1 already removed the two category-4 unresolved features. Keep the
# explicit empty gate so the persisted contract records that fact.
ZERO_GATED_FEATURES = frozenset()


def train_nail_v132_dynamic_additive_profiles(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit a forward feature-specific, mean-reverting additive state."""
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=build_centered_value_conditioned_aging_gap_returner_priors,
        player_prior_description=(
            "NAIL-RAPM v1.2 value-conditioned aging and exposure-gated cold starts, "
            f"with {GAP_RETURNER_METHOD}; v1.3.1's ten additive profiles use a "
            "forward feature-specific mean-reverting empirical-Bayes state"
        ),
        context_fit=fit_mean_reverting_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_fit_kwargs={
            "additive_features": ADDITIVE_FEATURES,
            "stable_features": STABLE_MATERIAL_FEATURES,
            "regime_features": REGIME_FEATURES,
            "zero_gated_features": ZERO_GATED_FEATURES,
            "mean_reversion_prior_strength": MEAN_REVERSION_PRIOR_STRENGTH,
            "process_variance_floor_ratio": PROCESS_VARIANCE_FLOOR_RATIO,
        },
        context_feature_set=CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
        profile_builder=partial(
            build_contextual_player_profiles,
            padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
            use_last_observed_profile=True,
        ),
        profile_contract_metadata={
            **MEDVEDOVSKY_2020_PROFILE_PADDING.metadata(),
            "assisted_shot_profile_padding": {
                trait: {"pseudo_possessions": value[0], "center_mode": value[1]}
                for trait, value in ASSISTED_SHOT_PROFILE_PADDING.items()
            },
            "gap_returner_profile_method": "last_observed_padded_profile",
            "dynamic_additive_state": {
                "state": "raw additive coefficient posterior mean and covariance",
                "transition": (
                    "feature-specific mean-reverting AR(1) toward prior-only running mean"
                ),
                "stable_material_features": sorted(STABLE_MATERIAL_FEATURES),
                "regime_features": sorted(REGIME_FEATURES),
                "zero_gated_features": sorted(ZERO_GATED_FEATURES),
                "mean_reversion_prior_strength": MEAN_REVERSION_PRIOR_STRENGTH,
                "process_variance_floor_ratio": PROCESS_VARIANCE_FLOOR_RATIO,
            },
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAIL-RAPM v1.3.2 dynamic additive profiles")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_v132_dynamic_additive_profiles(
                    through_season=args.through_season,
                    context_alpha=args.context_alpha,
                )
                print(f"NAIL-RAPM v1.3.2 dynamic additive profiles: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_v132_dynamic_additive_profiles(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
    )
    print(f"NAIL-RAPM v1.3.2 dynamic additive profiles: run={run.run_dir}")


if __name__ == "__main__":
    main()
