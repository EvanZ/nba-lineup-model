"""End-to-end equal-variance parity replay for State-Precision NAIL."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
)
from nba_lineup_model.modeling.contextual_profiles import (
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
    fit_linear_ridge_matchup_contextual_model,
    model_metadata,
)
from nba_lineup_model.modeling.state_precision import PlayerStatePrecisionConfig

MODEL_NAME = "forward_nail_state_precision_parity"
RUN_PREFIX = "forward-nail-state-precision-parity"


def train_nail_state_precision_parity(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    schedule_alpha: float | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    state_precision_config: PlayerStatePrecisionConfig | None = None,
    model_name: str = MODEL_NAME,
    run_prefix: str = RUN_PREFIX,
) -> ForwardPortableMatchupContextualRapmRun:
    """Run NAIL with either parity or posterior-variance player precision."""

    state_contract = (
        {
            "enabled": True,
            "mode": "posterior_variance",
            "process_variance_per_season": state_precision_config.process_variance_per_season,
            "initial_variance": state_precision_config.initial_variance,
        }
        if state_precision_config is not None
        else {
            "enabled": True,
            "mode": "uniform_parity",
            "relative_precision": 1.0,
            "expected_equivalence": "NAIL-RAPM v1.2.1.2",
        }
    )

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        include_back_to_back_control=True,
        schedule_alpha=schedule_alpha,
        model_name=model_name,
        run_prefix=run_prefix,
        player_prior_builder=build_centered_value_conditioned_aging_gap_returner_priors,
        player_prior_description=(
            "State-Precision NAIL equal-variance parity replay: v1.2.1.2 prior, "
            f"{GAP_RETURNER_METHOD}, standard USG%, and lagged B2B control"
        ),
        context_fit=fit_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        profile_builder=partial(
            build_contextual_player_profiles,
            padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
            use_last_observed_profile=True,
        ),
        profile_contract_metadata={
            **MEDVEDOVSKY_2020_PROFILE_PADDING.metadata(),
            "state_precision_contract": state_contract,
        },
        use_player_state_precision=True,
        player_state_precision_config=state_precision_config,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run State-Precision NAIL equal-variance parity replay"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--schedule-alpha", type=float)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_state_precision_parity(
                    through_season=args.through_season,
                    context_alpha=args.context_alpha,
                    schedule_alpha=args.schedule_alpha,
                )
                print(f"State-Precision NAIL parity replay: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_state_precision_parity(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
        schedule_alpha=args.schedule_alpha,
    )
    print(f"State-Precision NAIL parity replay: run={run.run_dir}")


if __name__ == "__main__":
    main()
