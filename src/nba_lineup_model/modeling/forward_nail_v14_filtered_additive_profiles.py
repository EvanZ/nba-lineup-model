"""NAIL-RAPM v1.4 with a Kalman-filtered additive profile state."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import CONTEXT_FEATURE_SET_NAIL_V13
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
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    ForwardPortableMatchupContextualRapmRun,
    train_forward_portable_matchup_contextual_rapm,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    GAP_RETURNER_METHOD,
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_kalman_filtered_linear_ridge_matchup_contextual_model,
    model_metadata,
)

MODEL_NAME = "forward_nail_rapm_v14_kalman_additive_profiles"
RUN_PREFIX = "forward-nail-rapm-v14-kalman-additive-profiles"
DEFAULT_PROCESS_VARIANCE_MULTIPLIER = 1.0


class _Tee:
    """Mirror training progress to a durable local log file."""

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)  # type: ignore[union-attr]
            stream.flush()  # type: ignore[union-attr]
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()  # type: ignore[union-attr]


def train_nail_v14_kalman_additive_profiles(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    process_variance_multiplier: float = DEFAULT_PROCESS_VARIANCE_MULTIPLIER,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit v1.4 with a proper forward Kalman state on additive coefficients."""

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
            f"with {GAP_RETURNER_METHOD}; twelve additive profiles use a prior-season "
            "Kalman coefficient posterior with a diagonal random-walk process covariance"
        ),
        context_fit=fit_kalman_filtered_linear_ridge_matchup_contextual_model,
        context_metadata=model_metadata,
        context_fit_kwargs={"process_variance_multiplier": process_variance_multiplier},
        context_feature_set=CONTEXT_FEATURE_SET_NAIL_V13,
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
            "additive_kalman_contract": {
                "state": "raw additive coefficient posterior mean and covariance",
                "transition": "random walk with diagonal process covariance",
                "process_variance_multiplier": process_variance_multiplier,
                "non_additive_terms": "six independent zero-centered Ridge coefficients",
            },
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NAIL-RAPM v1.4 Kalman additive profiles"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument(
        "--process-variance-multiplier",
        type=float,
        default=DEFAULT_PROCESS_VARIANCE_MULTIPLIER,
    )
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_v14_kalman_additive_profiles(
                    through_season=args.through_season,
                    context_alpha=args.context_alpha,
                    process_variance_multiplier=args.process_variance_multiplier,
                )
                print(f"NAIL-RAPM v1.4 Kalman additive profiles: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_v14_kalman_additive_profiles(
        through_season=args.through_season,
        context_alpha=args.context_alpha,
        process_variance_multiplier=args.process_variance_multiplier,
    )
    print(f"NAIL-RAPM v1.4 Kalman additive profiles: run={run.run_dir}")


if __name__ == "__main__":
    main()
