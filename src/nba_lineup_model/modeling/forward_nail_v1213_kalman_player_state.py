"""NAIL-RAPM v1.2.1.3: forward Kalman-filtered player RAPM priors."""

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
from nba_lineup_model.modeling.kalman_player_prior import (
    DEFAULT_PLAYER_KALMAN_CONFIG,
    PlayerKalmanConfig,
    build_centered_value_conditioned_aging_gap_returner_kalman_priors,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_linear_ridge_matchup_contextual_model,
    model_metadata,
)

MODEL_NAME = "forward_nail_rapm_v1213_kalman_player_state"
RUN_PREFIX = "forward-nail-rapm-v1213-kalman-player-state"


def train_nail_v1213_kalman_player_state(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    process_variance: float = DEFAULT_PLAYER_KALMAN_CONFIG.process_variance_per_season,
    observation_variance_scale: float = (
        DEFAULT_PLAYER_KALMAN_CONFIG.observation_variance_possession_scale
    ),
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit NAIL with a filtered player-state prior and unchanged context terms."""

    kalman_config = PlayerKalmanConfig(
        process_variance_per_season=process_variance,
        observation_variance_possession_scale=observation_variance_scale,
    )
    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        context_alpha=context_alpha,
        context_curvature_alpha=0.0,
        context_temporal_alpha=0.0,
        include_back_to_back_control=True,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_prior_builder=partial(
            build_centered_value_conditioned_aging_gap_returner_kalman_priors,
            config=kalman_config,
        ),
        player_prior_description=(
            "NAIL-RAPM v1.2.1.3 forward Kalman player state with value-conditioned "
            "aging, gap-returner bridges, exposure-gated cold starts, standard USG%, "
            "and a lagged B2B schedule adjustment"
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
            "kalman_player_prior": {
                "state": "player RAPM posterior mean and scalar variance",
                "transition": (
                    "forward value-conditioned aging projection plus random walk variance"
                ),
                "initial_variance": kalman_config.initial_variance,
                "process_variance_per_season": kalman_config.process_variance_per_season,
                "observation_variance": "observation_variance_scale / on_court_possessions",
                "observation_variance_scale": (
                    kalman_config.observation_variance_possession_scale
                ),
            },
            "gap_returner_profile_method": "last_observed_padded_profile",
            "usage_contract": {
                "column": "usage_pct",
                "definition": "conventional game-level USG%",
                "formula": (
                    "100 * (FGA + 0.44 * FTA + TOV) / "
                    "estimated_team_opportunities_on_court"
                ),
            },
            "schedule_control_contract": {
                "feature": "home_back_to_back - away_back_to_back",
                "availability": "known before tipoff from the game calendar",
                "state": "source-season weighted Ridge coefficient",
            },
            "nonadditive_contract": {
                "retained": ["top_two_assists", "usage_concentration"],
            },
        },
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NAIL-RAPM v1.2.1.3 forward Kalman player-state candidate"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--process-variance", type=float, default=1.0)
    parser.add_argument("--observation-variance-scale", type=float, default=4000.0)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    kwargs = {
        "through_season": args.through_season,
        "context_alpha": args.context_alpha,
        "process_variance": args.process_variance,
        "observation_variance_scale": args.observation_variance_scale,
    }
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_v1213_kalman_player_state(**kwargs)
                print(f"NAIL-RAPM v1.2.1.3 Kalman player state: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_v1213_kalman_player_state(**kwargs)
    print(f"NAIL-RAPM v1.2.1.3 Kalman player state: run={run.run_dir}")


if __name__ == "__main__":
    main()
