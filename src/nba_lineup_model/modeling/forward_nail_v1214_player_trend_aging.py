"""NAIL-RAPM v1.2.1.4: value-conditioned aging with a lagged player trend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_TARGET_SEASON,
)
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import (
    train_nail_v1212_back_to_back,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    RESIDUALIZED_LAMBDA_GRID,
)
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    ForwardPortableMatchupContextualRapmRun,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    GAP_RETURNER_METHOD,
    build_centered_value_conditioned_trend_aging_gap_returner_priors,
)

MODEL_NAME = "forward_nail_rapm_v1214_player_trend_aging"
RUN_PREFIX = "forward-nail-rapm-v1214-player-trend-aging"


def train_nail_v1214_player_trend_aging(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    context_alpha: float = DEFAULT_CONTEXT_ALPHA,
    schedule_alpha: float | None = None,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Train the production NAIL contract with only lagged trend aging added."""

    return train_nail_v1212_back_to_back(
        through_season=through_season,
        context_alpha=context_alpha,
        schedule_alpha=schedule_alpha,
        player_lambda_mode="residualized_cv",
        residualized_lambda_grid=RESIDUALIZED_LAMBDA_GRID,
        player_prior_builder=build_centered_value_conditioned_trend_aging_gap_returner_priors,
        player_prior_description=(
            "NAIL-RAPM v1.2.1.4 value-conditioned aging with a strictly lagged "
            "player RAPM trend, exposure-gated cold starts, "
            f"{GAP_RETURNER_METHOD}, standard USG%, and a lagged home-minus-away "
            "back-to-back schedule adjustment"
        ),
        profile_contract_metadata_updates={
            "player_trend_aging_contract": {
                "signal": "completed prior RAPM minus completed two-seasons-prior RAPM",
                "availability": "only players observed in both consecutive completed seasons",
                "reliability": (
                    "change multiplied by log(1 + min(prior, two-seasons-prior possessions))"
                ),
                "regularization": "selected forward aging Ridge regularization",
            }
        },
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NAIL-RAPM v1.2.1.4 player-trend aging candidate"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--context-alpha", type=float, default=DEFAULT_CONTEXT_ALPHA)
    parser.add_argument("--schedule-alpha", type=float)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    kwargs = {
        "through_season": args.through_season,
        "context_alpha": args.context_alpha,
        "schedule_alpha": args.schedule_alpha,
    }
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_v1214_player_trend_aging(**kwargs)
                print(f"NAIL-RAPM v1.2.1.4 player trend aging: run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_v1214_player_trend_aging(**kwargs)
    print(f"NAIL-RAPM v1.2.1.4 player trend aging: run={run.run_dir}")


if __name__ == "__main__":
    main()
