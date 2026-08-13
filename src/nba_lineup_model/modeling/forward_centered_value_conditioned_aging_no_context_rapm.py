"""Controlled no-context ablation of the centered value-conditioned HPM prior."""

from __future__ import annotations

import argparse
from pathlib import Path

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

MODEL_NAME = "forward_centered_value_conditioned_aging_no_context_rapm"
RUN_PREFIX = "forward-centered-value-conditioned-aging-no-context-rapm"


def train_centered_value_conditioned_aging_no_context_rapm(
    *,
    through_season: str = DEFAULT_TARGET_SEASON,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> ForwardPortableMatchupContextualRapmRun:
    """Fit the HPM player-prior state with all contextual corrections disabled."""

    return train_forward_portable_matchup_contextual_rapm(
        through_season=through_season,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        use_context=False,
        player_prior_builder=build_centered_value_conditioned_aging_exposure_gated_priors,
        player_prior_description=(
            "prior-season possession-centered value-conditioned aging-adjusted "
            "returning RAPM plus exposure-gated cold-start prior; controlled no-context ablation"
        ),
        player_season_panel_path=player_season_panel_path,
        analytical_dir=analytical_dir,
        curated_dir=curated_dir,
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    """Run the controlled no-context HPM ablation."""

    parser = argparse.ArgumentParser(
        description="Train the centered value-conditioned aging RAPM no-context control"
    )
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run = train_centered_value_conditioned_aging_no_context_rapm(
        through_season=args.through_season
    )
    print(f"Controlled no-context RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
