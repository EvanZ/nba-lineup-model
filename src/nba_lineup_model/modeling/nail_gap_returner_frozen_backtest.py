"""Three-season frozen comparison for NAIL gap-returner priors."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_nail_profile_padding import (
    PUBLISHED_MODEL_NAME as INCUMBENT_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_nail_gap_returners import MODEL_NAME as CANDIDATE_MODEL_NAME
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)


OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/nail_gap_returner_frozen_backtest")


def run_nail_gap_returner_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay v1.1 and the age-bridged returner candidate without refitting."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                INCUMBENT_MODEL_NAME,
                "NAIL-RAPM v1.1 stat-specific padding",
                profile_builder=partial(
                    build_contextual_player_profiles,
                    padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
                ),
            ),
            BacktestModel(
                CANDIDATE_MODEL_NAME,
                "NAIL-RAPM v1.2 gap-returner priors",
                profile_builder=partial(
                    build_contextual_player_profiles,
                    padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
                    use_last_observed_profile=True,
                ),
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay frozen NAIL gap-returner candidate")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_nail_gap_returner_frozen_backtest(seasons=tuple(args.seasons))
    print(f"NAIL gap-returner frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
