"""Frozen three-season evaluation for the additive profile-prior RAPM experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.forward_additive_profile_prior_rapm import MODEL_NAME
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/analysis/additive_profile_prior_frozen")


def run_additive_profile_prior_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay the player-prior-only candidate without refitting any season."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                MODEL_NAME,
                "Additive profile-prior RAPM",
                uses_context=False,
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen multi-season backtest for additive profile-prior RAPM"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_additive_profile_prior_frozen_backtest(seasons=tuple(args.seasons))
    print(f"Additive profile-prior frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
