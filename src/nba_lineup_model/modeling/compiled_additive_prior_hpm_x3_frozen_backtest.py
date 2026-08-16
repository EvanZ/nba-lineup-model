"""Focused frozen comparison for the recursive compiled-additive-prior HPM x3 test."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.forward_compiled_additive_prior_hpm_x3 import (
    MODEL_NAME as CANDIDATE_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty import (
    MODEL_NAME as BASELINE_MODEL_NAME,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/analysis/compiled_additive_prior_hpm_x3_frozen")


def run_compiled_additive_prior_hpm_x3_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay the canonical x3 baseline and additive-prior candidate without refitting."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                BASELINE_MODEL_NAME,
                "Canonical compiled-linear HPM x3",
            ),
            BacktestModel(
                CANDIDATE_MODEL_NAME,
                "Forward compiled-additive-prior HPM x3",
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen compiled-additive-prior HPM x3 comparison"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_compiled_additive_prior_hpm_x3_frozen_backtest(seasons=tuple(args.seasons))
    print(f"Compiled-additive-prior HPM x3 frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
