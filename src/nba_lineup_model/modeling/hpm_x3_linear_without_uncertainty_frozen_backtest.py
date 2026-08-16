"""Frozen comparison of compiled-linear HPM x3 with and without uncertainty context."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge import MODEL_NAME as FULL_X3_MODEL_NAME
from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty import (
    MODEL_NAME as ABLATED_MODEL_NAME,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/analysis/hpm_x3_linear_without_uncertainty_frozen")


def run_hpm_x3_linear_without_uncertainty_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay the exact full-versus-uncertainty-free model pair without refitting."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(FULL_X3_MODEL_NAME, "Compiled-additive linear HPM x3"),
            BacktestModel(ABLATED_MODEL_NAME, "Compiled-linear HPM x3 without uncertainty"),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen compiled-linear HPM x3 uncertainty ablation"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_hpm_x3_linear_without_uncertainty_frozen_backtest(
        seasons=tuple(args.seasons)
    )
    print(f"Linear HPM x3 uncertainty ablation: run={run.run_dir}")


if __name__ == "__main__":
    main()
