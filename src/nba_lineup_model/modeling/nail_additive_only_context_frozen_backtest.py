"""Frozen NAIL-RAPM comparison with the non-additive bundle ablated."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty import (
    MODEL_NAME as NAIL_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_nail_additive_only_context import (
    MODEL_NAME as ADDITIVE_ONLY_MODEL_NAME,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/analysis/nail_additive_only_context_frozen")


def run_nail_additive_only_context_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay the exact NAIL-versus-additive-only pair without refitting."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(NAIL_MODEL_NAME, "NAIL-RAPM v1.0"),
            BacktestModel(ADDITIVE_ONLY_MODEL_NAME, "NAIL additive-only context"),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen NAIL-RAPM additive-only-context ablation"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_nail_additive_only_context_frozen_backtest(seasons=tuple(args.seasons))
    print(f"NAIL additive-only frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
