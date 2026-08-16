"""Frozen comparison of an additive player prior with and without shape context."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.forward_additive_profile_linear_shape_context import MODEL_NAME
from nba_lineup_model.modeling.forward_additive_profile_prior_rapm import (
    MODEL_NAME as ADDITIVE_PRIOR_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path(
    "artifacts/models/analysis/additive_profile_linear_shape_context_frozen"
)


def run_additive_profile_linear_shape_context_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay the controlled pair without refitting target-season state."""

    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                ADDITIVE_PRIOR_MODEL_NAME,
                "Additive profile-prior RAPM",
                uses_context=False,
            ),
            BacktestModel(
                MODEL_NAME,
                "Additive prior plus linear shape context",
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen additive-prior versus linear-shape-context comparison"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_additive_profile_linear_shape_context_frozen_backtest(
        seasons=tuple(args.seasons)
    )
    print(f"Additive-prior linear-shape frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
