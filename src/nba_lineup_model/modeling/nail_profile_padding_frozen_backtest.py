"""Three-season frozen comparison for NAIL profile-padding contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_hpm_x3_linear_ridge_without_uncertainty import (
    MODEL_NAME as NAIL_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_nail_profile_padding import (
    profile_builder_for_contract,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/nail_profile_padding_frozen_backtest")


def run_nail_profile_padding_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    published_builder, _, published_model, published_label = profile_builder_for_contract(
        "published", artifacts_dir=artifacts_dir
    )
    learned_builder, _, learned_model, learned_label = profile_builder_for_contract(
        "cross-season", artifacts_dir=artifacts_dir
    )
    uniform_builder, _, uniform_model, uniform_label = profile_builder_for_contract(
        "uniform-season", artifacts_dir=artifacts_dir
    )
    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(NAIL_MODEL_NAME, "NAIL-RAPM v1.0"),
            BacktestModel(
                uniform_model,
                uniform_label,
                profile_builder=uniform_builder,
            ),
            BacktestModel(
                published_model,
                published_label,
                profile_builder=published_builder,
            ),
            BacktestModel(
                learned_model,
                learned_label,
                profile_builder=learned_builder,
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen NAIL profile-padding comparison")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    run = run_nail_profile_padding_frozen_backtest(seasons=tuple(args.seasons))
    print(f"NAIL profile-padding frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
