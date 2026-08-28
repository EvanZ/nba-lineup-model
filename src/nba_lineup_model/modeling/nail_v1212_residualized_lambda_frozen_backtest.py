"""Frozen comparison of production NAIL and residualized-lambda NAIL."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import MODEL_NAME as PRODUCTION_MODEL
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)


OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/nail_v1212_residualized_lambda_frozen_backtest")


def run_nail_v1212_residualized_lambda_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Score the two otherwise-identical recursive artifacts on shared support."""

    profiles = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )
    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                PRODUCTION_MODEL,
                "NAIL-RAPM v1.2.1.2 production imported lambda schedule",
                profile_builder=profiles,
                uses_schedule_control=True,
            ),
            BacktestModel(
                MODEL_NAME,
                "NAIL-RAPM v1.2.1.2 residualized-target lambda CV",
                profile_builder=profiles,
                uses_schedule_control=True,
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen production-versus-residualized-lambda NAIL comparison"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle, redirect_stdout(handle):
            run = run_nail_v1212_residualized_lambda_frozen_backtest(
                seasons=tuple(args.seasons)
            )
            print(f"NAIL residualized-lambda frozen backtest: run={run.run_dir}", flush=True)
        return
    run = run_nail_v1212_residualized_lambda_frozen_backtest(seasons=tuple(args.seasons))
    print(f"NAIL residualized-lambda frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
