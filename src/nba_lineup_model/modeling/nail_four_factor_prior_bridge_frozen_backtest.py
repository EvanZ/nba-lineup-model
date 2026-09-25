"""Frozen production-versus-Four-Factors-prior-bridge NAIL comparison."""

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
from nba_lineup_model.modeling.forward_nail_four_factor_prior_bridge import (
    MODEL_NAME,
    ZERO_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME as PRODUCTION_MODEL
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)


OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/analysis/nail_four_factor_prior_bridge_frozen_backtest")


def run_nail_four_factor_prior_bridge_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Score unchanged NAIL, its zero bridge identity, and the learned bridge."""

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
                "NAIL-RAPM v1.2.1.3 production",
                profile_builder=profiles,
                uses_schedule_control=True,
            ),
            BacktestModel(
                ZERO_MODEL_NAME,
                "NAIL-RAPM Four Factors bridge (zero-weight parity)",
                profile_builder=profiles,
                uses_schedule_control=True,
            ),
            BacktestModel(
                MODEL_NAME,
                "NAIL-RAPM Four Factors prior bridge",
                profile_builder=profiles,
                uses_schedule_control=True,
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Four Factors prior bridge comparison")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle, redirect_stdout(handle):
            run = run_nail_four_factor_prior_bridge_frozen_backtest(seasons=tuple(args.seasons))
            print(f"NAIL Four Factors prior bridge frozen backtest: run={run.run_dir}", flush=True)
        return
    run = run_nail_four_factor_prior_bridge_frozen_backtest(seasons=tuple(args.seasons))
    print(f"NAIL Four Factors prior bridge frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
