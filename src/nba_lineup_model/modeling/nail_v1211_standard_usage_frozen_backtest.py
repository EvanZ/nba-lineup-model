"""Three-season frozen comparison of NAIL-RAPM v1.2.1.1 against v1.2.1."""

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
from nba_lineup_model.modeling.forward_nail_v121_pruned_nonadditive import (
    MODEL_NAME as INCUMBENT_MODEL_NAME,
)
from nba_lineup_model.modeling.forward_nail_v1211_standard_usage import (
    MODEL_NAME as CANDIDATE_MODEL_NAME,
)
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    DEFAULT_SEASONS,
    BacktestModel,
    FrozenMultiseasonBacktestRun,
    run_frozen_multiseason_backtest,
)

OUTPUT_ARTIFACTS_DIR = Path("artifacts/models/nail_v1211_standard_usage_frozen_backtest")


def run_nail_v1211_standard_usage_frozen_backtest(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_artifacts_dir: Path | str = OUTPUT_ARTIFACTS_DIR,
) -> FrozenMultiseasonBacktestRun:
    """Replay standard-USG% NAIL against its legacy usage-rate predecessor."""

    profiles = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )
    return run_frozen_multiseason_backtest(
        seasons=seasons,
        models=(
            BacktestModel(
                INCUMBENT_MODEL_NAME,
                "NAIL-RAPM v1.2.1 legacy usage-events rate",
                profile_builder=profiles,
            ),
            BacktestModel(
                CANDIDATE_MODEL_NAME,
                "NAIL-RAPM v1.2.1.1 standard USG%",
                profile_builder=profiles,
            ),
        ),
        artifacts_dir=artifacts_dir,
        output_artifacts_dir=output_artifacts_dir,
        docs_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen standard-USG% NAIL v1.2.1.1 backtest"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle, redirect_stdout(handle):
            run = run_nail_v1211_standard_usage_frozen_backtest(
                seasons=tuple(args.seasons)
            )
            print(f"NAIL-RAPM v1.2.1.1 frozen backtest: run={run.run_dir}", flush=True)
        return
    run = run_nail_v1211_standard_usage_frozen_backtest(seasons=tuple(args.seasons))
    print(f"NAIL-RAPM v1.2.1.1 frozen backtest: run={run.run_dir}")


if __name__ == "__main__":
    main()
