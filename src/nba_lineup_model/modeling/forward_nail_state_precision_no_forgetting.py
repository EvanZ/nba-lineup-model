"""NAIL candidate with posterior-variance precision and no offseason forgetting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nba_lineup_model.modeling.forward_nail_state_precision_parity import (
    train_nail_state_precision_parity,
)
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import _Tee
from nba_lineup_model.modeling.state_precision import PlayerStatePrecisionConfig

MODEL_NAME = "forward_nail_state_precision_no_forgetting"
RUN_PREFIX = "forward-nail-state-precision-no-forgetting"
NO_FORGETTING_CONFIG = PlayerStatePrecisionConfig(process_variance_per_season=0.0)


def train_nail_state_precision_no_forgetting(**kwargs: object):
    """Fit state-aware NAIL while preserving posterior variance across seasons."""

    return train_nail_state_precision_parity(
        **kwargs,
        state_precision_config=NO_FORGETTING_CONFIG,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train no-forgetting State-Precision NAIL")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as handle:
            original_stdout = sys.stdout
            sys.stdout = _Tee(original_stdout, handle)  # type: ignore[assignment]
            try:
                run = train_nail_state_precision_no_forgetting(
                    through_season=args.through_season
                )
                print(f"State-Precision NAIL no-forgetting run={run.run_dir}")
            finally:
                sys.stdout = original_stdout
        return
    run = train_nail_state_precision_no_forgetting(through_season=args.through_season)
    print(f"State-Precision NAIL no-forgetting run={run.run_dir}")


if __name__ == "__main__":
    main()
