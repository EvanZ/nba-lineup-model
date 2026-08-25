"""Compare State-Precision NAIL's equal-variance replay to production NAIL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StatePrecisionParityAudit:
    """Maximum absolute deltas for artifacts that must match under unit precision."""

    player_coefficient_max_abs_delta: float
    player_prior_max_abs_delta: float
    possession_prediction_max_abs_delta: float
    game_prediction_max_abs_delta: float

    @property
    def max_abs_delta(self) -> float:
        return max(
            self.player_coefficient_max_abs_delta,
            self.player_prior_max_abs_delta,
            self.possession_prediction_max_abs_delta,
            self.game_prediction_max_abs_delta,
        )


def audit_state_precision_parity(
    production_run_dir: Path | str,
    parity_run_dir: Path | str,
) -> StatePrecisionParityAudit:
    """Require equivalent fitted state and predictions under unit precision."""

    production = Path(production_run_dir)
    parity = Path(parity_run_dir)
    coefficient_delta = _maximum_delta(
        production / "historical_player_coefficients.parquet",
        parity / "historical_player_coefficients.parquet",
        keys=["season", "player_id"],
        columns=["rapm", "prior_rapm", "rapm_adjustment_from_prior", "selected_lambda"],
    )
    prior_delta = _maximum_delta(
        production / "season_player_priors.parquet",
        parity / "season_player_priors.parquet",
        keys=["season", "player_id"],
        columns=["prior_rapm"],
    )
    possession_delta = _maximum_delta(
        production / "possession_predictions.parquet",
        parity / "possession_predictions.parquet",
        keys=["cohort", "game_id", "possession_id"],
        columns=["prediction_offense_margin", "prediction_home_margin"],
    )
    game_delta = _maximum_delta(
        production / "game_predictions.parquet",
        parity / "game_predictions.parquet",
        keys=["cohort", "game_id"],
        columns=["predicted_home_margin"],
    )
    return StatePrecisionParityAudit(
        player_coefficient_max_abs_delta=coefficient_delta,
        player_prior_max_abs_delta=prior_delta,
        possession_prediction_max_abs_delta=possession_delta,
        game_prediction_max_abs_delta=game_delta,
    )


def assert_state_precision_parity(
    audit: StatePrecisionParityAudit,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Fail if a unit-precision replay differs materially from production."""

    if audit.max_abs_delta > tolerance:
        raise ValueError(
            "State-Precision NAIL equal-variance parity failed: "
            f"max_abs_delta={audit.max_abs_delta:.3e}, tolerance={tolerance:.3e}"
        )


def _maximum_delta(
    production_path: Path,
    parity_path: Path,
    *,
    keys: list[str],
    columns: list[str],
) -> float:
    production = pd.read_parquet(production_path)
    parity = pd.read_parquet(parity_path)
    joined = production.loc[:, [*keys, *columns]].merge(
        parity.loc[:, [*keys, *columns]],
        on=keys,
        how="outer",
        suffixes=("_production", "_parity"),
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Parity artifact keys differ for {production_path.name}")
    deltas = [
        np.abs(
            joined[f"{column}_production"].to_numpy(dtype=float)
            - joined[f"{column}_parity"].to_numpy(dtype=float)
        ).max()
        for column in columns
    ]
    return float(max(deltas, default=0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit State-Precision NAIL parity")
    parser.add_argument("production_run_dir", type=Path)
    parser.add_argument("parity_run_dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    audit = audit_state_precision_parity(args.production_run_dir, args.parity_run_dir)
    assert_state_precision_parity(audit, tolerance=args.tolerance)
    print(
        "State-Precision NAIL parity passed: "
        f"max_abs_delta={audit.max_abs_delta:.3e}; "
        f"player_coefficients={audit.player_coefficient_max_abs_delta:.3e}; "
        f"player_priors={audit.player_prior_max_abs_delta:.3e}; "
        f"possessions={audit.possession_prediction_max_abs_delta:.3e}; "
        f"games={audit.game_prediction_max_abs_delta:.3e}"
    )


if __name__ == "__main__":
    main()
