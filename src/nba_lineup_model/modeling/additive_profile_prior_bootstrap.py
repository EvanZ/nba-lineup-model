"""Paired game-block bootstrap for additive profile-prior RAPM comparisons."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.modeling.frozen_model_tournament import (
    DEFAULT_DRAWS,
    _latest_run,
    _paired_metrics,
)

DEFAULT_SEED = 20_260_815
ADDITIVE_ROOT = Path(
    "artifacts/models/analysis/additive_profile_prior_frozen/"
    "frozen_multiseason_backtest/2023-24_to_2025-26"
)
COMPLETE_ROOT = Path("artifacts/models/forward_complete_player_prior_rapm/2023-24_to_2025-26")
LINEAR_X3_ROOT = Path(
    "artifacts/models/analysis/hpm_x3_linear_ridge_frozen/"
    "frozen_multiseason_backtest/2023-24_to_2025-26"
)
OUTPUT_ROOT = Path("artifacts/models/analysis/additive_profile_prior_bootstrap/2023-24_to_2025-26")

COMPARISONS = (
    ("complete_player_prior", "Complete player-prior RAPM", COMPLETE_ROOT),
    ("linear_hpm_x3", "Linear-Ridge HPM x3 context", LINEAR_X3_ROOT),
)


def run_additive_profile_prior_bootstrap(
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    additive_root: Path | str = ADDITIVE_ROOT,
    output_root: Path | str = OUTPUT_ROOT,
) -> Path:
    """Compare the additive player prior with the two relevant frozen controls."""

    if draws < 1:
        raise ValueError("draws must be positive")
    candidate_run = _latest_run(Path(additive_root))
    candidate = _prediction_sources(candidate_run)
    rows: list[pd.DataFrame] = []
    source_rows: list[dict[str, str]] = [
        {"role": "candidate", "label": "Additive profile-prior RAPM", "run_dir": str(candidate_run)}
    ]
    for index, (comparison_id, label, root) in enumerate(COMPARISONS):
        reference_run = _latest_run(root)
        metrics = _paired_metrics(
            _prediction_sources(reference_run),
            candidate,
            draws=draws,
            seed=seed + index * 10,
        )
        metrics.insert(0, "comparison_id", comparison_id)
        metrics.insert(1, "reference_label", label)
        rows.append(metrics)
        source_rows.append({"role": comparison_id, "label": label, "run_dir": str(reference_run)})
    report = pd.concat(rows, ignore_index=True)
    return _write_report(
        report,
        pd.DataFrame(source_rows),
        draws=draws,
        seed=seed,
        output_root=Path(output_root),
    )


def _prediction_sources(run_dir: Path) -> dict[str, pd.DataFrame]:
    games = pd.read_parquet(run_dir / "regular_game_predictions.parquet")
    possessions = pd.read_parquet(run_dir / "possession_predictions.parquet")
    if "cohort" in possessions:
        possessions = possessions.loc[possessions["cohort"].eq("regular_season")].copy()
    return {"games": games, "possessions": possessions}


def _write_report(
    metrics: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    output_root: Path,
) -> Path:
    run_id = (
        f"additive-profile-prior-bootstrap-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-"
        f"{uuid4().hex[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(run_dir / "paired_bootstrap_metrics.parquet", index=False)
    sources.to_parquet(run_dir / "sources.parquet", index=False)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "resampling_unit": "complete regular-season game, stratified by season",
        "difference": "additive_profile_prior_minus_reference",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap additive profile-prior RAPM against frozen controls"
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()
    run_dir = run_additive_profile_prior_bootstrap(draws=args.draws)
    print(f"Additive profile-prior bootstrap: run={run_dir}")


if __name__ == "__main__":
    main()
