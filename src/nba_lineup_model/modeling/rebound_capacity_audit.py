"""Audit empirical HPM rebound-calibration response curves."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_hpm_v21 import MODEL_NAME


def audit_rebound_capacity(
    *,
    model_run_dir: Path,
    artifacts_dir: Path = Path("artifacts/analysis/rebound_capacity_audit"),
) -> Path:
    """Write central-support curve points and directional diagnostics by season."""

    models = joblib.load(model_run_dir / "season_context_models.joblib")
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for season, context_model in sorted(models.items()):
        rebound = getattr(context_model, "rebound_model", None)
        if rebound is None:
            continue
        defensive_grid = np.linspace(
            rebound.reference_defensive_claims[0], rebound.reference_defensive_claims[-1], 101
        )
        offensive_grid = np.linspace(
            rebound.reference_offensive_claims[0], rebound.reference_offensive_claims[-1], 101
        )
        defensive_probability = rebound.predict_defensive_rebound_probability(
            defensive_grid, np.median(rebound.reference_offensive_claims)
        )
        offensive_probability = rebound.predict_defensive_rebound_probability(
            np.median(rebound.reference_defensive_claims), offensive_grid
        )
        defensive_delta = 100.0 * np.diff(defensive_probability)
        offensive_delta = 100.0 * np.diff(offensive_probability)
        summary_rows.append(
            {
                "season": season,
                "training_opportunity_count": rebound.training_opportunity_count,
                "defensive_claim_p5": defensive_grid[0],
                "defensive_claim_p95": defensive_grid[-1],
                "defensive_probability_change_pp": 100.0
                * (defensive_probability[-1] - defensive_probability[0]),
                "defensive_wrong_way_step_count": int((defensive_delta < -1e-5).sum()),
                "largest_defensive_reverse_pp": float(max(0.0, -defensive_delta.min())),
                "offensive_claim_p5": offensive_grid[0],
                "offensive_claim_p95": offensive_grid[-1],
                "offensive_probability_change_pp": 100.0
                * (offensive_probability[-1] - offensive_probability[0]),
                "offensive_wrong_way_step_count": int((offensive_delta > 1e-5).sum()),
                "largest_offensive_reverse_pp": float(max(0.0, offensive_delta.max())),
            }
        )
        curve_rows.extend(
            {
                "season": season,
                "curve": "defensive_claim",
                "claim": claim,
                "defensive_rebound_probability": probability,
            }
            for claim, probability in zip(defensive_grid, defensive_probability, strict=True)
        )
        curve_rows.extend(
            {
                "season": season,
                "curve": "offensive_claim",
                "claim": claim,
                "defensive_rebound_probability": probability,
            }
            for claim, probability in zip(offensive_grid, offensive_probability, strict=True)
        )
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError("No stored rebound calibrations found in model artifact")
    now = datetime.now(UTC)
    run_id = f"rebound-capacity-audit-{now:%Y%m%dT%H%M%SZ}"
    output = artifacts_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    summary.to_parquet(output / "season_summary.parquet", index=False)
    pd.DataFrame(curve_rows).to_parquet(output / "curve_points.parquet", index=False)
    metadata = {
        "run_id": run_id,
        "model_run_dir": str(model_run_dir),
        "season_count": len(summary),
        "created_at": now.isoformat(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (artifacts_dir / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return output


def _latest_model_run() -> Path:
    root = Path("artifacts/models") / MODEL_NAME / "2025-26"
    run_id = json.loads((root / "latest.json").read_text())["run_id"]
    return root / str(run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HPM v2.1 rebound calibrations")
    parser.add_argument("--model-run-dir", type=Path, default=_latest_model_run())
    args = parser.parse_args()
    output = audit_rebound_capacity(model_run_dir=args.model_run_dir)
    print(f"Rebound-capacity audit: run={output}")


if __name__ == "__main__":
    main()
