"""Audit NAIL-RAPM v1.3.1's ten retained additive-profile coefficients."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
    LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v131_pruned_additive_profiles import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v131_pruned_additive_weight_audit")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-v131/pruned-additive-profile-weight-trajectories.svg"
)


@dataclass(frozen=True)
class NailV131PrunedAdditiveWeightAuditRun:
    """Persisted ten-panel standardized-coefficient audit."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path


def build_nail_v131_pruned_additive_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> NailV131PrunedAdditiveWeightAuditRun:
    """Extract and chart coefficients from every completed v1.3.1 source model."""
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Pruned additive-weight audit requires a NAIL-RAPM v1.3.1 artifact")

    features = tuple(LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES)
    weights = standardized_additive_weights(
        joblib.load(source / "season_context_models.joblib"),
        feature_set=CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
        features=features,
        accent_features=frozenset(),
    )
    summary = summarize_additive_weights(weights)
    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        weights,
        summary,
        rendered_chart,
        title="NAIL-RAPM v1.3.1 retained additive weights by completed source season",
        features=features,
        accent_features=frozenset(),
        legend="All traces: retained v1.3.1 additive profile terms",
    )

    root = Path(output_root)
    run_id = (
        "nail-v131-pruned-additive-weight-audit-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(run_dir / "seasonal_standardized_weights.parquet", index=False)
    summary.to_parquet(run_dir / "feature_stability_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "feature_set": CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
                "feature_count": len(features),
                "removed_features": ["three_pa_per_100", "usage_per_100"],
                "weight_definition": (
                    "Ridge coefficient after StandardScaler of the home-minus-away "
                    "five-man aggregate feature"
                ),
                "chart_path": str(rendered_chart),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV131PrunedAdditiveWeightAuditRun(run_dir, source, rendered_chart)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NAIL-RAPM v1.3.1 retained additive weights")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v131_pruned_additive_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(f"NAIL-RAPM v1.3.1 additive-weight audit: run={run.run_dir}; chart={run.chart_path}")


if __name__ == "__main__":
    main()
