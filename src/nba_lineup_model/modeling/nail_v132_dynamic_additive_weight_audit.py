"""Audit NAIL-RAPM v1.3.2's forward dynamic additive-profile weights."""

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
from nba_lineup_model.modeling.forward_nail_v132_dynamic_additive_profiles import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v132_dynamic_additive_weight_audit")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-v132/dynamic-additive-profile-weight-trajectories.svg"
)


@dataclass(frozen=True)
class NailV132DynamicAdditiveWeightAuditRun:
    """Persisted ten-panel dynamic additive-coefficient audit."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path


def build_nail_v132_dynamic_additive_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> NailV132DynamicAdditiveWeightAuditRun:
    """Extract and chart every completed v1.3.2 source-state coefficient."""
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Dynamic additive-weight audit requires a NAIL-RAPM v1.3.2 artifact")

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
        title="NAIL-RAPM v1.3.2 dynamic additive weights by completed source season",
        features=features,
        accent_features=frozenset(),
        legend="All traces: forward dynamic v1.3.2 additive profile terms",
    )

    root = Path(output_root)
    run_id = (
        "nail-v132-dynamic-additive-weight-audit-"
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
                "dynamic_state": "forward feature-specific mean-reverting empirical-Bayes",
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
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV132DynamicAdditiveWeightAuditRun(run_dir, source, rendered_chart)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NAIL-RAPM v1.3.2 dynamic additive weights")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v132_dynamic_additive_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(
        "NAIL-RAPM v1.3.2 dynamic additive-weight audit: "
        f"run={run.run_dir}; chart={run.chart_path}"
    )


if __name__ == "__main__":
    main()
