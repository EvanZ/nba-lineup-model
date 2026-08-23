"""Audit all persisted NAIL-RAPM v1.2 context coefficients."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_gap_returners import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v12_context_weight_audit")
DEFAULT_CHART_PATH = Path("docs/assets/images/nail-v12/context-weight-trajectories.svg")
ADDITIVE_FEATURES = frozenset(
    {
        "three_pa_per_100",
        "three_pm_per_100",
        "assists_per_100",
        "turnovers_per_100",
        "usage_per_100",
        "steals_per_100",
        "blocks_per_100",
        "offensive_rebound_claim_total",
    }
)
CONTEXT_FEATURES = tuple(side_context_feature_columns(CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY))


@dataclass(frozen=True)
class NailV12ContextWeightAuditRun:
    """Persisted v1.2 context-coefficient audit."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path


def build_nail_v12_context_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> NailV12ContextWeightAuditRun:
    """Extract and chart all 14 persisted v1.2 context coefficients."""
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Context-weight audit requires a NAIL-RAPM v1.2 artifact")

    weights = standardized_additive_weights(
        joblib.load(source / "season_context_models.joblib"),
        feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
        features=CONTEXT_FEATURES,
        accent_features=frozenset(CONTEXT_FEATURES) - ADDITIVE_FEATURES,
    )
    summary = summarize_additive_weights(weights)
    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        weights,
        summary,
        rendered_chart,
        title="NAIL-RAPM v1.2 context weights by completed source season",
        features=CONTEXT_FEATURES,
        accent_features=frozenset(CONTEXT_FEATURES) - ADDITIVE_FEATURES,
        legend="Blue: additive player-profile total | Orange: non-additive lineup term",
    )

    root = Path(output_root)
    run_id = f"nail-v12-context-weight-audit-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(run_dir / "seasonal_standardized_weights.parquet", index=False)
    summary.to_parquet(run_dir / "feature_stability_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "feature_set": CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
                "feature_count": len(CONTEXT_FEATURES),
                "additive_features": sorted(ADDITIVE_FEATURES),
                "nonadditive_features": sorted(set(CONTEXT_FEATURES) - ADDITIVE_FEATURES),
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
    (root / "latest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV12ContextWeightAuditRun(run_dir, source, rendered_chart)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NAIL-RAPM v1.2 context weights")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v12_context_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(f"NAIL-RAPM v1.2 context-weight audit: run={run.run_dir}; chart={run.chart_path}")


if __name__ == "__main__":
    main()
