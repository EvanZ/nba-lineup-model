"""Coefficient trajectories for the lead-secondary usage-gap NAIL candidate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_LEAD_SECONDARY_USAGE_GAP,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_lead_secondary_usage_gap import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_lead_secondary_usage_gap_weight_audit")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nonadditive-feature-screens/lead-secondary-usage-gap-weight-trajectories.svg"
)
FEATURES = ("usage_concentration", "top_two_assists", "lead_secondary_usage_gap")


def build_nail_lead_secondary_usage_gap_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> Path:
    """Extract the three retained/candidate non-additive standardized weights."""

    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / "2025-26")
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Lead-secondary weight audit requires its candidate artifact")
    models = joblib.load(source / "season_context_models.joblib")
    weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_LEAD_SECONDARY_USAGE_GAP,
        features=FEATURES,
        accent_features=frozenset(FEATURES),
    )
    summary = summarize_additive_weights(weights)
    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        weights,
        summary,
        rendered_chart,
        title="Lead-secondary usage-gap candidate: non-additive weights by source season",
        features=FEATURES,
        accent_features=frozenset(FEATURES),
        legend="Orange: retained or candidate non-additive term",
    )

    run_id = (
        "nail-lead-secondary-usage-gap-weight-audit-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(run_dir / "seasonal_standardized_weights.parquet", index=False)
    summary.to_parquet(run_dir / "feature_stability_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "features": list(FEATURES),
                "weight_definition": (
                    "Ridge coefficient after StandardScaler of the home-minus-away "
                    "five-man context feature"
                ),
                "chart_path": str(rendered_chart),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit lead-secondary usage-gap weights")
    parser.add_argument("--source-run-dir")
    args = parser.parse_args()
    run_dir = build_nail_lead_secondary_usage_gap_weight_audit(source_run_dir=args.source_run_dir)
    print(f"Lead-secondary usage-gap weight audit: run={run_dir}")


if __name__ == "__main__":
    main()
