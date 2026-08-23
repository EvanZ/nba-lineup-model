"""Audit the two retained non-additive NAIL-RAPM v1.2.1 coefficients."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v121_pruned_nonadditive import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v121_pruned_nonadditive_weight_audit")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-v121/pruned-nonadditive-weight-trajectories.svg"
)
DEFAULT_ADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v121/additive-profile-weight-trajectories.svg"
)
ADDITIVE_FEATURES = (
    "three_pa_per_100",
    "three_pm_per_100",
    "assists_per_100",
    "turnovers_per_100",
    "usage_per_100",
    "steals_per_100",
    "blocks_per_100",
    "offensive_rebound_claim_total",
)
RETAINED_FEATURES = ("usage_concentration", "top_two_assists")


@dataclass(frozen=True)
class NailV121PrunedNonadditiveWeightAuditRun:
    """Persisted standardized-coefficient audit for the v1.2.1 contract."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path
    additive_chart_path: Path


def build_nail_v121_pruned_nonadditive_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
    additive_chart_path: Path | str = DEFAULT_ADDITIVE_CHART_PATH,
) -> NailV121PrunedNonadditiveWeightAuditRun:
    """Extract and chart the two retained v1.2.1 lineup-context coefficients."""
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Weight audit requires a NAIL-RAPM v1.2.1 artifact")

    models = joblib.load(source / "season_context_models.joblib")
    weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
        features=RETAINED_FEATURES,
        accent_features=frozenset(RETAINED_FEATURES),
    )
    summary = summarize_additive_weights(weights)
    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        weights,
        summary,
        rendered_chart,
        title="NAIL-RAPM v1.2.1 retained non-additive weights by completed source season",
        features=RETAINED_FEATURES,
        accent_features=frozenset(RETAINED_FEATURES),
        legend="Orange: retained non-additive lineup term",
    )
    additive_weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
        features=ADDITIVE_FEATURES,
        accent_features=frozenset(),
    )
    additive_summary = summarize_additive_weights(additive_weights)
    rendered_additive_chart = Path(additive_chart_path)
    rendered_additive_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        additive_weights,
        additive_summary,
        rendered_additive_chart,
        title="NAIL-RAPM v1.2.1 additive profile weights by completed source season",
        features=ADDITIVE_FEATURES,
        accent_features=frozenset(),
        legend="Blue: player-attributable additive profile term",
    )

    root = Path(output_root)
    run_id = (
        "nail-v121-pruned-nonadditive-weight-audit-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(run_dir / "seasonal_standardized_weights.parquet", index=False)
    summary.to_parquet(run_dir / "feature_stability_summary.parquet", index=False)
    additive_weights.to_parquet(
        run_dir / "additive_profile_seasonal_standardized_weights.parquet",
        index=False,
    )
    additive_summary.to_parquet(
        run_dir / "additive_profile_feature_stability_summary.parquet",
        index=False,
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "feature_set": CONTEXT_FEATURE_SET_NAIL_V121_PRUNED_NONADDITIVE,
                "retained_features": list(RETAINED_FEATURES),
                "additive_features": list(ADDITIVE_FEATURES),
                "weight_definition": (
                    "Ridge coefficient after StandardScaler of the home-minus-away "
                    "five-man aggregate feature"
                ),
                "chart_path": str(rendered_chart),
                "additive_chart_path": str(rendered_additive_chart),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV121PrunedNonadditiveWeightAuditRun(
        run_dir,
        source,
        rendered_chart,
        rendered_additive_chart,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit NAIL-RAPM v1.2.1 retained non-additive weights"
    )
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    parser.add_argument("--additive-chart-path", default=str(DEFAULT_ADDITIVE_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v121_pruned_nonadditive_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
        additive_chart_path=args.additive_chart_path,
    )
    print(
        "NAIL-RAPM v1.2.1 coefficient audit: "
        f"run={run.run_dir}; nonadditive_chart={run.chart_path}; "
        f"additive_chart={run.additive_chart_path}"
    )


if __name__ == "__main__":
    main()
