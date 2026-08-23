"""Audit all non-additive terms in the lower-quintile spacing candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE,
)
from nba_lineup_model.modeling.forward_nail_critical_spacing_quintile import MODEL_NAME
from nba_lineup_model.modeling.nail_critical_spacing_weight_audit import (
    NailCriticalSpacingWeightAuditRun,
    build_nail_critical_spacing_weight_audit,
)

DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/models/analysis/nail_critical_spacing_quintile_weight_audit"
)
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-critical-spacing-quintile/"
    "critical-spacing-quintile-weight-trajectory.svg"
)
FEATURES = ("usage_concentration", "top_two_assists", "critical_spacing")


def build_nail_critical_spacing_quintile_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> NailCriticalSpacingWeightAuditRun:
    """Render all three non-additive trajectories for the quintile candidate."""

    return build_nail_critical_spacing_weight_audit(
        source_run_dir=source_run_dir,
        output_root=output_root,
        chart_path=chart_path,
        model_name=MODEL_NAME,
        feature_set=CONTEXT_FEATURE_SET_NAIL_CRITICAL_SPACING_QUINTILE,
        features=FEATURES,
        run_label="nail-critical-spacing-quintile-weight-audit",
        chart_title=(
            "Lower-Quintile Critical Spacing non-additive weights by completed "
            "source season"
        ),
        legend="Orange: non-additive lineup-context term in the quintile candidate fit",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit lower-quintile Critical Spacing non-additive weights"
    )
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_critical_spacing_quintile_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(
        "NAIL lower-quintile Critical Spacing non-additive audit: "
        f"run={run.run_dir}; chart={run.chart_path}"
    )


if __name__ == "__main__":
    main()
