"""Audit NAIL-RAPM v1.2.2 additive and non-additive coefficient trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE,
    LINEAR_NAIL_V122_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v122_defensive_rebound_profile import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/models/analysis/nail_v122_defensive_rebound_profile_weight_audit"
)
DEFAULT_ADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v122/additive-profile-weight-trajectories.svg"
)
DEFAULT_NONADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v122/retained-nonadditive-weight-trajectories.svg"
)
RETAINED_NONADDITIVE_FEATURES = ("usage_concentration", "top_two_assists")


@dataclass(frozen=True)
class NailV122DefensiveReboundProfileWeightAuditRun:
    """Persisted v1.2.2 standardized coefficient audit."""

    run_dir: Path
    source_run_dir: Path
    additive_chart_path: Path
    nonadditive_chart_path: Path


def build_nail_v122_defensive_rebound_profile_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    additive_chart_path: Path | str = DEFAULT_ADDITIVE_CHART_PATH,
    nonadditive_chart_path: Path | str = DEFAULT_NONADDITIVE_CHART_PATH,
) -> NailV122DefensiveReboundProfileWeightAuditRun:
    """Extract each v1.2.2 source-season Ridge coefficient and render both panels."""
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Weight audit requires a NAIL-RAPM v1.2.2 artifact")
    models = joblib.load(source / "season_context_models.joblib")
    additive_features = tuple(LINEAR_NAIL_V122_BASKETBALL_ADDITIVE_FEATURES)
    additive_weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE,
        features=additive_features,
        accent_features=frozenset(),
    )
    additive_summary = summarize_additive_weights(additive_weights)
    rendered_additive_chart = Path(additive_chart_path)
    rendered_additive_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        additive_weights,
        additive_summary,
        rendered_additive_chart,
        title="NAIL-RAPM v1.2.2 additive profile weights by completed source season",
        features=additive_features,
        accent_features=frozenset(),
        legend="Blue: player-attributable additive profile term",
    )

    nonadditive_weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE,
        features=RETAINED_NONADDITIVE_FEATURES,
        accent_features=frozenset(RETAINED_NONADDITIVE_FEATURES),
    )
    nonadditive_summary = summarize_additive_weights(nonadditive_weights)
    rendered_nonadditive_chart = Path(nonadditive_chart_path)
    rendered_nonadditive_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        nonadditive_weights,
        nonadditive_summary,
        rendered_nonadditive_chart,
        title="NAIL-RAPM v1.2.2 retained non-additive weights by completed source season",
        features=RETAINED_NONADDITIVE_FEATURES,
        accent_features=frozenset(RETAINED_NONADDITIVE_FEATURES),
        legend="Orange: retained non-additive lineup term",
    )

    root = Path(output_root)
    run_id = (
        "nail-v122-defensive-rebound-profile-weight-audit-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    additive_weights.to_parquet(
        run_dir / "additive_profile_seasonal_standardized_weights.parquet",
        index=False,
    )
    additive_summary.to_parquet(
        run_dir / "additive_profile_feature_stability_summary.parquet",
        index=False,
    )
    nonadditive_weights.to_parquet(
        run_dir / "nonadditive_seasonal_standardized_weights.parquet",
        index=False,
    )
    nonadditive_summary.to_parquet(
        run_dir / "nonadditive_feature_stability_summary.parquet",
        index=False,
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "feature_set": CONTEXT_FEATURE_SET_NAIL_V122_DEFENSIVE_REBOUND_PROFILE,
                "additive_features": list(additive_features),
                "retained_nonadditive_features": list(RETAINED_NONADDITIVE_FEATURES),
                "weight_definition": (
                    "Ridge coefficient after StandardScaler of the home-minus-away "
                    "five-man aggregate feature"
                ),
                "additive_chart_path": str(rendered_additive_chart),
                "nonadditive_chart_path": str(rendered_nonadditive_chart),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV122DefensiveReboundProfileWeightAuditRun(
        run_dir,
        source,
        rendered_additive_chart,
        rendered_nonadditive_chart,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NAIL-RAPM v1.2.2 coefficient weights")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--additive-chart-path", default=str(DEFAULT_ADDITIVE_CHART_PATH))
    parser.add_argument("--nonadditive-chart-path", default=str(DEFAULT_NONADDITIVE_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v122_defensive_rebound_profile_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        additive_chart_path=args.additive_chart_path,
        nonadditive_chart_path=args.nonadditive_chart_path,
    )
    print(
        "NAIL-RAPM v1.2.2 coefficient audit: "
        f"run={run.run_dir}; additive_chart={run.additive_chart_path}; "
        f"nonadditive_chart={run.nonadditive_chart_path}"
    )


if __name__ == "__main__":
    main()
