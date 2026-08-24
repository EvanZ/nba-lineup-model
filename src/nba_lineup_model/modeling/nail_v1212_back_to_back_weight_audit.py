"""Persist and visualize annual NAIL back-to-back schedule-control weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import MODEL_NAME
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    render_additive_weight_trajectories,
    standardized_additive_weights,
    summarize_additive_weights,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v1212_back_to_back_weight_audit")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-v1212/back-to-back-weight-trajectory.svg"
)
DEFAULT_ADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v1212/additive-profile-weight-trajectories.svg"
)
DEFAULT_NONADDITIVE_CHART_PATH = Path(
    "docs/assets/images/nail-v1212/retained-nonadditive-weight-trajectories.svg"
)
RETAINED_NONADDITIVE_FEATURES = ("usage_concentration", "top_two_assists")


@dataclass(frozen=True)
class BackToBackWeightAuditRun:
    run_dir: Path
    source_run_dir: Path
    chart_path: Path
    additive_chart_path: Path
    nonadditive_chart_path: Path


def build_nail_v1212_back_to_back_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
    additive_chart_path: Path | str = DEFAULT_ADDITIVE_CHART_PATH,
    nonadditive_chart_path: Path | str = DEFAULT_NONADDITIVE_CHART_PATH,
) -> BackToBackWeightAuditRun:
    """Render the source-season raw B2B effect in net-rating units."""

    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("B2B audit requires a NAIL-RAPM v1.2.1.2 artifact")
    weights = pd.read_parquet(source / "season_schedule_control_metadata.parquet")
    required = {"season", "schedule_control_raw_weight", "schedule_control_standardized_weight"}
    missing = required - set(weights)
    if missing:
        raise ValueError(f"Schedule-control artifact lacks: {sorted(missing)}")
    weights = (
        weights.loc[:, sorted(required)]
        .sort_values("season", kind="stable")
        .reset_index(drop=True)
    )
    rendered = Path(chart_path)
    rendered.parent.mkdir(parents=True, exist_ok=True)
    _render(weights, rendered)
    models = joblib.load(source / "season_context_models.joblib")
    additive_weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        features=LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
        accent_features=frozenset(),
    )
    nonadditive_weights = standardized_additive_weights(
        models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        features=RETAINED_NONADDITIVE_FEATURES,
        accent_features=frozenset(RETAINED_NONADDITIVE_FEATURES),
    )
    rendered_additive = Path(additive_chart_path)
    rendered_nonadditive = Path(nonadditive_chart_path)
    rendered_additive.parent.mkdir(parents=True, exist_ok=True)
    rendered_nonadditive.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(
        additive_weights,
        summarize_additive_weights(additive_weights),
        rendered_additive,
        title="NAIL-RAPM v1.2.1.2 additive profile weights by completed source season",
        features=LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
        accent_features=frozenset(),
        legend="Blue: player-attributable additive profile term",
    )
    render_additive_weight_trajectories(
        nonadditive_weights,
        summarize_additive_weights(nonadditive_weights),
        rendered_nonadditive,
        title="NAIL-RAPM v1.2.1.2 retained non-additive weights by completed source season",
        features=RETAINED_NONADDITIVE_FEATURES,
        accent_features=frozenset(RETAINED_NONADDITIVE_FEATURES),
        legend="Orange: retained non-additive lineup term",
    )
    root = Path(output_root)
    run_id = (
        "nail-v1212-back-to-back-weight-audit-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(run_dir / "season_schedule_control_weights.parquet", index=False)
    additive_weights.to_parquet(
        run_dir / "additive_profile_seasonal_standardized_weights.parquet", index=False
    )
    nonadditive_weights.to_parquet(
        run_dir / "nonadditive_seasonal_standardized_weights.parquet", index=False
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "weight_definition": (
                    "Raw weighted-Ridge coefficient in home net-rating points for "
                    "home back-to-back minus away back-to-back."
                ),
                "chart_path": str(rendered),
                "additive_chart_path": str(rendered_additive),
                "nonadditive_chart_path": str(rendered_nonadditive),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return BackToBackWeightAuditRun(
        run_dir,
        source,
        rendered,
        rendered_additive,
        rendered_nonadditive,
    )


def _render(weights: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.8), layout="constrained")
    x = range(len(weights))
    axis.axhline(0.0, color="#7c837c", linewidth=1.0, linestyle="--")
    axis.plot(
        x,
        weights["schedule_control_raw_weight"],
        color="#e4572e",
        marker="o",
        linewidth=2.2,
        markersize=4,
    )
    axis.set_xticks(list(x), weights["season"], rotation=45, ha="right")
    axis.set_ylabel("Home net-rating points")
    axis.set_title("NAIL-RAPM v1.2.1.2 back-to-back effect by completed source season")
    axis.text(
        0.01,
        0.02,
        "Negative: a home back-to-back is predicted to reduce the home edge",
        transform=axis.transAxes,
        color="#626a64",
        fontsize=9,
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit NAIL-RAPM v1.2.1.2 back-to-back schedule weights"
    )
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    parser.add_argument("--additive-chart-path", default=str(DEFAULT_ADDITIVE_CHART_PATH))
    parser.add_argument("--nonadditive-chart-path", default=str(DEFAULT_NONADDITIVE_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v1212_back_to_back_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
        additive_chart_path=args.additive_chart_path,
        nonadditive_chart_path=args.nonadditive_chart_path,
    )
    print(
        "NAIL-RAPM v1.2.1.2 B2B weight audit: "
        f"run={run.run_dir}; chart={run.chart_path}; "
        f"additive_chart={run.additive_chart_path}; nonadditive_chart={run.nonadditive_chart_path}"
    )


if __name__ == "__main__":
    main()
