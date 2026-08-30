"""Persist and render the teammate-continuity context coefficient history."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_PRIOR_TEAMMATE_CONTINUITY,
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_teammate_continuity import MODEL_NAME
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    MODEL_NAME as INCUMBENT_MODEL_NAME,
)
from nba_lineup_model.modeling.nail_critical_spacing_weight_audit import (
    NailCriticalSpacingWeightAuditRun,
    build_nail_critical_spacing_weight_audit,
)
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    standardized_additive_weights,
)

DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/models/analysis/nail_teammate_continuity_weight_audit"
)
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-teammate-continuity/nonadditive-weight-trajectory.svg"
)
FEATURES = ("usage_concentration", "top_two_assists", "prior_teammate_continuity")


def build_nail_teammate_continuity_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> NailCriticalSpacingWeightAuditRun:
    run = build_nail_critical_spacing_weight_audit(
        source_run_dir=source_run_dir,
        artifacts_dir=artifacts_dir,
        output_root=output_root,
        chart_path=chart_path,
        model_name=MODEL_NAME,
        feature_set=CONTEXT_FEATURE_SET_NAIL_PRIOR_TEAMMATE_CONTINUITY,
        features=FEATURES,
        run_label="nail-teammate-continuity-weight-audit",
        chart_title="Teammate-continuity context weights by completed source season",
        legend="Orange: non-additive or relationship feature in the candidate fit",
    )
    candidate = pd.read_parquet(run.run_dir / "seasonal_standardized_weights.parquet")
    incumbent_source = _latest_run(
        Path(artifacts_dir) / INCUMBENT_MODEL_NAME / "2025-26"
    )
    incumbent_models = joblib.load(incumbent_source / "season_context_models.joblib")
    incumbent = standardized_additive_weights(
        incumbent_models,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        features=("usage_concentration", "top_two_assists"),
        accent_features=frozenset(("usage_concentration", "top_two_assists")),
    )
    displacement_rows: list[dict[str, object]] = []
    for feature in ("usage_concentration", "top_two_assists"):
        matched = (
            incumbent.loc[
                incumbent["feature"].eq(feature), ["season", "standardized_weight"]
            ]
            .rename(columns={"standardized_weight": "incumbent_weight"})
            .merge(
                candidate.loc[
                    candidate["feature"].eq(feature), ["season", "standardized_weight"]
                ].rename(columns={"standardized_weight": "candidate_weight"}),
                on="season",
                how="inner",
                validate="one_to_one",
            )
        )
        delta = matched["candidate_weight"] - matched["incumbent_weight"]
        displacement_rows.append(
            {
                "feature": feature,
                "season_count": len(matched),
                "pearson_correlation": matched["candidate_weight"].corr(
                    matched["incumbent_weight"]
                ),
                "mean_absolute_change": delta.abs().mean(),
                "mean_signed_change": delta.mean(),
            }
        )
    pd.DataFrame(displacement_rows).to_parquet(
        run.run_dir / "incumbent_feature_displacement.parquet",
        index=False,
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit teammate-continuity weights")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_teammate_continuity_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(f"Teammate-continuity weight audit: run={run.run_dir}; chart={run.chart_path}")


if __name__ == "__main__":
    main()
