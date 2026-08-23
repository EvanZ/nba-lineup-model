"""Audit the seasonal stability of NAIL-RAPM v1.3 additive profile weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V13,
    LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES,
)
from nba_lineup_model.modeling.forward_contextual_rapm import DEFAULT_ARTIFACTS_DIR
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v13_additive_profiles import MODEL_NAME
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v13_additive_weight_audit")
DEFAULT_CHART_PATH = Path("docs/assets/images/nail-v13/additive-profile-weight-trajectories.svg")

FEATURE_LABELS = {
    "three_pa_per_100": "Three-point attempts / 100",
    "three_pm_per_100": "Three-point makes / 100",
    "assists_per_100": "Assists / 100",
    "turnovers_per_100": "Turnovers / 100",
    "usage_per_100": "Usage / 100",
    "steals_per_100": "Steals / 100",
    "blocks_per_100": "Blocks / 100",
    "offensive_rebound_claim_total": "Offensive-rebound claim total",
    "defensive_rebound_pct": "Defensive rebound percentage",
    "free_throw_attempts_per_100": "Free-throw attempts / 100",
    "unassisted_rim_makes_per_100": "Unassisted rim makes / 100",
    "unassisted_three_makes_per_100": "Unassisted three makes / 100",
    "bottom_two_three_pm": "Bottom-two three-point makes / 100",
    "credible_shooter_count": "Credible shooter count",
    "top_two_assists": "Top-two assists / 100",
    "usage_concentration": "Usage concentration",
    "critical_spacing": "Critical spacing",
    "shooting_usage_interaction": "Shooting-by-usage",
    "shooter_passing_interaction": "Shooter-by-passing",
}
NEW_FEATURES = frozenset(LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES[8:])


@dataclass(frozen=True)
class NailV13AdditiveWeightAuditRun:
    """Persisted v1.3 additive-coefficient audit."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path


def build_nail_v13_additive_weight_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> NailV13AdditiveWeightAuditRun:
    """Extract all persisted standardized additive Ridge weights and chart them."""
    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Additive-weight audit requires a NAIL-RAPM v1.3 artifact")
    models = joblib.load(source / "season_context_models.joblib")
    weights = standardized_additive_weights(models)
    summary = summarize_additive_weights(weights)
    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    render_additive_weight_trajectories(weights, summary, rendered_chart)

    root = Path(output_root)
    run_id = f"nail-v13-additive-weight-audit-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights.to_parquet(run_dir / "seasonal_standardized_weights.parquet", index=False)
    summary.to_parquet(run_dir / "feature_stability_summary.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "feature_set": CONTEXT_FEATURE_SET_NAIL_V13,
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
    return NailV13AdditiveWeightAuditRun(
        run_dir=run_dir,
        source_run_dir=source,
        chart_path=rendered_chart,
    )


def standardized_additive_weights(
    models: dict[str, MatchupContextualModel],
    *,
    feature_set: str = CONTEXT_FEATURE_SET_NAIL_V13,
    features: tuple[str, ...] = LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES,
    accent_features: frozenset[str] = NEW_FEATURES,
) -> pd.DataFrame:
    """Return standardized Ridge weights for all additive terms in every season."""
    rows: list[dict[str, object]] = []
    for season, model in sorted(models.items()):
        if model.feature_set != feature_set:
            raise ValueError(f"Unexpected feature set for {season}: {model.feature_set}")
        scale = model.pipeline.named_steps["scale"]
        ridge = model.pipeline.named_steps["ridge"]
        for feature_name, coefficient in zip(scale.feature_names_in_, ridge.coef_, strict=True):
            feature = str(feature_name).removeprefix("home_minus_away_")
            if feature not in features:
                continue
            rows.append(
                {
                    "season": season,
                    "season_start_year": int(str(season)[:4]),
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "is_v13_addition": feature in accent_features,
                    "standardized_weight": float(coefficient),
                }
            )
    weights = pd.DataFrame(rows)
    expected = len(models) * len(features)
    if len(weights) != expected:
        raise ValueError("Persisted v1.3 context states lack additive profile coefficients")
    return weights.sort_values(["feature", "season_start_year"], kind="stable").reset_index(
        drop=True
    )


def summarize_additive_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarize magnitude and directional consistency for each feature trajectory."""
    required = {"feature", "label", "is_v13_addition", "standardized_weight"}
    if required - set(weights):
        raise ValueError("Weight ledger is missing required columns")
    rows: list[dict[str, object]] = []
    for feature, group in weights.groupby("feature", sort=False):
        values = group["standardized_weight"].to_numpy(dtype=float)
        positive_share = float(np.mean(values > 0))
        negative_share = float(np.mean(values < 0))
        positive_mass = float(np.clip(values, a_min=0.0, a_max=None).sum())
        negative_mass = float(np.clip(-values, a_min=0.0, a_max=None).sum())
        total_directional_mass = positive_mass + negative_mass
        positive_directional_mass_share = (
            positive_mass / total_directional_mass if total_directional_mass else 0.5
        )
        rows.append(
            {
                "feature": feature,
                "label": str(group["label"].iloc[0]),
                "is_v13_addition": bool(group["is_v13_addition"].iloc[0]),
                "season_count": int(len(values)),
                "median_standardized_weight": float(np.median(values)),
                "mean_absolute_standardized_weight": float(np.mean(np.abs(values))),
                "positive_season_share": positive_share,
                "dominant_sign_share": max(positive_share, negative_share),
                "positive_directional_mass": positive_mass,
                "negative_directional_mass": negative_mass,
                "positive_directional_mass_share": positive_directional_mass_share,
                "dominant_directional_mass_share": max(
                    positive_directional_mass_share,
                    1.0 - positive_directional_mass_share,
                ),
                "standard_deviation": float(np.std(values, ddof=0)),
                "minimum_weight": float(np.min(values)),
                "maximum_weight": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dominant_directional_mass_share", "mean_absolute_standardized_weight"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def render_additive_weight_trajectories(
    weights: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    *,
    title: str = "NAIL-RAPM v1.3 additive-profile weights by completed source season",
    features: tuple[str, ...] = LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES,
    accent_features: frozenset[str] = NEW_FEATURES,
    legend: str = "Blue: inherited feature | Orange: v1.3 addition",
) -> None:
    """Render one historical weight trajectory per additive player profile feature."""
    import matplotlib.pyplot as plt

    columns = 3
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3.05), squeeze=False)
    x_ticks = [1998, 2005, 2012, 2019, 2025]
    for axis, feature in zip(axes.flat, features, strict=False):
        series = weights.loc[weights["feature"].eq(feature)].sort_values("season_start_year")
        color = "#e8502f" if feature in accent_features else "#2f6ea8"
        axis.plot(
            series["season_start_year"],
            series["standardized_weight"],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=2.8,
        )
        axis.axhline(0, color="#767d76", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_title(FEATURE_LABELS[feature], fontsize=10, loc="left", fontweight="bold")
        axis.set_xlim(
            series["season_start_year"].min() - 0.4,
            series["season_start_year"].max() + 0.4,
        )
        axis.set_xticks(x_ticks)
        axis.tick_params(axis="both", labelsize=7)
        statistic = summary.loc[summary["feature"].eq(feature)].iloc[0]
        axis.text(
            0.98,
            0.04,
            (
                f"one-sided mass {statistic.dominant_directional_mass_share:.0%}\n"
                f"mean |w| {statistic.mean_absolute_standardized_weight:.2f}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#5f6860",
        )
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle(
        title,
        fontsize=15,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    figure.supxlabel("Source season start year", fontsize=10)
    figure.supylabel("Standardized Ridge weight", fontsize=10)
    figure.text(
        0.98,
        0.975,
        legend,
        ha="right",
        va="top",
        fontsize=9,
        color="#5f6860",
    )
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit NAIL-RAPM v1.3 additive profile weights")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v13_additive_weight_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(f"NAIL v1.3 additive-weight audit: run={run.run_dir}; chart={run.chart_path}")


if __name__ == "__main__":
    main()
