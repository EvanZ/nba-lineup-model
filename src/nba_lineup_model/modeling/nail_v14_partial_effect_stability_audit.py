"""Classify the forward stability of NAIL-RAPM v1.4 additive partial effects."""

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
from nba_lineup_model.modeling.forward_nail_v14_filtered_additive_profiles import MODEL_NAME
from nba_lineup_model.modeling.matchup_contextual import MatchupContextualModel
from nba_lineup_model.modeling.nail_v13_additive_weight_audit import (
    FEATURE_LABELS,
    NEW_FEATURES,
)

DEFAULT_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/nail_v14_partial_effect_stability_audit")
DEFAULT_CHART_PATH = Path(
    "docs/assets/images/nail-v14/kalman-additive-partial-effect-stability.svg"
)
RESOLVED_Z_SCORE = 1.645
MATERIAL_STANDARDIZED_WEIGHT = 0.10
STABLE_SIGN_SHARE = 0.80
REGIME_MIN_RUN = 3


@dataclass(frozen=True)
class NailV14PartialEffectStabilityAuditRun:
    """Persisted uncertainty-aware v1.4 partial-effect audit."""

    run_dir: Path
    source_run_dir: Path
    chart_path: Path


def kalman_standardized_additive_effects(
    models: dict[str, MatchupContextualModel],
) -> pd.DataFrame:
    """Extract standardized Kalman means, standard errors, and marginal intervals."""

    rows: list[dict[str, object]] = []
    features = tuple(LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES)
    for season, model in sorted(models.items()):
        if model.feature_set != CONTEXT_FEATURE_SET_NAIL_V13:
            raise ValueError(f"Unexpected feature set for {season}: {model.feature_set}")
        raw_mean = np.asarray(model.additive_kalman_mean_raw, dtype=float)
        raw_covariance = np.asarray(model.additive_kalman_covariance_raw, dtype=float)
        if raw_mean.shape != (len(features),) or raw_covariance.shape != (
            len(features),
            len(features),
        ):
            raise ValueError(f"Missing or invalid Kalman additive state for {season}")
        scale = model.pipeline.named_steps["scale"]
        columns = list(scale.feature_names_in_)
        feature_scale = np.asarray(
            [
                scale.scale_[columns.index(f"home_minus_away_{feature}")]
                for feature in features
            ],
            dtype=float,
        )
        standardized_mean = raw_mean * feature_scale
        standardized_variance = np.diag(raw_covariance) * np.square(feature_scale)
        standardized_se = np.sqrt(np.maximum(standardized_variance, 0.0))
        for feature, mean, standard_error in zip(
            features, standardized_mean, standardized_se, strict=True
        ):
            z_score = float(mean / standard_error) if standard_error > 0 else np.nan
            resolved = bool(np.isfinite(z_score) and abs(z_score) >= RESOLVED_Z_SCORE)
            rows.append(
                {
                    "season": season,
                    "season_start_year": int(str(season)[:4]),
                    "feature": feature,
                    "label": FEATURE_LABELS[feature],
                    "is_v13_addition": feature in NEW_FEATURES,
                    "standardized_weight": float(mean),
                    "standardized_standard_error": float(standard_error),
                    "z_score": z_score,
                    "interval_90_lower": float(mean - RESOLVED_Z_SCORE * standard_error),
                    "interval_90_upper": float(mean + RESOLVED_Z_SCORE * standard_error),
                    "is_resolved": resolved,
                    "resolved_sign": int(np.sign(mean)) if resolved else 0,
                }
            )
    effects = pd.DataFrame(rows)
    expected = len(models) * len(features)
    if len(effects) != expected:
        raise ValueError("Persisted v1.4 states lack complete additive partial effects")
    return effects.sort_values(["feature", "season_start_year"], kind="stable").reset_index(
        drop=True
    )


def _longest_run(values: np.ndarray, value: int) -> int:
    longest = current = 0
    for observed in values:
        if observed == value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _resolved_sign_changes(values: np.ndarray) -> int:
    nonzero = values[values != 0]
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0


def classify_partial_effects(effects: pd.DataFrame) -> pd.DataFrame:
    """Classify effects using only sequential posterior estimates and uncertainty.

    ``stable_material`` requires a material median magnitude, resolved
    evidence in at least half of seasons, and a single resolved sign in at
    least 80% of those resolved seasons. ``stable_weak`` has no material
    median effect. ``sustained_regime_shift`` has resolved runs of at least
    three seasons on both sides of zero. The remaining trajectories are not
    sufficiently resolved to retain a directional partial-effect claim.
    """

    required = {
        "feature",
        "label",
        "is_v13_addition",
        "season_start_year",
        "standardized_weight",
        "standardized_standard_error",
        "is_resolved",
        "resolved_sign",
    }
    missing = required - set(effects)
    if missing:
        raise ValueError(f"Partial-effect ledger is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for feature, group in effects.groupby("feature", sort=False):
        ordered = group.sort_values("season_start_year", kind="stable")
        weights = ordered["standardized_weight"].to_numpy(dtype=float)
        signs = ordered["resolved_sign"].to_numpy(dtype=int)
        resolved = signs != 0
        positive_share = float(np.mean(signs > 0))
        negative_share = float(np.mean(signs < 0))
        dominant_sign = 1 if positive_share >= negative_share else -1
        dominant_sign_share = max(positive_share, negative_share)
        median_weight = float(np.median(weights))
        median_absolute_weight = float(np.median(np.abs(weights)))
        resolved_share = float(np.mean(resolved))
        resolved_direction_share = (
            float(np.mean(signs[resolved] == dominant_sign)) if resolved.any() else 0.0
        )
        longest_positive_run = _longest_run(signs, 1)
        longest_negative_run = _longest_run(signs, -1)
        if median_absolute_weight < MATERIAL_STANDARDIZED_WEIGHT:
            classification = "stable_weak"
        elif (
            resolved_share >= 0.50
            and resolved_direction_share >= STABLE_SIGN_SHARE
        ):
            classification = "stable_material"
        elif (
            longest_positive_run >= REGIME_MIN_RUN
            and longest_negative_run >= REGIME_MIN_RUN
        ):
            classification = "sustained_regime_shift"
        else:
            classification = "insufficiently_resolved"
        autocorrelation = (
            float(np.corrcoef(weights[:-1], weights[1:])[0, 1])
            if len(weights) > 2
            and np.std(weights[:-1]) > 0
            and np.std(weights[1:]) > 0
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "label": str(ordered["label"].iloc[0]),
                "is_v13_addition": bool(ordered["is_v13_addition"].iloc[0]),
                "classification": classification,
                "season_count": int(len(ordered)),
                "resolved_season_share": resolved_share,
                "resolved_positive_share": positive_share,
                "resolved_negative_share": negative_share,
                "dominant_resolved_sign": "positive" if dominant_sign > 0 else "negative",
                "dominant_resolved_sign_share": dominant_sign_share,
                "dominant_direction_among_resolved_share": resolved_direction_share,
                "median_standardized_weight": median_weight,
                "median_absolute_standardized_weight": median_absolute_weight,
                "mean_absolute_standardized_weight": float(np.mean(np.abs(weights))),
                "lag_one_autocorrelation": autocorrelation,
                "resolved_sign_changes": _resolved_sign_changes(signs),
                "longest_positive_sign_run": longest_positive_run,
                "longest_negative_sign_run": longest_negative_run,
                "minimum_weight": float(np.min(weights)),
                "maximum_weight": float(np.max(weights)),
            }
        )
    order = {
        "stable_material": 0,
        "stable_weak": 1,
        "sustained_regime_shift": 2,
        "insufficiently_resolved": 3,
    }
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["classification", "median_absolute_standardized_weight"],
        key=lambda values: values.map(order) if values.name == "classification" else values,
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def render_partial_effect_stability_chart(
    effects: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Render uncertainty bands and stability classifications for all twelve effects."""

    import matplotlib.pyplot as plt

    colors = {
        "stable_material": "#245a47",
        "stable_weak": "#6c786e",
        "sustained_regime_shift": "#b36b22",
        "insufficiently_resolved": "#b8442c",
    }
    columns = 3
    features = tuple(LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES)
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3.1), squeeze=False)
    for axis, feature in zip(axes.flat, features, strict=False):
        series = effects.loc[effects["feature"].eq(feature)].sort_values("season_start_year")
        statistic = summary.loc[summary["feature"].eq(feature)].iloc[0]
        color = colors[str(statistic["classification"])]
        x = series["season_start_year"].to_numpy(dtype=float)
        y = series["standardized_weight"].to_numpy(dtype=float)
        lower = series["interval_90_lower"].to_numpy(dtype=float)
        upper = series["interval_90_upper"].to_numpy(dtype=float)
        resolved = series["is_resolved"].to_numpy(dtype=bool)
        axis.fill_between(x, lower, upper, color=color, alpha=0.15, linewidth=0)
        axis.plot(x, y, color=color, linewidth=1.7)
        axis.scatter(x[resolved], y[resolved], color=color, s=11, zorder=3)
        axis.scatter(
            x[~resolved],
            y[~resolved],
            facecolors="white",
            edgecolors=color,
            linewidths=0.9,
            s=11,
            zorder=3,
        )
        axis.axhline(0, color="#767d76", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_title(str(statistic["label"]), fontsize=10, loc="left", fontweight="bold")
        axis.set_xlim(x.min() - 0.4, x.max() + 0.4)
        axis.set_xticks([1998, 2005, 2012, 2019, 2025])
        axis.tick_params(axis="both", labelsize=7)
        axis.text(
            0.98,
            0.04,
            f"{str(statistic['classification']).replace('_', ' ')}\n"
            f"resolved {float(statistic['resolved_season_share']):.0%}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#4f5a51",
        )
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle(
        "NAIL-RAPM v1.4 additive partial-effect stability by completed source season",
        fontsize=15,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    figure.supxlabel("Source season start year", fontsize=10)
    figure.supylabel("Standardized partial effect", fontsize=10)
    figure.text(
        0.98,
        0.975,
        "Band: 90% marginal Kalman interval | filled marker: interval excludes zero",
        ha="right",
        va="top",
        fontsize=9,
        color="#5f6860",
    )
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def build_nail_v14_partial_effect_stability_audit(
    *,
    source_run_dir: Path | str | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str = DEFAULT_CHART_PATH,
) -> NailV14PartialEffectStabilityAuditRun:
    """Persist the posterior-aware stability audit and rendered diagnostic chart."""

    source = (
        Path(source_run_dir)
        if source_run_dir is not None
        else _latest_run(Path(artifacts_dir) / MODEL_NAME / DEFAULT_ARTIFACT_SEASON)
    )
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Partial-effect stability audit requires a NAIL-RAPM v1.4 artifact")
    models = joblib.load(source / "season_context_models.joblib")
    effects = kalman_standardized_additive_effects(models)
    summary = classify_partial_effects(effects)
    rendered_chart = Path(chart_path)
    rendered_chart.parent.mkdir(parents=True, exist_ok=True)
    render_partial_effect_stability_chart(effects, summary, rendered_chart)
    root = Path(output_root)
    run_id = (
        "nail-v14-partial-effect-stability-audit-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    effects.to_parquet(run_dir / "seasonal_partial_effects.parquet", index=False)
    summary.to_parquet(run_dir / "feature_stability_classification.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_run_dir": str(source),
                "effect_definition": "Kalman-filtered standardized additive partial effect",
                "interval": "90% marginal Kalman posterior interval",
                "resolved_z_score": RESOLVED_Z_SCORE,
                "material_standardized_weight": MATERIAL_STANDARDIZED_WEIGHT,
                "stable_sign_share": STABLE_SIGN_SHARE,
                "regime_min_run": REGIME_MIN_RUN,
                "chart_path": str(rendered_chart),
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return NailV14PartialEffectStabilityAuditRun(run_dir, source, rendered_chart)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit NAIL-RAPM v1.4 additive partial-effect stability"
    )
    parser.add_argument("--source-run-dir")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path", default=str(DEFAULT_CHART_PATH))
    args = parser.parse_args()
    run = build_nail_v14_partial_effect_stability_audit(
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(
        "NAIL-RAPM v1.4 partial-effect stability audit: "
        f"run={run.run_dir}; chart={run.chart_path}"
    )


if __name__ == "__main__":
    main()
