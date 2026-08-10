"""No-refit audit of saved contextual spline response functions."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import contextual_feature_columns
from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import MODEL_NAME
from nba_lineup_model.modeling.lineup_context_case_study import _feature_label
from nba_lineup_model.modeling.matchup_contextual import (
    MatchupContextualModel,
    isolated_feature_component,
)

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_AUDITS_DIR = Path("artifacts/analysis/context_function_audit")
DEFAULT_DOCS_ASSETS_DIR = Path("docs/assets/images/context-function-audit")
DEFAULT_DOCS_PAGE = Path("docs/models/context-function-audit.md")
MODEL_ARTIFACT = "forward_portable_matchup_contextual_rapm"
PERCENTILES = np.linspace(0.05, 0.95, 61)
REFERENCE_PAIR_SAMPLE_SIZE = 30_000
SECTION_START = "<!-- context-function-audit:start -->"
SECTION_END = "<!-- context-function-audit:end -->"


@dataclass(frozen=True)
class ContextFunctionAuditRun:
    """Materialized audit tied to one immutable contextual model artifact."""

    run_dir: Path
    run_id: str


def build_context_function_audit(
    *,
    through_season: str = "2025-26",
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    audits_dir: Path | str = DEFAULT_AUDITS_DIR,
    docs_assets_dir: Path | str = DEFAULT_DOCS_ASSETS_DIR,
    docs_page: Path | str = DEFAULT_DOCS_PAGE,
) -> ContextFunctionAuditRun:
    """Render seasonal response atlases from stored models without retraining."""

    source_dir, source_run_id = _latest_context_run(Path(artifacts_dir), through_season)
    models = joblib.load(source_dir / "season_context_models.joblib")
    if not isinstance(models, dict) or not models:
        raise ValueError("Contextual artifact has no seasonal context model map")
    seasonal_models = {
        str(season): model
        for season, model in models.items()
        if isinstance(model, MatchupContextualModel)
    }
    if len(seasonal_models) != len(models):
        raise ValueError("Contextual artifact contains an incompatible seasonal model")

    curves = _curve_records(seasonal_models)
    summary = _summarize_curves(curves)
    run_id = f"context-function-audit-{through_season}-{source_run_id[-12:]}"
    root = Path(audits_dir) / through_season / run_id
    root.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(root / "response_curves.parquet", index=False)
    summary.to_parquet(root / "feature_summary.parquet", index=False)
    metadata = {
        "run_id": run_id,
        "source_model": MODEL_NAME,
        "source_run_id": source_run_id,
        "through_season": through_season,
        "season_count": int(curves["season"].nunique()),
        "feature_count": int(curves["feature"].nunique()),
        "percentiles": [float(value) for value in PERCENTILES],
        "reference_pair_sample_size": REFERENCE_PAIR_SAMPLE_SIZE,
        "response_contract": "r_s,k(z) = [f_s,k(z) - f_s,k(-z)] / 2",
        "support_contract": "Possession-weighted independent reference-side differences",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    assets = Path(docs_assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    _render_atlas(
        curves,
        summary,
        _main_feature_columns(),
        assets / "main-effects.svg",
        "Main profile-feature response atlas",
    )
    _render_atlas(
        curves,
        summary,
        _composition_feature_columns(),
        assets / "composition-features.svg",
        "Composition-summary response atlas",
    )
    _render_temporal_contrast_atlas(
        curves,
        summary,
        assets / "temporal-contrast.svg",
    )
    _update_docs_page(Path(docs_page), summary, metadata, root)
    return ContextFunctionAuditRun(run_dir=root, run_id=run_id)


def _latest_context_run(artifacts_dir: Path, season: str) -> tuple[Path, str]:
    latest_path = artifacts_dir / MODEL_ARTIFACT / season / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Contextual latest pointer is missing: {latest_path}")
    run_id = str(json.loads(latest_path.read_text())["run_id"])
    root = latest_path.parent / run_id
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Contextual latest pointer does not identify the portable-matchup model")
    return root, run_id


def _curve_records(models: dict[str, MatchupContextualModel]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = contextual_feature_columns()
    for season, model in sorted(models.items()):
        support = _relative_support(model, season)
        for index, feature in enumerate(features):
            values = np.quantile(support[:, index], PERCENTILES)
            response = isolated_feature_component(model, index, values)
            for percentile, value, contribution in zip(PERCENTILES, values, response, strict=True):
                rows.append(
                    {
                        "season": season,
                        "feature": feature,
                        "label": _feature_label(feature),
                        "feature_group": _feature_group(feature),
                        "percentile": float(percentile),
                        "feature_difference": float(value),
                        "response_net_rating": float(contribution),
                    }
                )
    return pd.DataFrame(rows)


def _relative_support(model: MatchupContextualModel, season: str) -> np.ndarray:
    """Draw stable possession-weighted independent unit differences for one season."""

    reference = model.reference_features.to_numpy(dtype=float)
    digest = hashlib.sha256(season.encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    indices = generator.choice(
        len(reference),
        size=(2, REFERENCE_PAIR_SAMPLE_SIZE),
        replace=True,
        p=model.reference_weights,
    )
    return reference[indices[0]] - reference[indices[1]]


def _summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature, group in curves.groupby("feature", sort=False):
        seasons = []
        for season, season_curve in group.groupby("season", sort=False):
            ordered = season_curve.sort_values("percentile", kind="stable")
            response = ordered["response_net_rating"].to_numpy(dtype=float)
            chord = np.linspace(response[0], response[-1], len(response))
            seasons.append(
                {
                    "season": season,
                    "response_range": float(response.max() - response.min()),
                    "curvature": float(np.abs(response - chord).max()),
                    "turning_points": _turning_points(response),
                    "support_low": float(ordered["feature_difference"].iloc[0]),
                    "support_high": float(ordered["feature_difference"].iloc[-1]),
                }
            )
        season_summary = pd.DataFrame(seasons)
        pivot = group.pivot(index="season", columns="percentile", values="response_net_rating")
        latest = season_summary.loc[season_summary["season"].eq(season_summary["season"].max())]
        contrast = pivot[0.95] - pivot[0.05]
        years = np.asarray([int(str(season)[:4]) for season in contrast.index], dtype=float)
        trend = np.polyfit(years, contrast.to_numpy(dtype=float), 1)
        fitted = np.polyval(trend, years)
        variation = float(((contrast - contrast.mean()) ** 2).sum())
        trend_r_squared = (
            1 - float(((contrast.to_numpy(dtype=float) - fitted) ** 2).sum()) / variation
            if variation
            else 0.0
        )
        records.append(
            {
                "feature": feature,
                "label": str(group["label"].iloc[0]),
                "feature_group": str(group["feature_group"].iloc[0]),
                "season_count": int(len(season_summary)),
                "median_response_range": float(season_summary["response_range"].median()),
                "median_curvature": float(season_summary["curvature"].median()),
                "median_turning_points": float(season_summary["turning_points"].median()),
                "median_cross_season_sd": float(pivot.std(axis=0).median()),
                "latest_support_low": float(latest["support_low"].iloc[0]),
                "latest_support_high": float(latest["support_high"].iloc[0]),
                "contrast_slope_per_decade": float(trend[0] * 10),
                "contrast_linear_r_squared": float(trend_r_squared),
                "contrast_season_sd": float(contrast.std(ddof=1)),
                "contrast_min": float(contrast.min()),
                "contrast_max": float(contrast.max()),
            }
        )
    return pd.DataFrame(records).sort_values(["feature_group", "label"], kind="stable")


def _turning_points(values: np.ndarray) -> int:
    slopes = np.diff(values)
    signs = np.sign(slopes[np.abs(slopes) > 1e-8])
    return int(np.count_nonzero(np.diff(signs) != 0)) if len(signs) > 1 else 0


def _main_feature_columns() -> tuple[str, ...]:
    return tuple(f"home_minus_away_{column}" for column in PROFILE_RATE_COLUMNS)


def _composition_feature_columns() -> tuple[str, ...]:
    main = set(_main_feature_columns())
    return tuple(feature for feature in contextual_feature_columns() if feature not in main)


def _feature_group(feature: str) -> str:
    return "profile rate" if feature in _main_feature_columns() else "composition summary"


def _render_atlas(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    features: tuple[str, ...],
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    columns = 3
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3.2), squeeze=False)
    latest_season = str(curves["season"].max())
    for axis, feature in zip(axes.flat, features, strict=False):
        feature_curves = curves.loc[curves["feature"].eq(feature)]
        for _season, season_curve in feature_curves.groupby("season", sort=False):
            ordered = season_curve.sort_values("percentile", kind="stable")
            axis.plot(
                ordered["percentile"] * 100,
                ordered["response_net_rating"],
                color="#b8c8c0",
                linewidth=0.65,
                alpha=0.55,
                zorder=1,
            )
        median = feature_curves.groupby("percentile", as_index=False)[
            "response_net_rating"
        ].median()
        current = feature_curves.loc[feature_curves["season"].eq(latest_season)]
        axis.plot(
            median["percentile"] * 100,
            median["response_net_rating"],
            color="#e8502f",
            linewidth=1.8,
            label="Historical median",
            zorder=3,
        )
        axis.plot(
            current["percentile"] * 100,
            current["response_net_rating"],
            color="#245a47",
            linewidth=2.1,
            label=latest_season,
            zorder=4,
        )
        axis.axhline(0, color="#767d76", linewidth=0.75, linestyle="--", zorder=0)
        axis.set_title(_feature_label(feature), fontsize=10, loc="left", fontweight="bold")
        axis.set_xlim(5, 95)
        axis.set_xticks([5, 50, 95])
        axis.tick_params(axis="both", labelsize=8)
        stat = summary.loc[summary["feature"].eq(feature)].iloc[0]
        axis.text(
            0.98,
            0.04,
            f"turns {stat.median_turning_points:.0f} | sd {stat.median_cross_season_sd:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#5f6860",
        )
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=15, x=0.02, ha="left", fontweight="bold")
    figure.supxlabel("Within-season relative-difference percentile", fontsize=10)
    figure.supylabel("Orientation-symmetrized context contribution", fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _render_temporal_contrast_atlas(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    features = tuple(summary["feature"])
    columns = 4
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 2.8), squeeze=False)
    for axis, feature in zip(axes.flat, features, strict=False):
        feature_curves = curves.loc[curves["feature"].eq(feature)]
        pivot = feature_curves.pivot(
            index="season", columns="percentile", values="response_net_rating"
        )
        years = np.asarray([int(str(season)[:4]) for season in pivot.index], dtype=float)
        contrast = (pivot[0.95] - pivot[0.05]).to_numpy(dtype=float)
        trend = np.polyfit(years, contrast, 1)
        axis.plot(years, contrast, color="#9eaba4", linewidth=1.1, marker="o", markersize=2.5)
        axis.plot(years, np.polyval(trend, years), color="#e8502f", linewidth=1.5)
        axis.axhline(0, color="#767d76", linewidth=0.7, linestyle="--")
        axis.set_title(_feature_label(feature), fontsize=9, loc="left", fontweight="bold")
        axis.set_xticks([years.min(), years.max()])
        axis.tick_params(axis="both", labelsize=7)
        stat = summary.loc[summary["feature"].eq(feature)].iloc[0]
        axis.text(
            0.98,
            0.04,
            f"R² {stat.contrast_linear_r_squared:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#5f6860",
        )
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Temporal central-response contrasts: 95th percentile minus 5th percentile",
        fontsize=15,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    figure.supxlabel("Season start year", fontsize=10)
    figure.supylabel("Central response contrast", fontsize=10)
    figure.text(
        0.98, 0.975, "Gray: season | Orange: linear trend", ha="right", va="top", fontsize=9
    )
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _update_docs_page(
    page: Path,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    artifact_root: Path,
) -> None:
    source = page.read_text()
    if SECTION_START not in source or SECTION_END not in source:
        raise ValueError("Context-function audit page is missing generated-section markers")
    rows = [
        "| Feature | Type | Median range | Median curvature | Median turns | Cross-season SD | "
        "Latest support |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.itertuples(index=False):
        rows.append(
            f"| {row.label} | {row.feature_group} | {row.median_response_range:.2f} | "
            f"{row.median_curvature:.2f} | {row.median_turning_points:.0f} | "
            f"{row.median_cross_season_sd:.2f} | "
            f"[{row.latest_support_low:.2f}, {row.latest_support_high:.2f}] |"
        )
    temporal_rows = [
        "| Feature | Contrast trend / decade | Linear R² | Season SD | Observed contrast range |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary.itertuples(index=False):
        temporal_rows.append(
            f"| {row.label} | {row.contrast_slope_per_decade:+.2f} | "
            f"{row.contrast_linear_r_squared:.2f} | {row.contrast_season_sd:.2f} | "
            f"[{row.contrast_min:+.2f}, {row.contrast_max:+.2f}] |"
        )
    generated = [
        SECTION_START,
        "",
        "## Current Audit",
        "",
        f"Source artifact: `{metadata['source_run_id']}`. The audit evaluates "
        f"{metadata['season_count']} saved seasonal states without refitting. "
        "Faint curves are individual seasons, orange is the historical median, and green is "
        "2025-26.",
        "",
        "The x-axis is each season's 5th-to-95th percentile of possession-weighted "
        "independent reference-unit differences. This makes curve shapes comparable across eras "
        "without treating raw feature scales as stationary.",
        "",
        "![Main profile-feature response atlas]"
        "(../assets/images/context-function-audit/main-effects.svg)",
        "",
        "![Composition-summary response atlas]"
        "(../assets/images/context-function-audit/composition-features.svg)",
        "",
        "### Temporal Direction Versus Scatter",
        "",
        "Each point below is a season's signed central response contrast: the 95th-percentile "
        "response minus the 5th-percentile response. Orange is a least-squares time trend. "
        "A low R² means the annual movement is primarily scatter rather than a sustained "
        "directional evolution.",
        "",
        "![Temporal central-response contrast atlas]"
        "(../assets/images/context-function-audit/temporal-contrast.svg)",
        "",
        *temporal_rows,
        "",
        "### Stability Summary",
        "",
        "`Median range` is the within-season 5th-to-95th response range. `Median curvature` "
        "is the largest departure from that season's endpoint chord. `Median turns` counts "
        "direction reversals across the central response interval. `Cross-season SD` is the "
        "median seasonal standard deviation at matched percentiles. These are screening "
        "diagnostics, not automatic feature-selection rules.",
        "",
        *rows,
        "",
        f"Immutable generated artifact: `{artifact_root}`.",
        "",
        SECTION_END,
    ]
    before, remainder = source.split(SECTION_START, maxsplit=1)
    _, after = remainder.split(SECTION_END, maxsplit=1)
    page.write_text(before + "\n".join(generated) + after)


def main() -> None:
    """Build the no-refit context-function audit and documentation report."""

    parser = argparse.ArgumentParser(description="Audit saved contextual spline response functions")
    parser.add_argument("--through-season", default="2025-26")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--audits-dir", default=str(DEFAULT_AUDITS_DIR))
    parser.add_argument("--docs-assets-dir", default=str(DEFAULT_DOCS_ASSETS_DIR))
    parser.add_argument("--docs-page", default=str(DEFAULT_DOCS_PAGE))
    args = parser.parse_args()
    result = build_context_function_audit(
        through_season=args.through_season,
        artifacts_dir=args.artifacts_dir,
        audits_dir=args.audits_dir,
        docs_assets_dir=args.docs_assets_dir,
        docs_page=args.docs_page,
    )
    print(f"Wrote context-function audit: {result.run_dir}")


if __name__ == "__main__":
    main()
