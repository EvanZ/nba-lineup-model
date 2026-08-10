"""Audit agreement between portable and relative contextual RAPM fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling import (
    forward_bounded_hierarchical_portable_matchup_contextual_rapm as portable_model,
)
from nba_lineup_model.modeling.contextual_features import (
    contextual_feature_columns,
    lineup_context_features,
    lineup_side_context_features,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.forward_bounded_hierarchical_contextual_rapm import (
    BoundedRelativeContextModel,
)
from nba_lineup_model.modeling.lineup_context_case_study import (
    _feature_contributions,
    _feature_label,
)
from nba_lineup_model.modeling.matchup_contextual import (
    BoundedMatchupContextualModel,
    MatchupContextualModel,
)
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints

DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_AUDITS_DIR = Path("artifacts/analysis/portable_relative_context_agreement")
DEFAULT_DOCS_ASSETS_DIR = Path("docs/assets/images/portable-relative-context-agreement")
DEFAULT_DOCS_PAGE = Path(
    "docs/models/forward-bounded-hierarchical-portable-matchup-contextual-rapm.md"
)
PORTABLE_ARTIFACT = "forward_bounded_hierarchical_portable_matchup_contextual_rapm"
RELATIVE_ARTIFACT = "forward_bounded_hierarchical_pspline_contextual_rapm"
DEFAULT_SEASON = "2025-26"
PLOT_SAMPLE_FRACTION = 0.10
PLOT_SAMPLE_SEED = 20250809
SECTION_START = "<!-- portable-relative-context-agreement:start -->"
SECTION_END = "<!-- portable-relative-context-agreement:end -->"


@dataclass(frozen=True)
class AgreementAuditRun:
    """One immutable portable-versus-relative agreement audit."""

    run_dir: Path
    run_id: str


@dataclass(frozen=True)
class CompletedContextualState:
    """The coefficients, profiles, and contextual function for one completed season."""

    run_id: str
    coefficients: pd.DataFrame
    profiles: pd.DataFrame
    context_model: MatchupContextualModel | BoundedRelativeContextModel


def build_portable_relative_context_agreement_audit(
    *,
    season: str = DEFAULT_SEASON,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    audits_dir: Path | str = DEFAULT_AUDITS_DIR,
    docs_assets_dir: Path | str = DEFAULT_DOCS_ASSETS_DIR,
    docs_page: Path | str = DEFAULT_DOCS_PAGE,
    analytical_dir: Path | str = "data/analytical",
) -> AgreementAuditRun:
    """Score observed regular-season stints with two completed contextual fits.

    This is a model-agreement audit, not a frozen predictive evaluation: both
    models use their completed state for ``season`` and score the same observed
    lineup matchups from that season.
    """

    artifacts_root = Path(artifacts_dir)
    portable = _load_completed_state(
        artifacts_root, PORTABLE_ARTIFACT, season, MatchupContextualModel
    )
    relative = _load_completed_state(
        artifacts_root, RELATIVE_ARTIFACT, season, BoundedRelativeContextModel
    )
    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    paired = _paired_predictions(stints, portable, relative)
    metrics = agreement_metrics(paired)
    feature_metrics = feature_agreement_metrics(paired)

    now = datetime.now(UTC)
    run_id = f"portable-relative-context-agreement-{season}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    root = Path(audits_dir) / season
    output = root / run_id
    temporary = root / f".{run_id}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        paired.to_parquet(temporary / "reference_matchup_predictions.parquet", index=False)
        metrics.to_parquet(temporary / "agreement_metrics.parquet", index=False)
        feature_metrics.to_parquet(temporary / "feature_agreement_metrics.parquet", index=False)
        _render_agreement_scatter_plots(
            paired, metrics, temporary / "agreement-scatter.png"
        )
        _render_feature_agreement_atlas(
            paired,
            feature_metrics,
            contextual_feature_columns()[:10],
            temporary / "feature-agreement-main.png",
            "Main profile-feature agreement",
        )
        _render_feature_agreement_atlas(
            paired,
            feature_metrics,
            contextual_feature_columns()[10:],
            temporary / "feature-agreement-composition.png",
            "Composition-summary feature agreement",
        )
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "season": season,
            "portable_model": portable_model.MODEL_NAME,
            "portable_run_id": portable.run_id,
            "relative_model": RELATIVE_ARTIFACT,
            "relative_run_id": relative.run_id,
            "evaluation_population": "Observed regular-season RAPM stints",
            "weighting": "Stint possessions",
            "comparison_contract": (
                "portable player edge versus relative player edge; portable total context "
                "C(A,B)=h(A)-h(B)+q(A,B) versus relative context g(x(A)-x(B))"
            ),
            "feature_comparison_contract": (
                "For each original contextual feature, compare portable composition plus "
                "matchup contribution against the relative-model total contribution"
            ),
            "created_at": now.isoformat(),
            "code_version": modeling_code_fingerprint((Path(__file__),)),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records = [
            {
                "filename": path.name,
                "byte_count": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file()
        ]
        (temporary / "manifest.json").write_text(
            json.dumps({**metadata, "artifacts": records}, indent=2) + "\n"
        )
        temporary.rename(output)
    except Exception:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    docs_assets = {
        "agreement": Path(docs_assets_dir) / f"{season}-agreement-scatter.png",
        "main": Path(docs_assets_dir) / f"{season}-feature-agreement-main.png",
        "composition": Path(docs_assets_dir) / f"{season}-feature-agreement-composition.png",
    }
    for key, docs_asset in docs_assets.items():
        docs_asset.parent.mkdir(parents=True, exist_ok=True)
        source_name = {
            "agreement": "agreement-scatter.png",
            "main": "feature-agreement-main.png",
            "composition": "feature-agreement-composition.png",
        }[key]
        shutil.copy2(output / source_name, docs_asset)
    _update_docs_page(Path(docs_page), metrics, feature_metrics, metadata, output, docs_assets)
    return AgreementAuditRun(run_dir=output, run_id=run_id)


def agreement_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return possession-weighted agreement metrics for each model component."""

    required = {
        "possessions",
        "portable_player_edge",
        "relative_player_edge",
        "portable_context_edge",
        "relative_context_edge",
        "portable_predicted_net_rating",
        "relative_predicted_net_rating",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Agreement predictions missing columns: {sorted(missing)}")
    weights = predictions["possessions"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Agreement weights must be finite and positive")

    components = (
        ("Net player rating edge", "portable_player_edge", "relative_player_edge"),
        ("Net context edge", "portable_context_edge", "relative_context_edge"),
        (
            "Predicted net rating",
            "portable_predicted_net_rating",
            "relative_predicted_net_rating",
        ),
    )
    rows: list[dict[str, object]] = []
    for label, portable_column, relative_column in components:
        rows.append(_agreement_metric_row(label, predictions, portable_column, relative_column))
    return pd.DataFrame(rows)


def feature_agreement_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return possession-weighted agreement metrics for every context feature."""

    required = {
        "possessions",
        *(_portable_feature_column(feature) for feature in contextual_feature_columns()),
        *(_relative_feature_column(feature) for feature in contextual_feature_columns()),
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Feature agreement predictions missing columns: {sorted(missing)}")
    rows = []
    for feature in contextual_feature_columns():
        row = _agreement_metric_row(
            _feature_label(feature),
            predictions,
            _portable_feature_column(feature),
            _relative_feature_column(feature),
        )
        row["feature"] = feature
        row["label"] = _feature_label(feature)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "weighted_pearson_correlation", ascending=False, kind="stable"
    ).reset_index(drop=True)


def _agreement_metric_row(
    label: str, predictions: pd.DataFrame, portable_column: str, relative_column: str
) -> dict[str, object]:
    weights = predictions["possessions"].to_numpy(dtype=float)
    portable = predictions[portable_column].to_numpy(dtype=float)
    relative = predictions[relative_column].to_numpy(dtype=float)
    difference = relative - portable
    return {
        "component": label,
        "stint_count": int(len(predictions)),
        "possession_count": float(weights.sum()),
        "weighted_pearson_correlation": _weighted_correlation(portable, relative, weights),
        "weighted_spearman_correlation": _weighted_correlation(
            _average_ranks(portable), _average_ranks(relative), weights
        ),
        "weighted_rmse": float(np.sqrt(np.average(difference**2, weights=weights))),
        "weighted_mae": float(np.average(np.abs(difference), weights=weights)),
        "portable_weighted_mean": float(np.average(portable, weights=weights)),
        "relative_weighted_mean": float(np.average(relative, weights=weights)),
        "relative_minus_portable_mean": float(np.average(difference, weights=weights)),
    }


def _load_completed_state(
    artifacts_dir: Path,
    artifact_name: str,
    season: str,
    expected_type: type[MatchupContextualModel] | type[BoundedRelativeContextModel],
) -> CompletedContextualState:
    latest_path = artifacts_dir / artifact_name / season / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Model latest pointer is missing: {latest_path}")
    run_id = str(json.loads(latest_path.read_text())["run_id"])
    run_dir = latest_path.parent / run_id
    coefficients = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
    coefficients = coefficients.loc[
        coefficients["season"].eq(season), ["player_id", "rapm"]
    ].copy()
    profiles = pd.read_parquet(run_dir / "target_player_profiles.parquet")
    models = joblib.load(run_dir / "season_context_models.joblib")
    context_model = models.get(season)
    if coefficients.empty or coefficients["player_id"].duplicated().any():
        raise ValueError(f"{artifact_name} has invalid completed player coefficients")
    if profiles["player_id"].duplicated().any():
        raise ValueError(f"{artifact_name} has duplicate player profiles")
    if not isinstance(context_model, expected_type):
        raise ValueError(f"{artifact_name} has an incompatible contextual model")
    return CompletedContextualState(run_id, coefficients, profiles, context_model)


def _paired_predictions(
    stints: pd.DataFrame,
    portable: CompletedContextualState,
    relative: CompletedContextualState,
) -> pd.DataFrame:
    home_lineups = stints["home_player_ids"].tolist()
    away_lineups = stints["away_player_ids"].tolist()
    _require_player_coverage(home_lineups, away_lineups, portable, "portable")
    _require_player_coverage(home_lineups, away_lineups, relative, "relative")
    if not isinstance(portable.context_model, MatchupContextualModel):
        raise TypeError("Portable state does not expose a portable-matchup context model")
    if not isinstance(relative.context_model, BoundedRelativeContextModel):
        raise TypeError("Relative state does not expose a bounded relative context model")
    portable_player_edge = _player_edges(home_lineups, away_lineups, portable.coefficients)
    relative_player_edge = _player_edges(home_lineups, away_lineups, relative.coefficients)
    portable_home = lineup_side_context_features(home_lineups, portable.profiles)
    portable_away = lineup_side_context_features(away_lineups, portable.profiles)
    portable_relative = _portable_relative_features(
        portable.context_model, portable_home, portable_away
    )
    portable_context_edge = portable.context_model.predict_side_pairs(
        portable_home, portable_away
    )
    portable_feature_values = _orientation_symmetric_feature_contributions(
        portable.context_model.pipeline, portable_relative
    )
    relative_features = lineup_context_features(home_lineups, away_lineups, relative.profiles)
    relative_context_edge = relative.context_model.predict(relative_features)
    relative_bounded = relative_features.clip(
        relative.context_model.lower, relative.context_model.upper, axis=1
    )
    relative_feature_values = _feature_contributions(
        relative.context_model.pipeline, relative_bounded
    )
    output = stints.loc[:, ["game_id", "stint_index", "possessions"]].assign(
        portable_player_edge=portable_player_edge,
        relative_player_edge=relative_player_edge,
        portable_context_edge=portable_context_edge,
        relative_context_edge=relative_context_edge,
        portable_predicted_net_rating=portable_player_edge + portable_context_edge,
        relative_predicted_net_rating=relative_player_edge + relative_context_edge,
    )
    for index, feature in enumerate(contextual_feature_columns()):
        output[_portable_feature_column(feature)] = portable_feature_values[:, index]
        output[_relative_feature_column(feature)] = relative_feature_values[:, index]
    portable_residual = portable_feature_values.sum(axis=1) - portable_context_edge
    if not np.allclose(portable_residual, 0.0, atol=1e-6):
        raise ValueError(
            "Portable feature contributions do not sum to the context edge: "
            f"maximum absolute residual={np.abs(portable_residual).max():.12f}"
        )
    relative_residual = relative_feature_values.sum(axis=1) - relative_context_edge
    if not np.allclose(relative_residual, 0.0, atol=1e-5):
        raise ValueError(
            "Relative feature contributions do not sum to the context edge: "
            f"maximum absolute residual={np.abs(relative_residual).max():.12f}"
        )
    return output


def _portable_relative_features(
    context_model: MatchupContextualModel,
    home: pd.DataFrame,
    away: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact bounded relative feature inputs of the portable model."""

    home_values = home.loc[:, side_context_feature_columns()].copy()
    away_values = away.loc[:, side_context_feature_columns()].copy()
    if isinstance(context_model, BoundedMatchupContextualModel):
        home_values = home_values.clip(
            context_model.side_lower, context_model.side_upper, axis=1
        )
        away_values = away_values.clip(
            context_model.side_lower, context_model.side_upper, axis=1
        )
    relative = pd.DataFrame(
        {
            feature: (
                home_values[side_context_feature_columns()[index]].to_numpy(dtype=float)
                - away_values[side_context_feature_columns()[index]].to_numpy(dtype=float)
            )
            for index, feature in enumerate(contextual_feature_columns())
        },
        columns=contextual_feature_columns(),
    )
    if isinstance(context_model, BoundedMatchupContextualModel):
        return relative.clip(-context_model.relative_cap, context_model.relative_cap, axis=1)
    return relative


def _orientation_symmetric_feature_contributions(
    pipeline: object, features: pd.DataFrame
) -> np.ndarray:
    """Return per-feature contributions under the exact antisymmetric prediction."""

    forward = _feature_contributions(pipeline, features)
    reverse = _feature_contributions(pipeline, -features)
    return 0.5 * (forward - reverse)


def _portable_feature_column(feature: str) -> str:
    return f"portable_feature_{feature}"


def _relative_feature_column(feature: str) -> str:
    return f"relative_feature_{feature}"


def _require_player_coverage(
    home_lineups: list[list[int]],
    away_lineups: list[list[int]],
    state: CompletedContextualState,
    label: str,
) -> None:
    stint_players = set().union(*home_lineups, *away_lineups)
    available = set(state.coefficients["player_id"].astype(int)) & set(
        state.profiles["player_id"].astype(int)
    )
    missing = sorted(stint_players - available)
    if missing:
        raise ValueError(f"{label} model is missing {len(missing)} observed stint players")


def _player_edges(
    home_lineups: list[list[int]], away_lineups: list[list[int]], coefficients: pd.DataFrame
) -> np.ndarray:
    values = dict(
        zip(
            coefficients["player_id"].astype(int), coefficients["rapm"].astype(float), strict=True
        )
    )
    return np.asarray(
        [
            sum(values[int(player_id)] for player_id in home)
            - sum(values[int(player_id)] for player_id in away)
            for home, away in zip(home_lineups, away_lineups, strict=True)
        ],
        dtype=float,
    )


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    left_centered = left - np.average(left, weights=weights)
    right_centered = right - np.average(right, weights=weights)
    denominator = np.sqrt(
        np.average(left_centered**2, weights=weights)
        * np.average(right_centered**2, weights=weights)
    )
    return float(np.average(left_centered * right_centered, weights=weights) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_agreement_scatter_plots(
    predictions: pd.DataFrame, metrics: pd.DataFrame, path: Path
) -> None:
    """Render compact possession-weighted agreement plots for the documentation."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    sample_count = max(1, int(np.ceil(len(predictions) * PLOT_SAMPLE_FRACTION)))
    plotted = predictions.sample(n=sample_count, random_state=PLOT_SAMPLE_SEED)
    components = (
        (
            "Player-rating edge",
            "Net player rating edge",
            "relative_player_edge",
            "portable_player_edge",
        ),
        ("Context edge", "Net context edge", "relative_context_edge", "portable_context_edge"),
    )
    weights = predictions["possessions"].to_numpy(dtype=float)
    by_component = metrics.set_index("component")
    for axis, (label, metric_component, relative_column, portable_column) in zip(
        axes, components, strict=True
    ):
        relative = predictions[relative_column].to_numpy(dtype=float)
        portable = predictions[portable_column].to_numpy(dtype=float)
        plotted_relative = plotted[relative_column].to_numpy(dtype=float)
        plotted_portable = plotted[portable_column].to_numpy(dtype=float)
        plotted_sizes = np.clip(
            np.sqrt(plotted["possessions"].to_numpy(dtype=float)), 1.0, 5.0
        )
        metric = by_component.loc[metric_component]
        low = float(min(relative.min(), portable.min()))
        high = float(max(relative.max(), portable.max()))
        padding = max((high - low) * 0.05, 0.1)
        low -= padding
        high += padding
        slope, intercept = _weighted_linear_fit(relative, portable, weights)
        line = np.asarray([low, high])
        axis.scatter(
            plotted_relative,
            plotted_portable,
            s=plotted_sizes,
            color="#245a47",
            alpha=0.14,
            edgecolors="none",
            rasterized=True,
        )
        axis.plot(line, line, color="#767d76", linestyle="--", linewidth=1.0, label="1:1")
        axis.plot(
            line,
            intercept + slope * line,
            color="#e8502f",
            linewidth=1.6,
            label="Weighted fit",
        )
        axis.axhline(0, color="#aab0ab", linewidth=0.6, zorder=0)
        axis.axvline(0, color="#aab0ab", linewidth=0.6, zorder=0)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{label}\nr = {metric.weighted_pearson_correlation:.4f} | "
            f"rho = {metric.weighted_spearman_correlation:.4f}",
            fontweight="bold",
            loc="left",
        )
        axis.set_xlabel("Relative-context model (per 100 possessions)")
        axis.set_ylabel("Portable model (per 100 possessions)")
        axis.tick_params(labelsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    figure.suptitle(
        "Completed 2025-26 model agreement on observed regular-season matchup stints",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_feature_agreement_atlas(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    features: tuple[str, ...],
    path: Path,
    title: str,
) -> None:
    """Render a compact feature-attribution agreement atlas."""

    import matplotlib.pyplot as plt

    columns = 2
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(12, rows * 3.5), constrained_layout=True, squeeze=False
    )
    sample_count = max(1, int(np.ceil(len(predictions) * PLOT_SAMPLE_FRACTION)))
    plotted = predictions.sample(n=sample_count, random_state=PLOT_SAMPLE_SEED)
    weights = predictions["possessions"].to_numpy(dtype=float)
    metric_by_feature = metrics.set_index("feature")
    for axis, feature in zip(axes.flat, features, strict=False):
        relative = predictions[_relative_feature_column(feature)].to_numpy(dtype=float)
        portable = predictions[_portable_feature_column(feature)].to_numpy(dtype=float)
        plotted_relative = plotted[_relative_feature_column(feature)].to_numpy(dtype=float)
        plotted_portable = plotted[_portable_feature_column(feature)].to_numpy(dtype=float)
        plotted_sizes = np.clip(
            np.sqrt(plotted["possessions"].to_numpy(dtype=float)), 1.0, 4.0
        )
        low = float(min(relative.min(), portable.min()))
        high = float(max(relative.max(), portable.max()))
        padding = max((high - low) * 0.06, 0.02)
        low -= padding
        high += padding
        slope, intercept = _weighted_linear_fit(relative, portable, weights)
        line = np.asarray([low, high])
        metric = metric_by_feature.loc[feature]
        axis.scatter(
            plotted_relative,
            plotted_portable,
            s=plotted_sizes,
            color="#245a47",
            alpha=0.14,
            edgecolors="none",
            rasterized=True,
        )
        axis.plot(line, line, color="#767d76", linestyle="--", linewidth=0.8)
        axis.plot(line, intercept + slope * line, color="#e8502f", linewidth=1.2)
        axis.axhline(0, color="#aab0ab", linewidth=0.5, zorder=0)
        axis.axvline(0, color="#aab0ab", linewidth=0.5, zorder=0)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{_feature_label(feature)}\nr = {metric.weighted_pearson_correlation:.3f} | "
            f"rho = {metric.weighted_spearman_correlation:.3f}",
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        axis.set_xlabel("Relative", fontsize=8)
        axis.set_ylabel("Portable", fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
    for axis in axes.flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{title}: completed 2025-26 context-feature contributions",
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _weighted_linear_fit(
    predictor: np.ndarray, response: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    predictor_mean = float(np.average(predictor, weights=weights))
    response_mean = float(np.average(response, weights=weights))
    centered = predictor - predictor_mean
    variance = float(np.average(centered**2, weights=weights))
    if variance == 0:
        return 0.0, response_mean
    covariance = float(np.average(centered * (response - response_mean), weights=weights))
    slope = covariance / variance
    return slope, response_mean - slope * predictor_mean


def _update_docs_page(
    docs_page: Path,
    metrics: pd.DataFrame,
    feature_metrics: pd.DataFrame,
    metadata: dict[str, object],
    output: Path,
    docs_assets: dict[str, Path],
) -> None:
    metric_rows = []
    for row in metrics.itertuples(index=False):
        metric_rows.append(
            f"| {row.component} | {row.weighted_pearson_correlation:.4f} | "
            f"{row.weighted_spearman_correlation:.4f} | {row.weighted_rmse:.3f} | "
            f"{row.weighted_mae:.3f} |"
        )
    section = "\n".join(
        [
            SECTION_START,
            "## Relative-Context Agreement Audit",
            "",
            "This is a possession-weighted agreement audit on the **observed 2025-26 "
            "regular-season matchup stints**. It is not a frozen predictive comparison: "
            "both completed 2025-26 fits score the same lineups with their own player "
            "coefficients, profiles, and context functions.",
            "",
            "The portable model's net context edge is "
            "\\(C(A,B)=h(A)-h(B)+q(A,B)\\). The relative-context reference's net "
            "context edge is \\(g(x(A)-x(B))\\). Thus the table tests whether the "
            "two separately trained decompositions agree on the same realized lineup "
            "field; it does not establish that either estimate is correct.",
            "",
            "![Portable and relative-context correlation plots]"
            f"(../assets/images/portable-relative-context-agreement/{docs_assets['agreement'].name})",
            "",
            "Each dot is one observed stint. The axes use each model's net edge in "
            "points per 100 possessions; dot area is capped square-root stint possessions. "
            "The plot uses a deterministic 10% sample, while the gray dashed line, orange "
            "possession-weighted least-squares fit, and table all use the full stint field.",
            "",
            "| Component | Weighted Pearson | Weighted Spearman | RMSE | MAE |",
            "| --- | ---: | ---: | ---: | ---: |",
            *metric_rows,
            "",
            "### Feature-Level Agreement",
            "",
            "Both contextual functions are additive over the same 20 original features. "
            "For each feature, this audit compares the portable model's total feature "
            "contribution, which is its composition contribution plus its matchup residual, "
            "with the relative model's contribution to \\(g(x(A)-x(B))\\). Each model's "
            "20 contributions sum exactly to its net context edge. This is therefore an "
            "attribution-agreement diagnostic, not a comparison of the portable-only "
            "composition term \\(h(A)-h(B)\\), for which the relative model has no analog.",
            "",
            "| Feature | Weighted Pearson | Weighted Spearman | RMSE | MAE |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                f"| {row.label} | {row.weighted_pearson_correlation:.4f} | "
                f"{row.weighted_spearman_correlation:.4f} | {row.weighted_rmse:.3f} | "
                f"{row.weighted_mae:.3f} |"
                for row in feature_metrics.itertuples(index=False)
            ],
            "",
            "![Main profile-feature agreement atlas]"
            f"(../assets/images/portable-relative-context-agreement/{docs_assets['main'].name})",
            "",
            "![Composition-summary feature agreement atlas]"
            f"(../assets/images/portable-relative-context-agreement/{docs_assets['composition'].name})",
            "",
            "The atlases use the same deterministic 10% stint sample for dots, but every "
            "reported metric and orange fit uses the full possession-weighted stint field. "
            "The gray dashed line marks exact contribution agreement.",
            "",
            "The audit includes {stints:,} stints representing {possessions:,.1f} "
            "possessions. RMSE and MAE are in net-rating points per 100 possessions. "
            "Artifact: `{artifact}`. Source runs: portable `{portable}`; relative "
            "`{relative}`.".format(
                stints=int(metrics["stint_count"].iloc[0]),
                possessions=float(metrics["possession_count"].iloc[0]),
                artifact=output,
                portable=metadata["portable_run_id"],
                relative=metadata["relative_run_id"],
            ),
            SECTION_END,
        ]
    )
    source = docs_page.read_text()
    if SECTION_START in source and SECTION_END in source:
        before, _, remainder = source.partition(SECTION_START)
        _, _, after = remainder.partition(SECTION_END)
        docs_page.write_text(before.rstrip() + "\n\n" + section + after)
    else:
        docs_page.write_text(source.rstrip() + "\n\n" + section + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit agreement between completed portable and relative contextual RAPM"
    )
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--audits-dir", type=Path, default=DEFAULT_AUDITS_DIR)
    parser.add_argument("--docs-assets-dir", type=Path, default=DEFAULT_DOCS_ASSETS_DIR)
    parser.add_argument("--docs-page", type=Path, default=DEFAULT_DOCS_PAGE)
    parser.add_argument("--analytical-dir", type=Path, default=Path("data/analytical"))
    args = parser.parse_args()
    run = build_portable_relative_context_agreement_audit(
        season=args.season,
        artifacts_dir=args.artifacts_dir,
        audits_dir=args.audits_dir,
        docs_assets_dir=args.docs_assets_dir,
        docs_page=args.docs_page,
        analytical_dir=args.analytical_dir,
    )
    print(f"Portable-relative context agreement audit: run={run.run_dir}")


if __name__ == "__main__":
    main()
