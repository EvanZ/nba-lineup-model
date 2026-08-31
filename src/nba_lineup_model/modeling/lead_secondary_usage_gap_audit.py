"""Condition the lead-secondary usage-gap screen on frozen superstar imbalance.

The frozen residual screen shows whether a proposed coordinate is associated
with unexplained target-season scoring. This companion audit asks a narrower
mechanistic question: does the association remain after accounting for the
largest frozen player prior on each side? It reuses the immutable screen
residuals and never refits NAIL.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.frozen_feature_screen import DEFAULT_SEASONS
from nba_lineup_model.modeling.stints import read_rapm_stints

DEFAULT_SCREEN_ROOT = Path("artifacts/models/analysis/frozen_feature_screen")
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/lead_secondary_usage_gap_conditioning")
DEFAULT_MODEL_ARTIFACT_SEASON = "2025-26"


@dataclass(frozen=True)
class ConditionalRegression:
    """Possession-weighted standardized residual-regression summary."""

    intercept: float
    lead_secondary_usage_gap: float
    max_frozen_prior_edge: float
    predictor_correlation: float


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = _weighted_mean(values, weights)
    variance = _weighted_mean(np.square(values - mean), weights)
    if variance <= 0.0:
        raise ValueError("Conditional audit predictor has no weighted variation")
    return (values - mean) / np.sqrt(variance)


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    centered_left = left - _weighted_mean(left, weights)
    centered_right = right - _weighted_mean(right, weights)
    denominator = np.sqrt(
        _weighted_mean(np.square(centered_left), weights)
        * _weighted_mean(np.square(centered_right), weights)
    )
    return float(_weighted_mean(centered_left * centered_right, weights) / denominator)


def fit_conditional_regression(frame: pd.DataFrame) -> ConditionalRegression:
    """Fit weighted residuals on the standardized candidate and star edges."""

    weights = frame["possessions"].to_numpy(dtype=float)
    residual = frame["frozen_residual_net_rating"].to_numpy(dtype=float)
    gap = _weighted_standardize(frame["feature_edge"].to_numpy(dtype=float), weights)
    star = _weighted_standardize(frame["max_frozen_prior_edge"].to_numpy(dtype=float), weights)
    design = np.column_stack((np.ones(len(frame)), gap, star))
    coefficients, *_ = np.linalg.lstsq(
        design * np.sqrt(weights)[:, None], residual * np.sqrt(weights), rcond=None
    )
    return ConditionalRegression(
        intercept=float(coefficients[0]),
        lead_secondary_usage_gap=float(coefficients[1]),
        max_frozen_prior_edge=float(coefficients[2]),
        predictor_correlation=_weighted_correlation(gap, star, weights),
    )


def fit_unadjusted_usage_gap_weight(frame: pd.DataFrame) -> float:
    """Fit the frozen residual on the standardized usage-gap edge alone."""

    weights = frame["possessions"].to_numpy(dtype=float)
    residual = frame["frozen_residual_net_rating"].to_numpy(dtype=float)
    gap = _weighted_standardize(frame["feature_edge"].to_numpy(dtype=float), weights)
    design = np.column_stack((np.ones(len(frame)), gap))
    coefficients, *_ = np.linalg.lstsq(
        design * np.sqrt(weights)[:, None], residual * np.sqrt(weights), rcond=None
    )
    return float(coefficients[1])


def _latest_screen(screen_root: Path) -> Path:
    pointer = screen_root / "lead_secondary_usage_gap" / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"Lead-secondary screen pointer does not exist: {pointer}")
    return pointer.parent / json.loads(pointer.read_text())["run_id"]


def _max_frozen_prior_edge(stints: pd.DataFrame, prior_map: dict[int, float]) -> np.ndarray:
    def unit_max(player_ids: list[int]) -> float:
        return max(
            (float(prior_map.get(int(player_id), 0.0)) for player_id in player_ids), default=0.0
        )

    home = np.asarray([unit_max(ids) for ids in stints["home_player_ids"]], dtype=float)
    away = np.asarray([unit_max(ids) for ids in stints["away_player_ids"]], dtype=float)
    return home - away


def build_lead_secondary_usage_gap_conditioning_audit(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    screen_root: Path | str = DEFAULT_SCREEN_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    model_artifact_season: str = DEFAULT_MODEL_ARTIFACT_SEASON,
) -> Path:
    """Persist a no-refit conditional audit for the lead-secondary candidate."""

    target_seasons = tuple(sorted({str(season) for season in seasons}))
    screen_dir = _latest_screen(Path(screen_root))
    screen = pd.read_parquet(screen_dir / "stint_residuals.parquet")
    model_dir = _latest_run(Path(artifacts_dir) / MODEL_NAME / model_artifact_season)
    priors = pd.read_parquet(model_dir / "season_player_priors.parquet")

    rows: list[pd.DataFrame] = []
    for season in target_seasons:
        stints = read_rapm_stints(season, analytical_dir=analytical_dir)
        season_screen = screen.loc[screen["season"].eq(season)].copy()
        keys = ["season", "game_id", "stint_index"]
        aligned = season_screen.merge(
            stints.loc[:, keys + ["home_player_ids", "away_player_ids"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if aligned[["home_player_ids", "away_player_ids"]].isna().any().any():
            raise ValueError(f"Could not align all screen stints for {season}")
        target_priors = priors.loc[priors["season"].eq(season), ["player_id", "prior_rapm"]]
        prior_map = dict(
            zip(target_priors["player_id"].astype(int), target_priors["prior_rapm"], strict=True)
        )
        aligned["max_frozen_prior_edge"] = _max_frozen_prior_edge(aligned, prior_map)
        aligned["source_season"] = _previous_season(season)
        rows.append(aligned)

    audited = pd.concat(rows, ignore_index=True)
    summaries: list[dict[str, float | int | str]] = []
    for label, group in [
        *((season, audited.loc[audited["season"].eq(season)]) for season in target_seasons),
        ("pooled", audited),
    ]:
        result = fit_conditional_regression(group)
        summaries.append(
            {
                "season": label,
                "source_season": "multiple" if label == "pooled" else _previous_season(label),
                "stint_count": int(len(group)),
                "possession_count": float(group["possessions"].sum()),
                "intercept": result.intercept,
                "unadjusted_usage_gap_weight_per_sd": fit_unadjusted_usage_gap_weight(group),
                "lead_secondary_usage_gap_weight_per_sd": result.lead_secondary_usage_gap,
                "max_frozen_prior_edge_weight_per_sd": result.max_frozen_prior_edge,
                "predictor_correlation": result.predictor_correlation,
            }
        )

    run_id = (
        "lead-secondary-usage-gap-conditioning-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    audited.to_parquet(run_dir / "stint_conditioning.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(run_dir / "conditional_coefficients.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "candidate": "lead_secondary_usage_gap",
                "target_seasons": list(target_seasons),
                "source_screen": str(screen_dir),
                "source_model": MODEL_NAME,
                "source_model_run": model_dir.name,
                "method": (
                    "Possession-weighted least squares of the frozen target-season residual on "
                    "within-season standardized lead-secondary usage-gap edge and within-season "
                    "standardized max frozen player-prior edge. No ratings or context models "
                    "are refit."
                ),
                "information_boundary": (
                    "Both predictors use the frozen target-season NAIL prior state available "
                    "before the target season; target outcomes remain evaluation-only."
                ),
            },
            indent=2,
        )
    )
    latest = Path(output_root) / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps({"run_id": run_id}, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Condition lead-secondary usage-gap residual screen on frozen superstar imbalance"
        )
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    run_dir = build_lead_secondary_usage_gap_conditioning_audit(
        seasons=tuple(args.seasons),
        output_root=args.output_root,
    )
    print(f"Lead-secondary conditioning audit: run={run_dir}")


if __name__ == "__main__":
    main()
