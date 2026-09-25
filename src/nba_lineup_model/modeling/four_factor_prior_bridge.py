"""Strictly forward Four Factors adjustment for a NAIL player prior.

The bridge is intentionally a narrow candidate: it starts with the established
NAIL player prior and estimates whether the eight *previous-season* Four
Factors O/D player states explain the later completed NAIL residual.  It never
uses total NAIL as an input to the Four Factors state and never observes the
target season when building its forecast adjustment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


FACTOR_SIDE_FEATURE_COLUMNS = tuple(
    f"{factor}_{side}"
    for factor in ("efg_pct", "tov_pct", "orb_pct", "ftr")
    for side in ("offense", "defense")
)
DEFAULT_BRIDGE_ALPHA_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MINIMUM_TRAINING_SEASONS = 6
DEFAULT_EFFECTS_PATH = Path(
    "artifacts/models/analysis/forward_four_factor_offense_defense/2025-26/"
    "forward-four-factor-od-2025-26-20260924T062408Z-11a10c95/"
    "season_factor_side_effects.parquet"
)


@dataclass(frozen=True)
class FourFactorPriorBridge:
    """A no-intercept Ridge bridge whose zero factor vector has zero effect."""

    coefficients: np.ndarray
    feature_scales: np.ndarray
    selected_alpha: float
    training_row_count: int
    training_seasons: tuple[str, ...]
    cv_rows: tuple[dict[str, object], ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix = features.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].to_numpy(dtype=float)
        return (matrix / self.feature_scales) @ self.coefficients


def load_factor_side_effects(path: Path | str = DEFAULT_EFFECTS_PATH) -> pd.DataFrame:
    """Return one strictly completed factor-side state vector per player-season."""

    effects = pd.read_parquet(path)
    required = {
        "season",
        "factor",
        "player_id",
        "offense_effect_percentage_points",
        "defense_effect_percentage_points",
    }
    missing = required - set(effects)
    if missing:
        raise ValueError(f"Four Factors effect artifact lacks: {sorted(missing)}")
    offense = effects.loc[:, ["season", "player_id", "factor", "offense_effect_percentage_points"]]
    offense = offense.rename(columns={"offense_effect_percentage_points": "value"})
    offense["feature"] = offense["factor"] + "_offense"
    defense = effects.loc[:, ["season", "player_id", "factor", "defense_effect_percentage_points"]]
    defense = defense.rename(columns={"defense_effect_percentage_points": "value"})
    defense["feature"] = defense["factor"] + "_defense"
    wide = (
        pd.concat([offense, defense], ignore_index=True)
        .pivot(index=["season", "player_id"], columns="feature", values="value")
        .reindex(columns=FACTOR_SIDE_FEATURE_COLUMNS, fill_value=0.0)
        .reset_index()
    )
    wide.loc[:, FACTOR_SIDE_FEATURE_COLUMNS] = wide.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].fillna(0.0)
    return wide


def build_four_factor_prior_bridge(
    *,
    completed_results: Sequence[ForwardLaggedRapmSeason],
    exposure_history: Sequence[pd.DataFrame] | None = None,
    effects: pd.DataFrame,
    alpha_grid: tuple[float, ...] = DEFAULT_BRIDGE_ALPHA_GRID,
) -> FourFactorPriorBridge | None:
    """Fit only on fully completed historical NAIL residuals.

    An observation for completed NAIL season ``s`` joins the Factor state from
    ``s-1`` to ``NAIL_s - prior_s``.  Consequently the bridge fitted before a
    target season can use results through its source season and no further.
    """

    if not alpha_grid or any(value <= 0 for value in alpha_grid):
        raise ValueError("Four Factors bridge alpha grid must contain positive values")
    training = _training_frame(completed_results, effects, exposure_history=exposure_history)
    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    if len(seasons) < MINIMUM_TRAINING_SEASONS:
        return None
    selected_alpha, cv_rows = _select_alpha(training, alpha_grid)
    matrix = training.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].to_numpy(dtype=float)
    weights = training["on_court_possessions"].to_numpy(dtype=float)
    scales = _weighted_rms(matrix, weights)
    model = Ridge(alpha=selected_alpha, fit_intercept=False)
    model.fit(matrix / scales, training["residual"].to_numpy(dtype=float), sample_weight=weights)
    return FourFactorPriorBridge(
        coefficients=np.asarray(model.coef_, dtype=float),
        feature_scales=scales,
        selected_alpha=float(selected_alpha),
        training_row_count=len(training),
        training_seasons=seasons,
        cv_rows=tuple(cv_rows),
    )


def apply_four_factor_prior_bridge(
    *,
    priors: pd.DataFrame,
    target_season: str,
    effects: pd.DataFrame,
    bridge: FourFactorPriorBridge | None,
    weight: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add a source-state Four Factors adjustment, preserving the zero path."""

    if not 0.0 <= weight <= 1.0:
        raise ValueError("Four Factors bridge weight must be between zero and one")
    if bridge is None or weight == 0.0:
        return priors.copy(), {
            "four_factor_prior_bridge_enabled": False,
            "four_factor_prior_bridge_weight": float(weight),
            "four_factor_prior_bridge_reason": "zero_weight" if weight == 0.0 else "insufficient_history",
        }
    source = _previous_season(target_season)
    source_effects = effects.loc[effects["season"].eq(source)].copy()
    if source_effects.empty:
        return priors.copy(), {
            "four_factor_prior_bridge_enabled": False,
            "four_factor_prior_bridge_weight": float(weight),
            "four_factor_prior_bridge_reason": "missing_source_factor_state",
        }
    output = priors.merge(
        source_effects.drop(columns="season"), on="player_id", how="left", validate="one_to_one"
    )
    available = output.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].notna().all(axis=1)
    features = output.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].fillna(0.0)
    adjustment = bridge.predict(features) * weight
    # Players with no completed source factor state are cold starts for this bridge.
    adjustment = np.where(available.to_numpy(dtype=bool), adjustment, 0.0)
    output[PRIOR_MEAN_COLUMN] = output[PRIOR_MEAN_COLUMN].to_numpy(dtype=float) + adjustment
    output = output.drop(columns=list(FACTOR_SIDE_FEATURE_COLUMNS))
    return output, {
        "four_factor_prior_bridge_enabled": True,
        "four_factor_prior_bridge_weight": float(weight),
        "four_factor_prior_bridge_source_season": source,
        "four_factor_prior_bridge_alpha": bridge.selected_alpha,
        "four_factor_prior_bridge_training_row_count": bridge.training_row_count,
        "four_factor_prior_bridge_training_seasons": list(bridge.training_seasons),
        "four_factor_prior_bridge_available_player_count": int(available.sum()),
        "four_factor_prior_bridge_mean_adjustment": float(np.mean(adjustment)),
        "four_factor_prior_bridge_sd_adjustment": float(np.std(adjustment)),
        "four_factor_prior_bridge_coefficients": {
            feature: float(value)
            for feature, value in zip(FACTOR_SIDE_FEATURE_COLUMNS, bridge.coefficients, strict=True)
        },
        "four_factor_prior_bridge_feature_scales": {
            feature: float(value)
            for feature, value in zip(FACTOR_SIDE_FEATURE_COLUMNS, bridge.feature_scales, strict=True)
        },
        "four_factor_prior_bridge_cv": list(bridge.cv_rows),
    }


def _training_frame(
    completed_results: Sequence[ForwardLaggedRapmSeason],
    effects: pd.DataFrame,
    *,
    exposure_history: Sequence[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    if exposure_history is not None and len(completed_results) != len(exposure_history):
        raise ValueError("Four Factors bridge needs aligned NAIL exposure history")
    histories = exposure_history or [None] * len(completed_results)
    for result, exposure in zip(completed_results, histories, strict=True):
        source = _previous_season(result.season)
        state = effects.loc[effects["season"].eq(source)].drop(columns="season")
        estimates = result.player_estimates.loc[:, ["player_id", "rapm", "prior_rapm"]].copy()
        if exposure is None:
            estimates["on_court_possessions"] = 1.0
        else:
            estimates = estimates.merge(
                exposure.loc[:, ["player_id", "on_court_possessions"]],
                on="player_id",
                how="inner",
                validate="one_to_one",
            )
        frame = estimates.merge(state, on="player_id", how="inner", validate="one_to_one")
        frame["season"] = result.season
        frame["residual"] = frame["rapm"] - frame["prior_rapm"]
        records.append(frame)
    if not records:
        return pd.DataFrame(columns=["season", "player_id", "residual", "on_court_possessions", *FACTOR_SIDE_FEATURE_COLUMNS])
    output = pd.concat(records, ignore_index=True)
    output = output.loc[
        np.isfinite(output["residual"])
        & np.isfinite(output["on_court_possessions"])
        & output["on_court_possessions"].gt(0)
    ].copy()
    return output


def _select_alpha(training: pd.DataFrame, alpha_grid: tuple[float, ...]) -> tuple[float, list[dict[str, object]]]:
    seasons = tuple(sorted(training["season"].unique(), key=lambda value: int(value[:4])))
    validation_seasons = seasons[-3:]
    rows: list[dict[str, object]] = []
    scores: dict[float, list[float]] = {float(alpha): [] for alpha in alpha_grid}
    for validation in validation_seasons:
        earlier = tuple(season for season in seasons if int(season[:4]) < int(validation[:4]))
        train = training.loc[training["season"].isin(earlier)]
        validation_frame = training.loc[training["season"].eq(validation)]
        if train.empty or validation_frame.empty:
            continue
        train_x = train.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].to_numpy(dtype=float)
        train_weight = train["on_court_possessions"].to_numpy(dtype=float)
        scales = _weighted_rms(train_x, train_weight)
        validation_x = validation_frame.loc[:, FACTOR_SIDE_FEATURE_COLUMNS].to_numpy(dtype=float)
        validation_y = validation_frame["residual"].to_numpy(dtype=float)
        validation_weight = validation_frame["on_court_possessions"].to_numpy(dtype=float)
        for alpha in alpha_grid:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(train_x / scales, train["residual"].to_numpy(dtype=float), sample_weight=train_weight)
            error = validation_y - model.predict(validation_x / scales)
            mse = float(np.average(np.square(error), weights=validation_weight))
            scores[float(alpha)].append(mse)
            rows.append(
                {
                    "validation_season": validation,
                    "alpha": float(alpha),
                    "weighted_mse": mse,
                    "validation_player_count": int(len(validation_frame)),
                    "training_season_count": len(earlier),
                }
            )
    selected = min(alpha_grid, key=lambda alpha: (float(np.mean(scores[float(alpha)])), alpha))
    return float(selected), rows


def _weighted_rms(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scales = np.sqrt(np.average(np.square(matrix), axis=0, weights=weights))
    return np.where(scales > 1e-12, scales, 1.0)
