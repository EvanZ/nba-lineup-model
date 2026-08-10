"""Identifiable side-specific spline context for five-player lineups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.contextual_features import (
    lineup_side_context_features,
    side_context_feature_columns,
)


@dataclass(frozen=True)
class DecomposedContextualModel:
    """A centered side function ``h`` used as ``h(home) - h(away)``."""

    spline: SplineTransformer
    scale: StandardScaler
    ridge: Ridge
    feature_columns: tuple[str, ...]

    def predict_side_features(self, features: pd.DataFrame) -> np.ndarray:
        """Return the centered context value for each individual lineup feature vector."""

        basis = self.spline.transform(features.loc[:, self.feature_columns])
        scaled = self.scale.transform(basis)
        return np.asarray(scaled @ self.ridge.coef_, dtype=float)

    def predict_lineups(
        self,
        home_lineups: list[tuple[int, ...]] | list[list[int]],
        away_lineups: list[tuple[int, ...]] | list[list[int]],
        profiles: pd.DataFrame,
    ) -> np.ndarray:
        """Return the exactly antisymmetric matchup term ``h(home) - h(away)``."""

        home = lineup_side_context_features(home_lineups, profiles)
        away = lineup_side_context_features(away_lineups, profiles)
        return self.predict_matchup_features(home, away)

    def predict_matchup_features(
        self, home_features: pd.DataFrame, away_features: pd.DataFrame
    ) -> np.ndarray:
        """Score already encoded side features without reconstructing player lineups."""

        return self.predict_side_features(home_features) - self.predict_side_features(away_features)

    def side_feature_contributions(self, features: pd.DataFrame) -> np.ndarray:
        """Return per-original-feature contributions to a centered side value."""

        basis = self.spline.transform(features.loc[:, self.feature_columns])
        scaled = self.scale.transform(basis)
        coefficient = np.asarray(self.ridge.coef_, dtype=float)
        feature_count = len(self.feature_columns)
        if scaled.shape[1] % feature_count or len(coefficient) != scaled.shape[1]:
            raise ValueError("Spline basis is incompatible with decomposed context attribution")
        basis_count = scaled.shape[1] // feature_count
        return (scaled * coefficient).reshape(len(features), feature_count, basis_count).sum(axis=2)


def fit_decomposed_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
) -> DecomposedContextualModel:
    """Fit a shared side function to paired home-minus-away residual targets.

    The spline and scaler are fit on the pooled side-feature population. Ridge
    then receives the difference of the transformed home and away vectors with
    no intercept, enforcing ``h(home) - h(away)`` exactly.
    """

    columns = side_context_feature_columns()
    _validate_training_inputs(home_features, away_features, target, sample_weight, columns, alpha)
    combined = pd.concat(
        [home_features.loc[:, columns], away_features.loc[:, columns]], ignore_index=True
    )
    spline = SplineTransformer(n_knots=4, degree=2, extrapolation="linear")
    basis = spline.fit_transform(combined)
    scale = StandardScaler().fit(basis)
    row_count = len(home_features)
    design = scale.transform(basis[:row_count]) - scale.transform(basis[row_count:])
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design, target, sample_weight=sample_weight)
    return DecomposedContextualModel(
        spline=spline,
        scale=scale,
        ridge=ridge,
        feature_columns=columns,
    )


def _validate_training_inputs(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    columns: tuple[str, ...],
    alpha: float,
) -> None:
    if alpha <= 0:
        raise ValueError("Decomposed contextual alpha must be positive")
    if len(home_features) == 0 or len(home_features) != len(away_features):
        raise ValueError("Decomposed context requires aligned non-empty home and away features")
    missing = (set(columns) - set(home_features)) | (set(columns) - set(away_features))
    if missing:
        raise ValueError(f"Decomposed context features are missing columns: {sorted(missing)}")
    if len(target) != len(home_features) or len(sample_weight) != len(home_features):
        raise ValueError("Decomposed context target and weights must align with features")
    if not np.isfinite(target).all() or not np.isfinite(sample_weight).all():
        raise ValueError("Decomposed context target and weights must be finite")
    if (sample_weight <= 0).any():
        raise ValueError("Decomposed context weights must be positive")


def model_metadata(model: DecomposedContextualModel) -> dict[str, Any]:
    """Return compact serializable estimator metadata for an artifact manifest."""

    return {
        "feature_columns": list(model.feature_columns),
        "spline_knots": int(model.spline.n_knots),
        "spline_degree": int(model.spline.degree),
        "ridge_fit_intercept": bool(model.ridge.fit_intercept),
        "side_reference": "pooled transformed lineup-feature mean",
    }
