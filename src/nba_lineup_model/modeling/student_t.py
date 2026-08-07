"""Sparse Student-t MAP RAPM fitted by iteratively reweighted ridge regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from nba_lineup_model.models.baselines import PriorCenteredRidgeLineupModel, RidgeLineupModel


@dataclass(frozen=True)
class StudentTPriorCenteredFit:
    """A fitted robust prior-centered lineup model and convergence diagnostics."""

    model: PriorCenteredRidgeLineupModel
    scale: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class StudentTCoefficientPriorFit:
    """A Gaussian-error fit with a Student-t prior on coefficient adjustments."""

    model: StudentTCoefficientPriorLineupModel
    iterations: int
    converged: bool


class StudentTCoefficientPriorLineupModel:
    """Prior-centered ridge with a distinct IRLS shrinkage multiplier per player."""

    def __init__(
        self,
        *,
        prior: np.ndarray,
        adjustment: np.ndarray,
        intercept: float,
    ) -> None:
        self.prior = prior
        self.adjustment = adjustment
        self.intercept = intercept

    def predict(self, features: sparse.spmatrix | np.ndarray) -> np.ndarray:
        return self.intercept_ + np.asarray(features @ self.coef_, dtype=float).reshape(-1)

    @property
    def intercept_(self) -> float:
        return self.intercept

    @property
    def coef_(self) -> np.ndarray:
        return self.prior + self.adjustment

    @property
    def adjustment_(self) -> np.ndarray:
        return self.adjustment


def fit_student_t_prior_centered_ridge(
    features: sparse.spmatrix | np.ndarray,
    target: np.ndarray,
    possessions: np.ndarray,
    coefficient_prior: np.ndarray,
    *,
    regularization: float,
    degrees_of_freedom: float = 5.0,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
) -> StudentTPriorCenteredFit:
    """Fit a Student-t observation model with Gaussian coefficient shrinkage.

    The coefficient penalty is identical to prior-centered ridge. IRLS only
    changes the observation weights according to Student-t residual influence.
    """

    if degrees_of_freedom <= 2:
        raise ValueError("Student-t degrees of freedom must exceed two")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Student-t IRLS iterations and tolerance must be positive")
    values = np.asarray(target, dtype=float)
    base_weights = np.asarray(possessions, dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(base_weights).all()
        or (base_weights <= 0).any()
    ):
        raise ValueError("Student-t fitting requires finite targets and positive possessions")
    model = PriorCenteredRidgeLineupModel(regularization).fit(
        features, values, base_weights, coefficient_prior
    )
    prediction = model.predict(features)
    scale = _weighted_mad(values - prediction, base_weights)
    previous = np.concatenate(([model.intercept_], model.coef_))
    converged = False
    for iteration in range(1, max_iterations + 1):
        residual = values - model.predict(features)
        standardized = residual / scale
        robust_weights = (degrees_of_freedom + 1.0) / (degrees_of_freedom + standardized**2)
        model = PriorCenteredRidgeLineupModel(regularization).fit(
            features,
            values,
            base_weights * robust_weights,
            coefficient_prior,
        )
        residual = values - model.predict(features)
        new_scale = np.sqrt(
            np.average(
                residual**2 * robust_weights,
                weights=base_weights,
            )
        )
        new_scale = max(float(new_scale), 1e-6)
        current = np.concatenate(([model.intercept_], model.coef_))
        if np.max(np.abs(current - previous)) <= tolerance and abs(new_scale - scale) <= tolerance:
            converged = True
            return StudentTPriorCenteredFit(model, new_scale, iteration, converged)
        previous = current
        scale = new_scale
    return StudentTPriorCenteredFit(model, scale, max_iterations, converged)


def fit_student_t_coefficient_prior_ridge(
    features: sparse.spmatrix | np.ndarray,
    target: np.ndarray,
    possessions: np.ndarray,
    coefficient_prior: np.ndarray,
    *,
    regularization: float,
    degrees_of_freedom: float = 3.0,
    prior_scale: float = 3.0,
    max_iterations: int = 120,
    tolerance: float = 1e-5,
) -> StudentTCoefficientPriorFit:
    """Fit Gaussian stint errors with a Student-t prior on player adjustments.

    The Student-t prior is represented as a normal-scale mixture. IRLS retains
    the Gaussian ridge penalty locally at zero adjustment and progressively
    relaxes it for coefficients farther than ``prior_scale`` from their prior.
    """

    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if prior_scale <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("Student-t coefficient-prior settings must be positive")
    values = np.asarray(target, dtype=float)
    base_weights = np.asarray(possessions, dtype=float)
    prior = np.asarray(coefficient_prior, dtype=float)
    if features.shape[0] != len(values) or features.shape[1] != len(prior):
        raise ValueError("Student-t coefficient-prior dimensions do not match")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(base_weights).all()
        or (base_weights <= 0).any()
        or not np.isfinite(prior).all()
    ):
        raise ValueError("Student-t coefficient-prior fitting requires finite inputs")

    initial = PriorCenteredRidgeLineupModel(regularization).fit(
        features, values, base_weights, prior
    )
    adjustment = initial.adjustment_
    intercept = initial.intercept_
    converged = False
    iterations = 0
    for current_iteration in range(1, max_iterations + 1):
        iterations = current_iteration
        penalty_multiplier = 1.0 / (1.0 + (adjustment / prior_scale) ** 2 / degrees_of_freedom)
        transformed = _scale_feature_columns(features, penalty_multiplier)
        residual_target = values - np.asarray(features @ prior, dtype=float).reshape(-1)
        ridge = RidgeLineupModel(regularization).fit(transformed, residual_target, base_weights)
        new_adjustment = ridge.coef_ / np.sqrt(penalty_multiplier)
        new_intercept = ridge.intercept_
        delta = max(
            float(np.max(np.abs(new_adjustment - adjustment))),
            abs(new_intercept - intercept),
        )
        adjustment = new_adjustment
        intercept = new_intercept
        if delta <= tolerance:
            converged = True
            break
    model = StudentTCoefficientPriorLineupModel(
        prior=prior,
        adjustment=adjustment,
        intercept=intercept,
    )
    return StudentTCoefficientPriorFit(model, iterations, converged)


def _scale_feature_columns(
    features: sparse.spmatrix | np.ndarray,
    penalty_multiplier: np.ndarray,
) -> sparse.spmatrix | np.ndarray:
    scale = 1.0 / np.sqrt(penalty_multiplier)
    if sparse.issparse(features):
        return features @ sparse.diags(scale, format="csr")
    return np.asarray(features, dtype=float) * scale


def _weighted_mad(values: np.ndarray, weights: np.ndarray) -> float:
    median = _weighted_quantile(values, weights, 0.5)
    mad = _weighted_quantile(np.abs(values - median), weights, 0.5)
    return max(1.4826 * float(mad), 1e-6)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered_values[np.searchsorted(cumulative, probability * cumulative[-1])])
