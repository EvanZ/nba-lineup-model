from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.stats import t


@dataclass(frozen=True)
class MarginalPosterior:
    """Univariate summaries from one marginal Student-t posterior."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    probability_positive: np.ndarray


@dataclass(frozen=True)
class PredictivePosterior:
    """Posterior predictive location and interval for observed outcomes."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    standard_deviation: np.ndarray


@dataclass(frozen=True)
class ConjugateBayesianRidge:
    """Exact weighted Gaussian posterior corresponding to a ridge fit.

    The likelihood variance for row ``i`` is ``sigma_squared / weight_i``.
    Player coefficients have a zero-centered Gaussian prior conditional on
    ``sigma_squared``. The intercept is unpenalized, and the residual variance
    uses the scale-invariant prior ``p(sigma_squared) proportional to 1 / sigma_squared``.
    """

    location: np.ndarray
    precision_cholesky: np.ndarray
    precision_inverse: np.ndarray
    residual_quadratic: float
    degrees_of_freedom: int
    regularization: float
    ridge_alpha: float
    training_weight_mean: float

    @classmethod
    def fit(
        cls,
        features: sparse.spmatrix | np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        regularization: float,
    ) -> ConjugateBayesianRidge:
        """Fit the exact conjugate posterior with an unpenalized intercept."""

        matrix = sparse.csr_matrix(features, dtype=np.float64)
        values = np.asarray(target, dtype=np.float64)
        weights = np.asarray(sample_weight, dtype=np.float64)
        if matrix.shape[0] != len(values):
            raise ValueError("Feature and target rows must match")
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Target must be one-dimensional and finite")
        if weights.ndim != 1 or len(weights) != len(values):
            raise ValueError("Sample weights must be one-dimensional and match rows")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Sample weights must be finite and positive")
        if regularization <= 0:
            raise ValueError("Conjugate Bayesian ridge requires positive regularization")
        if len(values) <= 3:
            raise ValueError("At least four observations are required")

        weight_mean = float(np.mean(weights))
        normalized_weights = weights / weight_mean
        design = _with_intercept(matrix)
        weighted_design = design.multiply(np.sqrt(normalized_weights)[:, None])
        precision = np.asarray((weighted_design.T @ weighted_design).toarray())
        ridge_alpha = float(regularization * len(values))
        precision[1:, 1:] += ridge_alpha * np.eye(matrix.shape[1])
        precision_cholesky = cholesky(
            precision,
            lower=True,
            overwrite_a=False,
            check_finite=False,
        )
        right_hand_side = np.asarray(
            design.T @ (normalized_weights * values),
            dtype=np.float64,
        ).reshape(-1)
        location = cho_solve(
            (precision_cholesky, True),
            right_hand_side,
            check_finite=False,
        )
        precision_inverse = cho_solve(
            (precision_cholesky, True),
            np.eye(precision.shape[0]),
            check_finite=False,
        )
        weighted_target_sum_squares = float(np.dot(normalized_weights, values**2))
        residual_quadratic = weighted_target_sum_squares - float(
            np.dot(right_hand_side, location)
        )
        tolerance = np.finfo(float).eps * max(1.0, weighted_target_sum_squares) * 100
        if residual_quadratic <= tolerance:
            raise ValueError("Posterior residual quadratic must be positive")

        return cls(
            location=location,
            precision_cholesky=precision_cholesky,
            precision_inverse=precision_inverse,
            residual_quadratic=residual_quadratic,
            degrees_of_freedom=len(values) - 1,
            regularization=float(regularization),
            ridge_alpha=ridge_alpha,
            training_weight_mean=weight_mean,
        )

    @property
    def intercept_mean(self) -> float:
        """Posterior location for the unpenalized intercept."""

        return float(self.location[0])

    @property
    def coefficient_mean(self) -> np.ndarray:
        """Posterior locations for all penalized coefficients."""

        return self.location[1:].copy()

    @property
    def residual_variance_mean(self) -> float:
        """Posterior mean of the residual variance."""

        return self.residual_quadratic / (self.degrees_of_freedom - 2)

    def marginal_summary(
        self,
        *,
        interval_probability: float = 0.90,
    ) -> MarginalPosterior:
        """Return exact marginal summaries for every model parameter."""

        if not 0 < interval_probability < 1:
            raise ValueError("Interval probability must be between zero and one")
        diagonal = np.diag(self.precision_inverse)
        marginal_scale = np.sqrt(
            self.residual_quadratic / self.degrees_of_freedom * diagonal
        )
        posterior_sd = np.sqrt(self.residual_variance_mean * diagonal)
        tail = (1.0 - interval_probability) / 2.0
        quantile = float(t.ppf(1.0 - tail, self.degrees_of_freedom))
        probability_positive = t.cdf(
            self.location / marginal_scale,
            self.degrees_of_freedom,
        )
        return MarginalPosterior(
            mean=self.location.copy(),
            standard_deviation=posterior_sd,
            lower=self.location - quantile * marginal_scale,
            upper=self.location + quantile * marginal_scale,
            probability_positive=np.asarray(probability_positive, dtype=np.float64),
        )

    def draw_parameters(
        self,
        draw_count: int,
        *,
        seed: int,
    ) -> np.ndarray:
        """Draw jointly from the multivariate Student-t posterior."""

        if draw_count < 1:
            raise ValueError("draw_count must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        rng = np.random.default_rng(seed)
        standard_normal = rng.standard_normal((len(self.location), draw_count))
        correlated = solve_triangular(
            self.precision_cholesky.T,
            standard_normal,
            lower=False,
            check_finite=False,
        )
        chi_squared = rng.chisquare(self.degrees_of_freedom, size=draw_count)
        scales = np.sqrt(self.residual_quadratic / chi_squared)
        return (self.location[:, None] + correlated * scales).T

    def predict_mean(
        self,
        features: sparse.spmatrix | np.ndarray,
    ) -> np.ndarray:
        """Predict from the posterior location."""

        design = _validated_prediction_design(features, len(self.location) - 1)
        return np.asarray(design @ self.location, dtype=np.float64).reshape(-1)

    def predictive_summary(
        self,
        features: sparse.spmatrix | np.ndarray,
        sample_weight: np.ndarray,
        *,
        interval_probability: float = 0.90,
        batch_size: int = 2048,
    ) -> PredictivePosterior:
        """Return marginal posterior predictive intervals for new observations."""

        if not 0 < interval_probability < 1:
            raise ValueError("Interval probability must be between zero and one")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        design = _validated_prediction_design(features, len(self.location) - 1)
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != design.shape[0]:
            raise ValueError("Prediction weights must be one-dimensional and match rows")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Prediction weights must be finite and positive")
        normalized_weights = weights / self.training_weight_mean
        leverage = np.empty(design.shape[0], dtype=np.float64)
        for start in range(0, design.shape[0], batch_size):
            stop = min(start + batch_size, design.shape[0])
            block = design[start:stop]
            leverage[start:stop] = np.sum(
                np.asarray(block @ self.precision_inverse) * block.toarray(),
                axis=1,
            )
        predictive_scale = np.sqrt(
            self.residual_quadratic
            / self.degrees_of_freedom
            * (1.0 / normalized_weights + leverage)
        )
        predictive_sd = predictive_scale * np.sqrt(
            self.degrees_of_freedom / (self.degrees_of_freedom - 2)
        )
        tail = (1.0 - interval_probability) / 2.0
        quantile = float(t.ppf(1.0 - tail, self.degrees_of_freedom))
        mean = np.asarray(design @ self.location, dtype=np.float64).reshape(-1)
        return PredictivePosterior(
            mean=mean,
            lower=mean - quantile * predictive_scale,
            upper=mean + quantile * predictive_scale,
            standard_deviation=predictive_sd,
        )


def _with_intercept(features: sparse.csr_matrix) -> sparse.csr_matrix:
    intercept = sparse.csr_matrix(np.ones((features.shape[0], 1), dtype=np.float64))
    return sparse.hstack((intercept, features), format="csr")


def _validated_prediction_design(
    features: sparse.spmatrix | np.ndarray,
    expected_columns: int,
) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(features, dtype=np.float64)
    if matrix.shape[1] != expected_columns:
        raise ValueError("Prediction feature columns do not match fitted model")
    return _with_intercept(matrix)
