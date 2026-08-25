from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class FittedMeanModel:
    """Possession-weighted intercept-only prediction."""

    mean: float

    @classmethod
    def fit(
        cls,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> FittedMeanModel:
        values = np.asarray(target, dtype=float)
        weights = _validated_weights(sample_weight, len(values))
        return cls(mean=float(np.average(values, weights=weights)))

    def predict(self, row_count: int) -> np.ndarray:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        return np.full(row_count, self.mean, dtype=float)


class RidgeLineupModel:
    """Sparse ridge model with a sample-size-normalized lambda convention."""

    def __init__(self, regularization: float = 1.0) -> None:
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self.regularization = float(regularization)
        self.model: Ridge | None = None
        self.sklearn_alpha: float | None = None

    def fit(
        self,
        features: sparse.spmatrix | np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
    ) -> RidgeLineupModel:
        values = np.asarray(target, dtype=float)
        if features.shape[0] != len(values):
            raise ValueError("Feature and target rows must match")
        weights = _validated_weights(sample_weight, len(values))
        normalized_weights = weights / np.mean(weights)
        self.sklearn_alpha = self.regularization * len(values)
        self.model = Ridge(
            alpha=self.sklearn_alpha,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-8,
        )
        self.model.fit(
            features,
            values,
            sample_weight=normalized_weights,
        )
        return self

    def predict(self, features: sparse.spmatrix | np.ndarray) -> np.ndarray:
        return self._fitted_model().predict(features)

    @property
    def intercept_(self) -> float:
        return float(self._fitted_model().intercept_)

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self._fitted_model().coef_, dtype=float)

    def _fitted_model(self) -> Ridge:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        return self.model


class PriorCenteredRidgeLineupModel:
    """Sparse ridge whose coefficient penalty is centered on a prior vector.

    Fitting ridge to ``y - X @ prior`` gives the coefficient adjustment around
    the prior. Adding the prior back to that adjustment is equivalent to
    penalizing ``||coefficient - prior||^2`` in the original objective.
    """

    def __init__(self, regularization: float = 1.0) -> None:
        self.regularization = regularization
        self.residual_model = RidgeLineupModel(regularization)
        self.prior: np.ndarray | None = None

    def fit(
        self,
        features: sparse.spmatrix | np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        coefficient_prior: np.ndarray,
    ) -> PriorCenteredRidgeLineupModel:
        prior = np.asarray(coefficient_prior, dtype=float)
        if prior.ndim != 1 or features.shape[1] != len(prior):
            raise ValueError("Coefficient prior must match the feature columns")
        if not np.isfinite(prior).all():
            raise ValueError("Coefficient prior must be finite")
        values = np.asarray(target, dtype=float)
        if features.shape[0] != len(values):
            raise ValueError("Feature and target rows must match")
        offset = np.asarray(features @ prior, dtype=float).reshape(-1)
        self.residual_model.fit(features, values - offset, sample_weight)
        self.prior = prior
        return self

    def predict(self, features: sparse.spmatrix | np.ndarray) -> np.ndarray:
        prior = self._prior()
        offset = np.asarray(features @ prior, dtype=float).reshape(-1)
        return self.residual_model.predict(features) + offset

    @property
    def intercept_(self) -> float:
        return self.residual_model.intercept_

    @property
    def coef_(self) -> np.ndarray:
        return self._prior() + self.residual_model.coef_

    @property
    def adjustment_(self) -> np.ndarray:
        return self.residual_model.coef_

    @property
    def sklearn_alpha(self) -> float:
        alpha = self.residual_model.sklearn_alpha
        if alpha is None:
            raise ValueError("Model has not been fitted")
        return alpha

    def _prior(self) -> np.ndarray:
        if self.prior is None:
            raise ValueError("Model has not been fitted")
        return self.prior


class PriorPrecisionRidgeLineupModel:
    """Prior-centered ridge with a separate relative precision per coefficient.

    The penalty is ``regularization * sum_i precision_i *
    (coefficient_i - prior_i)^2`` under the same sample-size-normalized
    convention as :class:`RidgeLineupModel`. Reparameterizing each adjustment
    by ``sqrt(precision_i)`` permits an exact implementation through the
    existing uniform-penalty solver.
    """

    def __init__(self, regularization: float = 1.0) -> None:
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self.regularization = float(regularization)
        self.residual_model = RidgeLineupModel(regularization)
        self.prior: np.ndarray | None = None
        self.relative_precision: np.ndarray | None = None
        self.adjustment: np.ndarray | None = None
        self.posterior_variance: np.ndarray | None = None

    def fit(
        self,
        features: sparse.spmatrix | np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        coefficient_prior: np.ndarray,
        relative_precision: np.ndarray,
    ) -> PriorPrecisionRidgeLineupModel:
        prior = np.asarray(coefficient_prior, dtype=float)
        precision = np.asarray(relative_precision, dtype=float)
        if prior.ndim != 1 or features.shape[1] != len(prior):
            raise ValueError("Coefficient prior must match the feature columns")
        if precision.ndim != 1 or features.shape[1] != len(precision):
            raise ValueError("Relative precision must match the feature columns")
        if not np.isfinite(prior).all():
            raise ValueError("Coefficient prior must be finite")
        if not np.isfinite(precision).all() or not np.all(precision > 0):
            raise ValueError("Relative precision must be finite and positive")
        values = np.asarray(target, dtype=float)
        if features.shape[0] != len(values):
            raise ValueError("Feature and target rows must match")

        adjustment_scale = 1.0 / np.sqrt(precision)
        # Preserve the incumbent solver input exactly for the parity case.
        transformed = (
            features
            if np.array_equal(precision, np.ones_like(precision))
            else _scale_feature_columns(features, adjustment_scale)
        )
        offset = np.asarray(features @ prior, dtype=float).reshape(-1)
        self.residual_model.fit(transformed, values - offset, sample_weight)
        self.prior = prior
        self.relative_precision = precision
        self.adjustment = self.residual_model.coef_ * adjustment_scale
        self.posterior_variance = _ridge_posterior_variance(
            features,
            values,
            sample_weight,
            self.coef_,
            self.intercept_,
            self.sklearn_alpha,
            precision,
        )
        return self

    def predict(self, features: sparse.spmatrix | np.ndarray) -> np.ndarray:
        prior = self._prior()
        precision = self._relative_precision()
        transformed = (
            features
            if np.array_equal(precision, np.ones_like(precision))
            else _scale_feature_columns(features, 1.0 / np.sqrt(precision))
        )
        offset = np.asarray(features @ prior, dtype=float).reshape(-1)
        return self.residual_model.predict(transformed) + offset

    @property
    def intercept_(self) -> float:
        return self.residual_model.intercept_

    @property
    def coef_(self) -> np.ndarray:
        return self._prior() + self.adjustment_

    @property
    def adjustment_(self) -> np.ndarray:
        if self.adjustment is None:
            raise ValueError("Model has not been fitted")
        return self.adjustment.copy()

    @property
    def sklearn_alpha(self) -> float:
        alpha = self.residual_model.sklearn_alpha
        if alpha is None:
            raise ValueError("Model has not been fitted")
        return alpha

    @property
    def posterior_variance_(self) -> np.ndarray:
        if self.posterior_variance is None:
            raise ValueError("Model has not been fitted")
        return self.posterior_variance.copy()

    def _prior(self) -> np.ndarray:
        if self.prior is None:
            raise ValueError("Model has not been fitted")
        return self.prior

    def _relative_precision(self) -> np.ndarray:
        if self.relative_precision is None:
            raise ValueError("Model has not been fitted")
        return self.relative_precision


def _scale_feature_columns(
    features: sparse.spmatrix | np.ndarray,
    scale: np.ndarray,
) -> sparse.spmatrix | np.ndarray:
    """Scale a dense or sparse feature matrix without changing its row shape."""

    if sparse.issparse(features):
        return features @ sparse.diags(scale, format="csr")
    return np.asarray(features, dtype=float) * scale


def _ridge_posterior_variance(
    features: sparse.spmatrix | np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    alpha: float,
    relative_precision: np.ndarray,
) -> np.ndarray:
    """Return a diagonal Laplace covariance for a prior-precision ridge fit."""

    values = np.asarray(target, dtype=float)
    weights = _validated_weights(sample_weight, len(values))
    normalized = weights / np.mean(weights)
    matrix = sparse.csr_matrix(features, dtype=float)
    weighted_matrix = matrix.multiply(normalized[:, None])
    gram = (matrix.T @ weighted_matrix).toarray()
    feature_mean = np.asarray(matrix.T @ normalized, dtype=float).reshape(-1) / normalized.sum()
    gram -= normalized.sum() * np.outer(feature_mean, feature_mean)
    precision = gram + alpha * np.diag(relative_precision)
    factor, lower = linalg.cho_factor(precision, check_finite=False)
    inverse = linalg.cho_solve(
        (factor, lower),
        np.eye(precision.shape[0]),
        check_finite=False,
    )
    prediction = np.asarray(matrix @ coefficients, dtype=float).reshape(-1) + intercept
    residual = values - prediction
    effective_df = float(np.trace(gram @ inverse) + 1.0)
    degrees_of_freedom = max(float(len(values)) - effective_df, 1.0)
    noise_variance = float(np.dot(normalized, residual**2) / degrees_of_freedom)
    return noise_variance * np.diag(inverse)


def entity_vocabulary(
    frame: pd.DataFrame,
    positive_column: str,
    negative_column: str,
    *,
    multiple: bool,
) -> tuple[int, ...]:
    """Return sorted entity IDs appearing on either side of a signed design."""

    if multiple:
        identifiers = {
            int(identifier)
            for column in (positive_column, negative_column)
            for values in frame[column]
            for identifier in values
        }
    else:
        identifiers = {
            int(identifier)
            for column in (positive_column, negative_column)
            for identifier in frame[column]
        }
    if not identifiers:
        raise ValueError("Signed entity vocabulary cannot be empty")
    return tuple(sorted(identifiers))


def signed_entity_matrix(
    frame: pd.DataFrame,
    positive_column: str,
    negative_column: str,
    entity_to_column: Mapping[int, int],
    *,
    multiple: bool,
) -> sparse.csr_matrix:
    """Encode positive and negative entities in a SciPy CSR matrix."""

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, (positive, negative) in enumerate(
        zip(frame[positive_column], frame[negative_column], strict=True)
    ):
        positive_ids = positive if multiple else (positive,)
        negative_ids = negative if multiple else (negative,)
        for identifier in positive_ids:
            column = entity_to_column.get(int(identifier))
            if column is not None:
                row_indices.append(row_index)
                column_indices.append(column)
                values.append(1.0)
        for identifier in negative_ids:
            column = entity_to_column.get(int(identifier))
            if column is not None:
                row_indices.append(row_index)
                column_indices.append(column)
                values.append(-1.0)
    return sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(frame), len(entity_to_column)),
        dtype=np.float64,
    ).tocsr()


def vocabulary_mapping(identifiers: Sequence[int]) -> dict[int, int]:
    """Map unique entity IDs to stable contiguous sparse columns."""

    normalized = tuple(int(identifier) for identifier in identifiers)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Entity vocabulary contains duplicate IDs")
    return {identifier: column for column, identifier in enumerate(normalized)}


def _validated_weights(weights: np.ndarray, expected_length: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError("Sample weights must be one-dimensional and match rows")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Sample weights must be finite and positive")
    return values
