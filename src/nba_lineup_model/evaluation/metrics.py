from __future__ import annotations

import numpy as np
import pandas as pd


def mean_squared_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Return optional-weighted mean squared error."""

    residual = _residual(actual, predicted)
    return float(np.average(residual**2, weights=_weights(sample_weight, len(residual))))


def rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Return optional-weighted root mean squared error."""

    return float(np.sqrt(mean_squared_error(actual, predicted, sample_weight)))


def mean_absolute_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Return optional-weighted mean absolute error."""

    residual = _residual(actual, predicted)
    return float(np.average(np.abs(residual), weights=_weights(sample_weight, len(residual))))


def game_margin_rmse(
    game_ids: np.ndarray | pd.Series,
    actual_net_rating: np.ndarray,
    predicted_net_rating: np.ndarray,
    possessions: np.ndarray,
) -> float:
    """Aggregate stint net ratings into game margins before computing RMSE."""

    identifiers = np.asarray(game_ids, dtype=str)
    actual = np.asarray(actual_net_rating, dtype=float)
    predicted = np.asarray(predicted_net_rating, dtype=float)
    exposure = _weights(possessions, len(actual))
    if exposure is None:
        raise ValueError("Possessions are required")
    if len(identifiers) != len(actual):
        raise ValueError("Game IDs must match metric rows")
    frame = pd.DataFrame(
        {
            "game_id": identifiers,
            "actual_margin": actual * exposure / 100.0,
            "predicted_margin": predicted * exposure / 100.0,
        }
    )
    games = frame.groupby("game_id", sort=False)[["actual_margin", "predicted_margin"]].sum()
    return rmse(
        games["actual_margin"].to_numpy(),
        games["predicted_margin"].to_numpy(),
    )


def skill_score(model_mse: float, baseline_mse: float) -> float:
    """Return out-of-sample skill relative to a baseline MSE."""

    if baseline_mse <= 0:
        raise ValueError("Baseline MSE must be positive")
    return float(1.0 - model_mse / baseline_mse)


def _residual(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if actual_values.ndim != 1:
        raise ValueError("Metrics require one-dimensional arrays")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Metric inputs must be finite")
    return actual_values - predicted_values


def _weights(
    sample_weight: np.ndarray | None,
    expected_length: int,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    values = np.asarray(sample_weight, dtype=float)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError("Sample weights must be one-dimensional and match rows")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Sample weights must be finite and positive")
    return values
