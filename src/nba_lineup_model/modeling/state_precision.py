"""Forward uncertainty helpers for state-precision RAPM.

The player prior mean is supplied by the existing forward aging and gap-returner
pipeline. This module carries only uncertainty: a completed season's diagonal
Laplace posterior variance is advanced through a random-walk transition, then
converted into relative ridge precision for the following season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlayerStatePrecisionConfig:
    """Variance assumptions for a forward player-state RAPM fit."""

    initial_variance: float = 9.0
    process_variance_per_season: float = 1.0
    variance_floor: float = 1e-6

    def validate(self) -> None:
        if self.initial_variance <= 0:
            raise ValueError("Initial state variance must be positive")
        if self.process_variance_per_season < 0:
            raise ValueError("Process variance must be non-negative")
        if self.variance_floor <= 0:
            raise ValueError("Variance floor must be positive")


DEFAULT_PLAYER_STATE_PRECISION_CONFIG = PlayerStatePrecisionConfig()


def advance_state_variance(
    posterior_variance: np.ndarray,
    *,
    elapsed_seasons: np.ndarray | float,
    config: PlayerStatePrecisionConfig = DEFAULT_PLAYER_STATE_PRECISION_CONFIG,
) -> np.ndarray:
    """Advance diagonal posterior variances without using future outcomes."""

    config.validate()
    posterior = np.asarray(posterior_variance, dtype=float)
    elapsed = np.asarray(elapsed_seasons, dtype=float)
    if not np.isfinite(posterior).all() or not np.all(posterior > 0):
        raise ValueError("Posterior variance must be finite and positive")
    if not np.isfinite(elapsed).all() or not np.all(elapsed >= 0):
        raise ValueError("Elapsed seasons must be finite and non-negative")
    return np.maximum(
        posterior + elapsed * config.process_variance_per_season,
        config.variance_floor,
    )


def relative_precision_from_variance(
    prior_variance: np.ndarray,
    *,
    reference_variance: float | None = None,
    config: PlayerStatePrecisionConfig = DEFAULT_PLAYER_STATE_PRECISION_CONFIG,
) -> np.ndarray:
    """Map state variance to a unit-median relative ridge precision vector."""

    config.validate()
    variance = np.asarray(prior_variance, dtype=float)
    if variance.ndim != 1 or not np.isfinite(variance).all() or not np.all(variance > 0):
        raise ValueError("Prior variance must be a finite positive vector")
    reference = float(np.median(variance)) if reference_variance is None else reference_variance
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Reference variance must be finite and positive")
    return reference / np.maximum(variance, config.variance_floor)
