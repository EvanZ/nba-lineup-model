from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeLineupModel:
    """Thin wrapper for the first linear plus-minus baseline."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, features: np.ndarray, target: np.ndarray) -> RidgeLineupModel:
        self.model.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
