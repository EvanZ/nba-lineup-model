from __future__ import annotations

import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(residual**2)))
