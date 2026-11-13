"""Only the required prediction calculation, with no legacy runtime imports."""

import io
import sys
from hashlib import sha256

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gestalt_pipelines.ratings.utils import context as module


def test_saved_values_and_prediction_parity(context_values, monkeypatch):
    values = context_values
    scaler = StandardScaler()
    scaler.mean_ = np.array(values["scaler"]["mean"])
    scaler.scale_ = np.array(values["scaler"]["scale"])
    scaler.var_ = np.array(values["scaler"]["variance"])
    scaler.n_features_in_ = 10
    scaler.n_samples_seen_ = 100
    scaler.feature_names_in_ = np.array(module.FEATURES)
    ridge = Ridge()
    ridge.coef_ = np.array(values["coefficients"])
    ridge.intercept_ = values["intercept"]
    ridge.n_features_in_ = 10
    saved = module._SavedContextState()
    saved.pipeline = Pipeline([("scale", scaler), ("ridge", ridge)])
    saved.feature_set = values["feature_set"]
    saved.rebound_model = saved.usage_model = None
    stream = io.BytesIO(); joblib.dump({"2025-26": saved}, stream)
    raw = stream.getvalue()
    monkeypatch.setitem(module.SOURCES, "context", ("test", sha256(raw).hexdigest()))
    decoded = module.export_context(raw)
    assert decoded["coefficients"] == values["coefficients"]
    assert decoded["scaler"] == values["scaler"]
    x = np.random.default_rng(76).normal(size=(128, 10))
    x[:, :7] = 0  # Already compiled into player ratings.
    frame = pd.DataFrame(x, columns=module.FEATURES)
    expected = 0.5*(saved.pipeline.predict(frame)-saved.pipeline.predict(-frame))
    actual = [module.nonadditive_context(decoded, row) for row in frame.to_dict(orient="records")]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert not any(name == "nba_lineup_model" or name.startswith("nba_lineup_model.") for name in sys.modules)


def test_additive_player_terms_are_not_counted_twice(context_values):
    differences = {name: 999.0 if i < 7 else 0.0 for i, name in enumerate(module.FEATURES)}
    assert module.nonadditive_context(context_values, differences) == 0


def test_unverified_context_is_never_decoded():
    with pytest.raises(ValueError, match="checksum"):
        module.export_context(b"unverified pickle")
