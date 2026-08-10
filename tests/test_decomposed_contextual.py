from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import side_context_feature_columns
from nba_lineup_model.modeling.decomposed_contextual import fit_decomposed_contextual_model


def test_decomposed_context_is_exactly_antisymmetric() -> None:
    columns = side_context_feature_columns()
    values = np.arange(24 * len(columns), dtype=float).reshape(24, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = np.linspace(-2.0, 2.0, len(home))
    model = fit_decomposed_contextual_model(
        home,
        away,
        target,
        np.ones(len(home)),
        alpha=10.0,
    )

    forward = model.predict_matchup_features(home, away)
    reverse = model.predict_matchup_features(away, home)

    np.testing.assert_allclose(forward, -reverse, atol=1e-12)
    np.testing.assert_allclose(
        forward,
        model.predict_side_features(home) - model.predict_side_features(away),
        atol=1e-12,
    )
