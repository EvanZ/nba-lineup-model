"""Tests for portable-relative context agreement summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import contextual_feature_columns
from nba_lineup_model.modeling.portable_relative_context_agreement import (
    _weighted_linear_fit,
    agreement_metrics,
    feature_agreement_metrics,
)


def test_agreement_metrics_weight_identical_player_edges() -> None:
    predictions = pd.DataFrame(
        {
            "possessions": [1.0, 2.0, 3.0],
            "portable_player_edge": [-2.0, 0.0, 2.0],
            "relative_player_edge": [-4.0, 0.0, 4.0],
            "portable_context_edge": [1.0, -1.0, 0.0],
            "relative_context_edge": [2.0, -2.0, 0.0],
            "portable_predicted_net_rating": [-1.0, -1.0, 2.0],
            "relative_predicted_net_rating": [-2.0, -2.0, 4.0],
        }
    )

    metrics = agreement_metrics(predictions).set_index("component")

    player = metrics.loc["Net player rating edge"]
    assert player["stint_count"] == 3
    assert player["possession_count"] == 6.0
    assert np.isclose(player["weighted_pearson_correlation"], 1.0)
    assert np.isclose(player["weighted_spearman_correlation"], 1.0)
    assert np.isclose(player["weighted_rmse"], np.sqrt(8.0 / 3.0))
    assert np.isclose(player["weighted_mae"], 4.0 / 3.0)


def test_weighted_linear_fit_recovers_a_line() -> None:
    slope, intercept = _weighted_linear_fit(
        np.asarray([-1.0, 0.0, 2.0]),
        np.asarray([-1.0, 1.0, 5.0]),
        np.asarray([1.0, 2.0, 3.0]),
    )

    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, 1.0)


def test_feature_agreement_metrics_reports_every_context_feature() -> None:
    predictions = pd.DataFrame({"possessions": [1.0, 2.0, 3.0]})
    for index, feature in enumerate(contextual_feature_columns()):
        portable = np.asarray([-1.0, 0.0, 1.0]) * (index + 1)
        predictions[f"portable_feature_{feature}"] = portable
        predictions[f"relative_feature_{feature}"] = portable * 2.0

    metrics = feature_agreement_metrics(predictions)

    assert len(metrics) == len(contextual_feature_columns())
    assert set(metrics["feature"]) == set(contextual_feature_columns())
    assert np.allclose(metrics["weighted_pearson_correlation"], 1.0)
