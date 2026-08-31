from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.short_rest_travel_screen import _summarize


def test_travel_screen_reports_per_standard_deviation_and_raw_weights() -> None:
    frame = pd.DataFrame(
        {
            "possessions": [10.0, 10.0, 10.0],
            "feature_edge": [-1.0, 0.0, 1.0],
            "frozen_residual_net_rating": [-2.0, 0.0, 2.0],
        }
    )

    summary, _ = _summarize("2025-26", "2024-25", frame)

    assert summary["weighted_correlation"] == 1.0
    assert summary["standardized_residual_weight"] > 1.0
    assert summary["raw_residual_weight_per_thousand_miles"] > 0.0
