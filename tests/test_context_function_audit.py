from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.context_function_audit import _summarize_curves


def test_curve_summary_reports_stability_statistics() -> None:
    curves = pd.DataFrame(
        {
            "season": ["2024-25"] * 3 + ["2025-26"] * 3,
            "feature": ["home_minus_away_usage_per_100"] * 6,
            "label": ["Usage events"] * 6,
            "feature_group": ["profile rate"] * 6,
            "percentile": [0.05, 0.5, 0.95, 0.05, 0.5, 0.95],
            "feature_difference": [-2.0, 0.0, 2.0, -3.0, 0.0, 3.0],
            "response_net_rating": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0],
        }
    )

    summary = _summarize_curves(curves).iloc[0]

    assert summary["season_count"] == 2
    assert summary["median_turning_points"] == 0
    assert summary["median_response_range"] == 3.0
    assert summary["latest_support_low"] == -3.0
    assert summary["latest_support_high"] == 3.0
    assert summary["contrast_slope_per_decade"] > 0
    assert summary["contrast_linear_r_squared"] == 1.0
