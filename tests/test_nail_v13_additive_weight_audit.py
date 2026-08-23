from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.nail_v13_additive_weight_audit import summarize_additive_weights


def test_weight_summary_ranks_consistent_direction_before_magnitude() -> None:
    weights = pd.DataFrame(
        {
            "feature": ["steady", "steady", "mixed", "mixed"],
            "label": ["Steady", "Steady", "Mixed", "Mixed"],
            "is_v13_addition": [False, False, True, True],
            "standardized_weight": [0.3, 0.4, -2.0, 2.0],
        }
    )

    summary = summarize_additive_weights(weights)

    assert summary["feature"].tolist() == ["steady", "mixed"]
    steady = summary.iloc[0]
    assert steady["dominant_sign_share"] == 1.0
    assert steady["dominant_directional_mass_share"] == 1.0
    assert steady["mean_absolute_standardized_weight"] == 0.35
    mixed = summary.iloc[1]
    assert mixed["dominant_sign_share"] == 0.5
    assert mixed["positive_directional_mass"] == 2.0
    assert mixed["negative_directional_mass"] == 2.0
    assert mixed["dominant_directional_mass_share"] == 0.5
    assert mixed["is_v13_addition"]
