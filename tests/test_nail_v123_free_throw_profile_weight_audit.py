from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.nail_v123_free_throw_profile_weight_audit import (
    summarize_directional_mass,
)


def test_directional_mass_downweights_small_opposite_sign_season() -> None:
    weights = pd.DataFrame(
        {
            "feature": ["free_throw_attempts_per_100"] * 4,
            "label": ["Free-throw attempts / 100"] * 4,
            "season_start_year": [2020, 2021, 2022, 2023],
            "standardized_weight": [0.40, 0.35, -0.02, 0.25],
        }
    )

    summary = summarize_directional_mass(weights, recent_season_count=2)

    record = summary.iloc[0]
    assert record["positive_directional_mass"] == 1.0
    assert record["negative_directional_mass"] == 0.02
    assert record["positive_directional_mass_share"] > 0.98
    assert record["recent_positive_directional_mass_share"] > 0.92
