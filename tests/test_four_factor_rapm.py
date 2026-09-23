from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.four_factor_rapm import build_factor_design


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["001", "002"],
            "home_player_ids": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10], [6, 7, 8, 9, 10]],
            "home_minus_away_back_to_back": [0.0, 1.0],
            "home_efg_pct": [0.6, 0.5],
            "away_efg_pct": [0.5, 0.6],
            "home_efg_denominator": [10.0, 4.0],
            "away_efg_denominator": [10.0, 12.0],
            "home_tov_pct": [0.1, 0.2],
            "away_tov_pct": [0.2, 0.1],
            "home_tov_denominator": [10.0, 4.0],
            "away_tov_denominator": [10.0, 12.0],
            "home_orb_pct": [0.3, 0.2],
            "away_orb_pct": [0.2, 0.3],
            "home_orb_denominator": [10.0, 4.0],
            "away_orb_denominator": [10.0, 12.0],
            "home_ftr": [0.3, 0.1],
            "away_ftr": [0.2, 0.2],
            "home_ftr_denominator": [10.0, 4.0],
            "away_ftr_denominator": [10.0, 12.0],
        }
    )


def test_factor_design_uses_harmonic_opportunity_and_turnover_avoidance() -> None:
    design = build_factor_design(_stints(), "tov_pct")

    assert design.target.tolist() == [0.1, -0.1]
    assert design.weights.tolist() == [5.0, 3.0]
    assert design.player_opportunities[0] == 14.0
    assert design.player_opportunities[-1] == 22.0
    assert design.features.shape == (2, 10)
    assert np.allclose(design.features.toarray()[0, :5], 1.0)
    assert np.allclose(design.features.toarray()[0, 5:], -1.0)
