from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.forward_draft_history_cold_start import forward_rookie_rate_training


def test_forward_rookie_rate_training_uses_post_fit_forward_coefficients() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2024-25", "2025-26"],
            "player_id": [1, 2],
            "is_rookie": [True, True],
            "rapm": [99.0, 99.0],
            "rapm_possessions": [100.0, 200.0],
            "draft_year": [2024, 2025],
            "draft_round": [1, 1],
            "draft_number": [1, 2],
            "is_undrafted": [False, False],
            "age": [20.0, 21.0],
            "height_inches": [78.0, 80.0],
            "weight_pounds": [210.0, 220.0],
        }
    )
    coefficients = pd.DataFrame(
        {
            "season": ["2024-25", "2025-26"],
            "player_id": [1, 2],
            "rapm": [1.25, -0.75],
        }
    )

    training = forward_rookie_rate_training(panel, coefficients, through_season="2025-26")

    assert training["rapm"].tolist() == [1.25, -0.75]
    assert training["season_start_year"].tolist() == [2024, 2025]
