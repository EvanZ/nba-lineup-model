from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.frozen_window_rapm import _target_player_priors, _training_seasons


def test_training_windows_end_before_target_season():
    assert _training_seasons("2025-26", 1) == ("2024-25",)
    assert _training_seasons("2025-26", 3) == ("2022-23", "2023-24", "2024-25")


def test_target_priors_preserve_zero_cold_start_policy():
    coefficients = pd.DataFrame(
        {
            "player_id": [2, 1],
            "prior_rapm_mean": [1.5, -0.5],
        }
    )

    priors = _target_player_priors(
        coefficients,
        [1, 2, 3],
        target_season="2025-26",
        source_season="2024-25",
    ).set_index("player_id")

    assert priors.loc[1, "prior_rapm_mean"] == -0.5
    assert priors.loc[3, "prior_rapm_mean"] == 0.0
    assert priors.loc[1, "prior_branch"] == "window_rapm"
    assert priors.loc[3, "prior_branch"] == "zero_cold_start"
