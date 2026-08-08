from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.frozen_game_outcomes import score_full_game_outcomes


def test_full_game_outcomes_score_margins_and_tied_forecasts_without_side_bias() -> None:
    predictions = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "actual_home_margin": [4.0, -2.0, 6.0],
            "predicted_home_margin": [2.0, 3.0, 0.0],
            "actual_home_win": [True, False, True],
        }
    )

    metrics = score_full_game_outcomes(predictions)

    assert metrics["regular_game_count"] == 3
    assert metrics["full_game_margin_rmse"] == pytest.approx(np.sqrt(65.0 / 3.0))
    assert metrics["full_game_margin_mae"] == pytest.approx(13.0 / 3.0)
    assert metrics["game_winner_accuracy"] == pytest.approx(0.5)
    assert metrics["predicted_tie_count"] == 1


def test_full_game_outcomes_reject_duplicate_games_and_inconsistent_winner_labels() -> None:
    duplicate_games = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "actual_home_margin": [2.0, 3.0],
            "predicted_home_margin": [1.0, 1.0],
            "actual_home_win": [True, True],
        }
    )
    with pytest.raises(ValueError, match="one non-null row per game"):
        score_full_game_outcomes(duplicate_games)

    bad_label = duplicate_games.iloc[:1].assign(actual_home_win=False)
    with pytest.raises(ValueError, match="must agree"):
        score_full_game_outcomes(bad_label)
