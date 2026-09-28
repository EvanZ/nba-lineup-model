from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.possession_state_screen import (
    build_live_ball_interaction_design,
    fit_conditional_state_model,
    predict_conditional_state_models,
)
from nba_lineup_model.models.baselines import vocabulary_mapping


def _rows() -> pd.DataFrame:
    rows = []
    for game in range(40):
        live = game % 2 == 0
        for possession in range(6):
            first_team_a = 1 if possession % 3 == 0 else 11
            first_team_b = 6 if possession % 2 == 0 else 16
            team_a = (first_team_a, 2, 3, 4, 5)
            team_b = (first_team_b, 7, 8, 9, 10)
            offense, defense = (team_a, team_b) if possession % 2 == 0 else (team_b, team_a)
            # Player 1 gets a +2.0 live-ball offensive deviation; player 6's
            # defense gets +1.0, which enters the scoring design with a minus.
            signal = (2.0 if live and 1 in offense else 0.0) - (1.0 if live and 6 in defense else 0.0)
            rows.append(
                {
                    "season": "2025-26",
                    "game_id": f"{game:03d}",
                    "possession_id": f"{game:03d}:{possession:03d}",
                    "home_offense": possession % 2 == 0,
                    "offense_player_ids": offense,
                    "defense_player_ids": defense,
                    "target": 100.0 + 3.0 * live + signal,
                    "baseline_prediction": 100.0,
                    "is_live_ball": live,
                    "coarse_start_state": "live_ball" if live else "dead_ball",
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_live_ball_interactions_are_zero_for_dead_ball_rows() -> None:
    rows = _rows().iloc[[0, 6]].copy()
    columns = vocabulary_mapping(tuple(range(1, 17)))
    design = build_live_ball_interaction_design(rows, columns).toarray()
    np.testing.assert_allclose(design[1], 0.0)
    for player_id in rows.iloc[0]["offense_player_ids"]:
        assert design[0, columns[player_id]] == 1.0
    for player_id in rows.iloc[0]["defense_player_ids"]:
        assert design[0, len(columns) + columns[player_id]] == -1.0


def test_conditional_model_separates_state_control_and_live_player_deviations() -> None:
    rows = _rows()
    fit = fit_conditional_state_model(rows, regularization=0.001)
    predicted = predict_conditional_state_models(rows, fit)
    baseline_error = np.mean(np.square(predicted["target"] - predicted["baseline_prediction"]))
    candidate_error = np.mean(np.square(predicted["target"] - predicted["state_player_prediction"]))
    assert candidate_error < baseline_error * 0.01
    player_index = fit.player_ids.index(1)
    assert fit.full_model.coef_[1 + player_index] > 0.5
