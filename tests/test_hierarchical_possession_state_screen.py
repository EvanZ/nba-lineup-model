from __future__ import annotations

import numpy as np
import pandas as pd

import nba_lineup_model.modeling.hierarchical_possession_state_screen as screen


def _rows() -> pd.DataFrame:
    states = (
        ("after_defensive_rebound", "live_ball"),
        ("after_live_ball_turnover_steal", "live_ball"),
        ("after_made_2pt", "dead_ball"),
        ("after_made_3pt", "dead_ball"),
    )
    rows = []
    for game in range(16):
        for possession, (state, coarse) in enumerate(states):
            team_a = (1, 2, 3, 4, 5)
            team_b = (6, 7, 8, 9, 10)
            offense, defense = (team_a, team_b) if possession % 2 == 0 else (team_b, team_a)
            signal = 4.0 if state == "after_live_ball_turnover_steal" and 6 in offense else 0.0
            rows.append(
                {
                    "season": "2025-26",
                    "game_id": str(game),
                    "possession_id": f"{game}:{possession}",
                    "home_offense": possession % 2 == 0,
                    "offense_player_ids": offense,
                    "defense_player_ids": defense,
                    "target": 100.0 + signal,
                    "baseline_prediction": 100.0,
                    "start_state": state,
                    "coarse_start_state": coarse,
                    "weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_source_centered_parent_and_child_designs_sum_to_zero(monkeypatch) -> None:
    monkeypatch.setattr(screen, "MINIMUM_STATE_SOURCE_POSSESSIONS", 1)
    specification = screen.build_fine_state_specification(_rows())
    prepared = screen.prepare_fine_state_features(_rows(), specification)

    np.testing.assert_allclose(np.asarray(prepared.parent.sum(axis=0)).ravel(), 0.0)
    np.testing.assert_allclose(np.asarray(prepared.children.sum(axis=0)).ravel(), 0.0)


def test_hierarchical_fit_separates_supported_child_state_signal(monkeypatch) -> None:
    monkeypatch.setattr(screen, "MINIMUM_STATE_SOURCE_POSSESSIONS", 1)
    specification = screen.build_fine_state_specification(_rows())
    prepared = screen.prepare_fine_state_features(_rows(), specification)
    fit = screen.fit_hierarchical_state_model(prepared, child_relative_precision=16.0)
    predictions = screen.predict_hierarchical_state_models(prepared, fit)

    baseline_error = np.mean(np.square(predictions["target"] - predictions["baseline_prediction"]))
    candidate_error = np.mean(
        np.square(predictions["target"] - predictions["hierarchical_player_prediction"])
    )
    assert candidate_error < baseline_error
    diagnostics = screen._player_child_diagnostics(prepared, fit)
    assert not diagnostics.empty
    assert set(diagnostics["start_state"]) == set(specification.child_states)
