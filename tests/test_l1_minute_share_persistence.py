"""Tests for the parameter-free rotation baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l1_minute_share_persistence import (
    evaluate_l1_minute_share_persistence,
)


def test_l1_copies_prior_team_game_shares_and_exposes_new_player_error() -> None:
    minutes = pd.DataFrame(
        {
            "game_id": ["001", "001", "002", "002"],
            "team_id": [1, 1, 1, 1],
            "team_tricode": ["TST"] * 4,
            "player_id": [1, 2, 1, 3],
            "player_name": ["One", "Two", "One", "Three"],
            "game_time_utc": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-03", "2025-01-03"], utc=True
            ),
            "minutes": [30.0, 18.0, 24.0, 24.0],
            "team_minutes": [48.0] * 4,
            "minute_share": [0.625, 0.375, 0.5, 0.5],
        }
    )

    predictions, metrics = evaluate_l1_minute_share_persistence(minutes)

    assert set(predictions["player_id"]) == {1, 2, 3}
    predicted = predictions.set_index("player_id")["predicted_minute_share"]
    actual = predictions.set_index("player_id")["actual_minute_share"]
    assert predicted.to_dict() == {1: 0.625, 2: 0.375, 3: 0.0}
    assert actual.to_dict() == {1: 0.5, 2: 0.0, 3: 0.5}
    assert metrics.loc[0, "allocation_total_variation"] == pytest.approx(0.5)
    assert metrics.loc[0, "brier_score"] == pytest.approx(0.40625)
    assert metrics.loc[0, "cross_entropy"] == pytest.approx(float("inf"))
    assert bool(metrics.loc[0, "has_zero_probability_active_player"])
    assert metrics.loc[0, "new_player_actual_share"] == pytest.approx(0.5)
    assert metrics.loc[0, "departed_player_predicted_share"] == pytest.approx(0.375)


def test_l1_uses_actual_team_minutes_so_overtime_shares_remain_normalized() -> None:
    minutes = pd.DataFrame(
        {
            "game_id": ["001", "001", "002", "002"],
            "team_id": [1, 1, 1, 1],
            "team_tricode": ["TST"] * 4,
            "player_id": [1, 2, 1, 2],
            "player_name": ["One", "Two", "One", "Two"],
            "game_time_utc": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-03", "2025-01-03"], utc=True
            ),
            "minutes": [132.5, 132.5, 159.0, 106.0],
            "team_minutes": [265.0, 265.0, 265.0, 265.0],
            "minute_share": [0.5, 0.5, 0.6, 0.4],
        }
    )

    predictions, metrics = evaluate_l1_minute_share_persistence(minutes)

    assert predictions["actual_minute_share"].sum() == pytest.approx(1.0)
    assert predictions["predicted_minute_share"].sum() == pytest.approx(1.0)
    assert metrics.loc[0, "brier_score"] == pytest.approx(0.02)
    assert metrics.loc[0, "cross_entropy"] == pytest.approx(0.693147)
    assert not bool(metrics.loc[0, "has_zero_probability_active_player"])
