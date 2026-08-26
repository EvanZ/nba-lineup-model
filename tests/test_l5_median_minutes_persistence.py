"""Tests for the last-five median-minutes rotation baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l5_median_minutes_persistence import (
    evaluate_l5_median_minutes_persistence,
    median_minute_share_forecast,
)


def _game(game_id: str, player_one_minutes: float, player_two_minutes: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id],
            "team_id": [1, 1],
            "team_tricode": ["TST", "TST"],
            "player_id": [1, 2],
            "player_name": ["One", "Two"],
            "game_time_utc": pd.to_datetime([f"2025-01-{game_id}"] * 2, utc=True),
            "minutes": [player_one_minutes, player_two_minutes],
            "minute_share": [player_one_minutes / 48.0, player_two_minutes / 48.0],
        }
    )


def test_l5_forecast_uses_median_minutes_then_normalizes() -> None:
    history = [
        _game("01", 30.0, 18.0),
        _game("02", 24.0, 24.0),
        _game("03", 36.0, 12.0),
        _game("04", 12.0, 36.0),
        _game("05", 24.0, 24.0),
    ]

    forecast = median_minute_share_forecast(history)

    assert forecast == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}


def test_l5_evaluates_only_after_five_completed_team_games() -> None:
    games = [
        _game("01", 30.0, 18.0),
        _game("02", 24.0, 24.0),
        _game("03", 36.0, 12.0),
        _game("04", 12.0, 36.0),
        _game("05", 24.0, 24.0),
        _game("06", 30.0, 18.0),
    ]
    minutes = pd.concat(games, ignore_index=True)

    predictions, metrics = evaluate_l5_median_minutes_persistence(minutes)

    assert set(predictions["game_id"]) == {"06"}
    assert metrics.loc[0, "history_start_game_id"] == "01"
    assert metrics.loc[0, "previous_game_id"] == "05"
    assert metrics.loc[0, "allocation_total_variation"] == pytest.approx(0.125)
    assert metrics.loc[0, "brier_score"] == pytest.approx(0.03125)
