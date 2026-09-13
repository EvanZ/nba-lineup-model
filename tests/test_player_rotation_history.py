"""Tests for the compact availability and FCM player-bio cache."""

from __future__ import annotations

import pandas as pd

import nba_lineup_model.web_api.player_rotation_history as rotation_history


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2024-25", "2025-26"],
            "season_start_year": [2024, 2025],
            "player_id": [7, 7],
            "player_name": ["Test Player", "Test Player"],
            "age": [23.0, 24.0],
            "available_games": [72, 68],
            "known_player_games": [82, 82],
            "available_share": [72 / 82, 68 / 82],
            "injury_or_illness_games": [8, 11],
            "rest_games": [2, 3],
            "minutes_per_available_game": [27.0, 28.0],
            "total_nba_minutes": [1944.0, 1904.0],
        }
    )


def test_rotation_history_keeps_observed_outcomes_and_forward_forecasts(monkeypatch) -> None:
    summary = _summary()

    def predict_availability(*_args, target_season: str, **_kwargs):
        return (
            pd.DataFrame(
                {
                    "season": [target_season],
                    "player_id": [7],
                    "predicted_available_share": [0.8],
                }
            ),
            None,
            {},
        )

    def predict_minutes(*_args, target_season: str, **_kwargs):
        return (
            pd.DataFrame(
                {
                    "season": [target_season],
                    "player_id": [7],
                    "predicted_minutes_per_available_game": [29.0],
                }
            ),
            None,
        )

    monkeypatch.setattr(rotation_history, "predict_availability_season", predict_availability)
    monkeypatch.setattr(rotation_history, "predict_conditional_minutes_season", predict_minutes)

    history = rotation_history.build_player_rotation_history(
        summary,
        forecast_players=pd.DataFrame(
            {
                "player_id": [7],
                "player_name": ["Test Player"],
                "age": [25.0],
                "availability_probability": [0.75],
                "conditional_minutes_per_game": [30.0],
                "baseline_minutes_per_game": [18.5],
            }
        ),
        forecast_season="2026-27",
    )

    observed = history.loc[history["season"].eq("2025-26")].iloc[0]
    forecast = history.loc[history["season"].eq("2026-27")].iloc[0]

    assert observed["actual_available_games"] == 68
    assert observed["actual_minutes_per_available_game"] == 28.0
    assert observed["predicted_available_share"] == 0.8
    assert observed["predicted_minutes_per_available_game"] == 29.0
    assert bool(observed["is_preseason_forecast"]) is False
    assert pd.isna(observed["projected_total_minutes"])
    assert pd.isna(forecast["actual_available_games"])
    assert forecast["predicted_available_share"] == 0.75
    assert forecast["predicted_minutes_per_available_game"] == 30.0
    assert forecast["projected_total_minutes"] == 1517.0
    assert bool(forecast["is_preseason_forecast"]) is True
