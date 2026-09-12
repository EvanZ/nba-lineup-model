from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.forward_plackett_luce_availability import (
    predict_availability_integrated_fcm_control,
    predict_availability_integrated_plackett_luce,
    prepare_availability_integrated_panel,
    summarize_integrated_rotation_metrics,
)
from nba_lineup_model.rotation.forward_plackett_luce_rotation import ForwardPlackettLuceConfig


def _roster_minutes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    games = (
        ("001", "2025-01-01", ((1, 96.0), (2, 72.0), (3, 48.0), (4, 24.0), (5, 0.0), (6, 0.0))),
        ("002", "2025-01-03", ((1, 90.0), (2, 78.0), (3, 42.0), (4, 30.0), (5, 0.0), (6, 0.0))),
    )
    for game_id, date, players in games:
        for player_id, minutes in players:
            rows.append(
                {
                    "season": "2024-25",
                    "game_id": game_id,
                    "game_date": pd.Timestamp(date, tz="UTC"),
                    "team_id": 1,
                    "team": "TST",
                    "player_id": player_id,
                    "player_name": str(player_id),
                    "minutes": minutes,
                    "available": player_id < 6,
                    "actual_minute_share": minutes / 240.0,
                }
            )
    return pd.DataFrame(rows)


def _conditional_prior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": list(range(1, 7)),
            "predicted_minutes_per_available_game": [30.0, 25.0, 20.0, 15.0, 10.0, 5.0],
        }
    )


def _availability_prior() -> pd.DataFrame:
    return pd.DataFrame(
        {"player_id": list(range(1, 7)), "predicted_available_share": [1.0] * 5 + [0.0]}
    )


def test_integrated_control_respects_availability_probability_and_240_minutes() -> None:
    panel = prepare_availability_integrated_panel(
        _roster_minutes(), _conditional_prior(), _availability_prior()
    )
    predictions = predict_availability_integrated_fcm_control(panel, scenario_count=8)
    zero_probability = predictions.loc[predictions["player_id"].eq(6)]
    assert zero_probability["predicted_minutes"].eq(0.0).all()
    totals = predictions.groupby(["game_id", "team_id"])["predicted_minutes"].sum()
    assert np.allclose(totals, 240.0)
    assert predictions["mean_scenario_available_players"].eq(5.0).all()


def test_current_game_outcome_does_not_change_integrated_prediction() -> None:
    panel = prepare_availability_integrated_panel(
        _roster_minutes(), _conditional_prior(), _availability_prior()
    )
    changed = panel.copy()
    current = changed["game_id"].eq("002")
    changed.loc[current, "minutes"] = [30.0, 30.0, 60.0, 120.0, 0.0, 0.0]
    changed.loc[current, "actual_minute_share"] = [
        30 / 240,
        30 / 240,
        60 / 240,
        120 / 240,
        0.0,
        0.0,
    ]
    original = predict_availability_integrated_plackett_luce(
        panel, config=ForwardPlackettLuceConfig(10.0), scenario_count=8
    )
    revised = predict_availability_integrated_plackett_luce(
        changed, config=ForwardPlackettLuceConfig(10.0), scenario_count=8
    )
    original_game = original.loc[original["game_id"].eq("002"), "predicted_minutes"].to_numpy()
    revised_game = revised.loc[revised["game_id"].eq("002"), "predicted_minutes"].to_numpy()
    assert np.allclose(original_game, revised_game)


def test_integrated_metrics_use_complete_roster_support() -> None:
    panel = prepare_availability_integrated_panel(
        _roster_minutes(), _conditional_prior(), _availability_prior()
    )
    predictions = predict_availability_integrated_fcm_control(panel, scenario_count=8)
    metrics = summarize_integrated_rotation_metrics(
        predictions, model="test", season="2024-25"
    )
    assert metrics.loc[0, "evaluated_team_games"] == 2
    assert metrics.loc[0, "mean_roster_candidates"] == pytest.approx(6.0)
