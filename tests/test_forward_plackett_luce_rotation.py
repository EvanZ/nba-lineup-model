from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.forward_plackett_luce_rotation import (
    MAX_ROTATION_PLAYERS,
    ForwardPlackettLuceConfig,
    predict_forward_conditional_minutes_control,
    predict_forward_plackett_luce,
    prepare_forward_plackett_luce_panel,
    summarize_rotation_metrics,
)


def _available_minutes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    games = (
        ("A-1", "2025-01-01", 1, ((1, "Mover", 180.0), (2, "Home One", 60.0))),
        ("A-2", "2025-01-03", 1, ((1, "Mover", 170.0), (2, "Home One", 70.0))),
        ("B-1", "2025-01-05", 2, ((1, "Mover", 12.0), (3, "New Star", 228.0))),
    )
    for game_id, date, team_id, players in games:
        for player_id, name, minutes in players:
            rows.append(
                {
                    "season": "2024-25",
                    "game_id": game_id,
                    "game_date": pd.Timestamp(date, tz="UTC"),
                    "team_id": team_id,
                    "team": "AAA" if team_id == 1 else "BBB",
                    "player_id": player_id,
                    "player_name": name,
                    "minutes": minutes,
                    "actual_minute_share": minutes / 240.0,
                }
            )
    return pd.DataFrame(rows)


def _prior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "predicted_minutes_per_available_game": [24.0, 20.0, 32.0],
        }
    )


def test_destination_team_resets_role_but_keeps_player_prior() -> None:
    panel = prepare_forward_plackett_luce_panel(_available_minutes(), _prior())
    predictions = predict_forward_plackett_luce(
        panel, config=ForwardPlackettLuceConfig(role_prior_precision=1.0)
    )
    old_team_second_game = predictions.loc[
        predictions["game_id"].eq("A-2") & predictions["player_id"].eq(1)
    ].iloc[0]
    destination_first_game = predictions.loc[
        predictions["game_id"].eq("B-1") & predictions["player_id"].eq(1)
    ].iloc[0]
    assert old_team_second_game["team_role_adjustment"] > 0.0
    assert destination_first_game["team_role_games_completed"] == 0
    assert destination_first_game["team_role_adjustment"] == pytest.approx(0.0)
    assert destination_first_game["predicted_minutes_per_available_game"] == pytest.approx(24.0)


def test_current_game_outcome_cannot_change_its_pregame_prediction() -> None:
    panel = prepare_forward_plackett_luce_panel(_available_minutes(), _prior())
    changed = panel.copy()
    mask = changed["game_id"].eq("A-2")
    changed.loc[mask, "minutes"] = [20.0, 220.0]
    changed.loc[mask, "actual_minute_share"] = [20.0 / 240.0, 220.0 / 240.0]
    original = predict_forward_plackett_luce(
        panel, config=ForwardPlackettLuceConfig(role_prior_precision=1.0)
    )
    revised = predict_forward_plackett_luce(
        changed, config=ForwardPlackettLuceConfig(role_prior_precision=1.0)
    )
    before = original.loc[original["game_id"].eq("A-2"), "predicted_minutes"].to_numpy()
    after = revised.loc[revised["game_id"].eq("A-2"), "predicted_minutes"].to_numpy()
    assert np.allclose(before, after)


def test_allocations_obey_current_roster_and_top_15_cap() -> None:
    rows = []
    for player_id in range(1, 18):
        rows.append(
            {
                "season": "2024-25",
                "game_id": "wide",
                "game_date": pd.Timestamp("2025-01-01", tz="UTC"),
                "team_id": 1,
                "team": "AAA",
                "player_id": player_id,
                "player_name": str(player_id),
                "minutes": 240.0 / 17.0,
                "actual_minute_share": 1.0 / 17.0,
            }
        )
    prior = pd.DataFrame(
        {
            "player_id": list(range(1, 18)),
            "predicted_minutes_per_available_game": list(range(17, 0, -1)),
        }
    )
    panel = prepare_forward_plackett_luce_panel(pd.DataFrame(rows), prior)
    for predictions in (
        predict_forward_conditional_minutes_control(panel),
        predict_forward_plackett_luce(panel, config=ForwardPlackettLuceConfig(1.0)),
    ):
        assert predictions["predicted_minutes"].sum() == pytest.approx(240.0)
        assert predictions["selected_top_15"].sum() == MAX_ROTATION_PLAYERS
        assert predictions["predicted_minutes"].gt(0.0).sum() == MAX_ROTATION_PLAYERS
        metrics = summarize_rotation_metrics(predictions, model="test", season="2024-25")
        assert metrics.loc[0, "evaluated_team_games"] == 1
