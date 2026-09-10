from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.ac_minute_share_projection import (
    ACMSPConfig,
    build_conditional_minute_panel,
    fit_ac_msp,
    load_regulation_available_minutes,
    predict_ac_msp,
    predict_conditional_l5_control,
    summarize_allocation_metrics,
)


def _available_minutes() -> pd.DataFrame:
    rows = []
    games = (
        ("001", "2025-01-01", ((1, "One", 30.0), (2, "Two", 18.0))),
        ("002", "2025-01-03", ((1, "One", 24.0), (2, "Two", 18.0), (3, "Three", 6.0))),
        ("003", "2025-01-05", ((1, "One", 20.0), (2, "Two", 16.0), (3, "Three", 12.0))),
    )
    for game_id, game_date, players in games:
        for player_id, player_name, minutes in players:
            rows.append(
                {
                    "season": "2024-25",
                    "game_id": game_id,
                    "game_date": pd.Timestamp(game_date, tz="UTC"),
                    "team_id": 1,
                    "team": "TST",
                    "player_id": player_id,
                    "player_name": player_name,
                    "minutes": minutes,
                    "actual_minute_share": minutes / 48.0,
                }
            )
    return pd.DataFrame(rows)


def test_panel_only_uses_earlier_team_games() -> None:
    panel = build_conditional_minute_panel(_available_minutes())
    second_game = panel.loc[panel["game_id"].eq("002")].set_index("player_id")
    assert second_game.loc[1, "lag_1_minutes"] == pytest.approx(30.0)
    assert second_game.loc[2, "lag_5_mean_minutes"] == pytest.approx(18.0)
    assert second_game.loc[3, "lag_1_minutes"] == 0.0
    assert second_game.loc[3, "team_games_completed"] == 1


def test_softmax_and_control_allocate_to_current_available_candidates() -> None:
    panel = build_conditional_minute_panel(_available_minutes())
    model = fit_ac_msp(panel, config=ACMSPConfig(alpha=0.1))
    predictions = predict_ac_msp(panel, model=model)
    control = predict_conditional_l5_control(panel)
    for frame in (predictions, control):
        totals = frame.groupby(["game_id", "team_id"])["predicted_minute_share"].sum()
        assert np.isclose(totals, 1.0).all()
        assert frame["predicted_minute_share"].gt(0.0).all()
    metrics = summarize_allocation_metrics(predictions, model="test", season="2024-25")
    assert metrics.loc[0, "evaluated_team_games"] == 3
    assert np.isfinite(metrics.loc[0, "mean_cross_entropy"])


def test_loader_excludes_overtime_and_ambiguous_rosters(tmp_path) -> None:
    root = tmp_path / "player_availability" / "2024-25"
    root.mkdir(parents=True)
    rows = []
    for game_id, team_id, minutes, known in (
        ("regulation", 1, (120.0, 120.0), True),
        ("ambiguous", 2, (120.0, 120.0), False),
        ("overtime", 3, (132.5, 132.5), True),
    ):
        for player_id, player_minutes in enumerate(minutes, start=1):
            rows.append(
                {
                    "season": "2024-25",
                    "game_id": game_id,
                    "game_date": pd.Timestamp("2025-01-01", tz="UTC"),
                    "team_id": team_id,
                    "team": "TST",
                    "player_id": player_id,
                    "player_name": str(player_id),
                    "minutes": player_minutes,
                    "available": True,
                    "availability_state_known": known or player_id == 1,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "part-00000.parquet", index=False)

    loaded = load_regulation_available_minutes("2024-25", curated_dir=tmp_path)

    assert set(loaded["game_id"]) == {"regulation"}
    assert loaded["actual_minute_share"].sum() == pytest.approx(1.0)
