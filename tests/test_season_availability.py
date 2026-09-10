from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation.l20_source_aware_preseason_minute_share import (
    SourceAwareSeasonInputs,
)
from nba_lineup_model.rotation.season_availability import (
    AvailabilityConfig,
    _season_appearance_counts,
    evaluate_availability,
    fit_availability_model,
)


def _role_state() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 1], "team": ["MIN", "MIN"], "player_id": [1, 2],
            "player_name": ["Available", "Absent"], "player_state": ["continuous", "cold_start"],
            "gap_seasons": [0.0, 0.0], "source_minutes": [1800.0, 0.0],
            "source_games": [70.0, 0.0], "source_starts": [60.0, 0.0],
            "preseason_nail": [2.0, 0.0], "draft_capital": [0.0, 0.9],
            "is_undrafted": [False, False], "age_offset": [2.0, -2.0],
            "is_guard": [1.0, 1.0], "is_forward": [0.0, 0.0], "is_center": [0.0, 0.0],
        }
    )


def _game_minutes() -> pd.DataFrame:
    rows = []
    for game in range(3):
        rows.append({"team_id": 1, "game_id": f"a{game}", "player_id": 1})
    rows.append({"team_id": 2, "game_id": "b0", "player_id": 1})
    rows.append({"team_id": 2, "game_id": "b1", "player_id": 2})
    rows.append({"team_id": 2, "game_id": "b2", "player_id": 2})
    rows.append({"team_id": 2, "game_id": "b3", "player_id": 2})
    return pd.DataFrame(rows)


def test_appearance_count_is_player_season_total_even_across_teams() -> None:
    appearances, scheduled_games = _season_appearance_counts(_game_minutes())
    assert scheduled_games == 4
    assert appearances.loc[1] == pytest.approx(4.0)
    assert appearances.loc[2] == pytest.approx(3.0)


def test_availability_fit_and_evaluation_use_full_season_games() -> None:
    state = _role_state()
    inputs = SourceAwareSeasonInputs(
        season="2025-26", target_game_minutes=_game_minutes(),
        opening_candidates=state.loc[:, ["team_id", "team", "player_id", "player_name"]],
        role_state=state, preseason_nail_ratings=pd.Series({1: 2.0, 2: 0.0}),
    )
    model = fit_availability_model([inputs], config=AvailabilityConfig())
    predictions, metrics = evaluate_availability(inputs, model=model)
    assert predictions.loc[predictions.player_id.eq(1), "actual_games"].item() == 4.0
    assert predictions.loc[predictions.player_id.eq(1), "scheduled_games"].item() == 4.0
    assert metrics.loc[0, "scheduled_games"] == 4
