from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.usage_response import (
    build_pregame_usage_states_from_games,
    build_usage_response_features,
    fit_usage_response,
    game_usage_from_player_boxscores,
    lineup_usage_shock_features,
)
from nba_lineup_model.modeling.split_nail import (
    SPLIT_NAIL_ADDITIVE_FEATURES,
    SPLIT_NAIL_NONADDITIVE_FEATURES,
    build_split_nail_design_from_side_features,
)


def _boxscores() -> pd.DataFrame:
    rows = []
    for game_id, timestamp, attempts in (
        ("001", "2025-10-20T00:00:00Z", 30),
        ("002", "2025-10-22T00:00:00Z", 10),
    ):
        for player_id in range(1, 6):
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": 100,
                    "personId": player_id,
                    "game_time_utc": timestamp,
                    "statistics_minutes": "PT24M00.00S",
                    "statistics_fieldGoalsAttempted": attempts if player_id == 1 else 10,
                    "statistics_freeThrowsAttempted": 0,
                    "statistics_turnovers": 0,
                }
            )
    return pd.DataFrame(rows)


def test_pregame_usage_state_excludes_the_current_game_box_score() -> None:
    games = game_usage_from_player_boxscores(_boxscores())
    prior = pd.DataFrame(
        {
            "player_id": [1],
            "usage_events": [100.0],
            "usage_opportunities": [500.0],
        }
    )
    states = build_pregame_usage_states_from_games(
        games,
        prior_summary=prior,
        pseudo_opportunities=100.0,
    )
    first = states.loc[states["player_id"].eq(1)].iloc[0]
    second = states.loc[states["player_id"].eq(1)].iloc[1]
    assert first["pregame_usage_pct"] == pytest.approx(20.0)
    assert second["pregame_usage_pct"] > first["pregame_usage_pct"]

    mutated = games.copy()
    mutated.loc[
        mutated["game_id"].eq("002") & mutated["player_id"].eq(1), "usage_events"
    ] = 999.0
    no_leak = build_pregame_usage_states_from_games(
        mutated,
        prior_summary=prior,
        pseudo_opportunities=100.0,
    )
    assert no_leak.loc[no_leak["player_id"].eq(1)].iloc[1]["pregame_usage_pct"] == pytest.approx(
        second["pregame_usage_pct"]
    )


def test_usage_response_recovers_independent_offense_and_defense_slopes() -> None:
    generator = np.random.default_rng(4)
    features = generator.normal(size=(1_000, 2))
    target = 0.35 * features[:, 0] - 0.18 * features[:, 1]
    model = fit_usage_response(
        features,
        target,
        np.ones(len(features)),
        regularization=0.0,
    )
    assert model.raw_coefficients[0] == pytest.approx(0.35, abs=1e-10)
    assert model.raw_coefficients[1] == pytest.approx(-0.18, abs=1e-10)
    np.testing.assert_allclose(model.predict_response(features), target, atol=1e-10)


def test_lineup_shocks_follow_positive_exposure_split_scoring_rows() -> None:
    stints = pd.DataFrame(
        {
            "game_id": ["001", "001"],
            "home_player_ids": [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5)],
            "away_player_ids": [(6, 7, 8, 9, 10), (6, 7, 8, 9, 10)],
            "points_home": [3, 4],
            "points_away": [0, 2],
            "home_offensive_possessions": [1.0, 1.0],
            "away_offensive_possessions": [0.0, 1.0],
        }
    )
    profile_columns = (*SPLIT_NAIL_ADDITIVE_FEATURES, *SPLIT_NAIL_NONADDITIVE_FEATURES)
    profiles = pd.DataFrame({column: [0.0, 0.0] for column in profile_columns})
    design = build_split_nail_design_from_side_features(stints, profiles, profiles)
    states = pd.DataFrame(
        {
            "game_id": ["001"] * 10,
            "player_id": list(range(1, 11)),
            "usage_shock_pct": np.arange(1.0, 11.0),
        }
    )

    shocks = lineup_usage_shock_features(stints, states)
    features = build_usage_response_features(design, shocks)

    assert shocks["home_offense"].tolist() == [True, True, False]
    np.testing.assert_allclose(
        features,
        np.array([[15.0, -40.0], [15.0, -40.0], [40.0, -15.0]]),
    )
