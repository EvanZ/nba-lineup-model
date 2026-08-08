"""Contract tests for the NBA GESTALT local API."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.web_api.app import create_app
from nba_lineup_model.web_api.inference import LineupEvaluator


def _evaluator() -> LineupEvaluator:
    player_ids = list(range(1, 11))
    profiles = pd.DataFrame(
        {
            "player_id": player_ids,
            "three_pa_per_100": [6.0] * 10,
            "three_pm_per_100": [2.0] * 10,
            "assists_per_100": [5.0] * 10,
            "turnovers_per_100": [2.0] * 10,
            "usage_per_100": [20.0] * 10,
            "offensive_rebounds_per_100": [2.0] * 10,
            "defensive_rebounds_per_100": [6.0] * 10,
            "steals_per_100": [1.0] * 10,
            "blocks_per_100": [1.0] * 10,
            "profile_imputed": [0] * 10,
            "profile_replacement_weight": [0.0] * 10,
        }
    )
    players = profiles.assign(
        player_name=["Nikola Jokić", *[f"Player {player_id}" for player_id in player_ids[1:]]],
        team="TST",
        position="G",
        rapm=[float(player_id) / 10 for player_id in player_ids],
        possessions=1000.0,
        games=50,
        profile_source="prior_season",
    )
    training_features = pd.DataFrame(
        np.arange(100, dtype=float).reshape(5, 20),
        columns=[
            "home_minus_away_three_pa_per_100",
            "home_minus_away_three_pm_per_100",
            "home_minus_away_assists_per_100",
            "home_minus_away_turnovers_per_100",
            "home_minus_away_usage_per_100",
            "home_minus_away_offensive_rebounds_per_100",
            "home_minus_away_defensive_rebounds_per_100",
            "home_minus_away_steals_per_100",
            "home_minus_away_blocks_per_100",
            "home_minus_away_bottom_two_three_pm",
            "home_minus_away_credible_shooter_count",
            "home_minus_away_top_two_assists",
            "home_minus_away_usage_concentration",
            "home_minus_away_sqrt_offensive_rebounds",
            "home_minus_away_sqrt_defensive_rebounds",
            "home_minus_away_imputed_count",
            "home_minus_away_replacement_weight",
            "home_minus_away_shooting_usage_interaction",
            "home_minus_away_shooter_passing_interaction",
            "home_minus_away_rebounding_usage_interaction",
        ],
    )
    model = Pipeline(
        [
            ("spline", SplineTransformer(n_knots=4, degree=2, extrapolation="linear")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    ).fit(training_features, np.arange(5, dtype=float))
    return LineupEvaluator(
        season="2025-26",
        run_id="test-run",
        coefficients=players.loc[:, ["player_id", "rapm"]],
        profiles=profiles,
        players=players,
        context_model=model,
    )


def test_search_and_matchup_endpoints() -> None:
    client = TestClient(create_app(_evaluator()))

    search = client.get("/api/players", params={"q": "Jokic"})
    assert search.status_code == 200
    assert search.json()["players"][0]["player_name"] == "Nikola Jokić"

    default_opponent = client.get(
        "/api/default-opponent",
        params=[("exclude_player_id", player_id) for player_id in range(1, 6)],
    )
    assert default_opponent.status_code == 200
    assert len(default_opponent.json()["players"]) == 5
    default_ids = {player["player_id"] for player in default_opponent.json()["players"]}
    assert default_ids == {6, 7, 8, 9, 10}

    response = client.post(
        "/api/matchups",
        json={"unit_player_ids": [1, 2, 3, 4, 5], "opponent_player_ids": [6, 7, 8, 9, 10]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["additive_margin"] == -2.5
    assert len(payload["feature_contributions"]) == 20


def test_matchup_rejects_player_on_both_sides() -> None:
    client = TestClient(create_app(_evaluator()))

    response = client.post(
        "/api/matchups",
        json={"unit_player_ids": [1, 2, 3, 4, 5], "opponent_player_ids": [1, 6, 7, 8, 9]},
    )

    assert response.status_code == 422
    assert "both sides" in response.json()["detail"]
