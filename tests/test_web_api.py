"""Contract tests for the NBA GESTALT local API."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.matchup_contextual import (
    fit_bounded_hierarchical_matchup_contextual_model,
    fit_matchup_contextual_model,
)
from nba_lineup_model.web_api.app import create_app
from nba_lineup_model.web_api.inference import LineupEvaluator, _warm_response_cache


def _evaluator(*, bounded: bool = False) -> LineupEvaluator:
    player_ids = list(range(1, 11))
    profile_offset = player_ids if bounded else [0] * len(player_ids)
    profiles = pd.DataFrame(
        {
            "player_id": player_ids,
            "three_pa_per_100": [5.0 + value / 10 for value in profile_offset],
            "three_pm_per_100": [1.0 + value / 20 for value in profile_offset],
            "assists_per_100": [3.0 + value / 10 for value in profile_offset],
            "turnovers_per_100": [1.0 + value / 25 for value in profile_offset],
            "usage_per_100": [15.0 + value / 2 for value in profile_offset],
            "offensive_rebounds_per_100": [1.0 + value / 10 for value in profile_offset],
            "defensive_rebounds_per_100": [4.0 + value / 5 for value in profile_offset],
            "steals_per_100": [0.5 + value / 20 for value in profile_offset],
            "blocks_per_100": [0.25 + value / 25 for value in profile_offset],
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
    home_lineups = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
    ]
    away_lineups = [
        [6, 7, 8, 9, 10],
        [1, 7, 8, 9, 10],
        [1, 2, 8, 9, 10],
        [1, 2, 3, 9, 10],
        [1, 2, 3, 4, 10],
    ]
    fit = (
        fit_bounded_hierarchical_matchup_contextual_model
        if bounded
        else fit_matchup_contextual_model
    )
    model = fit(
        lineup_side_context_features(home_lineups, profiles),
        lineup_side_context_features(away_lineups, profiles),
        np.arange(5, dtype=float),
        np.ones(5, dtype=float),
        alpha=1.0,
    )
    return LineupEvaluator(
        season="2025-26",
        run_id="test-run",
        coefficients=players.loc[:, ["player_id", "rapm"]],
        profiles=profiles,
        players=players,
        context_model=model,
        response_cache=_warm_response_cache(model),
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
        json={
            "unit_player_ids": [1, 2, 3, 4, 5],
            "opponent_player_ids": [6, 7, 8, 9, 10],
            "include_response_curves": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["additive_margin"] == -2.5
    assert "relative_context_reference" not in payload
    assert len(payload["feature_contributions"]) == 20
    assert np.isclose(
        payload["contextual_adjustment"],
        payload["portable_composition_margin"] + payload["matchup_adjustment"],
    )
    assert np.isclose(
        payload["additive_margin"],
        payload["unit"]["additive_rating"] - payload["opponent"]["additive_rating"],
    )
    assert len(payload["composition_feature_contributions"]) == 20
    assert len(payload["matchup_feature_contributions"]) == 20
    composition_curves = payload["composition_response_curves"]
    matchup_curves = payload["matchup_response_curves"]
    assert len(composition_curves) == 20
    assert len(matchup_curves) == 20
    assert all(len(curve["points"]) == 33 for curve in composition_curves + matchup_curves)
    composition_by_id = {
        row["id"]: row["contribution"] for row in payload["composition_feature_contributions"]
    }
    matchup_by_id = {
        row["id"]: row["contribution"] for row in payload["matchup_feature_contributions"]
    }
    for curve in composition_curves:
        assert np.isclose(
            curve["unit_contribution"] - curve["opponent_contribution"],
            composition_by_id[curve["id"]],
        )
    for curve in matchup_curves:
        assert np.isclose(curve["unit_contribution"], matchup_by_id[curve["id"]])
    assert np.isclose(
        sum(row["contribution"] for row in payload["composition_feature_contributions"]),
        payload["portable_composition_margin"],
    )
    assert np.isclose(
        sum(row["contribution"] for row in payload["matchup_feature_contributions"]),
        payload["matchup_adjustment"],
    )


def test_matchup_rejects_player_on_both_sides() -> None:
    client = TestClient(create_app(_evaluator()))

    response = client.post(
        "/api/matchups",
        json={"unit_player_ids": [1, 2, 3, 4, 5], "opponent_player_ids": [1, 6, 7, 8, 9]},
    )

    assert response.status_code == 422
    assert "both sides" in response.json()["detail"]


def test_bounded_matchup_chart_contributions_match_the_model_cards() -> None:
    client = TestClient(create_app(_evaluator(bounded=True)))

    response = client.post(
        "/api/matchups",
        json={
            "unit_player_ids": [1, 2, 3, 4, 5],
            "opponent_player_ids": [6, 7, 8, 9, 10],
            "include_response_curves": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    cards = {
        row["id"]: row["contribution"] for row in payload["matchup_feature_contributions"]
    }
    curves = {row["id"]: row["unit_contribution"] for row in payload["matchup_response_curves"]}
    assert np.isclose(sum(cards.values()), payload["matchup_adjustment"])
    for feature_id, contribution in cards.items():
        assert np.isclose(curves[feature_id], contribution, atol=0.01)
