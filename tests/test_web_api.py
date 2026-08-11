"""Contract tests for the NBA GESTALT local API."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.matchup_contextual import (
    fit_bounded_hierarchical_matchup_contextual_model,
    fit_matchup_contextual_model,
)
import nba_lineup_model.web_api.app as web_app
from nba_lineup_model.web_api.app import create_app
from nba_lineup_model.web_api.inference import (
    LineupEvaluator,
    _historical_ranking_catalog,
    _player_rating_histories,
    _warm_response_cache,
)


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
        age=26.0,
        rating_history=[
            [
                {"season": "2023-24", "rating": float(player_id) / 20, "age": 24.0, "team_id": 1610612757, "team": "TST"},
                {"season": "2024-25", "rating": float(player_id) / 15, "age": 25.0, "team_id": 1610612757, "team": "TST"},
                {"season": "2025-26", "rating": float(player_id) / 10, "age": 26.0, "team_id": 1610612757, "team": "TST"},
            ]
            for player_id in player_ids
        ],
        rookie_season="2023-24",
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
    assert len(search.json()["players"][0]["rating_history"]) == 3
    assert search.json()["players"][0]["rookie_season"] == "2023-24"
    assert search.json()["players"][0]["age"] == 26.0

    rankings = client.get("/api/rankings")
    assert rankings.status_code == 200
    assert rankings.json()["season"] == "2025-26"
    assert rankings.json()["available_seasons"] == ["2025-26"]
    assert rankings.json()["players"][0]["rank"] == 1
    assert rankings.json()["players"][0]["player_id"] == 10

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


def test_player_rating_histories_include_seasonal_team_tricode(tmp_path) -> None:
    panel_path = tmp_path / "player_seasons.parquet"
    pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "player_id": [77, 77],
            "primary_team_id": [1610612757, 1610612749],
            "primary_team_tricode": ["POR", "MIL"],
            "rapm_possessions": [1200.0, 1100.0],
            "games": [60, 55],
            "games_started": [42, 39],
        }
    ).to_parquet(panel_path, index=False)
    ratings = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "player_id": [77, 77],
            "player_name": ["Test Player", "Test Player"],
            "rapm": [1.0, 2.0],
            "age": [25.0, 26.0],
        }
    )

    history = _player_rating_histories(ratings, panel_path=panel_path)

    assert [point["team"] for point in history[77]] == ["POR", "MIL"]
    assert [point["team_id"] for point in history[77]] == [1610612757, 1610612749]
    assert [point["possessions"] for point in history[77]] == [1200.0, 1100.0]
    assert [point["games"] for point in history[77]] == [60, 55]
    assert [point["games_started"] for point in history[77]] == [42, 39]


def test_historical_ranking_catalog_exposes_each_completed_fit_season(tmp_path) -> None:
    panel_path = tmp_path / "player_seasons.parquet"
    pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "player_id": [77, 77],
            "primary_team_tricode": ["POR", "MIL"],
            "listed_position": ["G", "G"],
            "rapm_possessions": [1200.0, 1100.0],
            "games": [60, 55],
        }
    ).to_parquet(panel_path, index=False)
    ratings = pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "player_id": [77, 77],
            "player_name": ["Test Player", "Test Player"],
            "rapm": [1.0, 2.0],
        }
    )

    catalog = _historical_ranking_catalog(ratings, panel_path=panel_path)
    evaluator = replace(
        _evaluator(),
        historical_rankings=catalog,
        player_rating_histories={
            77: [
                {
                    "season": "2023-24",
                    "rating": 1.0,
                    "age": 25.0,
                    "team": "POR",
                },
                {
                    "season": "2024-25",
                    "rating": 2.0,
                    "age": 26.0,
                    "team": "MIL",
                },
            ]
        },
    )
    client = TestClient(create_app(evaluator))

    response = client.get("/api/rankings", params={"season": "2023-24"})

    assert response.status_code == 200
    assert response.json()["available_seasons"] == ["2024-25", "2023-24"]
    assert response.json()["players"] == [
        {
            "rank": 1,
            "season": "2023-24",
            "player_id": 77,
            "player_name": "Test Player",
            "team": "POR",
            "position": "G",
            "rapm": 1.0,
            "possessions": 1200.0,
            "games": 60,
        }
    ]

    profile = client.get("/api/players/77")
    assert profile.status_code == 200
    assert profile.json()["rating_season"] == "2024-25"
    assert profile.json()["team"] == "MIL"
    assert profile.json()["profile_source"] == "career_history"
    assert profile.json()["three_pm_per_100"] is None


def test_headshot_endpoint_uses_cached_same_origin_image(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_headshot_png", lambda player_id: b"test-png")
    client = TestClient(create_app(_evaluator()))

    response = client.get("/api/headshots/201939.png")

    assert response.status_code == 200
    assert response.content == b"test-png"
    assert response.headers["content-type"] == "image/png"


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
