"""Contract tests for the NBA GESTALT local API."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import nba_lineup_model.web_api.app as web_app
import nba_lineup_model.web_api.inference as web_inference
import nba_lineup_model.web_api.preseason_rankings_cache as preseason_cache
from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    lineup_side_context_features,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_bounded_hierarchical_matchup_contextual_model,
    fit_linear_ridge_matchup_contextual_model,
    fit_matchup_contextual_model,
)
from nba_lineup_model.web_api.app import create_app
from nba_lineup_model.web_api.inference import (
    LineupEvaluator,
    SeasonLineupState,
    _historical_ranking_catalog,
    _player_latest_teams_by_season,
    _player_league_leader_histories,
    _player_rating_histories,
    _player_team_splits_by_season,
    _preseason_ranking_catalog,
    _warm_response_cache,
    build_player_team_splits,
)


def _evaluator(*, bounded: bool = False, compiled_linear: bool = False) -> LineupEvaluator:
    player_ids = list(range(1, 11))
    profile_offset = player_ids if bounded or compiled_linear else [0] * len(player_ids)
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
            "offensive_rebound_pct": [4.0 + value / 10 for value in profile_offset],
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
        draft_year=2014,
        draft_round=1,
        draft_number=41,
        is_undrafted=False,
        draft_class_year=2014,
        age=26.0,
        rating_history=[
            [
                {
                    "season": "2023-24",
                    "rating": float(player_id) / 20,
                    "age": 24.0,
                    "team_id": 1610612757,
                    "team": "TST",
                },
                {
                    "season": "2024-25",
                    "rating": float(player_id) / 15,
                    "age": 25.0,
                    "team_id": 1610612757,
                    "team": "TST",
                },
                {
                    "season": "2025-26",
                    "rating": float(player_id) / 10,
                    "age": 26.0,
                    "team_id": 1610612757,
                    "team": "TST",
                },
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
    feature_set = (
        CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY if compiled_linear else "v1"
    )
    fit = (
        fit_linear_ridge_matchup_contextual_model
        if compiled_linear
        else fit_bounded_hierarchical_matchup_contextual_model
        if bounded
        else fit_matchup_contextual_model
    )
    model = fit(
        lineup_side_context_features(home_lineups, profiles, feature_set=feature_set),
        lineup_side_context_features(away_lineups, profiles, feature_set=feature_set),
        np.arange(5, dtype=float),
        np.ones(5, dtype=float),
        alpha=1.0,
        feature_set=feature_set,
    )
    return LineupEvaluator(
        season="2025-26",
        run_id="test-run",
        coefficients=players.loc[:, ["player_id", "rapm"]],
        profiles=profiles,
        players=players,
        context_model=model,
        response_cache={} if compiled_linear else _warm_response_cache(model),
    )


def test_search_and_matchup_endpoints() -> None:
    client = TestClient(create_app(_evaluator()))

    search = client.get("/api/players", params={"q": "Jokic"})
    assert search.status_code == 200
    assert search.json()["players"][0]["player_name"] == "Nikola Jokić"
    assert len(search.json()["players"][0]["rating_history"]) == 3
    assert search.json()["players"][0]["rookie_season"] == "2023-24"
    assert search.json()["players"][0]["age"] == 26.0

    lineup_players = client.get(
        "/api/players/by-id",
        params=[("season", "2025-26"), *(("player_id", player_id) for player_id in range(1, 6))],
    )
    assert lineup_players.status_code == 200
    assert lineup_players.json()["season"] == "2025-26"
    assert [player["player_id"] for player in lineup_players.json()["players"]] == [1, 2, 3, 4, 5]

    profile = client.get("/api/players/1")
    assert profile.status_code == 200
    assert profile.json()["three_pa_per_100"] == 5.0
    assert profile.json()["three_pm_per_100"] == 1.0
    assert profile.json()["assists_per_100"] == 3.0
    assert profile.json()["turnovers_per_100"] == 1.0
    assert profile.json()["usage_per_100"] == 15.0
    assert profile.json()["steals_per_100"] == 0.5
    assert profile.json()["blocks_per_100"] == 0.25
    assert profile.json()["offensive_rebound_pct"] == 4.0
    assert profile.json()["draft_year"] == 2014
    assert profile.json()["draft_round"] == 1
    assert profile.json()["draft_number"] == 41
    assert profile.json()["is_undrafted"] is False
    assert profile.json()["draft_class_year"] == 2014

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


def test_compiled_linear_matchup_returns_nonadditive_side_scores() -> None:
    client = TestClient(create_app(_evaluator(compiled_linear=True)))

    response = client.post(
        "/api/matchups",
        json={
            "unit_player_ids": [1, 2, 3, 4, 5],
            "opponent_player_ids": [6, 7, 8, 9, 10],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_form"] == "compiled_linear_x3"
    assert np.isclose(payload["additive_margin"], -2.5)
    assert np.isclose(
        payload["unit_composition_rating"] - payload["opponent_composition_rating"],
        payload["contextual_adjustment"],
    )


def test_matchup_endpoint_accepts_season_scoped_units_and_neutral_environment() -> None:
    evaluator = _evaluator()
    historical_coefficients = pd.concat(
        [
            evaluator.coefficients.assign(season="2024-25"),
            evaluator.coefficients.assign(season="2025-26"),
        ],
        ignore_index=True,
    )
    seasonal_ratings = historical_coefficients.assign(
        player_name=lambda frame: frame["player_id"].map(
            dict(zip(evaluator.players["player_id"], evaluator.players["player_name"], strict=True))
        ),
        age=26.0,
    )
    seasonal_ratings = pd.concat(
        [seasonal_ratings.loc[seasonal_ratings["season"].eq("2024-25")], seasonal_ratings],
        ignore_index=True,
    )
    evaluator = replace(
        evaluator,
        historical_coefficients=historical_coefficients,
        seasonal_ratings=seasonal_ratings,
        season_context_models={
            "2024-25": evaluator.context_model,
            "2025-26": evaluator.context_model,
        },
    )
    evaluator.season_states["2024-25"] = SeasonLineupState(
        "2024-25", evaluator.coefficients, evaluator.profiles, evaluator.players
    )
    client = TestClient(create_app(evaluator))

    search = client.get("/api/players", params={"q": "Jokic", "season": "2024-25"})
    assert search.status_code == 200
    assert search.json()["season"] == "2024-25"

    response = client.post(
        "/api/matchups",
        json={
            "unit_player_ids": [1, 2, 3, 4, 5],
            "opponent_player_ids": [6, 7, 8, 9, 10],
            "unit_season": "2024-25",
            "opponent_season": "2025-26",
            "environment": "neutral",
            "include_response_curves": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["unit_season"] == "2024-25"
    assert payload["opponent_season"] == "2025-26"
    assert payload["environment"] == "neutral"
    assert payload["environment_seasons"] == ["2024-25", "2025-26"]


def test_historical_lab_state_uses_published_exposure_cache(monkeypatch) -> None:
    evaluator = _evaluator(compiled_linear=True)
    historical_coefficients = pd.concat(
        [
            evaluator.coefficients.assign(season="2024-25"),
            evaluator.coefficients.assign(season="2025-26"),
        ],
        ignore_index=True,
    )
    cached_cohort = pd.DataFrame(
        {
            "season": ["2024-25"] * 10,
            "player_id": list(range(1, 11)),
            "exposure_share": [0.2] * 10,
        }
    )
    evaluator = replace(
        evaluator,
        historical_coefficients=historical_coefficients,
        seasonal_ratings=historical_coefficients.assign(age=26.0, is_rookie=False),
        season_context_models={
            "2024-25": evaluator.context_model,
            "2025-26": evaluator.context_model,
        },
        exposure_cohort=cached_cohort,
        historical_profiles=evaluator.profiles.assign(season="2024-25"),
    )
    monkeypatch.setattr(
        web_inference,
        "prepare_player_exposure_cohort",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw stints requested")),
    )
    monkeypatch.setattr(
        web_inference,
        "build_contextual_player_profiles",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("profiles rebuilt")),
    )
    monkeypatch.setattr(
        web_inference,
        "_player_catalog",
        lambda *args, **kwargs: evaluator.players.copy(),
    )

    client = TestClient(create_app(evaluator))
    response = client.get("/api/default-opponent", params={"season": "2024-25"})

    assert response.status_code == 200, response.text
    assert len(response.json()["players"]) == 5


def test_lineup_rankings_endpoint_filters_by_possessions_and_players() -> None:
    evaluator = replace(
        _evaluator(),
        observed_lineups=pd.DataFrame(
            {
                "team_id": [1, 1],
                "team": ["TST", "TST"],
                "lineup_key": ["1|2|3|4|5", "1|6|7|8|9"],
                "player_ids": [[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]],
                "player_names": [
                    ["Nikola Jokić", "Player 2", "Player 3", "Player 4", "Player 5"],
                    ["Nikola Jokić", "Player 6", "Player 7", "Player 8", "Player 9"],
                ],
                "lineup_label": ["Nikola Jokić, Player 2", "Nikola Jokić, Player 6"],
                "possessions": [650.0, 300.0],
                "games": [24, 12],
                "player_rating": [1.5, 2.0],
                "player_edge": [0.3, 0.1],
                "composition_rating": [0.4, 0.2],
                "composition_edge": [0.5, -0.1],
                "matchup_bonus": [0.2, 0.3],
                "context_edge": [0.7, 0.2],
                "gestalt_score": [1.0, 0.3],
                "actual_net_rating": [4.2, -1.1],
            }
        ),
    )
    client = TestClient(create_app(evaluator))

    response = client.get("/api/lineups", params=[("minimum_possessions", 500), ("player_id", 1)])

    assert response.status_code == 200
    payload = response.json()
    assert payload["season"] == "2025-26"
    assert len(payload["lineups"]) == 1
    assert payload["lineups"][0]["rank"] == 1
    assert payload["lineups"][0]["player_rating"] == 1.5
    assert payload["lineups"][0]["context_edge"] == 0.7


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
            "additive_profile_adjustment": [0.25, -0.5],
            "age": [25.0, 26.0],
        }
    )

    splits_frame = pd.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2024-25"],
            "player_id": [77, 77, 77],
            "team_id": [1610612757, 1610612745, 1610612749],
            "team": ["POR", "HOU", "MIL"],
            "possessions": [900.0, 300.0, 1100.0],
            "games": [45, 15, 55],
            "last_game_time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z", "2025-04-01T00:00:00Z"]
            ),
            "is_primary_team": [True, False, True],
            "is_latest_team": [False, True, True],
        }
    )
    team_splits = _player_team_splits_by_season(splits_frame)
    latest_teams = _player_latest_teams_by_season(splits_frame)
    history = _player_rating_histories(
        ratings,
        panel_path=panel_path,
        player_team_splits=team_splits,
        player_latest_teams=latest_teams,
    )

    assert [point["team"] for point in history[77]] == ["HOU", "MIL"]
    assert [point["team_id"] for point in history[77]] == [1610612745, 1610612749]
    assert [point["possessions"] for point in history[77]] == [1200.0, 1100.0]
    assert [point["games"] for point in history[77]] == [60, 55]
    assert [point["games_started"] for point in history[77]] == [42, 39]
    assert [point["nail_rank"] for point in history[77]] == [1, 1]
    assert [point["additive_profile_adjustment"] for point in history[77]] == [0.25, -0.5]
    assert history[77][0]["team_splits"] == [
        {
            "team_id": 1610612757,
            "team": "POR",
            "possessions": 900.0,
            "games": 45,
            "is_primary_team": True,
            "is_latest_team": False,
        },
        {
            "team_id": 1610612745,
            "team": "HOU",
            "possessions": 300.0,
            "games": 15,
            "is_primary_team": False,
            "is_latest_team": True,
        },
    ]


def test_player_team_splits_distinguish_primary_from_latest_team(monkeypatch) -> None:
    stints = pd.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "game_id": ["early", "late"],
            "game_time_utc": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z"]
            ),
            "home_team_id": [1, 3],
            "home_team_tricode": ["AAA", "CCC"],
            "home_player_ids": [[77, 2, 3, 4, 5], [77, 6, 7, 8, 9]],
            "away_team_id": [2, 2],
            "away_team_tricode": ["BBB", "BBB"],
            "away_player_ids": [[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]],
            "possessions": [100.0, 10.0],
        }
    )
    monkeypatch.setattr(web_inference, "read_rapm_stints", lambda *args, **kwargs: stints)

    splits = build_player_team_splits(
        pd.DataFrame({"season": ["2023-24"], "player_id": [77]})
    )
    player = splits.loc[splits["player_id"].eq(77)].set_index("team")

    assert bool(player.loc["AAA", "is_primary_team"])
    assert not bool(player.loc["AAA", "is_latest_team"])
    assert not bool(player.loc["CCC", "is_primary_team"])
    assert bool(player.loc["CCC", "is_latest_team"])


def test_league_leader_history_retains_missed_player_seasons() -> None:
    ratings = pd.DataFrame(
        {
            "season": [
                "2018-19",
                "2018-19",
                "2019-20",
                "2019-20",
                "2020-21",
                "2020-21",
            ],
            "player_id": [7, 9, 9, 10, 7, 10],
            "player_name": [
                "Test Player",
                "Season One Leader",
                "Missed Season Leader",
                "Runner Up",
                "Test Player",
                "Season Three Leader",
            ],
            "rapm": [3.0, 5.0, 6.0, 2.0, 4.0, 7.0],
        }
    )
    histories = {
        7: [
            {"season": "2018-19", "rating": 3.0},
            {"season": "2020-21", "rating": 4.0},
        ]
    }

    leaders = _player_league_leader_histories(ratings, histories)

    assert leaders[7] == [
        {
            "season": "2018-19",
            "rating": 5.0,
            "player_id": 9,
            "player_name": "Season One Leader",
        },
        {
            "season": "2019-20",
            "rating": 6.0,
            "player_id": 9,
            "player_name": "Missed Season Leader",
        },
        {
            "season": "2020-21",
            "rating": 7.0,
            "player_id": 10,
            "player_name": "Season Three Leader",
        },
    ]


def test_league_leader_history_retains_active_terminal_dnp_season() -> None:
    ratings = pd.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2024-25", "2024-25", "2025-26"],
            "player_id": [7, 9, 7, 10, 11],
            "player_name": [
                "Active Player",
                "Season One Leader",
                "Active Player",
                "Season Two Leader",
                "Terminal Leader",
            ],
            "rapm": [2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )
    histories = {
        7: [
            {"season": "2023-24", "rating": 2.0},
            {"season": "2024-25", "rating": 3.0},
        ]
    }

    leaders = _player_league_leader_histories(
        ratings, histories, active_through_years={7: 2025}
    )

    assert [row["season"] for row in leaders[7]] == ["2023-24", "2024-25", "2025-26"]


def test_historical_ranking_catalog_exposes_each_completed_fit_season(tmp_path) -> None:
    panel_path = tmp_path / "player_seasons.parquet"
    pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "player_id": [77, 77],
            "primary_team_tricode": ["POR", "MIL"],
            "listed_position": ["G", "G"],
            "draft_year": [2013, 2013],
            "draft_round": [2, 2],
            "draft_number": [45, 45],
            "is_undrafted": [False, False],
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
            "prior_rapm": [0.5, 1.5],
            "rapm_adjustment_from_prior": [0.5, 0.5],
            "additive_profile_adjustment": [0.0, 0.0],
        }
    )

    catalog = _historical_ranking_catalog(
        ratings,
        panel_path=panel_path,
        player_latest_teams={
            ("2023-24", 77): {"team_id": 1610612745, "team": "HOU"}
        },
    )
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
            "team": "HOU",
            "position": "G",
            "draft_year": 2013,
            "draft_round": 2,
            "draft_number": 45,
            "is_undrafted": False,
            "draft_class_year": 2013,
            "rapm": 1.0,
            "prior_rating": 0.5,
            "season_update": 0.5,
            "additive_profile_adjustment": 0.0,
            "observed_context_exposure": None,
            "possessions": 1200.0,
            "games": 60,
        }
    ]

    profile = client.get("/api/players/77")
    assert profile.status_code == 200
    assert profile.json()["rating_season"] == "2024-25"
    assert profile.json()["team"] == "MIL"
    assert profile.json()["profile_source"] == "career_history"
    assert profile.json()["three_pa_per_100"] is None
    assert profile.json()["three_pm_per_100"] is None
    assert profile.json()["assists_per_100"] is None
    assert profile.json()["turnovers_per_100"] is None
    assert profile.json()["usage_per_100"] is None
    assert profile.json()["steals_per_100"] is None
    assert profile.json()["blocks_per_100"] is None
    assert profile.json()["offensive_rebound_pct"] is None


def test_preseason_rankings_include_returners_and_cold_starts(tmp_path) -> None:
    roster_path = tmp_path / "roster.parquet"
    draft_path = tmp_path / "draft.parquet"
    pd.DataFrame(
        {
            "player_id": [1, 99, 100],
            "player_name": ["Nikola Jokić", "Draft Rookie", "Undrafted Rookie"],
            "team_abbreviation": ["TST", "RKS", "UDR"],
            "listed_position": ["C", "F", "G"],
            "age": [31.0, 19.0, 22.0],
            "experience": [11, 0, 0],
        }
    ).to_parquet(roster_path, index=False)
    pd.DataFrame(
        {
            "player_id": [99],
            "draft_round": [1],
            "draft_number": [4],
            "cold_start_rapm_prior": [-0.5],
            "replacement_rapm": [-3.9],
        }
    ).to_parquet(draft_path, index=False)
    completed = pd.DataFrame(
        {
            "season": ["2025-26"],
            "player_id": [1],
            "player_name": ["Nikola Jokić"],
            "team": ["OLD"],
            "position": ["C"],
            "draft_year": [2014],
            "draft_round": [2],
            "draft_number": [41],
            "is_undrafted": [False],
            "draft_class_year": [2014],
            "rapm": [4.2],
            "prior_rating": [1.0],
            "season_update": [2.0],
            "additive_profile_adjustment": [1.2],
            "observed_context_exposure": [0.0],
            "possessions": [2000.0],
            "games": [70],
            "rookie_season": ["2015-16"],
        }
    )
    preview = _preseason_ranking_catalog(
        completed,
        roster_path=roster_path,
        draft_rankings_path=draft_path,
        completed_season="2025-26",
        preview_season="2026-27",
    )
    by_player = preview.set_index("player_id")
    assert by_player.loc[1, "rapm"] == 4.2
    assert by_player.loc[1, "team"] == "TST"
    assert pd.isna(by_player.loc[1, "prior_rating"])
    assert by_player.loc[99, "rapm"] == -0.5
    assert by_player.loc[99, "draft_number"] == 4
    assert by_player.loc[99, "profile_source"] == "draft_cold_start_prior"
    assert by_player.loc[100, "rapm"] == -3.9
    assert by_player.loc[100, "is_undrafted"] is True
    assert by_player.loc[100, "draft_class_year"] == 2026

    evaluator = replace(_evaluator(), preseason_rankings=preview)
    client = TestClient(create_app(evaluator))
    rankings = client.get("/api/rankings", params={"season": "2026-27"})
    assert rankings.status_code == 200
    assert rankings.json()["available_seasons"] == ["2026-27", "2025-26"]
    assert rankings.json()["players"][0]["player_id"] == 1
    rookie = client.get("/api/players/99")
    assert rookie.status_code == 200
    assert rookie.json()["rating_season"] == "2026-27"
    assert rookie.json()["rating_history"] == []


def test_preseason_draft_history_normalizes_string_player_ids() -> None:
    draft = pd.DataFrame(
        {
            "player_id": ["1642865"],
            "player_name": ["Yaxel Lendeborg"],
            "draft_number": [11],
        }
    )

    normalized = preseason_cache._normalize_draft_history_ids(draft)

    assert normalized["player_id"].dtype.kind in {"i", "u"}
    assert normalized.loc[0, "player_id"] == 1642865


def test_imputed_profile_adjustment_is_folded_into_historical_prior() -> None:
    ratings = pd.DataFrame(
        {
            "player_id": [1, 2],
            "prior_rapm": [-1.0, 0.5],
            "additive_profile_adjustment": [-0.4, 0.2],
            "rapm": [-0.2, 1.0],
        }
    )
    profiles = pd.DataFrame(
        {"player_id": [1, 2], "profile_imputed": [1, 0]}
    )

    folded = web_inference._fold_imputed_profile_into_prior(ratings, profiles)

    assert folded.loc[0, "prior_rapm"] == -1.4
    assert pd.isna(folded.loc[0, "additive_profile_adjustment"])
    assert folded.loc[1, "prior_rapm"] == 0.5
    assert folded.loc[1, "additive_profile_adjustment"] == 0.2


def test_additive_profile_breakdown_uses_weighted_forecast_reference() -> None:
    evaluator = _evaluator(compiled_linear=True)
    weights = pd.DataFrame(
        {"player_id": evaluator.profiles["player_id"], "possessions": [1.0, *[0.0] * 9]}
    )

    breakdown = web_inference.compiled_linear_x3_additive_profile_breakdown(
        evaluator.profiles,
        evaluator.context_model,
        weights,
    )

    player_one = breakdown.loc[breakdown["player_id"].eq(1)]
    assert len(player_one) == 8
    assert (player_one["reference_value"] == player_one["player_value"]).all()
    assert np.isclose(player_one["contribution"].sum(), 0.0)
    player_two = breakdown.loc[breakdown["player_id"].eq(2)]
    assert not np.isclose(player_two["contribution"].sum(), 0.0)


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
