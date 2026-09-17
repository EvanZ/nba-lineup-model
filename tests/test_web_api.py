"""Contract tests for the NBA GESTALT local API."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
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
    MeanRevertedScheduleControls,
    SeasonLineupState,
    _aggregate_observed_lineups,
    _build_team_gestalt_win_histories,
    _historical_ranking_catalog,
    _load_team_gestalt_win_history_cache,
    _observed_lineup_side_rows,
    _player_latest_teams_by_season,
    _player_league_leader_histories,
    _player_rating_histories,
    _player_team_splits_by_season,
    _preseason_ranking_catalog,
    _warm_response_cache,
    build_player_team_splits,
    build_published_player_ratings,
    team_win_history_path,
)


def test_team_gestalt_win_history_weights_player_ratings_by_team_exposure(monkeypatch) -> None:
    rankings = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "player_id": [1, 2, 3, 4],
            "rapm": [2.0, 0.0, 1.0, -1.0],
        }
    )
    splits = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "player_id": [1, 2, 3, 4],
            "team": ["TST", "TST", "OTH", "OTH"],
            "possessions": [300.0, 100.0, 200.0, 200.0],
        }
    )
    monkeypatch.setattr(
        web_inference,
        "_published_actual_regular_season_wins",
        lambda season: pd.DataFrame(
            {
                "season": [season, season],
                "team": ["TST", "OTH"],
                "games": [82, 82],
                "actual_wins": [50, 32],
            }
        ),
    )

    histories = _build_team_gestalt_win_histories(rankings, splits)

    assert histories["TST"] == [
        {
            "season": "2024-25",
            "games": 82,
            "actual_wins": 50,
            "gestalt_pywins": pytest.approx(59.569556),
            "gestalt_rating": pytest.approx(7.5),
        }
    ]
    assert histories["OTH"][0]["gestalt_rating"] == pytest.approx(0.0)


def test_team_gestalt_win_history_cache_round_trips_without_raw_schedules(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(web_inference, "DEFAULT_TEAM_WIN_HISTORY_CACHE_DIR", tmp_path)
    path = team_win_history_path("test-model", "test-run")
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "team": ["TST", "TST", "OTH"],
            "season": ["2023-24", "2024-25", "2024-25"],
            "games": [82, 82, 82],
            "actual_wins": [42, 50, 32],
            "gestalt_pywins": [43.2, 59.569556, 22.430444],
            "gestalt_rating": [0.1, 7.5, -7.5],
        }
    ).to_parquet(path, index=False)

    histories = _load_team_gestalt_win_history_cache(path)

    assert [row["season"] for row in histories["TST"]] == ["2023-24", "2024-25"]
    assert histories["TST"][1]["actual_wins"] == 50
    assert histories["OTH"][0]["gestalt_rating"] == pytest.approx(-7.5)


def test_win_projection_preview_cache_overrides_the_published_cache(
    tmp_path, monkeypatch
) -> None:
    preview_path = tmp_path / "preview.json"
    preview_path.write_text(
        json.dumps(
            {
                "minutes": {"model": "Local preview", "players": []},
                "win_projection": {"model": "Local preview wins", "teams": []},
            }
        )
    )
    monkeypatch.setenv("GESTALT_WIN_PROJECTION_CACHE_PATH", str(preview_path))

    response = TestClient(create_app(_evaluator())).get("/api/win-projections")

    assert response.status_code == 200
    assert response.json()["model"] == "Local preview"
    assert response.json()["win_projection"]["model"] == "Local preview wins"


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
        schedule_controls=MeanRevertedScheduleControls(
            home_court=0.0,
            back_to_back=0.0,
            source_season_count=1,
        ),
    )


def test_search_and_matchup_endpoints() -> None:
    client = TestClient(create_app(_evaluator()))

    search = client.get("/api/players", params={"q": "Jokic"})
    assert search.status_code == 200
    assert search.json()["players"][0]["player_name"] == "Nikola Jokić"
    assert len(search.json()["players"][0]["rating_history"]) == 3
    assert search.json()["players"][0]["rookie_season"] == "2023-24"
    assert search.json()["players"][0]["age"] == 26.0
    filtered_search = client.get("/api/players", params={"q": "Jokic", "team": "TST"})
    assert filtered_search.status_code == 200
    assert filtered_search.json()["players"][0]["team"] == "TST"
    comparison_search = client.get("/api/compare/players", params={"q": "Jokic"})
    assert comparison_search.status_code == 200
    assert comparison_search.json()["players"] == [
        {
            "player_id": 1,
            "player_name": "Nikola Jokić",
            "latest_season": "2025-26",
            "latest_team": "TST",
            "history_seasons": 3,
        }
    ]
    teams = client.get("/api/teams", params={"season": "2025-26"})
    assert teams.status_code == 200
    assert teams.json()["teams"] == ["TST"]

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

    partial_team_fill = client.get(
        "/api/default-opponent",
        params={"team": "TST", "count": 2, "exclude_player_id": 1},
    )
    assert partial_team_fill.status_code == 200
    partial_players = partial_team_fill.json()["players"]
    assert len(partial_players) == 2
    assert all(player["team"] == "TST" for player in partial_players)
    assert all(player["player_id"] != 1 for player in partial_players)

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


def test_global_search_and_team_season_endpoints_cover_historical_team_navigation() -> None:
    evaluator = _evaluator()
    players = evaluator.players.assign(team="DEN")
    observed_lineups = pd.DataFrame(
        {
            "team_id": [1610612743, 1610612743, 1610612743],
            "team": ["DEN", "DEN", "DEN"],
            "lineup_key": ["1|2|3|4|5", "1|2|3|4|6", "1|2|3|4|7"],
            "player_ids": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 7]],
            "player_names": [
                ["Nikola Jokić", "Player 2", "Player 3", "Player 4", "Player 5"],
                ["Nikola Jokić", "Player 2", "Player 3", "Player 4", "Player 6"],
                ["Nikola Jokić", "Player 2", "Player 3", "Player 4", "Player 7"],
            ],
            "lineup_label": [
                "Nikola Jokić, Player 2, Player 3, Player 4, Player 5",
                "Nikola Jokić, Player 2, Player 3, Player 4, Player 6",
                "Nikola Jokić, Player 2, Player 3, Player 4, Player 7",
            ],
            "possessions": [650.0, 700.0, 400.0],
            "games": [24, 26, 12],
            "player_rating": [1.5, 1.5, 1.5],
            "player_edge": [0.3, 0.3, 0.3],
            "offensive_edge": [0.8, 0.8, 0.8],
            "defensive_edge": [0.2, 0.2, 0.2],
            "composition_rating": [0.4, 0.4, 0.4],
            "composition_edge": [0.5, 0.5, 0.5],
            "matchup_bonus": [0.2, 0.2, 0.2],
            "context_edge": [0.7, 1.2, 1.5],
            "gestalt_score": [1.0, 1.0, 1.0],
            "actual_net_rating": [4.2, 8.1, 12.0],
            "actual_offensive_rating": [112.4, 117.1, 120.2],
            "actual_defensive_rating": [108.2, 109.0, 108.2],
        }
    )
    team_win_histories = {
        "DEN": [
            {
                "season": "2025-26",
                "games": 82,
                "actual_wins": 50,
                "gestalt_pywins": 54.2,
                "gestalt_rating": 2.1,
            }
        ]
    }
    client = TestClient(
        create_app(
            replace(
                evaluator,
                players=players,
                observed_lineups=observed_lineups,
                team_win_histories=team_win_histories,
            )
        )
    )

    player_search = client.get("/api/search", params={"q": "Jokic"})
    assert player_search.status_code == 200
    assert player_search.json()["players"][0]["player_name"] == "Nikola Jokić"

    team_search = client.get("/api/search", params={"q": "nuggets"})
    assert team_search.status_code == 200
    assert team_search.json()["teams"] == [
        {
            "team": "DEN",
            "display_name": "Denver Nuggets",
            "latest_season": "2025-26",
            "season_count": 1,
            "seasons": ["2025-26"],
        }
    ]

    team_page = client.get(
        "/api/teams/den",
        params={"season": "2025-26", "minimum_possessions": 500},
    )
    assert team_page.status_code == 200
    payload = team_page.json()
    assert payload["display_name"] == "Denver Nuggets"
    assert payload["available_seasons"] == ["2025-26"]
    assert len(payload["players"]) == 10
    assert payload["minimum_possessions"] == 500
    assert len(payload["lineups"]) == 2
    assert payload["lineups"][0]["rank"] == 1
    assert payload["lineups"][0]["team"] == "DEN"
    assert payload["lineups"][0]["actual_net_rating"] == 8.1
    assert payload["win_history"] == team_win_histories["DEN"]


def test_team_rotation_endpoint_returns_clustered_shared_floor_and_correlation_views(
    monkeypatch,
) -> None:
    stints = pd.DataFrame(
        {
            "game_id": ["game-1", "game-1", "game-2", "game-2"],
            "home_team_tricode": ["TST", "TST", "TST", "TST"],
            "away_team_tricode": ["OTH", "OTH", "OTH", "OTH"],
            "home_player_ids": [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 6],
                [1, 2, 3, 4, 6],
            ],
            "away_player_ids": [[6, 7, 8, 9, 10]] * 4,
            "possessions": [10.0, 5.0, 15.0, 10.0],
            "duration_seconds": [300.0, 120.0, 480.0, 180.0],
        }
    )
    monkeypatch.setattr(web_inference, "read_rapm_stints", lambda _: stints)

    response = TestClient(create_app(_evaluator())).get(
        "/api/teams/tst/rotation",
        params={"season": "2025-26", "max_players": 6},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["team"] == "TST"
    assert payload["season"] == "2025-26"
    assert payload["game_count"] == 2
    assert len(payload["players"]) == 6
    assert {player["player_id"] for player in payload["players"]} == {1, 2, 3, 4, 5, 6}
    assert all(player["player_name"] for player in payload["players"])

    positions = {player["player_id"]: index for index, player in enumerate(payload["players"])}
    first = positions[1]
    fifth = positions[5]
    sixth = positions[6]
    assert payload["shared_possessions"][first][fifth] == pytest.approx(15.0)
    assert payload["shared_possessions"][first][sixth] == pytest.approx(25.0)
    assert payload["shared_court_similarity"][first][fifth] == pytest.approx(0.375)
    assert payload["floor_correlation"][first][first] == pytest.approx(1.0)
    assert np.allclose(
        payload["shared_court_similarity"],
        np.asarray(payload["shared_court_similarity"]).T,
    )
    assert np.allclose(payload["floor_correlation"], np.asarray(payload["floor_correlation"]).T)
    assert len(payload["dendrogram"]["branches"]) == 5
    assert all(
        len(branch["x"]) == 4 and len(branch["height"]) == 4
        for branch in payload["dendrogram"]["branches"]
    )

    for ordering in ("aoe", "fpc", "hclust"):
        ordered_response = TestClient(create_app(_evaluator())).get(
            "/api/teams/tst/rotation",
            params={
                "season": "2025-26",
                "max_players": 6,
                "ordering": ordering,
                "order_metric": "floor",
            },
        )
        assert ordered_response.status_code == 200
        ordered_payload = ordered_response.json()
        assert ordered_payload["ordering"] == ordering
        assert ordered_payload["order_metric"] == "floor"
        assert {player["player_id"] for player in ordered_payload["players"]} == {1, 2, 3, 4, 5, 6}
        if ordering == "hclust":
            assert len(ordered_payload["dendrogram"]["branches"]) == 5
        else:
            assert ordered_payload["dendrogram"]["branches"] == []


def test_floor_correlation_uses_league_wide_possession_weighted_floor_time() -> None:
    stints = pd.DataFrame(
        {
            "home_player_ids": [[1], [1], [7], [10]],
            "away_player_ids": [[2], [8], [2], [9]],
            "possessions": [2.0, 5.0, 3.0, 7.0],
        }
    )

    correlation = web_inference._floor_time_correlation(stints, [1, 2])

    # The original chart's active cells hold stint possessions, not a literal
    # 1. The league-wide vectors are [2, 5, 0, 0] and [2, 0, 3, 0]. This guards
    # against restricting the matrix to team games or reverting to a binary or
    # game-minute correlation.
    expected = np.corrcoef([2.0, 5.0, 0.0, 0.0], [2.0, 0.0, 3.0, 0.0])[0, 1]
    assert correlation[0, 1] == pytest.approx(expected)
    assert np.allclose(correlation, correlation.T)
    assert np.allclose(np.diag(correlation), 1.0)


def test_team_rotation_does_not_require_a_historical_scoring_state(monkeypatch) -> None:
    stints = pd.DataFrame(
        {
            "game_id": ["game-1"],
            "home_team_tricode": ["TST"],
            "away_team_tricode": ["OTH"],
            "home_player_ids": [[1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10]],
            "possessions": [10.0],
            "duration_seconds": [300.0],
        }
    )
    evaluator = _evaluator()
    object.__setattr__(
        evaluator,
        "historical_rankings",
        evaluator.players.assign(season="2024-25"),
    )
    monkeypatch.setattr(web_inference, "read_rapm_stints", lambda _: stints)

    response = TestClient(create_app(evaluator)).get(
        "/api/teams/tst/rotation",
        params={"season": "2024-25"},
    )

    assert response.status_code == 200
    players_by_id = {
        player["player_id"]: player["player_name"] for player in response.json()["players"]
    }
    assert set(players_by_id) == {1, 2, 3, 4, 5}
    assert players_by_id[1] == "Nikola Jokić"


def test_player_profile_includes_rotation_history_when_published() -> None:
    rotation_history = {
        1: [
            {
                "season": "2025-26",
                "season_start_year": 2025,
                "player_id": 1,
                "player_name": "Nikola Jokić",
                "age": 26.0,
                "actual_available_games": 70,
                "known_roster_games": 82,
                "actual_availability_share": 70 / 82,
                "injury_or_illness_games": 8,
                "rest_games": 1,
                "actual_minutes_per_available_game": 31.0,
                "actual_total_minutes": 2170.0,
                "predicted_available_share": 0.82,
                "predicted_minutes_per_available_game": 30.0,
                "projected_total_minutes": None,
                "is_preseason_forecast": False,
            }
        ]
    }
    evaluator = replace(_evaluator(), player_rotation_histories=rotation_history)

    response = TestClient(create_app(evaluator)).get("/api/players/1")

    assert response.status_code == 200
    assert response.json()["rotation_history"] == rotation_history[1]


def test_descriptive_od_edges_reconstruct_the_scalar_player_edge() -> None:
    evaluator = _evaluator(compiled_linear=True)
    players = evaluator.players.copy()
    players["offense_rating"] = players["rapm"] * 0.6
    players["defense_rating"] = players["rapm"] * 0.4
    client = TestClient(create_app(replace(evaluator, players=players, season_states={})))

    response = client.post(
        "/api/matchups",
        json={
            "unit_player_ids": [1, 2, 3, 4, 5],
            "opponent_player_ids": [6, 7, 8, 9, 10],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["od_split_available"] is True
    assert np.isclose(
        payload["offensive_player_edge"] + payload["defensive_player_edge"],
        payload["additive_margin"],
    )
    assert np.isclose(
        payload["unit"]["offense_rating"] + payload["unit"]["defense_rating"],
        payload["unit"]["additive_rating"],
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
    usage = next(
        row
        for row in payload["composition_feature_contributions"]
        if row["id"] == "home_minus_away_usage_concentration"
    )
    detail = usage["detail"]
    assert detail["kind"] == "usage_concentration"
    assert len(detail["unit_top_players"]) == 2
    assert len(detail["opponent_top_players"]) == 2
    assert detail["unit_total"] > 0.0
    assert detail["standard_deviation"] > 0.0
    assert np.isclose(
        detail["standardized_difference"],
        detail["difference"] / detail["standard_deviation"],
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
        historical_realized_profiles=evaluator.profiles.assign(season="2024-25"),
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


def test_published_player_ratings_retains_initial_rapm_season_without_context_model() -> None:
    ratings = pd.DataFrame(
        {
            "season": ["1996-97"],
            "player_id": [23],
            "player_name": ["Michael Jordan"],
            "rapm": [5.4],
            "prior_rapm": [0.0],
            "rapm_adjustment_from_prior": [5.4],
        }
    )

    published = build_published_player_ratings(
        ratings,
        panel=pd.DataFrame(),
        models={},
    )

    assert published["season"].tolist() == ["1996-97"]
    assert published["rapm"].tolist() == [5.4]
    assert published["additive_profile_adjustment"].tolist() == [0.0]


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
                "offensive_edge": [0.8, 0.15],
                "defensive_edge": [0.2, 0.15],
                "composition_rating": [0.4, 0.2],
                "composition_edge": [0.5, -0.1],
                "matchup_bonus": [0.2, 0.3],
                "context_edge": [0.7, 0.2],
            "gestalt_score": [1.0, 0.3],
            "actual_net_rating": [4.2, -1.1],
            "actual_offensive_rating": [112.4, 106.3],
            "actual_defensive_rating": [108.2, 107.4],
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
    assert payload["lineups"][0]["offensive_edge"] == 0.8
    assert payload["lineups"][0]["defensive_edge"] == 0.2
    assert payload["lineups"][0]["actual_offensive_rating"] == 112.4
    assert payload["lineups"][0]["actual_defensive_rating"] == 108.2


def test_observed_lineup_od_edges_reconstruct_scalar_edge_after_aggregation() -> None:
    rows = _observed_lineup_side_rows(
        team_ids=np.asarray([1, 1]),
        teams=np.asarray(["TST", "TST"]),
        lineups=[(1, 2, 3, 4, 5), (1, 2, 3, 4, 5)],
        game_ids=np.asarray(["game-a", "game-b"]),
        possessions=np.asarray([100.0, 50.0]),
        player_rating=np.asarray([3.0, 3.0]),
        opponent_player_rating=np.asarray([0.0, 0.0]),
        offensive_edge=np.asarray([2.0, 1.0]),
        defensive_edge=np.asarray([1.5, 2.5]),
        composition_rating=np.asarray([0.5, 0.5]),
        opponent_composition_rating=np.asarray([0.0, 0.0]),
        matchup_bonus=np.asarray([0.0, 0.0]),
        actual_net_rating=np.asarray([2.0, -2.0]),
        actual_offensive_points=np.asarray([120.0, 40.0]),
        actual_defensive_points=np.asarray([100.0, 60.0]),
        actual_offensive_possessions=np.asarray([100.0, 40.0]),
        actual_defensive_possessions=np.asarray([100.0, 60.0]),
    )

    aggregated = _aggregate_observed_lineups(
        rows,
        {player_id: f"Player {player_id}" for player_id in range(1, 6)},
    )

    assert np.isclose(aggregated.loc[0, "offensive_edge"], 5.0 / 3.0)
    assert np.isclose(aggregated.loc[0, "defensive_edge"], 11.0 / 6.0)
    assert np.isclose(
        aggregated.loc[0, "offensive_edge"] + aggregated.loc[0, "defensive_edge"],
        aggregated.loc[0, "gestalt_score"],
    )
    assert np.isclose(aggregated.loc[0, "actual_offensive_rating"], 16000.0 / 140.0)
    assert np.isclose(aggregated.loc[0, "actual_defensive_rating"], 100.0)


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
            "offense_rating": None,
            "defense_rating": None,
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

    evaluator = _evaluator()
    evaluator = replace(
        evaluator,
        historical_rankings=evaluator.players.assign(season="2025-26"),
        preseason_rankings=preview,
        observed_lineups=pd.DataFrame({"placeholder": [1]}),
    )
    client = TestClient(create_app(evaluator))
    rankings = client.get("/api/rankings", params={"season": "2026-27"})
    assert rankings.status_code == 200
    assert rankings.json()["available_seasons"] == ["2026-27", "2025-26"]
    assert rankings.json()["players"][0]["player_id"] == 1

    team_search = client.get("/api/search", params={"q": "TST"})
    assert team_search.status_code == 200
    assert team_search.json()["teams"][0]["latest_season"] == "2025-26"
    preseason_team_page = client.get("/api/teams/TST", params={"season": "2026-27"})
    assert preseason_team_page.status_code == 404

    rookie = client.get("/api/players/99")
    assert rookie.status_code == 200
    assert rookie.json()["rating_season"] == "2026-27"
    assert rookie.json()["rating_history"] == []


def test_preseason_projection_state_is_available_to_the_matchup_lab() -> None:
    evaluator = _evaluator(compiled_linear=True)
    ranking_columns = [
        "player_id", "player_name", "team", "position", "draft_year", "draft_round",
        "draft_number", "is_undrafted", "draft_class_year", "age", "rapm", "possessions",
        "games", "profile_source", "rookie_season",
    ]
    preseason_rankings = evaluator.players.loc[:, ranking_columns].copy()
    preseason_rankings.insert(0, "season", "2026-27")
    preseason_profiles = evaluator.profiles.copy()
    preseason_profiles.insert(0, "season", "2026-27")
    evaluator = replace(
        evaluator,
        preseason_rankings=preseason_rankings,
        preseason_profiles=preseason_profiles,
    )
    client = TestClient(create_app(evaluator))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["lab_seasons"][0] == "2026-27"

    players = client.get("/api/players", params={"q": "Jokic", "season": "2026-27"})
    assert players.status_code == 200
    assert players.json()["players"][0]["team"] == "TST"

    response = client.post(
        "/api/matchups",
        json={
            "unit_player_ids": [1, 2, 3, 4, 5],
            "opponent_player_ids": [6, 7, 8, 9, 10],
            "unit_season": "2026-27",
            "opponent_season": "2026-27",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["retrospective"] is False
    assert payload["context_source_seasons"] == ["2025-26"]
    assert payload["unit_season"] == "2026-27"
    assert payload["unit"]["players"][0]["rapm"] == pytest.approx(0.1)


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
