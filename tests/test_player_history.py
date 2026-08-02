from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.player_history import (
    aggregate_box_score_features,
    player_season_frame,
    player_transition_frame,
)


def _boxscore_row(
    *,
    game_id: str,
    player_id: int = 1,
    played: str = "1",
    minutes: str = "PT30M00.00S",
    team_id: int = 10,
    team_tricode: str = "AAA",
    starter: str = "1",
    points: int = 20,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "personId": player_id,
        "name": "Player One",
        "team_id": team_id,
        "team_tricode": team_tricode,
        "played": played,
        "starter": starter,
        "statistics_minutes": minutes,
        "statistics_assists": 5,
        "statistics_blocks": 1,
        "statistics_fieldGoalsAttempted": 10,
        "statistics_fieldGoalsMade": 6,
        "statistics_freeThrowsAttempted": 8,
        "statistics_freeThrowsMade": 6,
        "statistics_foulsOffensive": 1,
        "statistics_foulsDrawn": 4,
        "statistics_foulsPersonal": 2,
        "statistics_plusMinusPoints": 3,
        "statistics_points": points,
        "statistics_reboundsDefensive": 5,
        "statistics_reboundsOffensive": 2,
        "statistics_reboundsTotal": 7,
        "statistics_steals": 2,
        "statistics_threePointersAttempted": 4,
        "statistics_threePointersMade": 2,
        "statistics_turnovers": 3,
        "statistics_twoPointersAttempted": 6,
        "statistics_twoPointersMade": 4,
    }


def test_aggregate_box_score_features_ignores_dnp_and_derives_rates():
    rows = [
        _boxscore_row(game_id="001"),
        _boxscore_row(
            game_id="002",
            minutes="PT15M00.00S",
            team_id=20,
            team_tricode="BBB",
            starter="0",
            points=10,
        ),
        _boxscore_row(
            game_id="003",
            player_id=2,
            played="0",
            minutes="PT00M00.00S",
            points=0,
        ),
        _boxscore_row(
            game_id="004",
            player_id=2,
            played="1",
            minutes="PT00M00.00S",
            points=0,
        ),
    ]

    features = aggregate_box_score_features(pd.DataFrame(rows))

    assert features["player_id"].tolist() == [1]
    player = features.iloc[0]
    assert player["games"] == 2
    assert player["games_started"] == 1
    assert player["minutes"] == pytest.approx(45.0)
    assert player["points"] == pytest.approx(30.0)
    assert player["points_per_36"] == pytest.approx(24.0)
    assert player["field_goal_percentage"] == pytest.approx(0.6)
    assert player["box_primary_team_id"] == 10
    assert player["box_primary_team_tricode"] == "AAA"


def test_player_season_frame_combines_outcomes_and_bio_context():
    boxscores = aggregate_box_score_features(pd.DataFrame([_boxscore_row(game_id="001")]))
    rankings = pd.DataFrame(
        [
            {
                "player_id": 1,
                "rapm": 2.5,
                "raw_on_court_net_rating": 4.0,
                "stint_count": 12,
                "possessions": 300.0,
                "seconds": 900.0,
                "exposure_eligible": True,
                "primary_team_id": 10,
                "primary_team_tricode": "AAA",
            }
        ]
    )
    bios = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Player One",
                "age": 25.0,
                "listed_position": "G",
                "height_inches": 75,
                "weight_pounds": 195,
                "college": "Example",
                "country": "USA",
                "draft_year": 2017,
                "draft_round": 1,
                "draft_number": 12,
                "is_undrafted": False,
            }
        ]
    )
    catalog = pd.DataFrame([{"player_id": 1, "from_year": 2017, "to_year": 2025}])

    panel = player_season_frame(
        "2020-21",
        boxscores,
        rankings,
        bios,
        catalog,
        rapm_run_id="rapm-test",
    )

    player = panel.iloc[0]
    assert player["season_start_year"] == 2020
    assert player["rapm_run_id"] == "rapm-test"
    assert player["rapm_possessions"] == pytest.approx(300.0)
    assert player["nba_experience_years"] == 3
    assert player["years_since_draft"] == pytest.approx(3.0)
    assert bool(player["is_rookie"]) is False
    assert bool(player["boxscore_features_available"]) is True


def test_player_season_frame_preserves_rapm_player_without_boxscore_features():
    rankings = pd.DataFrame(
        [
            {
                "player_id": 1,
                "rapm": 2.5,
                "raw_on_court_net_rating": 4.0,
                "stint_count": 12,
                "possessions": 300.0,
                "seconds": 900.0,
                "exposure_eligible": True,
                "primary_team_id": 10,
                "primary_team_tricode": "AAA",
            }
        ]
    )
    bios = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Player One",
                "age": 25.0,
                "listed_position": "G",
                "height_inches": 75,
                "weight_pounds": 195,
                "college": "Example",
                "country": "USA",
                "draft_year": 2017,
                "draft_round": 1,
                "draft_number": 12,
                "is_undrafted": False,
            }
        ]
    )
    catalog = pd.DataFrame([{"player_id": 1, "from_year": 2017, "to_year": 2025}])
    empty_boxscores = pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "games",
            "box_primary_team_id",
            "box_primary_team_tricode",
        ]
    )

    panel = player_season_frame(
        "2020-21",
        empty_boxscores,
        rankings,
        bios,
        catalog,
        rapm_run_id="rapm-test",
    )

    assert len(panel) == 1
    assert bool(panel.loc[0, "boxscore_features_available"]) is False
    assert pd.isna(panel.loc[0, "games"])


def test_player_transition_frame_lags_performance_and_preserves_cold_starts():
    rows: list[dict[str, object]] = []
    for season, start_year, player_id, rapm, points_per_36 in (
        ("2019-20", 2019, 1, 1.5, 18.0),
        ("2020-21", 2020, 1, 2.5, 20.0),
        ("2020-21", 2020, 2, -0.5, 8.0),
    ):
        row: dict[str, object] = {
            "season": season,
            "season_start_year": start_year,
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "age": 24.0 + player_id,
            "nba_experience_years": 2,
            "listed_position": "G",
            "height_inches": 75,
            "weight_pounds": 195,
            "draft_year": 2017,
            "draft_round": 1,
            "draft_number": 12,
            "is_undrafted": False,
            "is_rookie": False,
            "rapm": rapm,
            "rapm_possessions": 300.0,
            "rapm_seconds": 900.0,
            "rapm_exposure_eligible": True,
            "raw_on_court_net_rating": rapm + 1.0,
            "games": 20,
            "games_started": 10,
            "minutes": 500.0,
            "points_per_36": points_per_36,
            "field_goals_attempted_per_36": 12.0,
            "three_pointers_attempted_per_36": 5.0,
            "free_throws_attempted_per_36": 4.0,
            "assists_per_36": 6.0,
            "turnovers_per_36": 2.0,
            "rebounds_offensive_per_36": 1.0,
            "rebounds_defensive_per_36": 4.0,
            "steals_per_36": 1.0,
            "blocks_per_36": 0.5,
            "fouls_personal_per_36": 2.5,
            "plus_minus_per_36": 3.0,
            "field_goal_percentage": 0.5,
            "three_point_percentage": 0.35,
            "free_throw_percentage": 0.8,
            "effective_field_goal_percentage": 0.55,
            "true_shooting_percentage": 0.58,
        }
        rows.append(row)

    transitions = player_transition_frame(pd.DataFrame(rows))

    assert transitions["player_id"].tolist() == [1, 2]
    returning = transitions.loc[transitions["player_id"].eq(1)].iloc[0]
    rookie = transitions.loc[transitions["player_id"].eq(2)].iloc[0]
    assert returning["target_rapm"] == pytest.approx(2.5)
    assert returning["prior_rapm"] == pytest.approx(1.5)
    assert returning["prior_points_per_36"] == pytest.approx(18.0)
    assert bool(returning["has_prior_season"]) is True
    assert bool(rookie["has_prior_season"]) is False
    assert pd.isna(rookie["prior_points_per_36"])
    assert "points_per_36" not in transitions
