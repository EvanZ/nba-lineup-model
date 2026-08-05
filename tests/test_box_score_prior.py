from __future__ import annotations

import math

import pandas as pd
import pytest

from nba_lineup_model.modeling.box_score_prior import build_box_score_prior_features


def test_box_score_prior_features_use_only_the_immediately_prior_season() -> None:
    features, summaries, references = build_box_score_prior_features(_player_seasons())

    target = features.loc[
        features["target_season"].eq("2021-22") & features["player_id"].eq(1)
    ].iloc[0]
    cold_start = features.loc[
        features["target_season"].eq("2021-22") & features["player_id"].eq(2)
    ].iloc[0]

    assert target.prior_source_season == "2020-21"
    assert target.prior_rapm == pytest.approx(2.0)
    assert target.prior_fga_per_100_on_court_possessions == pytest.approx(20.0)
    assert target.prior_exposure_cohort == "established"
    assert not cold_start.has_prior_season
    assert cold_start.prior_exposure_cohort == "no_prior"
    assert pd.isna(cold_start.prior_rapm)
    assert summaries["player_count"].sum() == len(features)
    assert references["target_season"].tolist() == ["2020-21", "2021-22"]


def test_stabilized_shooting_uses_source_season_league_reference() -> None:
    features, _, references = build_box_score_prior_features(_player_seasons())

    player = features.loc[
        features["target_season"].eq("2021-22") & features["player_id"].eq(1)
    ].iloc[0]
    league = references.loc[references["target_season"].eq("2021-22")].iloc[0]
    expected = (20.0 + 150.0 * league.league_three_point_percentage) / (40.0 + 150.0)

    assert player.prior_stabilized_three_point_percentage == pytest.approx(expected)
    assert math.isfinite(player.prior_stabilized_effective_field_goal_percentage)


def _player_seasons() -> pd.DataFrame:
    rows = [
        _row("2019-20", 1, 1.0, 1_000.0, fga=100, fgm=50, three_pa=40, three_pm=20),
        _row("2019-20", 3, 0.0, 1_000.0, fga=100, fgm=40, three_pa=20, three_pm=5),
        _row("2020-21", 1, 2.0, 2_000.0, fga=400, fgm=200, three_pa=40, three_pm=20),
        _row("2020-21", 3, 0.0, 2_000.0, fga=300, fgm=120, three_pa=100, three_pm=30),
        _row("2021-22", 1, 9.0, 2_100.0, fga=999, fgm=999, three_pa=999, three_pm=999),
        _row("2021-22", 2, -1.0, 800.0, fga=50, fgm=25, three_pa=10, three_pm=4),
    ]
    return pd.DataFrame(rows)


def _row(
    season: str,
    player_id: int,
    rapm: float,
    possessions: float,
    *,
    fga: int,
    fgm: int,
    three_pa: int,
    three_pm: int,
) -> dict[str, object]:
    return {
        "season": season,
        "player_id": player_id,
        "player_name": f"Player {player_id}",
        "rapm": rapm,
        "rapm_possessions": possessions,
        "boxscore_features_available": True,
        "field_goals_attempted": fga,
        "field_goals_made": fgm,
        "three_pointers_attempted": three_pa,
        "three_pointers_made": three_pm,
        "free_throws_attempted": 50,
        "free_throws_made": 40,
        "assists": 100,
        "turnovers": 50,
        "rebounds_offensive": 40,
        "rebounds_defensive": 80,
        "steals": 20,
        "blocks": 10,
        "fouls_personal": 30,
        "age": 25.0,
        "nba_experience_years": 3,
        "is_rookie": False,
        "years_since_draft": 4,
        "draft_year": 2016,
        "draft_round": 1,
        "draft_number": 10,
        "is_undrafted": False,
        "height_inches": 78.0,
        "weight_pounds": 220.0,
        "listed_position": "G",
    }
