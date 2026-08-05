from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.offense_defense_rapm import (
    _side_player_rankings,
    build_side_design,
    score_frozen_offense_defense_possessions,
)


def test_side_design_encodes_offense_against_opponent_defense():
    design = build_side_design(_stints())

    assert design.player_ids == tuple(range(1, 11))
    assert design.features.shape == (2, 21)
    assert design.target.tolist() == pytest.approx([125.0, 800.0 / 7.0])
    assert design.weights.tolist() == [8.0, 7.0]
    assert design.home_offense.tolist() == [True, False]

    home_row = design.features.getrow(0).toarray().ravel()
    away_row = design.features.getrow(1).toarray().ravel()
    assert home_row[0] == 1.0
    assert home_row[10 + 5] == -1.0
    assert home_row[-1] == 1.0
    assert away_row[5] == 1.0
    assert away_row[10] == -1.0
    assert away_row[-1] == -1.0


def test_frozen_scoring_uses_distinct_offense_and_defense_coefficients():
    coefficients = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "offense_rapm": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "defense_rapm": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    possessions = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "possession_id": ["g1:1", "g1:2"],
            "home_offense_sign": [1.0, -1.0],
            "offense_player_ids": [list(range(1, 6)), list(range(6, 11))],
            "defense_player_ids": [list(range(6, 11)), list(range(1, 6))],
            "target_offense_margin": [2.0, 1.0],
        }
    )

    scored = score_frozen_offense_defense_possessions(
        possessions,
        coefficients,
        league_offensive_rating=110.0,
        home_offense_shift=1.0,
        cohort="regular_season",
    )

    assert scored["prediction_offense_margin"].tolist() == pytest.approx([1.10, 1.09])
    assert scored["unknown_player_exposures"].tolist() == [0, 0]


def test_side_rankings_apply_side_specific_exposure_thresholds():
    stints = _stints().assign(
        home_team_id=1,
        away_team_id=2,
        home_team_tricode="HOM",
        away_team_tricode="AWY",
    )
    coefficients = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "offense_rapm": list(range(10, 0, -1)),
            "defense_rapm": list(range(0, 10)),
        }
    )
    bios = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "player_name": [f"P{value}" for value in range(1, 11)],
        }
    )

    rankings = _side_player_rankings(
        stints,
        coefficients,
        player_bios=bios,
        minimum_possessions=8.0,
    )

    offense = rankings.loc[rankings["ranking_type"] == "offense"]
    defense = rankings.loc[rankings["ranking_type"] == "defense"]
    overall = rankings.loc[rankings["ranking_type"] == "overall"]
    assert offense.iloc[0]["player_id"] == 1
    assert defense.iloc[0]["player_id"] == 10
    assert overall.iloc[0]["rating"] == pytest.approx(10.0)
    assert overall.iloc[0]["rating"] == pytest.approx(
        overall.iloc[0]["offense_rapm"] + overall.iloc[0]["defense_rapm"]
    )
    assert offense["ranking_possessions"].unique().tolist() == [8.0, 7.0]
    assert defense["ranking_possessions"].unique().tolist() == [8.0, 7.0]
    assert bool(offense.loc[offense["player_id"] == 1, "exposure_eligible"].item())
    assert not bool(defense.loc[defense["player_id"] == 1, "exposure_eligible"].item())


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1"],
            "home_player_ids": [list(range(1, 6))],
            "away_player_ids": [list(range(6, 11))],
            "points_home": [10.0],
            "points_away": [8.0],
            "home_offensive_possessions": [8.0],
            "away_offensive_possessions": [7.0],
        }
    )
