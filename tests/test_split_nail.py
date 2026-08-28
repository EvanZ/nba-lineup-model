"""Tests for the unconstrained offense/defense Split NAIL design."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.split_nail import (
    SPLIT_NAIL_ADDITIVE_FEATURES,
    SPLIT_NAIL_NONADDITIVE_FEATURES,
    build_split_nail_design_from_side_features,
    fit_split_nail_season,
    split_nail_prior_vector,
)


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["001", "002"],
            "home_player_ids": [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5)],
            "away_player_ids": [(6, 7, 8, 9, 10), (6, 7, 8, 9, 10)],
            "points_home": [12, 8],
            "points_away": [10, 14],
            "home_offensive_possessions": [10, 10],
            "away_offensive_possessions": [10, 10],
        }
    )


def _features(value: float) -> pd.DataFrame:
    columns = (*SPLIT_NAIL_ADDITIVE_FEATURES, *SPLIT_NAIL_NONADDITIVE_FEATURES)
    return pd.DataFrame({column: [value, value] for column in columns})


def test_split_nail_exposes_every_feature_on_both_sides() -> None:
    design = build_split_nail_design_from_side_features(_stints(), _features(2.0), _features(1.0))

    assert design.features.shape == (4, 2 * 10 + 2 * 10 + 2)
    home_row = design.features.getrow(0).toarray().ravel()
    assert home_row[design.player_column(1, side="offense")] == 1.0
    assert home_row[design.player_column(6, side="defense")] == -1.0
    for feature in (*SPLIT_NAIL_ADDITIVE_FEATURES, *SPLIT_NAIL_NONADDITIVE_FEATURES):
        scale = design.feature_scale(feature)
        assert home_row[design.feature_column(feature, side="offense")] == 2.0 / scale
        assert home_row[design.feature_column(feature, side="defense")] == -1.0 / scale


def test_split_nail_prior_preserves_scalar_total_and_carries_side_difference() -> None:
    design = build_split_nail_design_from_side_features(_stints(), _features(0.0), _features(0.0))

    prior = split_nail_prior_vector(design, {1: 4.0}, {1: 1.5})

    offense = prior[design.player_column(1, side="offense")]
    defense = prior[design.player_column(1, side="defense")]
    assert offense == pytest.approx(2.75)
    assert defense == pytest.approx(1.25)
    assert offense + defense == pytest.approx(4.0)


def test_split_nail_fit_keeps_player_and_feature_sides_identifiable() -> None:
    design = build_split_nail_design_from_side_features(_stints(), _features(0.0), _features(0.0))

    fit = fit_split_nail_season(
        design,
        {player_id: 0.0 for player_id in design.player_ids},
        regularization=1.0,
    )

    assert len(fit.player_coefficients) == 10
    assert set(fit.feature_coefficients["feature"]) == set(
        (*SPLIT_NAIL_ADDITIVE_FEATURES, *SPLIT_NAIL_NONADDITIVE_FEATURES)
    )
    ratings = fit.player_coefficients[["offense_base_rating", "defense_base_rating"]]
    assert np.isfinite(ratings).all().all()


def test_split_nail_exposes_side_specific_back_to_back_controls() -> None:
    stints = _stints().assign(
        home_back_to_back=[1, 0],
        away_back_to_back=[0, 1],
    )
    design = build_split_nail_design_from_side_features(stints, _features(0.0), _features(0.0))

    assert design.includes_back_to_back
    assert design.features.shape == (4, 2 * 10 + 2 * 10 + 2 + 2)
    home_row = design.features.getrow(0).toarray().ravel()
    away_row = design.features.getrow(1).toarray().ravel()
    assert home_row[design.back_to_back_column(side="offense")] == pytest.approx(1.0)
    assert home_row[design.back_to_back_column(side="defense")] == pytest.approx(0.0)
    assert away_row[design.back_to_back_column(side="offense")] == pytest.approx(0.0)
    assert away_row[design.back_to_back_column(side="defense")] == pytest.approx(-1.0)

    fit = fit_split_nail_season(
        design,
        {player_id: 0.0 for player_id in design.player_ids},
        regularization=1.0,
    )
    assert fit.schedule_coefficients["schedule_control"].tolist() == ["back_to_back"]
    assert np.isfinite(
        fit.schedule_coefficients[["offense_raw_coefficient", "defense_raw_coefficient"]]
    ).all().all()


def test_split_nail_exposes_separate_home_offense_and_defense_terms() -> None:
    design = build_split_nail_design_from_side_features(_stints(), _features(0.0), _features(0.0))

    home_row = design.features.getrow(0).toarray().ravel()
    away_row = design.features.getrow(1).toarray().ravel()
    assert home_row[design.home_court_column(side="offense")] == 1.0
    assert home_row[design.home_court_column(side="defense")] == 0.0
    assert away_row[design.home_court_column(side="offense")] == 0.0
    assert away_row[design.home_court_column(side="defense")] == -1.0
