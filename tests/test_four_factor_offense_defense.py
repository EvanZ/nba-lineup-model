from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.four_factor_offense_defense import (
    SIDE_CONTROL_COLUMNS,
    build_offense_defense_factor_design,
    fit_prior_centered_offense_defense_factor,
)
from nba_lineup_model.modeling.forward_four_factor_offense_defense import (
    _component_exposure_history,
)
from nba_lineup_model.modeling.four_factor_prior_bridge import (
    FACTOR_SIDE_FEATURE_COLUMNS,
    _training_frame,
    apply_four_factor_prior_bridge,
)
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


def test_od_factor_design_has_two_side_rows_and_favorable_turnover_sign() -> None:
    stints = pd.DataFrame(
        {
            "game_id": ["001"],
            "home_player_ids": [[1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10]],
            "home_back_to_back": [1],
            "away_back_to_back": [0],
            "home_tov_pct": [0.10],
            "away_tov_pct": [0.20],
            "home_tov_denominator": [10.0],
            "away_tov_denominator": [20.0],
        }
    )

    design = build_offense_defense_factor_design(stints, "tov_pct")

    assert design.target.tolist() == [-0.10, -0.20]
    assert design.weights.tolist() == [10.0, 20.0]
    assert design.player_features.shape == (2, 20)
    assert np.allclose(design.player_features.toarray()[0, :5], 1.0)
    assert np.allclose(design.player_features.toarray()[0, 15:], -1.0)
    assert np.allclose(design.player_features.toarray()[1, 5:10], 1.0)
    assert np.allclose(design.player_features.toarray()[1, 10:15], -1.0)
    assert SIDE_CONTROL_COLUMNS == (
        "intercept",
        "home_court",
        "own_back_to_back",
        "opponent_back_to_back",
    )
    assert design.controls.tolist() == [[1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 0.0, 1.0]]


def test_offense_and_defense_penalties_can_differ() -> None:
    design = build_offense_defense_factor_design(
        pd.DataFrame(
            {
                "game_id": ["001"],
                "home_player_ids": [[1, 2, 3, 4, 5]],
                "away_player_ids": [[6, 7, 8, 9, 10]],
                "home_back_to_back": [0],
                "away_back_to_back": [0],
                "home_tov_pct": [0.10],
                "away_tov_pct": [0.20],
                "home_tov_denominator": [10.0],
                "away_tov_denominator": [20.0],
            }
        ),
        "tov_pct",
    )
    offense, defense, _ = fit_prior_centered_offense_defense_factor(
        design,
        regularization=(0.01, 0.3),
    )

    assert offense.shape == defense.shape == (10,)
    assert np.isfinite(offense).all()
    assert np.isfinite(defense).all()


def test_component_exposure_keeps_factor_reliability_and_rookie_gate_inputs() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2025-26"],
            "season_start_year": [2025],
            "player_id": [1],
            "player_name": ["Example"],
            "is_rookie": [True],
            "draft_year": [2025],
            "draft_round": [1],
            "draft_number": [5],
            "is_undrafted": [False],
            "age": [20.0],
            "height_inches": [78.0],
            "weight_pounds": [220.0],
        }
    )
    generic = pd.DataFrame(
        {
            "player_id": [1],
            "team_opportunity_possessions": [8000.0],
            "exposure_share": [0.25],
        }
    )

    result = _component_exposure_history(
        season="2025-26",
        player_ids=(1,),
        component_opportunity=np.asarray([321.0]),
        panel=panel,
        generic_exposure=generic,
    )

    assert result["on_court_possessions"].item() == 321.0
    assert result["is_rookie"].item() is True
    assert result["exposure_share"].item() == 0.25


def test_four_factor_bridge_zero_weight_is_an_exact_prior_identity() -> None:
    priors = pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [1.25, -0.75]})
    effects = pd.DataFrame(
        {
            "season": ["2024-25"],
            "player_id": [1],
            **{column: [0.1] for column in FACTOR_SIDE_FEATURE_COLUMNS},
        }
    )

    adjusted, metadata = apply_four_factor_prior_bridge(
        priors=priors,
        target_season="2025-26",
        effects=effects,
        bridge=None,
        weight=0.0,
    )

    pd.testing.assert_frame_equal(adjusted, priors)
    assert metadata["four_factor_prior_bridge_reason"] == "zero_weight"


def test_four_factor_bridge_uses_only_prior_season_factor_state() -> None:
    effects = pd.DataFrame(
        {
            "season": ["2021-22", "2022-23", "2023-24"],
            "player_id": [1, 1, 1],
            **{
                column: [0.1, 0.2, 99.0]
                for column in FACTOR_SIDE_FEATURE_COLUMNS
            },
        }
    )
    result = ForwardLaggedRapmSeason(
        season="2023-24",
        selected_lambda=0.1,
        cv_results=pd.DataFrame(),
        player_estimates=pd.DataFrame(
            {
                "player_id": [1],
                "rapm": [1.0],
                "prior_rapm": [0.0],
                "on_court_possessions": [100.0],
            }
        ),
        player_priors=pd.DataFrame(),
    )

    training = _training_frame(
        [result],
        effects,
        exposure_history=[pd.DataFrame({"player_id": [1], "on_court_possessions": [100.0]})],
    )

    # 2023-24 may use 2022-23's completed state, not 2023-24's 99.0 state.
    assert training.loc[0, "efg_pct_offense"] == 0.2
