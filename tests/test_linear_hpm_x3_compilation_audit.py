from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.linear_hpm_x3_compilation_audit import (
    ADDITIVE_FEATURE_TO_PLAYER_PROFILE,
    _additive_feature_map_for_contract,
    _home_away_lineups,
    _player_adjustment_frame,
    linear_raw_context_coefficients,
)
from nba_lineup_model.modeling.matchup_contextual import fit_linear_ridge_matchup_contextual_model


def test_raw_coefficients_undo_standardization() -> None:
    columns = side_context_feature_columns(CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT)
    home = pd.DataFrame(np.arange(64 * len(columns)).reshape(64, len(columns)), columns=columns)
    away = home.iloc[::-1].reset_index(drop=True)
    model = fit_linear_ridge_matchup_contextual_model(
        home,
        away,
        np.linspace(-2.0, 2.0, len(home)),
        np.ones(len(home)),
        alpha=10.0,
        feature_set=CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT,
    )

    raw = linear_raw_context_coefficients(model)
    pipeline = model.pipeline
    expected = pipeline.named_steps["ridge"].coef_ / pipeline.named_steps["scale"].scale_
    np.testing.assert_allclose(raw.to_numpy(), expected)


def test_compiled_player_adjustments_use_each_additive_profile_coordinate() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2],
            **{
                column: [index + 1.0, index + 2.0]
                for index, column in enumerate(ADDITIVE_FEATURE_TO_PLAYER_PROFILE.values())
            },
        }
    )
    coefficients = pd.Series(
        {feature: index + 0.5 for index, feature in enumerate(ADDITIVE_FEATURE_TO_PLAYER_PROFILE)}
    )

    output = _player_adjustment_frame(profiles, coefficients)
    expected = sum(
        coefficients[feature] * profiles[column].to_numpy()
        for feature, column in ADDITIVE_FEATURE_TO_PLAYER_PROFILE.items()
    )
    np.testing.assert_allclose(output["compiled_additive_context_adjustment"], expected)


def test_home_away_lineups_use_the_signed_possession_contract() -> None:
    possessions = pd.DataFrame(
        {
            "offense_player_ids": [(1, 2, 3, 4, 5), (11, 12, 13, 14, 15)],
            "defense_player_ids": [(6, 7, 8, 9, 10), (16, 17, 18, 19, 20)],
            "home_offense_sign": [1.0, -1.0],
        }
    )

    home, away = _home_away_lineups(possessions)

    assert home == [(1, 2, 3, 4, 5), (16, 17, 18, 19, 20)]
    assert away == [(6, 7, 8, 9, 10), (11, 12, 13, 14, 15)]


def test_canonical_compilation_contract_excludes_profile_quality_terms() -> None:
    legacy = _additive_feature_map_for_contract(
        CONTEXT_FEATURE_SET_X3_V1_ORB_CLAIM_REPLACEMENT
    )
    canonical = _additive_feature_map_for_contract(CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY)

    assert set(legacy) - set(canonical) == {"imputed_count", "replacement_weight"}
    assert len(canonical) == 8
