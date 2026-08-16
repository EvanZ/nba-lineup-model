from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES,
    lineup_side_context_features,
)
from nba_lineup_model.modeling.contextual_profiles import PROFILE_RATE_COLUMNS
from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    _add_compiled_additive_context_to_priors,
    _compiled_additive_context,
)
from nba_lineup_model.modeling.matchup_contextual import fit_linear_ridge_matchup_contextual_model


def test_transferred_prior_edge_plus_shape_residual_equals_full_context() -> None:
    profiles = pd.DataFrame(
        [
            {
                "player_id": player_id,
                **{
                    column: float(player_id * (index + 1))
                    for index, column in enumerate(PROFILE_RATE_COLUMNS)
                },
                "offensive_rebound_pct": float(player_id) / 100.0,
                "profile_imputed": 0,
                "profile_replacement_weight": 0.0,
            }
            for player_id in range(1, 11)
        ]
    )
    home = lineup_side_context_features(
        [(1, 2, 3, 4, 5), (1, 3, 5, 7, 9)],
        profiles,
        feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    )
    away = lineup_side_context_features(
        [(6, 7, 8, 9, 10), (2, 4, 6, 8, 10)],
        profiles,
        feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    )
    model = fit_linear_ridge_matchup_contextual_model(
        home,
        away,
        np.array([2.0, -1.0]),
        np.ones(2),
        alpha=1.0,
        feature_set=CONTEXT_FEATURE_SET_X3_WITHOUT_UNCERTAINTY,
    )
    priors = pd.DataFrame(
        {"player_id": range(1, 11), "lagged_rapm_prior": np.zeros(10)}
    )
    transferred, _ = _add_compiled_additive_context_to_priors(
        priors,
        model,
        profiles,
        previous_exposure=None,
    )
    prior_map = dict(
        zip(
            transferred["player_id"].astype(int),
            transferred["lagged_rapm_prior"],
            strict=True,
        )
    )
    transferred_edge = sum(prior_map[player_id] for player_id in (1, 2, 3, 4, 5)) - sum(
        prior_map[player_id] for player_id in (6, 7, 8, 9, 10)
    )
    additive_context = _compiled_additive_context(home.iloc[:1], away.iloc[:1], model)[0]
    full_context = model.predict_side_pairs(home.iloc[:1], away.iloc[:1])[0]

    assert tuple(LINEAR_X3_BASKETBALL_ADDITIVE_FEATURES) == (
        "three_pa_per_100",
        "three_pm_per_100",
        "assists_per_100",
        "turnovers_per_100",
        "usage_per_100",
        "steals_per_100",
        "blocks_per_100",
        "offensive_rebound_claim_total",
    )
    np.testing.assert_allclose(transferred_edge, additive_context)
    np.testing.assert_allclose(full_context, transferred_edge + full_context - additive_context)
