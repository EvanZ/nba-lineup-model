"""Tests for the constrained R/s Split NAIL v0.1 parameterization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.forward_split_nail_v01 import _fit_season


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_player_ids": [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)],
            "away_player_ids": [(6, 7, 8, 9, 10), (1, 2, 3, 4, 5)],
            "target_home_net_rating": [5.0, -5.0],
            "possessions": [10.0, 10.0],
        }
    )


def test_split_nail_v01_nests_scalar_rapm_when_specialization_is_zero() -> None:
    priors = {player_id: (1.0 if player_id <= 5 else 0.0) for player_id in range(1, 11)}

    state = _fit_season(
        _stints(),
        scalar_priors=priors,
        previous_specialization={},
        selected_lambda=1.0,
        specialization_relative_precision=4.0,
    )

    ratings = dict(zip(state.player_ids, state.combined_rating, strict=True))
    specialization = dict(zip(state.player_ids, state.specialization, strict=True))
    assert ratings[1] == pytest.approx(1.0)
    assert ratings[6] == pytest.approx(0.0)
    assert np.max(np.abs(list(specialization.values()))) == pytest.approx(0.0)
