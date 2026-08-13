from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.context_reattributed_rapm import fit_context_projection


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_player_ids": [[1], [1], [2], [2]],
            "away_player_ids": [[2], [3], [1], [3]],
            "possessions": [10.0, 20.0, 15.0, 25.0],
        }
    )


def test_context_projection_reconciles_to_context_target() -> None:
    stints = _stints()
    target = np.array([2.0, 1.0, -2.0, -1.0])

    projection = fit_context_projection(stints, target, regularization=0.0)

    assert np.allclose(target, projection.prediction + projection.residual)


def test_context_projection_uses_same_signed_player_orientation_as_rapm() -> None:
    stints = _stints()
    target = np.array([2.0, 5.0, -2.0, 3.0])

    projection = fit_context_projection(stints, target, regularization=0.0)
    coefficient = dict(zip(projection.player_ids, projection.coefficients, strict=True))

    assert coefficient[1] > coefficient[2] > coefficient[3]
