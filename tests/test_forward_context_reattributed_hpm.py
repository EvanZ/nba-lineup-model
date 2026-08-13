from __future__ import annotations

import numpy as np
import pandas as pd

import nba_lineup_model.modeling.forward_context_reattributed_hpm as cr_hpm
import nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm as portable
from nba_lineup_model.modeling.context_reattributed_rapm import ContextProjection
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN


def test_context_reattribution_prior_promotes_only_known_returners() -> None:
    priors = pd.DataFrame({"player_id": [1, 2, 3], PRIOR_MEAN_COLUMN: [1.0, 0.0, -1.0]})
    projection = ContextProjection(
        player_ids=(1, 2),
        coefficients=np.array([2.0, -4.0]),
        intercept=0.5,
        prediction=np.array([]),
        residual=np.array([]),
        selected_lambda=0.1,
    )

    actual, metadata = portable._add_context_reattribution_to_priors(
        priors, projection, reattribution_weight=0.5
    )

    assert actual[PRIOR_MEAN_COLUMN].tolist() == [2.0, -2.0, -1.0]
    assert metadata["context_reattribution_returning_player_count"] == 2


def test_residual_context_removes_transferred_player_projection() -> None:
    stints = pd.DataFrame(
        {"home_player_ids": [[1, 2]], "away_player_ids": [[3, 4]]}
    )
    projection = ContextProjection(
        player_ids=(1, 2, 3, 4),
        coefficients=np.array([1.0, 2.0, 3.0, 4.0]),
        intercept=0.0,
        prediction=np.array([]),
        residual=np.array([]),
        selected_lambda=0.1,
    )

    residual = portable._project_reattributed_player_context(stints, projection)

    assert np.array_equal(residual, np.array([-4.0]))


def test_wrapper_configures_bounded_residual_hpm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def train(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cr_hpm, "train_forward_portable_matchup_contextual_rapm", train)

    cr_hpm.train_forward_context_reattributed_hpm(context_reattribution_weight=0.25)

    assert captured["context_reattribution_weight"] == 0.25
    assert captured["model_name"] == cr_hpm.MODEL_NAME
