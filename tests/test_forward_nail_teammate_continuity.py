from __future__ import annotations

from unittest.mock import Mock

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_PRIOR_TEAMMATE_CONTINUITY,
)
from nba_lineup_model.modeling.forward_nail_teammate_continuity import (
    MODEL_NAME,
    train_nail_teammate_continuity,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    RESIDUALIZED_LAMBDA_GRID,
)


def test_candidate_changes_only_the_registered_context_feature(monkeypatch) -> None:
    train = Mock(return_value="run")
    monkeypatch.setattr(
        "nba_lineup_model.modeling.forward_nail_teammate_continuity."
        "train_nail_v1212_back_to_back",
        train,
    )

    assert train_nail_teammate_continuity(through_season="2025-26") == "run"

    kwargs = train.call_args.kwargs
    assert kwargs["model_name"] == MODEL_NAME
    assert kwargs["player_lambda_mode"] == "residualized_cv"
    assert kwargs["residualized_lambda_grid"] == RESIDUALIZED_LAMBDA_GRID
    assert kwargs["context_feature_set"] == CONTEXT_FEATURE_SET_NAIL_PRIOR_TEAMMATE_CONTINUITY
    assert kwargs["use_prior_teammate_continuity"] is True
