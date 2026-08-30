from __future__ import annotations

from unittest.mock import Mock

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_TEAMMATE_CONTINUITY_REPLACEMENT,
)
from nba_lineup_model.modeling.forward_nail_teammate_continuity_replacement import (
    MODEL_NAME,
    train_nail_teammate_continuity_replacement,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import (
    RESIDUALIZED_LAMBDA_GRID,
)


def test_replacement_candidate_changes_only_the_nonadditive_contract(monkeypatch) -> None:
    train = Mock(return_value="run")
    monkeypatch.setattr(
        "nba_lineup_model.modeling.forward_nail_teammate_continuity_replacement."
        "train_nail_v1212_back_to_back",
        train,
    )

    assert train_nail_teammate_continuity_replacement(through_season="2025-26") == "run"

    kwargs = train.call_args.kwargs
    assert kwargs["model_name"] == MODEL_NAME
    assert kwargs["player_lambda_mode"] == "residualized_cv"
    assert kwargs["residualized_lambda_grid"] == RESIDUALIZED_LAMBDA_GRID
    assert kwargs["context_feature_set"] == CONTEXT_FEATURE_SET_NAIL_TEAMMATE_CONTINUITY_REPLACEMENT
    assert kwargs["use_prior_teammate_continuity"] is True
    contract = kwargs["profile_contract_metadata_updates"]["nonadditive_replacement_contract"]
    assert contract["retained"] == ["usage_concentration", "prior_teammate_continuity"]
    assert contract["removed"] == ["top_two_assists"]
