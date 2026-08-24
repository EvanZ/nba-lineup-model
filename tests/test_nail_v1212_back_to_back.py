from __future__ import annotations

from unittest.mock import Mock

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
)
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import (
    MODEL_NAME,
    train_nail_v1212_back_to_back,
)


def test_back_to_back_candidate_keeps_standard_usage_and_enables_schedule_control(
    monkeypatch,
) -> None:
    train = Mock(return_value="run")
    monkeypatch.setattr(
        "nba_lineup_model.modeling.forward_nail_v1212_back_to_back."
        "train_forward_portable_matchup_contextual_rapm",
        train,
    )

    assert train_nail_v1212_back_to_back(through_season="2025-26") == "run"

    kwargs = train.call_args.kwargs
    assert kwargs["model_name"] == MODEL_NAME
    assert kwargs["include_back_to_back_control"] is True
    assert kwargs["context_feature_set"] == CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE
