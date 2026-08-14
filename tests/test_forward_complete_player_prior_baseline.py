from __future__ import annotations

from nba_lineup_model.modeling.forward_complete_player_prior_baseline import (
    LABEL,
    MODEL_NAME,
    _seasons_through,
)
from nba_lineup_model.modeling.forward_rapm_memory_baselines import RapmMemoryBaseline


def test_complete_player_prior_uses_continuous_historical_seasons() -> None:
    seasons = _seasons_through("2025-26")

    assert seasons[0] == "1996-97"
    assert seasons[-1] == "2025-26"
    assert len(seasons) == 30


def test_complete_player_prior_uses_shared_frozen_evaluator_metadata_shape() -> None:
    candidate = RapmMemoryBaseline(MODEL_NAME, LABEL, 1)

    assert candidate.model == MODEL_NAME
    assert candidate.label == LABEL
