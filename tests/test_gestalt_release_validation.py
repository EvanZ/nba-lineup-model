"""Tests for the public NBA GESTALT release contract."""

from __future__ import annotations

import pytest

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
)
from nba_lineup_model.web_api.inference import (
    LineupEvaluationError,
    MODEL_NAME,
    _published_profile_padding_contract,
)
from nba_lineup_model.web_api.release_validation import (
    EXPECTED_CONTEXT_ALPHA,
    ReleaseValidationError,
    _validate_model_contract,
)


def test_published_padding_contract_rejects_one_stale_coefficient() -> None:
    metadata = {
        "profile_padding_contract": MEDVEDOVSKY_2020_PROFILE_PADDING.metadata()
    }
    selected = _published_profile_padding_contract(metadata)
    assert selected is MEDVEDOVSKY_2020_PROFILE_PADDING

    stale = MEDVEDOVSKY_2020_PROFILE_PADDING.metadata()
    stale["rate_pseudo_possessions"] = {
        **stale["rate_pseudo_possessions"],
        "three_pa": 300.0,
    }
    with pytest.raises(LineupEvaluationError, match="statistic-specific"):
        _published_profile_padding_contract({"profile_padding_contract": stale})


def test_release_contract_rejects_a_different_context_alpha() -> None:
    metadata = {
        "model": MODEL_NAME,
        "target_season": "2025-26",
        "context_alpha": EXPECTED_CONTEXT_ALPHA,
    }
    _validate_model_contract(metadata, season="2025-26")

    metadata["context_alpha"] = 5_000.0
    with pytest.raises(ReleaseValidationError, match="context alpha"):
        _validate_model_contract(metadata, season="2025-26")
