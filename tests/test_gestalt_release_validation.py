"""Tests for the public NBA GESTALT release contract."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
)
from nba_lineup_model.web_api.inference import (
    MODEL_NAME,
    LineupEvaluationError,
    _published_profile_padding_contract,
)
from nba_lineup_model.web_api.release_validation import (
    EXPECTED_CONTEXT_ALPHA,
    ReleaseValidationError,
    _validate_model_contract,
    _validate_player_rotation_history,
    _validate_win_projection_cache,
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


def test_published_padding_contract_accepts_pre_usage_percentage_artifact() -> None:
    legacy = MEDVEDOVSKY_2020_PROFILE_PADDING.metadata()
    legacy.pop("usage_percentage_pseudo_possessions")

    selected = _published_profile_padding_contract({"profile_padding_contract": legacy})

    assert selected is MEDVEDOVSKY_2020_PROFILE_PADDING


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


def test_release_cache_requires_the_promoted_fcm_contract() -> None:
    cache = {
        "minutes": {
            "conditional_minutes_version": "v0.3",
            "incumbent_team_strength": {
                "league_strength_fallback": 0.0,
                "team_strength_log_minutes_coefficient": -0.2,
                "teams": [{"team": "TST"}],
            },
            "players": [{"player_id": 1}],
        },
        "win_projection": {"teams": [{"team": "TST"}]},
    }

    _validate_win_projection_cache(cache)
    cache["minutes"]["conditional_minutes_version"] = "v0.2"

    with pytest.raises(ReleaseValidationError, match="FCM v0.3"):
        _validate_win_projection_cache(cache)


def test_player_rotation_history_requires_forecast_parity_with_live_minutes() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2025-26", "2026-27"],
            "season_start_year": [2025, 2026],
            "player_id": [1, 1],
            "player_name": ["Test Player", "Test Player"],
            "age": [25.0, 26.0],
            "actual_available_games": [70.0, None],
            "known_roster_games": [82.0, None],
            "actual_availability_share": [70 / 82, None],
            "injury_or_illness_games": [8.0, None],
            "rest_games": [1.0, None],
            "actual_minutes_per_available_game": [30.0, None],
            "actual_total_minutes": [2100.0, None],
            "predicted_available_share": [0.8, 0.75],
            "predicted_minutes_per_available_game": [29.0, 28.0],
            "projected_total_minutes": [None, 1476.0],
            "is_preseason_forecast": [False, True],
        }
    )
    cache = {
        "minutes": {
            "players": [
                {
                    "player_id": 1,
                    "availability_probability": 0.75,
                    "conditional_minutes_per_game": 28.0,
                    "baseline_minutes_per_game": 18.0,
                }
            ]
        }
    }

    _validate_player_rotation_history(frame, win_cache=cache)
    frame.loc[
        frame["is_preseason_forecast"], "predicted_minutes_per_available_game"
    ] = 29.0

    with pytest.raises(ReleaseValidationError, match="FCM forecast"):
        _validate_player_rotation_history(frame, win_cache=cache)
