from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.season.causal_hca_evaluation import (
    _american_implied_probability,
    _market_home_probability,
    _metrics,
    _project_team_strength,
    _validate_frozen_source,
)


def _official_frozen() -> dict[str, object]:
    return {
        "game_id": "0022600001",
        "source_provenance": {
            "basketball_facts": "NBA official liveData",
            "injuries": "NBA official injury report",
            "market": "The Odds API",
            "provider_fallbacks": [],
            "schedule_identity": "NBA official schedule",
        },
    }


def test_official_frozen_source_rejects_any_espn_dependency() -> None:
    _validate_frozen_source(_official_frozen(), game_id="0022600001")
    contaminated = copy.deepcopy(_official_frozen())
    contaminated["provider_note"] = "ESPN fallback"
    with pytest.raises(ValueError, match="ESPN dependency"):
        _validate_frozen_source(contaminated, game_id="0022600001")


def test_project_team_strength_requires_240_minutes_and_complete_ratings() -> None:
    rows = [
        {"player_id": 1, "projected_minutes": 120.0},
        {"player_id": 2, "projected_minutes": 120.0},
    ]
    assert _project_team_strength(rows, {1: 2.0, 2: -1.0}) == 2.5
    with pytest.raises(ValueError, match="missing players"):
        _project_team_strength(rows, {1: 2.0})
    with pytest.raises(ValueError, match="sum to 240"):
        _project_team_strength(rows[:1], {1: 2.0})


def test_forward_metrics_use_all_games_and_exact_ats_results() -> None:
    frame = pd.DataFrame(
        {
            "baseline_margin": [2.0, -1.0],
            "candidate_margin": [1.0, 3.0],
            "actual_margin": [4.0, -2.0],
            "baseline_probability": [0.6, 0.4],
            "candidate_probability": [0.55, 0.65],
            "baseline_ats_result": ["W", "W"],
            "candidate_ats_result": ["W", "L"],
        }
    )
    baseline = _metrics(frame, "baseline")
    candidate = _metrics(frame, "candidate")
    assert baseline["games"] == candidate["games"] == 2
    assert baseline["winner_accuracy"] == 1.0
    assert candidate["winner_accuracy"] == 0.5
    assert baseline["ats_wins"] == 2
    assert candidate["ats_losses"] == 1
    assert np.isclose(baseline["brier_score"], 0.16)


def test_market_probability_removes_two_sided_vig() -> None:
    assert np.isclose(_american_implied_probability(-110), 110 / 210)
    assert np.isclose(_american_implied_probability(120), 100 / 220)
    assert np.isclose(
        _market_home_probability({"home_moneyline": -110, "away_moneyline": -110}),
        0.5,
    )
    assert np.isclose(
        _market_home_probability(
            {
                "home_win_probability": 0.61,
                "home_moneyline": -110,
                "away_moneyline": -110,
            }
        ),
        0.61,
    )
