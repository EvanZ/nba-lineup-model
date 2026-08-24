"""Tests for date-free home-court and rest scenarios in the Matchup Lab."""

from __future__ import annotations

import pytest

from nba_lineup_model.web_api.app import MatchupRequest
from nba_lineup_model.web_api.inference import (
    MeanRevertedScheduleControls,
    _apply_schedule_scenario,
)


def test_schedule_scenario_keeps_the_core_edge_separate() -> None:
    result = _apply_schedule_scenario(
        {"predicted_net_rating": 3.0},
        controls=MeanRevertedScheduleControls(
            home_court=2.8,
            back_to_back=-1.5,
            source_season_count=29,
        ),
        court="unit_home",
        unit_back_to_back=True,
        opponent_back_to_back=False,
    )

    assert result["base_predicted_net_rating"] == 3.0
    assert result["home_court_adjustment"] == 2.8
    assert result["back_to_back_adjustment"] == -1.5
    assert result["schedule_adjustment"] == pytest.approx(1.3)
    assert result["predicted_net_rating"] == pytest.approx(4.3)


def test_schedule_scenario_reverses_for_opponent_home_and_rest_disadvantage() -> None:
    result = _apply_schedule_scenario(
        {"predicted_net_rating": 1.0},
        controls=MeanRevertedScheduleControls(
            home_court=2.0,
            back_to_back=-1.0,
            source_season_count=29,
        ),
        court="opponent_home",
        unit_back_to_back=False,
        opponent_back_to_back=True,
    )

    assert result["home_court_adjustment"] == -2.0
    assert result["back_to_back_adjustment"] == 1.0
    assert result["predicted_net_rating"] == 0.0


def test_matchup_request_defaults_to_the_neutral_rest_scenario() -> None:
    request = MatchupRequest(
        unit_player_ids=[1, 2, 3, 4, 5],
        opponent_player_ids=[6, 7, 8, 9, 10],
    )

    assert request.court == "neutral"
    assert not request.unit_back_to_back
    assert not request.opponent_back_to_back
