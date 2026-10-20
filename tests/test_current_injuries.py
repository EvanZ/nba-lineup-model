from __future__ import annotations

import pytest

from nba_lineup_model.web_api.current_injuries import (
    InjuryGamesMissedRange,
    apply_current_injury_availability,
)


def test_injury_midpoint_replaces_availability_without_return_ramp() -> None:
    payload = {
        "model": "test",
        "contract": "baseline.",
        "initial_rotation_size": 3,
        "players": [
            {
                "player_id": 1,
                "player_name": "Injured",
                "team": "AAA",
                "availability_probability": 0.75,
                "conditional_minutes_per_game": 30.0,
            },
            {
                "player_id": 2,
                "player_name": "Healthy",
                "team": "AAA",
                "availability_probability": 0.8,
                "conditional_minutes_per_game": 20.0,
            },
        ],
    }
    estimates = {
        1: InjuryGamesMissedRange("Injured", 10.0, 20.0, "https://example.com/injury")
    }

    adjusted = apply_current_injury_availability(
        payload, estimates=estimates, as_of="2026-10-20"
    )
    players = {row["player_id"]: row for row in adjusted["players"]}

    assert players[1]["model_availability_probability"] == pytest.approx(0.75)
    assert players[1]["injury_expected_games_missed"] == pytest.approx(15.0)
    assert players[1]["availability_probability"] == pytest.approx(67.0 / 82.0)
    assert players[1]["raw_expected_total_minutes"] == pytest.approx(67.0 * 30.0)
    assert players[2]["availability_probability"] == pytest.approx(0.8)
    assert sum(row["baseline_minutes_per_game"] for row in adjusted["players"]) == pytest.approx(
        240.0
    )
    assert adjusted["injury_availability"]["method"] == (
        "midpoint expected games missed; no post-return ramp"
    )


def test_injury_adjustment_rejects_player_outside_roster() -> None:
    payload = {
        "initial_rotation_size": 1,
        "players": [
            {
                "player_id": 1,
                "player_name": "Healthy",
                "team": "AAA",
                "availability_probability": 1.0,
                "conditional_minutes_per_game": 20.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="outside the roster"):
        apply_current_injury_availability(
            payload,
            estimates={
                2: InjuryGamesMissedRange("Missing", 1.0, 2.0, "https://example.com/injury")
            },
        )
