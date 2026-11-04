"""Regression checks for canonical player-game values at the serving boundary."""
import pandas as pd
import pytest

from nba_lineup_model.web_api.current_season_serving import _player_game_rating_trajectories
from nba_lineup_model.web_api.win_projections import _signed_integer_or_none


@pytest.mark.parametrize("value,expected", [
    (14, 14), (14.0, 14), ("14.0", 14), ("+14", 14), (" +14.0 ", 14),
    (-18, -18), (-18.0, -18), ("-18.0", -18), (0, 0), (0.0, 0),
    ("0.0", 0), (None, None), ("", None), ("bad", None),
    (1.5, None), ("-1.5", None), (float("nan"), None),
    (float("inf"), None), ("-inf", None), (True, None),
])
def test_nba_plus_minus_accepts_integral_numeric_representations(value, expected):
    assert _signed_integer_or_none(value) == expected


@pytest.mark.parametrize("played", [True, False])
@pytest.mark.parametrize("delta", [0.25, -0.125, 0.0, None])
def test_game_delta_is_canonical_not_difference_since_previous_game(played, delta):
    rankings = pd.DataFrame([{
        "player_id": 1, "preseason_nail": 3.0, "rapm": 3.5,
        "current_season_possessions": 200.0 if played else 100.0,
        "rating_posterior_standard_error": 0.1,
        "in_season_delta": 0.5, "rating_evidence_weight": 0.02,
    }])
    trajectory = pd.DataFrame([{
        "player_id": 1, "game_number": index, "appearance_number": index if active else None,
        "played": active, "game_id": str(index), "game_date": f"2026-10-{20+index}",
        "game_time_utc": f"2026-10-{20+index}T22:00:00Z", "team": "GSW",
        "opponent": "LAL", "pregame_status": "Available" if active else "Out",
        "pregame_reason": None, "updated_rating": rating, "preseason_rating": 3.0,
        "in_season_delta": rating-3.0, "game_on_court_possessions": 100.0 if active else 0.0,
        "cumulative_on_court_possessions": 100.0 if index == 1 or not active else 200.0,
        "evidence_weight": 0.02, "rating_delta": change,
    } for index, rating, active, change in [(1, 3.4, True, 0.4), (2, 3.5, played, delta)]])
    before = trajectory.copy(deep=True)
    points = _player_game_rating_trajectories(
        rankings, trajectory, current_bundle_sha="a"*64,
        evidence_through_date="2026-10-22", as_of_cutoff_time="2026-10-23T08:00:00Z",
    )[1]
    assert points[-1]["rating_delta"] == delta
    assert points[-1]["rating"] == 3.5
    assert "rating_delta" not in points[0]  # preseason is not a game update
    pd.testing.assert_frame_equal(trajectory, before)
    # Legacy artifacts lacking the canonical field stay explicitly unknown.
    legacy = _player_game_rating_trajectories(
        rankings, trajectory.drop(columns=["rating_delta"]), current_bundle_sha="b"*64,
        evidence_through_date="2026-10-22", as_of_cutoff_time="2026-10-23T08:00:00Z",
    )[1]
    assert legacy[-1]["rating_delta"] is None
