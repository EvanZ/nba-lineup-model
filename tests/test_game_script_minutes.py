from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.game_script_minutes import (
    REGULATION_SECONDS,
    apply_expected_game_script,
    competitive_boundary_seconds,
    competitive_role_minutes,
    expected_garbage_minutes,
)


def _events(rows: list[tuple[int, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_index": index,
                "period": period,
                "elapsed_game_seconds": elapsed,
                "score_home": home,
                "score_away": away,
            }
            for index, (period, elapsed, home, away) in enumerate(rows)
        ]
    )


def test_close_game_has_no_garbage_boundary() -> None:
    events = _events(
        [
            (1, 0.0, 0, 0),
            (4, 2520.0, 90, 86),
            (4, 2760.0, 101, 96),
            (4, 2880.0, 110, 105),
        ]
    )
    assert competitive_boundary_seconds(events) == REGULATION_SECONDS


def test_persistent_late_blowout_starts_at_first_qualifying_event() -> None:
    events = _events(
        [
            (1, 0.0, 0, 0),
            (4, 2520.0, 100, 78),  # 6:00, threshold 21
            (4, 2640.0, 106, 84),
            (4, 2880.0, 120, 96),
        ]
    )
    assert competitive_boundary_seconds(events) == 2520.0


def test_comeback_invalidates_early_blowout_signal() -> None:
    events = _events(
        [
            (1, 0.0, 0, 0),
            (4, 2520.0, 100, 78),
            (4, 2760.0, 104, 99),
            (4, 2880.0, 111, 108),
        ]
    )
    assert competitive_boundary_seconds(events) == REGULATION_SECONDS


def test_overtime_game_is_competitive_through_final_event() -> None:
    events = _events(
        [
            (4, 2880.0, 100, 100),
            (5, 3180.0, 112, 109),
        ]
    )
    assert competitive_boundary_seconds(events) == 3180.0


def test_competitive_exposure_excludes_garbage_unit_and_reconciles() -> None:
    starters_home = [1, 2, 3, 4, 5]
    starters_away = [11, 12, 13, 14, 15]
    bench_home = [6, 7, 8, 9, 10]
    bench_away = [16, 17, 18, 19, 20]
    stints = pd.DataFrame(
        [
            {
                "start_elapsed_game_seconds": 0.0,
                "end_elapsed_game_seconds": 2520.0,
                "home_player_ids": starters_home,
                "away_player_ids": starters_away,
            },
            {
                "start_elapsed_game_seconds": 2520.0,
                "end_elapsed_game_seconds": 2880.0,
                "home_player_ids": bench_home,
                "away_player_ids": bench_away,
            },
        ]
    )
    result = competitive_role_minutes(stints, boundary_seconds=2520.0)
    assert sum(result[player_id] for player_id in starters_home) == pytest.approx(240.0)
    assert sum(result[player_id] for player_id in starters_away) == pytest.approx(240.0)
    assert not set(bench_home + bench_away).intersection(result)


def test_incomplete_lineup_coverage_fails_closed() -> None:
    stints = pd.DataFrame(
        [
            {
                "start_elapsed_game_seconds": 0.0,
                "end_elapsed_game_seconds": 2400.0,
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [11, 12, 13, 14, 15],
            }
        ]
    )
    with pytest.raises(ValueError, match="coverage is incomplete"):
        competitive_role_minutes(stints, boundary_seconds=2520.0)


def test_pregame_adjustment_is_symmetric_in_margin_and_preserves_total() -> None:
    values = [36, 34, 32, 30, 28, 24, 20, 16, 12, 8]
    roles = {player_id: value for player_id, value in enumerate(values, 1)}
    # Normalize the compact fixture to a legal team total.
    scale = 240.0 / sum(roles.values())
    roles = {player_id: value * scale for player_id, value in roles.items()}
    home = apply_expected_game_script(roles, projected_margin=14.0)
    away = apply_expected_game_script(roles, projected_margin=-14.0)

    assert home == pytest.approx(away)
    assert sum(home.values()) == pytest.approx(240.0)
    assert home[1] < roles[1]
    assert home[10] > roles[10]
    assert expected_garbage_minutes(14.0) > expected_garbage_minutes(0.0)
