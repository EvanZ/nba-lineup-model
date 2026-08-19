from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    UNIFORM_300_PROFILE_PADDING,
    _pad_rebound_percentages,
    _rate_frame,
)


def test_uniform_padding_preserves_the_existing_300_possession_formula() -> None:
    frame = _player_seasons()

    rates = _rate_frame(frame, padding_contract=UNIFORM_300_PROFILE_PADDING)

    expected_assists = 100.0 * (10.0 + 300.0 * 0.175) / 400.0
    assert rates.loc[0, "assists_per_100"] == expected_assists


def test_stat_specific_padding_builds_three_point_makes_from_volume_and_accuracy() -> None:
    frame = _player_seasons()

    rates = _rate_frame(frame, padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING)

    reference_attempt_rate = 100.0 * 70.0 / 400.0
    reference_percentage = 22.0 / 70.0
    attempt_rate = (
        100.0 * (10.0 + 29.29 * reference_attempt_rate / 100.0) / (100.0 + 29.29)
    )
    percentage = (4.0 + 242.61 * reference_percentage) / (10.0 + 242.61)
    assert np.isclose(rates.loc[0, "three_pm_per_100"], attempt_rate * percentage)


def test_stat_specific_padding_builds_usage_from_independently_padded_components() -> None:
    frame = _player_seasons()

    rates = _rate_frame(frame, padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING)

    fga = 100.0 * (20.0 + 89.62 * (90.0 / 400.0)) / (100.0 + 89.62)
    fta = 100.0 * (8.0 + 197.58 * (38.0 / 400.0)) / (100.0 + 197.58)
    tov = 100.0 * (3.0 + 425.69 * (15.0 / 400.0)) / (100.0 + 425.69)
    assert np.isclose(rates.loc[0, "usage_per_100"], fga + 0.44 * fta + tov)


def test_rebound_percentage_padding_uses_possession_weighted_season_reference() -> None:
    rates = pd.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "player_id": [1, 2],
            "_possessions": [100.0, 300.0],
            "offensive_rebound_pct": [0.20, 0.05],
            "defensive_rebound_pct": [0.30, 0.20],
        }
    )

    padded = _pad_rebound_percentages(
        rates,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
    )

    reference = (100.0 * 0.20 + 300.0 * 0.05) / 400.0
    expected = (100.0 * 0.20 + 98.55 * reference) / (100.0 + 98.55)
    assert np.isclose(padded.loc[0, "offensive_rebound_pct"], expected)


def _player_seasons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "player_id": [1, 2],
            "rapm_possessions": [100.0, 300.0],
            "three_pointers_attempted": [10.0, 60.0],
            "three_pointers_made": [4.0, 18.0],
            "assists": [10.0, 60.0],
            "turnovers": [3.0, 12.0],
            "field_goals_attempted": [20.0, 70.0],
            "free_throws_attempted": [8.0, 30.0],
            "rebounds_offensive": [5.0, 15.0],
            "rebounds_defensive": [15.0, 45.0],
            "steals": [2.0, 8.0],
            "blocks": [1.0, 6.0],
        }
    )
