from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.frozen_feature_screen import (
    FEATURE_CANDIDATES,
    available_feature_candidates,
    summarize_feature_bins,
    weighted_correlation,
)


def test_production_nonadditive_candidates_are_registered() -> None:
    assert set(available_feature_candidates()) >= {
        "median_three_pm_per_100",
        "rim_protection_ceiling",
        "top_two_assists",
        "usage_concentration",
    }
    assert FEATURE_CANDIDATES["usage_concentration"].production_context_column == (
        "usage_concentration"
    )
    assert FEATURE_CANDIDATES["top_two_assists"].production_context_column == "top_two_assists"


def test_median_three_pm_candidate_uses_the_third_ranked_player() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "three_pm_per_100": [0.5, 1.0, 2.0, 3.0, 5.0],
        }
    )

    values = FEATURE_CANDIDATES["median_three_pm_per_100"].side_feature(
        [(5, 1, 4, 2, 3)], profiles
    )

    assert values.tolist() == [2.0]


def test_rim_protection_ceiling_uses_the_highest_block_profile() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "blocks_per_100": [0.2, 1.1, 0.7, 3.4, 1.9],
        }
    )

    values = FEATURE_CANDIDATES["rim_protection_ceiling"].side_feature(
        [(5, 1, 4, 2, 3)], profiles
    )

    assert values.tolist() == [3.4]


def test_summarize_feature_bins_uses_deciles_for_continuous_candidates() -> None:
    frame = pd.DataFrame(
        {
            "feature_edge": np.arange(20, dtype=float),
            "frozen_residual_net_rating": np.arange(20, dtype=float),
            "possessions": np.ones(20),
        }
    )

    summary = summarize_feature_bins(frame)

    assert summary["bin_kind"].eq("decile").all()
    assert summary["bin"].tolist() == list(range(1, 11))
    assert summary["stint_count"].tolist() == [2] * 10
    assert summary["residual_mean"].iloc[0] == 0.5
    assert summary["residual_mean"].iloc[-1] == 18.5


def test_summarize_feature_bins_keeps_discrete_candidates_intact() -> None:
    frame = pd.DataFrame(
        {
            "feature_edge": [0.0, 0.0, 1.0, 1.0],
            "frozen_residual_net_rating": [-1.0, 1.0, 2.0, 4.0],
            "possessions": [1.0, 3.0, 2.0, 2.0],
        }
    )

    summary = summarize_feature_bins(frame)

    assert summary["bin_kind"].eq("discrete_value").all()
    assert summary["bin"].tolist() == [0.0, 1.0]
    assert summary["residual_mean"].tolist() == [0.5, 3.0]


def test_weighted_correlation_returns_zero_for_a_constant_feature() -> None:
    result = weighted_correlation(
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
    )

    assert result == 0.0
