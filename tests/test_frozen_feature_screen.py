from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.frozen_feature_screen import (
    FEATURE_CANDIDATES,
    _creation_spacing_alignment,
    _defensive_anchor_by_perimeter_pressure,
    _offensive_role_redundancy,
    _rim_pressure_by_spacing_floor,
    _secondary_creator_floor,
    _source_profile_scales,
    available_feature_candidates,
    summarize_feature_bins,
    weighted_correlation,
)
from nba_lineup_model.modeling.teammate_continuity import (
    build_teammate_pair_exposure,
    teammate_continuity_side_feature,
)


def test_production_nonadditive_candidates_are_registered() -> None:
    assert set(available_feature_candidates()) >= {
        "median_three_pm_per_100",
        "rim_protection_ceiling",
        "creation_spacing_alignment",
        "defensive_anchor_by_perimeter_pressure",
        "offensive_role_redundancy",
        "prior_teammate_continuity",
        "rim_pressure_by_spacing_floor",
        "secondary_creator_floor",
        "top_two_assists",
        "usage_concentration",
    }
    assert FEATURE_CANDIDATES["usage_concentration"].production_context_column == (
        "usage_concentration"
    )
    assert FEATURE_CANDIDATES["top_two_assists"].production_context_column == "top_two_assists"
    assert FEATURE_CANDIDATES["prior_teammate_continuity"].uses_prior_pair_exposure


def test_creation_spacing_alignment_weights_shooting_by_unit_usage_share() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "usage_per_100": [10.0, 20.0, 30.0, 20.0, 20.0],
            "three_pm_per_100": [1.0, 2.0, 4.0, 3.0, 0.0],
        }
    )

    values = _creation_spacing_alignment([[1, 2, 3, 4, 5]], profiles)

    assert values.tolist() == [2.3]


def test_secondary_creator_floor_uses_second_highest_assist_profile() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "assists_per_100": [1.0, 6.5, 2.0, 4.0, 3.0],
        }
    )

    values = _secondary_creator_floor([[1, 2, 3, 4, 5]], profiles)

    assert values.tolist() == [4.0]


def test_rim_pressure_by_spacing_floor_uses_two_weakest_spacers() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "unassisted_rim_makes_per_100": [1.0, 2.0, 3.0, 4.0, 5.0],
            "three_pm_per_100": [2.0, 1.0, 5.0, 3.0, 4.0],
        }
    )

    values = _rim_pressure_by_spacing_floor([[1, 2, 3, 4, 5]], profiles)

    assert values.tolist() == [22.5]


def test_defensive_anchor_by_perimeter_pressure_excludes_anchor_steals() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "blocks_per_100": [0.4, 2.5, 0.8, 0.2, 1.0],
            "steals_per_100": [1.2, 1.5, 2.0, 0.5, 0.8],
        }
    )

    values = _defensive_anchor_by_perimeter_pressure([[1, 2, 3, 4, 5]], profiles)

    assert values.tolist() == [11.25]


def test_offensive_role_redundancy_uses_mean_pairwise_cosine_similarity() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "usage_per_100": [1.0, 1.0, 0.0, 0.0, 0.0],
            "assists_per_100": [0.0, 0.0, 1.0, 1.0, 0.0],
            "three_pa_per_100": [0.0, 0.0, 0.0, 0.0, 1.0],
            "unassisted_rim_makes_per_100": [0.0] * 5,
            "offensive_rebounds_per_100": [0.0] * 5,
            "free_throw_attempts_per_100": [0.0] * 5,
        }
    )

    values = _offensive_role_redundancy([[1, 2, 3, 4, 5]], profiles)

    assert values.tolist() == [0.2]


def test_source_profile_scales_use_possession_weighted_ninetieth_percentiles() -> None:
    source_profiles = pd.DataFrame({"player_id": [1, 2, 3], "usage_per_100": [1.0, 2.0, 4.0]})
    source_panel = pd.DataFrame({"player_id": [1, 2, 3], "rapm_possessions": [1.0, 1.0, 8.0]})

    scales = _source_profile_scales(source_profiles, source_panel, ("usage_per_100",))

    assert np.isclose(scales["usage_per_100"], 3.75)


def test_median_three_pm_candidate_uses_the_third_ranked_player() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "three_pm_per_100": [0.5, 1.0, 2.0, 3.0, 5.0],
        }
    )

    values = FEATURE_CANDIDATES["median_three_pm_per_100"].side_feature([(5, 1, 4, 2, 3)], profiles)

    assert values.tolist() == [2.0]


def test_rim_protection_ceiling_uses_the_highest_block_profile() -> None:
    profiles = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "blocks_per_100": [0.2, 1.1, 0.7, 3.4, 1.9],
        }
    )

    values = FEATURE_CANDIDATES["rim_protection_ceiling"].side_feature([(5, 1, 4, 2, 3)], profiles)

    assert values.tolist() == [3.4]


def test_teammate_continuity_uses_all_ten_prior_season_pairs() -> None:
    source_stints = pd.DataFrame(
        {
            "home_player_ids": [(1, 2, 3, 4, 5), (1, 2, 3, 4, 11)],
            "away_player_ids": [(6, 7, 8, 9, 10), (6, 7, 8, 9, 12)],
            "possessions": [10.0, 5.0],
        }
    )
    pair_exposure = build_teammate_pair_exposure(source_stints)

    values = teammate_continuity_side_feature(
        [(1, 2, 5, 11, 13), (20, 21, 22, 23, 24)], pair_exposure
    )

    expected = (
        np.log1p(15.0) + np.log1p(10.0) + np.log1p(5.0) + np.log1p(10.0) + np.log1p(5.0)
    ) / 10.0
    assert np.isclose(values[0], expected)
    assert values[1] == 0.0


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
