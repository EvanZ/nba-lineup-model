"""Tests for the forward same-team allocation-residual squash candidate."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nba_lineup_model.rotation.forward_allocation_residual_roster_squash import (
    AllocationResidualConfig,
    AllocationResidualInputs,
    apply_forward_allocation_residual_tilt,
    build_forward_allocation_residual_features,
    derive_allocation_residual_observations,
    summarize_allocation_residual_exposure,
)
from nba_lineup_model.rotation.forward_conditional_team_strength_roster_squash import (
    apply_production_roster_squash,
)


def _raw_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team": ["ONE", "ONE", "ONE"],
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
            "raw_expected_total_minutes": [100.0, 80.0, 70.0],
        }
    )


def _residual_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "player_id": [1, 2, 3],
            "same_team_return": [True, True, False],
            "prior_actual_minutes": [960.0, 480.0, 0.0],
            "prior_allocation_residual": [0.4, -0.4, 0.0],
            "residual_reliability": [0.8, 2.0 / 3.0, 0.0],
            "carried_allocation_residual": [0.32, -(0.4 * 2.0 / 3.0), 0.0],
        }
    )


def test_zero_residual_beta_exactly_recovers_production_gate() -> None:
    raw = _raw_predictions()
    output = apply_forward_allocation_residual_tilt(
        raw,
        residual_features=_residual_features(),
        config=AllocationResidualConfig(residual_beta=0.0, prior_minutes=240.0),
        initial_rotation_size=2,
    )
    production = apply_production_roster_squash(raw, initial_rotation_size=2)

    assert output.set_index("player_id")["allocation_residual_tilt"].to_dict() == pytest.approx(
        {1: 1.0, 2: 1.0, 3: 1.0}
    )
    assert output.set_index("player_id")["is_rotation_candidate"].to_dict() == (
        production.set_index("player_id")["is_rotation_candidate"].to_dict()
    )
    assert_frame_equal(
        output.loc[:, ["player_id", "raw_expected_total_minutes"]]
        .sort_values("player_id")
        .reset_index(drop=True),
        production.loc[:, ["player_id", "raw_expected_total_minutes"]]
        .sort_values("player_id")
        .reset_index(drop=True),
    )


def test_positive_residual_can_change_the_selected_rotation() -> None:
    output = apply_forward_allocation_residual_tilt(
        _raw_predictions(),
        residual_features=_residual_features(),
        config=AllocationResidualConfig(residual_beta=2.0, prior_minutes=240.0),
        initial_rotation_size=2,
    ).set_index("player_id")

    assert output.loc[1, "allocation_residual_tilt"] > 1.0
    assert output.loc[2, "allocation_residual_tilt"] < 1.0
    assert bool(output.loc[1, "is_rotation_candidate"])
    assert not bool(output.loc[2, "is_rotation_candidate"])
    assert bool(output.loc[3, "is_rotation_candidate"])


def test_only_a_same_team_return_receives_a_carried_residual() -> None:
    inputs = AllocationResidualInputs(
        season="2025-26",
        opening_roster=pd.DataFrame(
            {
                "team_id": [1, 2],
                "team": ["ONE", "TWO"],
                "player_id": [11, 22],
                "player_name": ["Returner", "Mover"],
            }
        ),
        static_roster_bios=pd.DataFrame(),
        target_game_minutes=pd.DataFrame(),
    )
    observations = pd.DataFrame(
        {
            "season_start_year": [2024, 2024],
            "team_id": [1, 1],
            "player_id": [11, 22],
            "allocation_residual": [0.6, 0.8],
            "actual_minutes": [720.0, 720.0],
        }
    )

    features = build_forward_allocation_residual_features(
        inputs,
        residual_observations=observations,
        config=AllocationResidualConfig(residual_beta=1.0, prior_minutes=240.0),
    ).set_index("player_id")

    assert bool(features.loc[11, "same_team_return"])
    assert features.loc[11, "residual_reliability"] == pytest.approx(0.75)
    assert features.loc[11, "carried_allocation_residual"] == pytest.approx(0.45)
    assert not bool(features.loc[22, "same_team_return"])
    assert features.loc[22, "carried_allocation_residual"] == 0.0


def test_residual_observations_exclude_trades_and_unstable_rows() -> None:
    predictions = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25"],
            "season_start_year": [2024, 2024, 2024],
            "team_id": [1, 1, 1],
            "team": ["ONE", "ONE", "ONE"],
            "player_id": [1, 2, 3],
            "player_name": ["Eligible", "Traded", "Outside Rotation"],
            "actual_minute_share": [0.2, 0.2, 0.2],
            "predicted_minute_share": [0.1, 0.1, 0.1],
            "actual_minutes": [960.0, 960.0, 960.0],
            "is_trade_contaminated": [False, True, False],
            "has_full_team_roster_coverage": [True, True, True],
            "was_selected_rotation": [True, True, False],
            "is_opening_roster_player": [True, True, True],
        }
    )

    observations = derive_allocation_residual_observations(predictions)

    assert observations["player_id"].tolist() == [1]
    assert observations.loc[0, "allocation_residual"] == pytest.approx(0.69314718056)


def test_residual_exposure_diagnostics_partition_the_observations() -> None:
    observations = pd.DataFrame(
        {
            "actual_minutes": [240.0, 480.0, 960.0, 1_920.0],
            "allocation_residual": [-0.4, -0.2, 0.2, 0.4],
        }
    )

    diagnostics = summarize_allocation_residual_exposure(observations, bins=2)

    assert diagnostics["observation_count"].sum() == 4
    assert diagnostics["mean_actual_minutes"].is_monotonic_increasing
