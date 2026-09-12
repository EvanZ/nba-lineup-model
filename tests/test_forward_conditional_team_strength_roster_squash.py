"""Tests for the full incumbent-team-strength roster-squash candidate."""

from __future__ import annotations

import pandas as pd

from nba_lineup_model.rotation.forward_conditional_team_strength_roster_squash import (
    TEAM_STRENGTH_COLUMN,
    TeamStrengthRosterInputs,
    apply_production_roster_squash,
    attach_incumbent_team_strength_to_roster,
    paired_team_bootstrap,
)
from nba_lineup_model.rotation.forward_nail_roster_squash import (
    build_forward_conditional_raw_weights,
)


def test_roster_attachment_uses_league_fallback_for_missing_team() -> None:
    inputs = TeamStrengthRosterInputs(
        season="2025-26",
        opening_roster=pd.DataFrame(),
        static_roster_bios=pd.DataFrame(
            {
                "team_id": [1, 2],
                "player_id": [11, 22],
                "player_name": ["One", "Two"],
            }
        ),
        target_game_minutes=pd.DataFrame(),
    )
    diagnostics = pd.DataFrame(
        {
            "season": ["2025-26"],
            "team_id": [1],
            TEAM_STRENGTH_COLUMN: [2.0],
            "league_strength_fallback": [0.5],
        }
    )

    attached = attach_incumbent_team_strength_to_roster(
        inputs, team_strength_diagnostics=diagnostics
    )

    assert attached.static_roster_bios[TEAM_STRENGTH_COLUMN].tolist() == [2.0, 0.5]


def test_production_roster_squash_only_gates_raw_weights() -> None:
    raw = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
            "raw_expected_total_minutes": [30.0, 20.0, 10.0],
            "is_rookie": [True, False, True],
        }
    )

    squashed = apply_production_roster_squash(raw, initial_rotation_size=2)

    assert squashed["raw_expected_total_minutes"].tolist() == [30.0, 20.0, 0.0]
    assert squashed["is_rotation_candidate"].tolist() == [True, True, False]


def test_raw_weight_merge_retains_the_opening_roster_player_name(monkeypatch) -> None:
    inputs = TeamStrengthRosterInputs(
        season="2025-26",
        opening_roster=pd.DataFrame(),
        static_roster_bios=pd.DataFrame(
            {
                "team_id": [1],
                "team": ["ONE"],
                "player_id": [11],
                "player_name": ["Opening Name"],
                "age": [22.0],
            }
        ),
        target_game_minutes=pd.DataFrame(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.rotation.forward_nail_roster_squash.predict_availability_roster",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"player_id": [11], "predicted_available_share": [0.8]}),
            None,
            {},
        ),
    )
    monkeypatch.setattr(
        "nba_lineup_model.rotation.forward_nail_roster_squash.predict_conditional_minutes_roster",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "player_id": [11],
                    "player_name": ["Model Name"],
                    "predicted_minutes_per_available_game": [20.0],
                }
            ),
            None,
        ),
    )

    raw = build_forward_conditional_raw_weights(inputs, availability_summary=pd.DataFrame())

    assert raw["player_name"].tolist() == ["Opening Name"]


def test_paired_team_bootstrap_is_reproducible_and_stratified() -> None:
    comparison = pd.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2024-25", "2024-25"],
            "allocation_total_variation_delta": [-0.02, -0.01, -0.03, -0.01],
            "brier_score_delta": [-0.002, -0.001, -0.003, -0.001],
            "player_share_mae_delta": [-0.002, -0.001, -0.003, -0.001],
            "player_share_mse_delta": [-0.0002, -0.0001, -0.0003, -0.0001],
            "actual_top_rotation_overlap_delta": [0.0, 0.0, 0.1, 0.0],
        }
    )

    first = paired_team_bootstrap(comparison, draws=100, seed=7)
    second = paired_team_bootstrap(comparison, draws=100, seed=7)

    pd.testing.assert_frame_equal(first, second)
    primary = first.loc[
        (first["scope"] == "pooled_frozen")
        & (first["metric"] == "allocation_total_variation")
    ].iloc[0]
    assert primary["point_estimate_candidate_minus_control"] < 0.0
    assert primary["candidate_better_probability"] > 0.99
