"""Tests for the nested preseason coach-signals candidate."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nba_lineup_model.rotation import forward_nail_roster_squash
from nba_lineup_model.rotation.forward_coach_signal_roster_squash import (
    CoachSignalConfig,
    apply_forward_coach_signals,
    build_coach_signal_features,
    build_live_forward_coach_signal_inputs,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    build_forward_conditional_roster_profile,
    normalize_opening_roster_minutes,
)
from nba_lineup_model.web_api import preseason_minutes
from nba_lineup_model.web_api.preseason_minutes import (
    _apply_raw_expected_total_rotation_limit,
    build_forward_conditional_preseason_minutes_payload,
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


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team": ["ONE", "ONE", "ONE"],
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
            "start_rate_z": [-1.0, 0.0, 1.0],
            "draft_investment_z": [-1.0, 0.0, 1.0],
        }
    )


def test_zero_coach_signal_weights_exactly_recover_production_rotation_and_squash() -> None:
    raw = _raw_predictions()
    opening_roster = raw.loc[:, ["team_id", "team", "player_id", "player_name"]]
    output = apply_forward_coach_signals(
        raw,
        coach_signals=_signals(),
        config=CoachSignalConfig(0.0, 0.0, 15.0, 2.0),
        initial_rotation_size=2,
    )
    production = _apply_raw_expected_total_rotation_limit(
        raw,
        opening_roster=opening_roster,
        size=2,
    )

    assert output.set_index("player_id")["coach_signal_tilt"].to_dict() == pytest.approx(
        {1: 1.0, 2: 1.0, 3: 1.0}
    )
    assert output.set_index("player_id")["is_rotation_candidate"].to_dict() == (
        production.set_index("player_id")["is_baseline_rotation_candidate"].to_dict()
    )
    assert_frame_equal(
        output.loc[:, ["player_id", "raw_expected_total_minutes"]]
        .sort_values("player_id")
        .reset_index(drop=True),
        production.loc[:, ["player_id", "raw_expected_total_minutes"]]
        .sort_values("player_id")
        .reset_index(drop=True),
    )
    assert_frame_equal(
        normalize_opening_roster_minutes(output, opening_roster=opening_roster)
        .sort_values("player_id")
        .reset_index(drop=True),
        normalize_opening_roster_minutes(production, opening_roster=opening_roster)
        .sort_values("player_id")
        .reset_index(drop=True),
    )


def test_positive_coach_signal_weights_can_promote_a_high_signal_player() -> None:
    output = apply_forward_coach_signals(
        _raw_predictions(),
        coach_signals=_signals(),
        config=CoachSignalConfig(1.0, 1.0, 15.0, 2.0),
        initial_rotation_size=2,
    ).set_index("player_id")

    assert output.loc[3, "coach_signal_tilt"] > output.loc[1, "coach_signal_tilt"]
    assert bool(output.loc[3, "is_rotation_candidate"])
    assert not bool(output.loc[1, "is_rotation_candidate"])


def test_live_inputs_use_the_exact_production_roster_profile(tmp_path) -> None:
    roster_path = tmp_path / "roster.parquet"
    roster = pd.DataFrame(
        {
            "team_id": [1, 1],
            "team_abbreviation": ["ONE", "ONE"],
            "player_id": [1, 2],
            "player_name": ["Veteran", "Lottery Rookie"],
            "listed_position": ["G", "F"],
            "age": [28.0, 20.0],
            "experience": ["5", "R"],
        }
    )
    roster.to_parquet(roster_path, index=False)
    curated_dir = tmp_path / "curated"
    draft_path = curated_dir / "draft_history" / "2026-27" / "part-00000.parquet"
    draft_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "player_id": [2],
            "player_name": ["Lottery Rookie"],
            "draft_year": [2026],
            "draft_number": [11],
        }
    ).to_parquet(draft_path, index=False)
    panel = pd.DataFrame(
        {
            "season_start_year": [2025, 2025],
            "player_id": [1, 2],
            "games": [70, 0],
            "games_started": [50, 0],
        }
    )

    inputs = build_live_forward_coach_signal_inputs(
        season="2026-27",
        roster_path=roster_path,
        panel=panel,
        curated_dir=curated_dir,
    )
    production_profile = build_forward_conditional_roster_profile(
        roster, target_season="2026-27", curated_dir=curated_dir
    ).merge(
        pd.DataFrame({"player_id": [2], "draft_year": [2026]}),
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    profile_columns = [
        "player_id",
        "age",
        "listed_position",
        "is_rookie",
        "draft_number",
        "is_undrafted",
        "draft_year",
    ]

    assert_frame_equal(
        inputs.static_roster_bios.loc[:, profile_columns]
        .sort_values("player_id")
        .reset_index(drop=True),
        production_profile.loc[:, profile_columns]
        .sort_values("player_id")
        .reset_index(drop=True),
    )


def test_zero_coach_signal_matches_the_full_production_forecast(tmp_path, monkeypatch) -> None:
    roster_path = tmp_path / "roster.parquet"
    roster = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team_abbreviation": ["ONE", "ONE", "ONE"],
            "player_id": [1, 2, 3],
            "player_name": ["Veteran", "Lottery Rookie", "Reserve"],
            "listed_position": ["G", "F", "C"],
            "age": [28.0, 20.0, 25.0],
            "experience": ["5", "R", "2"],
        }
    )
    roster.to_parquet(roster_path, index=False)
    curated_dir = tmp_path / "curated"
    draft_path = curated_dir / "draft_history" / "2026-27" / "part-00000.parquet"
    draft_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "player_id": [2],
            "player_name": ["Lottery Rookie"],
            "draft_year": [2026],
            "draft_number": [11],
        }
    ).to_parquet(draft_path, index=False)
    summary = pd.DataFrame(
        {
            "season_start_year": [2025, 2025, 2025],
            "player_id": [1, 2, 3],
            "total_nba_minutes": [2_000.0, 0.0, 500.0],
        }
    )
    panel = pd.DataFrame(
        {
            "season_start_year": [2025, 2025, 2025],
            "player_id": [1, 2, 3],
            "games": [70, 0, 40],
            "games_started": [50, 0, 0],
        }
    )
    availability = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "predicted_available_share": [0.8, 0.75, 0.7],
            "has_prior_availability_state": [True, False, True],
        }
    )
    conditional = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "predicted_minutes_per_available_game": [30.0, 24.0, 15.0],
            "has_prior_minutes_state": [True, False, True],
        }
    )

    def predict_availability(*_args, **_kwargs):
        return availability, None, {}

    def predict_conditional(*_args, **_kwargs):
        return conditional, None

    monkeypatch.setattr(
        preseason_minutes, "build_availability_season_summary", lambda *_a, **_k: summary
    )
    monkeypatch.setattr(preseason_minutes, "attach_cold_start_biographies", lambda frame: frame)
    monkeypatch.setattr(preseason_minutes, "predict_availability_roster", predict_availability)
    monkeypatch.setattr(
        preseason_minutes, "predict_conditional_minutes_roster", predict_conditional
    )
    monkeypatch.setattr(
        forward_nail_roster_squash, "predict_availability_roster", predict_availability
    )
    monkeypatch.setattr(
        forward_nail_roster_squash,
        "predict_conditional_minutes_roster",
        predict_conditional,
    )

    rankings = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "rapm": [1.0, 0.0, -1.0],
            "profile_source": ["prior", "draft", "prior"],
        }
    )
    production = build_forward_conditional_preseason_minutes_payload(
        preseason_rankings=rankings,
        roster_path=roster_path,
        curated_dir=curated_dir,
        initial_rotation_size=2,
    )
    inputs = build_live_forward_coach_signal_inputs(
        season="2026-27",
        roster_path=roster_path,
        panel=panel,
        curated_dir=curated_dir,
    )
    raw = forward_nail_roster_squash.build_forward_conditional_raw_weights(
        inputs, availability_summary=summary
    )
    config = CoachSignalConfig(0.0, 0.0, 5.0, 2.0)
    adjusted = apply_forward_coach_signals(
        raw,
        coach_signals=build_coach_signal_features(inputs, config=config),
        config=config,
        initial_rotation_size=2,
    )
    candidate = normalize_opening_roster_minutes(
        adjusted, opening_roster=inputs.opening_roster
    ).loc[:, ["player_id", "projected_minute_share", "projected_total_minutes"]]
    production_minutes = pd.DataFrame(production["players"])
    production_minutes = production_minutes.loc[
        :, ["player_id", "baseline_minute_share", "baseline_minutes_per_game"]
    ].rename(
        columns={
            "baseline_minute_share": "projected_minute_share",
            "baseline_minutes_per_game": "projected_minutes_per_game",
        }
    )
    candidate["projected_minutes_per_game"] = candidate["projected_total_minutes"] / 82.0

    assert_frame_equal(
        candidate.loc[
            :, ["player_id", "projected_minute_share", "projected_minutes_per_game"]
        ]
        .sort_values("player_id")
        .reset_index(drop=True),
        production_minutes.sort_values("player_id").reset_index(drop=True),
    )
