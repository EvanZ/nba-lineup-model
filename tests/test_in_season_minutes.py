from __future__ import annotations

import json

import pandas as pd
import pytest

from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    INACTIVE,
    MEDICAL_ABSENCE,
    PLAYED,
    UNKNOWN,
    InSeasonMinutesConfig,
    advance_minutes_state,
    calibrate_prior_games,
    contribution_sha256,
    forecast_team_minutes,
    initialize_minutes_state,
    publish_minutes_contribution,
    regulation_equivalent_minutes,
)
from nba_lineup_model.modeling.in_season_minutes_monitor import build_game_observations


def _config() -> InSeasonMinutesConfig:
    return InSeasonMinutesConfig(
        updater_version="test-v1",
        participation_prior_games=8.0,
        conditional_prior_games=4.0,
        rotation_size=2,
    )


def _priors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
            "team": ["AAA", "AAA", "AAA"],
            "availability_probability": [0.8, 0.9, 0.5],
            "conditional_minutes_per_game": [30.0, 20.0, 10.0],
            "baseline_minutes_per_game": [144.0, 96.0, 0.0],
        }
    )


def _observations(game_id: str = "001") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "player_id": [1, 2, 3],
            "outcome_classification": [PLAYED, MEDICAL_ABSENCE, UNKNOWN],
            "played": [True, False, False],
            "regulation_equivalent_minutes": [36.0, 0.0, 0.0],
            "conditional_observation_weight": [1.0, 0.0, 0.0],
        }
    )


def test_zero_evidence_reproduces_frozen_preseason_distribution() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[],
        report_status="submitted",
        config=_config(),
    ).set_index("player_id")

    assert forecast.loc[1, "final_normalized_minutes"] == pytest.approx(144.0)
    assert forecast.loc[2, "final_normalized_minutes"] == pytest.approx(96.0)
    assert forecast.loc[3, "final_normalized_minutes"] == pytest.approx(0.0)
    assert forecast["final_normalized_minutes"].sum() == pytest.approx(240.0)


def test_injury_absence_does_not_erase_conditional_role() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    updated = advance_minutes_state(
        state,
        _observations(),
        config=_config(),
        contribution_sha256="a" * 64,
    ).set_index("player_id")

    assert updated.loc[2, "current_participation_estimate"] < 0.9
    assert updated.loc[2, "current_conditional_minutes_estimate"] == pytest.approx(20.0)
    assert updated.loc[2, "conditional_observations"] == 0
    assert updated.loc[3, "current_participation_estimate"] == pytest.approx(0.5)


@pytest.mark.parametrize("classification", [COACH_DNP, INACTIVE])
def test_known_nonparticipants_update_participation_only(classification: str) -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    observation = _observations().iloc[[0]].copy()
    observation["player_id"] = 2
    observation["outcome_classification"] = classification
    observation["played"] = False
    observation["regulation_equivalent_minutes"] = 0.0
    updated = advance_minutes_state(
        state, observation, config=_config(), contribution_sha256="b" * 64
    ).set_index("player_id")
    assert updated.loc[2, "current_participation_estimate"] < 0.9
    assert updated.loc[2, "current_conditional_minutes_estimate"] == pytest.approx(20.0)


def test_overtime_minutes_are_regulation_equivalent() -> None:
    normalized = regulation_equivalent_minutes(
        pd.Series([50.0, 50.0, 50.0, 50.0, 50.0]), team_total_minutes=250.0
    )
    assert normalized.sum() == pytest.approx(240.0)
    assert normalized.tolist() == pytest.approx([48.0] * 5)


def test_out_player_gets_zero_and_remaining_minutes_conserve_240() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[{"player_id": 1, "status": "Out", "reason": "Ankle"}],
        report_status="submitted",
        config=_config(),
    ).set_index("player_id")
    assert forecast.loc[1, "final_normalized_minutes"] == 0.0
    assert forecast["final_normalized_minutes"].sum() == pytest.approx(240.0)
    assert forecast.loc[1, "injury_reason"] == "Ankle"


def test_report_pending_is_preserved_without_available_claim() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[],
        report_status="not_yet_submitted",
        config=_config(),
    )
    assert set(forecast["injury_status"]) == {"Report pending"}
    assert forecast["game_participation_probability"].isna().all()
    assert forecast.set_index("player_id").loc[1, "final_normalized_minutes"] == pytest.approx(
        144.0
    )


def test_incremental_and_full_replay_match_and_duplicate_is_rejected() -> None:
    initial = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    first = _observations("001")
    second = _observations("002")
    incremental = advance_minutes_state(
        advance_minutes_state(
            initial, first, config=_config(), contribution_sha256="1" * 64
        ),
        second,
        config=_config(),
        contribution_sha256="2" * 64,
    )
    full = advance_minutes_state(
        initial,
        pd.concat([first, second], ignore_index=True),
        config=_config(),
        contribution_sha256="3" * 64,
    )
    columns = [
        "player_id",
        "current_participation_estimate",
        "current_conditional_minutes_estimate",
        "participation_observations",
        "conditional_observations",
    ]
    pd.testing.assert_frame_equal(
        incremental[columns].reset_index(drop=True), full[columns].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="cannot be applied twice"):
        advance_minutes_state(
            incremental, second, config=_config(), contribution_sha256="2" * 64
        )


def test_content_addressed_contribution_is_idempotent_and_revision_is_new(tmp_path) -> None:
    observations = _observations()
    metadata = {"game_id": "001", "source_sha256": "s1", "finalized_at": "now"}
    first, created, digest = publish_minutes_contribution(observations, metadata, tmp_path)
    second, created_again, same_digest = publish_minutes_contribution(
        observations, metadata, tmp_path
    )
    assert created is True
    assert created_again is False
    assert first == second
    assert digest == same_digest == contribution_sha256(observations, metadata)
    revised = observations.copy()
    revised.loc[0, "regulation_equivalent_minutes"] = 35.0
    third, revised_created, revised_digest = publish_minutes_contribution(
        revised, {**metadata, "source_sha256": "s2"}, tmp_path
    )
    assert revised_created is True
    assert third != first
    assert revised_digest != digest


def test_calibration_replay_is_strictly_prior_day() -> None:
    prior = pd.DataFrame(
        {
            "game_id": ["p1", "p1"],
            "game_date": pd.to_datetime(["2025-01-01", "2025-01-01"], utc=True),
            "team": ["AAA", "AAA"],
            "player_id": [1, 2],
            "minutes": [30.0, 18.0],
            "availability_state": [PLAYED, PLAYED],
        }
    )
    target = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "game_date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], utc=True
            ),
            "team": ["AAA"] * 4,
            "player_id": [1, 2, 1, 2],
            "minutes": [36.0, 0.0, 34.0, 20.0],
            "availability_state": [PLAYED, COACH_DNP, PLAYED, PLAYED],
        }
    )
    result = calibrate_prior_games(
        prior,
        target,
        participation_grid=(1.0, 4.0),
        conditional_grid=(1.0, 4.0),
    )
    assert result["target_player_games"] == 4
    assert result["participation_prior_games"] in {1.0, 4.0}
    assert result["conditional_prior_games"] in {1.0, 4.0}
    assert json.loads(json.dumps(result)) == result


def test_final_boxscore_classifies_injury_and_coach_dnp_without_role_zero() -> None:
    priors = _priors()
    summary = {
        "game_id": "001",
        "game_date": "2026-10-20",
        "home_team": "AAA",
        "away_team": "BBB",
        "pregame": {
            "projected_minutes": {
                "home": [
                    {
                        "player_id": 2,
                        "status": "Out",
                        "reason": "Injury/Illness - Ankle",
                    }
                ],
                "away": [],
            }
        },
    }
    espn = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "AAA"},
                    "statistics": [
                        {
                            "labels": ["MIN"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "One"},
                                    "stats": ["48"],
                                    "didNotPlay": False,
                                    "ejected": False,
                                },
                                {
                                    "athlete": {"displayName": "Two"},
                                    "stats": ["0"],
                                    "didNotPlay": True,
                                    "reason": "DNP-INJURY/ILLNESS",
                                    "ejected": False,
                                },
                                {
                                    "athlete": {"displayName": "Three"},
                                    "stats": ["0"],
                                    "didNotPlay": True,
                                    "reason": "COACH'S DECISION",
                                    "ejected": False,
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }
    observations = build_game_observations(summary, espn, priors=priors).set_index(
        "player_id"
    )
    assert observations.loc[1, "outcome_classification"] == PLAYED
    assert observations.loc[2, "outcome_classification"] == MEDICAL_ABSENCE
    assert observations.loc[3, "outcome_classification"] == COACH_DNP
    assert observations.loc[1, "regulation_equivalent_minutes"] == pytest.approx(240.0)
