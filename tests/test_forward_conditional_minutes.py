from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.forward_conditional_minutes import (
    AGE_ONLY_CONFIG,
    ForwardConditionalMinutesConfig,
    attach_expected_total_minutes,
    normalize_opening_roster_minutes,
    predict_conditional_minutes_roster,
    predict_conditional_minutes_season,
)


def _summary() -> pd.DataFrame:
    rows = []
    for season, year, one_minutes, two_minutes in (
        ("2020-21", 2020, 30.0, 8.0),
        ("2021-22", 2021, 31.0, 9.0),
        ("2022-23", 2022, 32.0, 10.0),
    ):
        for player_id, name, age, mpg in (
            (1, "One", 25 + year - 2020, one_minutes),
            (2, "Two", 25 + year - 2020, two_minutes),
        ):
            available = 60 if player_id == 1 else 20
            rows.append(
                {
                    "season": season,
                    "season_start_year": year,
                    "player_id": player_id,
                    "player_name": name,
                    "age": float(age),
                    "available_games": available,
                    "known_player_games": 82,
                    "available_share": available / 82,
                    "total_nba_minutes": available * mpg,
                    "minutes_per_available_game": mpg,
                }
            )
    return pd.DataFrame(rows)


def test_forward_conditional_minutes_uses_prior_state_only() -> None:
    predictions, _age = predict_conditional_minutes_season(
        _summary(),
        target_season="2022-23",
        config=ForwardConditionalMinutesConfig(1.0, 15.0, 15.0),
    )
    values = predictions.set_index("player_id")
    assert values.loc[1, "has_prior_minutes_state"]
    assert (
        values.loc[1, "predicted_minutes_per_available_game"]
        > values.loc[2, "predicted_minutes_per_available_game"]
    )
    assert values.loc[1, "prior_gap_years"] == 1


def test_age_only_control_does_not_retain_player_role_state() -> None:
    predictions, _age = predict_conditional_minutes_season(
        _summary(), target_season="2022-23", config=AGE_ONLY_CONFIG
    )
    values = predictions.set_index("player_id")
    assert values.loc[1, "predicted_minutes_per_available_game"] == pytest.approx(
        values.loc[2, "predicted_minutes_per_available_game"]
    )


def test_roster_forecast_requires_no_target_minutes_outcomes() -> None:
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Cold start"],
            "age": [28.0, 28.0, 20.0],
        }
    )
    predictions, _age = predict_conditional_minutes_roster(
        _summary(),
        roster=roster,
        target_season="2023-24",
        config=ForwardConditionalMinutesConfig(1.0, 15.0, 15.0),
    )

    values = predictions.set_index("player_id")
    assert values.loc[1, "has_prior_minutes_state"]
    assert not values.loc[3, "has_prior_minutes_state"]
    assert values.loc[1, "predicted_minutes_per_available_game"] > values.loc[
        2, "predicted_minutes_per_available_game"
    ]


def test_opening_roster_normalization_conserves_team_minutes() -> None:
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "raw_expected_total_minutes": [1_000.0, 2_000.0, 500.0],
        }
    )
    roster = pd.DataFrame(
        {
            "team_id": [1, 1, 2],
            "team": ["ONE", "ONE", "TWO"],
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
        }
    )
    normalized = normalize_opening_roster_minutes(predictions, opening_roster=roster)
    totals = normalized.groupby("team_id")["projected_total_minutes"].sum()
    assert np.isclose(totals, 82.0 * 240.0).all()
    assert normalized.loc[normalized["player_id"].eq(2), "projected_minute_share"].iloc[
        0
    ] == pytest.approx(2 / 3)


def test_combined_expected_minutes_multiplies_both_forward_components() -> None:
    conditional = pd.DataFrame(
        {
            "player_id": [1],
            "predicted_minutes_per_available_game": [20.0],
            "total_nba_minutes": [1_000.0],
        }
    )
    availability = pd.DataFrame({"player_id": [1], "predicted_available_share": [0.5]})
    combined = attach_expected_total_minutes(conditional, availability)
    assert combined.loc[0, "raw_expected_total_minutes"] == pytest.approx(820.0)
