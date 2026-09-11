from __future__ import annotations

import pandas as pd
import pytest
from scipy.special import logit

from nba_lineup_model.rotation.forward_availability import (
    ForwardAvailabilityConfig,
    _require_full_roster_source_contract,
    fit_age_availability_model,
    predict_availability_roster,
    predict_availability_season,
    summarize_availability_metrics,
)


def _summary() -> pd.DataFrame:
    rows = []
    for season, year, available_a, available_b in (
        ("2020-21", 2020, 80, 40),
        ("2021-22", 2021, 78, 38),
        ("2022-23", 2022, 76, 36),
    ):
        for player_id, name, age, available, minutes in (
            (1, "Durable", 25 + year - 2020, available_a, 2200.0),
            (2, "Low availability", 25 + year - 2020, available_b, 800.0),
        ):
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
                    "total_nba_minutes": minutes,
                    "minutes_per_available_game": minutes / available,
                    "log_workload": 3.0,
                    "injury_or_illness_games": 82 - available,
                    "rest_games": 0,
                }
            )
    return pd.DataFrame(rows)


def test_forward_prediction_uses_only_prior_season_state() -> None:
    predictions, _age, _metadata = predict_availability_season(
        _summary(),
        target_season="2022-23",
        config=ForwardAvailabilityConfig(
            persistence=1.0,
            prior_strength=15.0,
            initial_prior_strength=15.0,
            workload_weight=0.0,
        ),
    )
    by_id = predictions.set_index("player_id")
    assert by_id.loc[1, "has_prior_availability_state"]
    assert by_id.loc[1, "predicted_available_share"] > by_id.loc[2, "predicted_available_share"]
    assert by_id.loc[1, "prior_gap_years"] == 1


def test_first_observed_season_is_shrunk_to_the_age_baseline() -> None:
    summary = _summary()
    summary.loc[
        (summary["player_id"].eq(2)) & (summary["season"].eq("2020-21")),
        "available_games",
    ] = 0
    summary["available_share"] = summary["available_games"] / summary["known_player_games"]
    predictions, _age, _metadata = predict_availability_season(
        summary,
        target_season="2021-22",
        config=ForwardAvailabilityConfig(
            persistence=1.0,
            prior_strength=15.0,
            initial_prior_strength=60.0,
            workload_weight=0.0,
        ),
    )
    low_availability_prediction = predictions.set_index("player_id").loc[
        2, "predicted_available_share"
    ]
    assert low_availability_prediction > 0.1


def test_prediction_exposes_exact_forward_component_logits() -> None:
    predictions, _age, _metadata = predict_availability_season(
        _summary(),
        target_season="2022-23",
        config=ForwardAvailabilityConfig(
            persistence=0.5,
            prior_strength=60.0,
            initial_prior_strength=15.0,
            workload_weight=-0.25,
        ),
    )

    component_sum = (
        predictions["age_baseline_logit"]
        + predictions["carried_state_residual_logit"]
        + predictions["carried_workload_adjustment_logit"]
    )
    assert logit(predictions["predicted_available_share"].to_numpy()) == pytest.approx(
        component_sum.to_numpy()
    )


def test_roster_forecast_requires_no_target_availability_outcomes() -> None:
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Durable", "Low availability", "Cold start"],
            "age": [28.0, 28.0, 20.0],
        }
    )
    predictions, _age, _metadata = predict_availability_roster(
        _summary(),
        roster=roster,
        target_season="2023-24",
        config=ForwardAvailabilityConfig(
            persistence=1.0,
            prior_strength=15.0,
            initial_prior_strength=15.0,
            workload_weight=0.0,
        ),
    )

    values = predictions.set_index("player_id")
    assert values.loc[1, "has_prior_availability_state"]
    assert not values.loc[3, "has_prior_availability_state"]
    assert values.loc[1, "predicted_available_share"] > values.loc[2, "predicted_available_share"]


def test_constant_baseline_ablates_age_for_cold_starts() -> None:
    roster = pd.DataFrame(
        {
            "player_id": [3, 4],
            "player_name": ["Young cold start", "Older cold start"],
            "age": [20.0, 35.0],
        }
    )

    predictions, _baseline, metadata = predict_availability_roster(
        _summary(),
        roster=roster,
        target_season="2023-24",
        config=ForwardAvailabilityConfig(
            persistence=0.5,
            prior_strength=60.0,
            initial_prior_strength=15.0,
            workload_weight=0.0,
            use_age_baseline=False,
        ),
    )

    assert not metadata["use_age_baseline"]
    assert predictions["predicted_available_share"].nunique() == 1


def test_age_model_keeps_rows_without_a_known_age_on_population_baseline() -> None:
    summary = _summary()
    summary.loc[summary["player_id"].eq(2), "age"] = float("nan")
    model = fit_age_availability_model(summary)
    predicted = model.predict_logit(summary["age"].to_numpy())
    assert predicted[1] == pytest.approx(model.population_logit)


def test_availability_metrics_are_exposure_weighted() -> None:
    predictions = pd.DataFrame(
        {
            "available_share": [1.0, 0.0],
            "predicted_available_share": [0.5, 0.5],
            "available_games": [10, 0],
            "known_player_games": [10, 1],
        }
    )
    metrics = summarize_availability_metrics(predictions)
    assert metrics["player_mae"] == 0.5
    assert metrics["weighted_brier"] == 0.25


def test_rejects_listed_player_source_contract(tmp_path) -> None:
    pd.DataFrame(
        {
            "source_kind": ["stats_v3"],
            "source_player_table_contract": ["boxscore_listed_players"],
        }
    ).to_parquet(tmp_path / "game_coverage.parquet", index=False)

    with pytest.raises(ValueError, match="complete roster denominator"):
        _require_full_roster_source_contract(tmp_path, season="2024-25")


def test_accepts_historical_summary_roster_contract(tmp_path) -> None:
    pd.DataFrame(
        {
            "source_kind": ["stats_v3_summary_v2"],
            "source_player_table_contract": ["full_roster_membership"],
        }
    ).to_parquet(tmp_path / "game_coverage.parquet", index=False)

    _require_full_roster_source_contract(tmp_path, season="2024-25")
