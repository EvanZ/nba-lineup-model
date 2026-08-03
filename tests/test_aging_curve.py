from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.aging import (
    prepare_aging_transitions,
    run_aging_experiment,
)
from nba_lineup_model.modeling.aging_curve import (
    extract_partial_age_curve,
    season_block_bootstrap_age_curve,
)


def test_age_curve_is_centered_and_bootstrap_draws_match_the_grid():
    transitions = synthetic_transitions()
    experiment = run_aging_experiment(
        transitions,
        regularization_grid=(0.01, 1.0),
        age_spline_knots=3,
    )
    prepared = prepare_aging_transitions(transitions)
    training = prepared.loc[
        prepared["target_season"].isin(experiment.training_target_seasons)
    ].copy()

    curve = extract_partial_age_curve(
        experiment.fitted_model,
        training,
        reference_age=27,
    )
    draws, annual_draws = season_block_bootstrap_age_curve(
        training,
        regularization=experiment.selected_regularization,
        age_spline_knots=3,
        age_spline_degree=2,
        reference_age=27,
        bootstrap_samples=8,
        bootstrap_seed=7,
        ages=curve["age"].to_numpy(dtype=float),
    )

    assert curve.loc[curve["age"].eq(27), "partial_age_effect"].item() == 0.0
    assert curve["annual_change"].iloc[-1] != curve["annual_change"].iloc[-1]
    assert draws.shape == (8, len(curve))
    assert annual_draws.shape == (8, len(curve))
    assert np.isfinite(draws).all()


def test_draft_adjusted_curves_are_centered_at_the_reference_age():
    transitions = prepare_aging_transitions(synthetic_transitions())
    experiment = run_aging_experiment(
        transitions,
        regularization_grid=(0.01, 1.0),
        age_spline_knots=3,
    )
    training = transitions.loc[
        transitions["target_season"].isin(experiment.training_target_seasons)
    ].copy()

    for cohort in ("early_entry", "late_entry", "undrafted"):
        curve = extract_partial_age_curve(
            experiment.fitted_model,
            training,
            reference_age=27,
            draft_cohort=cohort,
        )
        assert curve.loc[curve["age"].eq(27), "partial_age_effect"].item() == 0.0


def synthetic_transitions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_index, season in enumerate(("2020-21", "2021-22", "2022-23", "2023-24")):
        year = int(season[:4])
        prior_season = f"{year - 1}-{str(year)[-2:]}"
        for player_id in range(1, 18):
            age = float(20 + (player_id % 13) + season_index)
            cold_start = player_id % 8 == 0
            prior_rapm = np.nan if cold_start else -1.5 + player_id * 0.19
            rows.append(
                {
                    "target_season": season,
                    "prior_season": prior_season,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "target_age": age,
                    "target_nba_experience_years": 0 if cold_start else max(1, int(age - 21)),
                    "is_rookie": cold_start,
                    "has_prior_season": not cold_start,
                    "prior_rapm": prior_rapm,
                    "prior_rapm_possessions": np.nan if cold_start else 700.0 + 10 * player_id,
                    "target_rapm": (0.0 if cold_start else 0.7 * float(prior_rapm))
                    + 0.16 * (27.0 - age)
                    - 0.012 * (age - 27.0) ** 2,
                    "target_rapm_possessions": 500.0 + 15 * player_id,
                    "target_rapm_seconds": 7_000.0 + 20 * player_id,
                    "target_rapm_exposure_eligible": True,
                }
            )
    return pd.DataFrame(rows)
