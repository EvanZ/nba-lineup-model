from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nba_lineup_model.modeling.bayesian import fit_bayesian_rapm_experiment
from nba_lineup_model.models.baselines import RidgeLineupModel, signed_entity_matrix
from nba_lineup_model.models.bayesian import ConjugateBayesianRidge


def test_conjugate_posterior_location_matches_weighted_ridge() -> None:
    features = sparse.csr_matrix(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    target = np.array([8.0, 5.0, -2.0, -7.0, -4.0, 3.0, 10.0, -9.0])
    weights = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 3.5])
    regularization = 0.2

    ridge = RidgeLineupModel(regularization).fit(features, target, weights)
    posterior = ConjugateBayesianRidge.fit(
        features,
        target,
        weights,
        regularization,
    )

    assert posterior.intercept_mean == pytest.approx(ridge.intercept_, abs=1e-8)
    assert posterior.coefficient_mean == pytest.approx(ridge.coef_, abs=1e-8)
    assert posterior.ridge_alpha == pytest.approx(ridge.sklearn_alpha)
    assert posterior.predict_mean(features) == pytest.approx(
        ridge.predict(features),
        abs=1e-8,
    )


def test_conjugate_posterior_summaries_and_draws_are_well_formed() -> None:
    features = sparse.csr_matrix(
        [
            [1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
    target = np.array([12.0, 8.0, -9.0, -11.0, 10.0, -8.0])
    weights = np.array([1.0, 2.0, 1.0, 2.0, 3.0, 3.0])
    posterior = ConjugateBayesianRidge.fit(features, target, weights, 0.1)

    marginal = posterior.marginal_summary(interval_probability=0.90)
    first = posterior.draw_parameters(100, seed=11)
    second = posterior.draw_parameters(100, seed=11)
    predictive = posterior.predictive_summary(
        features,
        weights,
        interval_probability=0.90,
    )

    assert np.array_equal(first, second)
    assert first.shape == (100, 3)
    assert (marginal.lower < marginal.mean).all()
    assert (marginal.mean < marginal.upper).all()
    assert ((marginal.probability_positive >= 0) & (marginal.probability_positive <= 1)).all()
    assert posterior.residual_variance_mean > 0
    assert (predictive.lower < predictive.mean).all()
    assert (predictive.mean < predictive.upper).all()


def test_bayesian_rapm_experiment_reuses_ridge_ranking_and_test_contract() -> None:
    stints = _bayesian_stints()
    player_columns = {identifier: column for column, identifier in enumerate(range(1, 7))}
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    regularization = 0.05
    full_ridge = RidgeLineupModel(regularization).fit(matrix, target, weights)
    rankings = _ridge_rankings(full_ridge.coef_)
    final_train_games = {f"{index:010d}" for index in range(16)}
    final_test_games = {f"{index:010d}" for index in range(16, 20)}
    game_splits = pd.DataFrame(
        [
            {"split": "final", "role": "train", "game_id": game_id}
            for game_id in sorted(final_train_games)
        ]
        + [
            {"split": "final", "role": "test", "game_id": game_id}
            for game_id in sorted(final_test_games)
        ]
    )
    train_mask = stints["game_id"].isin(final_train_games).to_numpy()
    test_mask = stints["game_id"].isin(final_test_games).to_numpy()
    test_ridge = RidgeLineupModel(regularization).fit(
        matrix[train_mask],
        target[train_mask],
        weights[train_mask],
    )
    ridge_test_predictions = stints.loc[
        test_mask,
        ["game_id", "stint_index"],
    ].copy()
    ridge_test_predictions["prediction_rapm"] = test_ridge.predict(matrix[test_mask])

    experiment = fit_bayesian_rapm_experiment(
        stints,
        player_columns=player_columns,
        ridge_rankings=rankings,
        game_splits=game_splits,
        ridge_test_predictions=ridge_test_predictions,
        ridge_intercept=full_ridge.intercept_,
        selected_lambda=regularization,
        posterior_draws=200,
        posterior_seed=9,
        credible_interval_probability=0.90,
    )

    comparison = experiment.comparison_metrics.iloc[0]
    assert comparison["max_absolute_coefficient_difference"] < 1e-7
    assert comparison["max_absolute_test_prediction_difference"] < 1e-7
    assert comparison["top_25_overlap"] == 6
    assert len(experiment.posterior_rankings) == 6
    assert experiment.posterior_rankings["posterior_top_25_probability"].eq(1.0).all()
    assert experiment.predictive_calibration["nominal_coverage"].tolist() == [
        0.50,
        0.80,
        0.90,
        0.95,
    ]
    assert len(experiment.test_predictions) == 4


def _bayesian_stints() -> pd.DataFrame:
    start = datetime(2025, 10, 1, tzinfo=UTC)
    lineups = (
        ([1, 2], [3, 4]),
        ([1, 3], [5, 6]),
        ([2, 5], [4, 6]),
        ([3, 6], [1, 4]),
    )
    player_values = {1: 3.0, 2: 2.0, 3: 1.0, 4: -1.0, 5: -2.0, 6: -3.0}
    rows = []
    for index in range(20):
        home, away = lineups[index % len(lineups)]
        possessions = float(5 + index % 4)
        target = 1.5 + sum(player_values[player] for player in home) - sum(
            player_values[player] for player in away
        )
        rows.append(
            {
                "game_id": f"{index:010d}",
                "game_time_utc": start + timedelta(days=index),
                "stint_index": 0,
                "home_team_id": 100,
                "away_team_id": 200,
                "home_player_ids": home,
                "away_player_ids": away,
                "possessions": possessions,
                "home_margin": target * possessions / 100.0,
                "target_home_net_rating": target + (-1) ** index * 0.5,
            }
        )
    return pd.DataFrame(rows)


def _ridge_rankings(coefficients: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "player_id": range(1, 7),
            "player_name": [f"Player {index}" for index in range(1, 7)],
            "primary_team_id": 100,
            "primary_team_tricode": "TST",
            "rapm": coefficients,
            "raw_on_court_net_rating": coefficients,
            "stint_count": 10,
            "possessions": 1000.0,
            "seconds": 1000.0,
            "point_margin": 10.0,
            "primary_team_possessions": 1000.0,
            "exposure_eligible": True,
        }
    ).sort_values("rapm", ascending=False, kind="stable")
    frame["rank"] = np.arange(1, 7)
    frame["eligible_rank"] = pd.Series(np.arange(1, 7), dtype="Int64").to_numpy()
    frame["percentile"] = 100.0 * (6 - frame["rank"]) / 5
    return frame.reset_index(drop=True)
