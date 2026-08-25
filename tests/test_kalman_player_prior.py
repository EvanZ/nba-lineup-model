from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    _advance_player_state_variance,
)
from nba_lineup_model.modeling.kalman_player_prior import (
    PlayerKalmanConfig,
    filter_completed_player_states,
)
from nba_lineup_model.modeling.prior_rapm import ForwardLaggedRapmSeason
from nba_lineup_model.modeling.state_precision import (
    PlayerStatePrecisionConfig,
    advance_state_variance,
    relative_precision_from_variance,
)
from nba_lineup_model.models.baselines import (
    PriorCenteredRidgeLineupModel,
    PriorPrecisionRidgeLineupModel,
)


def _result(season: str, prior: float, observed: float) -> ForwardLaggedRapmSeason:
    estimates = pd.DataFrame(
        {
            "season": [season],
            "player_id": [7],
            "rapm": [observed],
            "prior_rapm": [prior],
            "rapm_adjustment_from_prior": [observed - prior],
        }
    )
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=1.0,
        cv_results=pd.DataFrame(),
        player_estimates=estimates,
        player_priors=pd.DataFrame(),
    )


def test_filter_uses_current_season_prior_as_state_prediction() -> None:
    config = PlayerKalmanConfig(
        initial_variance=9.0,
        process_variance_per_season=1.0,
        observation_variance_possession_scale=4000.0,
    )
    states = filter_completed_player_states(
        [_result("2022-23", 1.0, 5.0), _result("2023-24", 2.0, 6.0)],
        [
            pd.DataFrame({"player_id": [7], "on_court_possessions": [1000.0]}),
            pd.DataFrame({"player_id": [7], "on_court_possessions": [1000.0]}),
        ],
        config=config,
    )

    first, second = states.itertuples(index=False)
    assert first.kalman_gain == pytest.approx(9.0 / 13.0)
    assert first.posterior_mean == pytest.approx(1.0 + (9.0 / 13.0) * 4.0)
    assert second.prior_mean == pytest.approx(2.0)
    assert second.prior_variance == pytest.approx(first.posterior_variance + 1.0)
    assert second.posterior_mean > second.prior_mean


def test_filter_downweights_low_exposure_observations() -> None:
    result = _result("2023-24", 0.0, 5.0)
    config = PlayerKalmanConfig()
    states = filter_completed_player_states(
        [result],
        [
            pd.DataFrame(
                {"player_id": [7], "on_court_possessions": [100.0]}
            )
        ],
        config=config,
    )

    assert float(states.loc[0, "kalman_gain"]) < 0.25
    assert float(states.loc[0, "posterior_mean"]) < 1.25


def test_filter_rejects_misaligned_histories() -> None:
    with pytest.raises(ValueError, match="align"):
        filter_completed_player_states([_result("2023-24", 0.0, 1.0)], [])


def test_prior_precision_ridge_matches_uniform_prior_centered_ridge() -> None:
    features = sparse.csr_matrix(
        [
            [1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    target = pd.Series([1.0, -0.5, 0.25, -0.75]).to_numpy()
    weights = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy()
    prior = pd.Series([0.3, -0.2, 0.1]).to_numpy()
    baseline = PriorCenteredRidgeLineupModel(regularization=2.5).fit(
        features, target, weights, prior
    )
    state_precision = PriorPrecisionRidgeLineupModel(regularization=2.5).fit(
        features,
        target,
        weights,
        prior,
        relative_precision=pd.Series([1.0, 1.0, 1.0]).to_numpy(),
    )

    assert state_precision.coef_ == pytest.approx(baseline.coef_)
    assert state_precision.adjustment_ == pytest.approx(baseline.adjustment_)
    assert state_precision.intercept_ == pytest.approx(baseline.intercept_)
    assert state_precision.predict(features) == pytest.approx(baseline.predict(features))


def test_prior_precision_ridge_matches_closed_form_one_player_posterior() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    target = np.array([-4.0, -2.0, 2.0, 4.0])
    weights = np.ones(len(target))
    prior = np.array([0.5])
    relative_precision = np.array([4.0])
    regularization = 2.0
    fitted = PriorPrecisionRidgeLineupModel(regularization=regularization).fit(
        features,
        target,
        weights,
        prior,
        relative_precision,
    )

    expected = (
        float(features[:, 0] @ target)
        + regularization * len(target) * relative_precision[0] * prior[0]
    ) / (
        float(features[:, 0] @ features[:, 0])
        + regularization * len(target) * relative_precision[0]
    )
    assert fitted.intercept_ == pytest.approx(0.0)
    assert fitted.coef_[0] == pytest.approx(expected)
    assert fitted.posterior_variance_[0] > 0


def test_no_forgetting_state_precision_carries_player_specific_uncertainty() -> None:
    config = PlayerStatePrecisionConfig(
        initial_variance=9.0,
        process_variance_per_season=0.0,
    )
    variances = _advance_player_state_variance(
        (7, 8, 9),
        season_index=2,
        posterior_variance_by_player={7: 1.0, 8: 4.0},
        last_seen_index={7: 1, 8: 1},
        config=config,
    )

    assert variances == pytest.approx([1.0, 4.0, 9.0])
    precision = relative_precision_from_variance(variances, config=config)
    assert precision[0] > precision[1] > precision[2]


def test_state_variance_transition_controls_relative_precision() -> None:
    config = PlayerStatePrecisionConfig(process_variance_per_season=0.5)
    prior_variance = advance_state_variance(
        np.array([0.5, 1.0, 2.0]),
        elapsed_seasons=np.array([1.0, 2.0, 3.0]),
        config=config,
    )

    assert prior_variance == pytest.approx([1.0, 2.0, 3.5])
    precision = relative_precision_from_variance(prior_variance, config=config)
    assert precision[0] > precision[1] > precision[2]
    assert np.median(precision) == pytest.approx(1.0)
