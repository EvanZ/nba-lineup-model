from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.rotation.forward_tweedie_availability_loss_cost import (
    _tweedie_deviance,
    fit_tweedie_loss_cost_model,
    predict_tweedie_loss_cost,
)


def _state_panel() -> pd.DataFrame:
    rows = []
    for year in range(2020, 2024):
        for player_id, state, losses in ((1, -2.0, 4.0), (2, 0.0, 15.0), (3, 1.0, 35.0)):
            exposure = 82.0
            rows.append(
                {
                    "season": f"{year}-{str(year + 1)[-2:]}",
                    "season_start_year": year,
                    "player_id": player_id,
                    "known_player_games": exposure,
                    "available_games": exposure - losses,
                    "available_share": 1.0 - losses / exposure,
                    "unavailable_games": losses,
                    "loss_state_logit": state,
                }
            )
    return pd.DataFrame(rows)


def test_tweedie_loss_cost_is_positive_and_bounded_for_availability_output() -> None:
    predictions, model = predict_tweedie_loss_cost(
        _state_panel(),
        target_season="2023-24",
        power=1.5,
    )

    assert model.power == pytest.approx(1.5)
    assert predictions["predicted_unavailable_games"].ge(0.0).all()
    assert (
        predictions["predicted_unavailable_games"] <= predictions["known_player_games"]
    ).all()
    assert predictions["predicted_available_share"].between(0.0, 1.0).all()


def test_tweedie_deviance_is_zero_when_mean_equals_observed_loss() -> None:
    losses = np.array([0.0, 4.0, 15.0])
    values = _tweedie_deviance(losses, losses + 1e-12, power=1.5)
    assert values[1:] == pytest.approx(0.0, abs=1e-5)
    model = fit_tweedie_loss_cost_model(_state_panel().iloc[:6], power=1.5)
    assert np.isfinite(model.intercept)
    assert np.isfinite(model.state_weight)
