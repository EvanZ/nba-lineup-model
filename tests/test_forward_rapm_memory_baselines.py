from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.forward_rapm_memory_baselines import build_rapm_memory_prior
from nba_lineup_model.modeling.prior_rapm import ForwardLaggedRapmSeason


def _result(season: str, values: dict[int, float]) -> ForwardLaggedRapmSeason:
    estimates = pd.DataFrame({"player_id": list(values), "rapm": list(values.values())})
    estimates["season"] = season
    estimates["prior_rapm"] = 0.0
    estimates["rapm_adjustment_from_prior"] = estimates["rapm"]
    estimates["prior_available"] = False
    estimates["selected_lambda"] = 0.1
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=0.1,
        cv_results=pd.DataFrame(),
        player_estimates=estimates,
        player_priors=pd.DataFrame(),
    )


def test_rapm_memory_prior_uses_only_completed_and_weighted_seasons() -> None:
    history = (
        _result("2021-22", {1: 1.0, 2: 5.0}),
        _result("2022-23", {1: 4.0, 3: -2.0}),
        _result("2023-24", {1: 7.0, 2: -1.0}),
    )
    exposures = {
        "2021-22": pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [10.0, 10.0]}),
        "2022-23": pd.DataFrame({"player_id": [1, 3], "on_court_possessions": [20.0, 10.0]}),
        "2023-24": pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [30.0, 10.0]}),
    }

    one_year = build_rapm_memory_prior(history, exposures, memory_seasons=1)
    three_year = build_rapm_memory_prior(history, exposures, memory_seasons=3)

    assert one_year.set_index("player_id")["lagged_rapm_prior"].to_dict() == {1: 7.0, 2: -1.0}
    assert three_year.set_index("player_id")["lagged_rapm_prior"].to_dict() == {
        1: 5.0,
        2: 2.0,
        3: -2.0,
    }
    assert three_year.set_index("player_id").loc[1, "prior_observed_season_count"] == 3
