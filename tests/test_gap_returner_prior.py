from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.gap_returner_prior import _gap_returner_priors
from nba_lineup_model.modeling.prior_rapm import ForwardLaggedRapmSeason


class _AnnualIncrementModel:
    """Simple deterministic aging model used to verify recursive bridge steps."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["prior_rapm_filled"].to_numpy(dtype=float) + 1.0


def _result(season: str, rows: list[tuple[int, float]]) -> ForwardLaggedRapmSeason:
    estimates = pd.DataFrame(rows, columns=["player_id", "rapm"])
    estimates.insert(0, "season", season)
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=0.1,
        cv_results=pd.DataFrame(),
        player_estimates=estimates,
        player_priors=pd.DataFrame(),
    )


def test_gap_returner_receives_one_annual_prior_per_missing_season() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2021-22", "2021-22"],
            "player_id": [1, 2],
            "player_name": ["Gap Returner", "Immediate Returner"],
            "age": [30.0, 28.0],
            "nba_experience_years": [9.0, 7.0],
            "is_rookie": [False, False],
            "draft_year": [2012.0, 2014.0],
            "draft_number": [15.0, 4.0],
            "height_inches": [78.0, 80.0],
            "weight_pounds": [220.0, 230.0],
            "is_undrafted": [False, False],
            "rapm_seconds": [0.0, 0.0],
            "rapm_exposure_eligible": [False, False],
        }
    )
    results = [
        _result("2018-19", [(1, 2.0), (2, 0.5)]),
        _result("2019-20", [(2, 1.0)]),
        _result("2020-21", [(2, 1.5)]),
    ]
    exposures = [
        pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [2000.0, 1000.0]}),
        pd.DataFrame({"player_id": [2], "on_court_possessions": [1100.0]}),
        pd.DataFrame({"player_id": [2], "on_court_possessions": [1200.0]}),
    ]

    priors, states = _gap_returner_priors(
        season="2021-22",
        panel=panel,
        completed_results=results,
        exposure_history=exposures,
        existing_prior_ids={2},
        aging_model=_AnnualIncrementModel(),
    )

    assert priors.to_dict("records") == [{"player_id": 1, "lagged_rapm_prior": 5.0}]
    assert states["projected_season"].tolist() == ["2019-20", "2020-21", "2021-22"]
    assert states["is_return_season"].tolist() == [False, False, True]
    assert states["projected_prior_rapm"].tolist() == [3.0, 4.0, 5.0]
    assert states["gap_seasons"].tolist() == [2, 2, 2]


def test_gap_returner_does_not_overwrite_an_immediate_prior() -> None:
    panel = pd.DataFrame(
        {
            "season": ["2021-22"],
            "player_id": [1],
            "player_name": ["Immediate Returner"],
            "age": [28.0],
            "nba_experience_years": [7.0],
            "is_rookie": [False],
            "draft_year": [2014.0],
            "draft_number": [4.0],
            "height_inches": [80.0],
            "weight_pounds": [230.0],
            "is_undrafted": [False],
            "rapm_seconds": [0.0],
            "rapm_exposure_eligible": [False],
        }
    )
    results = [_result("2020-21", [(1, 1.5)])]
    exposures = [pd.DataFrame({"player_id": [1], "on_court_possessions": [1200.0]})]

    priors, states = _gap_returner_priors(
        season="2021-22",
        panel=panel,
        completed_results=results,
        exposure_history=exposures,
        existing_prior_ids={1},
        aging_model=_AnnualIncrementModel(),
    )

    assert priors.empty
    assert states.empty
