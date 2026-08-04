from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling import forward_calibration
from nba_lineup_model.modeling.forward_calibration import (
    _player_team_exposure,
    _team_outcomes,
    build_team_season_inputs,
    fit_win_calibration,
)


def test_team_inputs_use_frozen_priors_and_realized_stint_seconds(monkeypatch) -> None:
    monkeypatch.setattr(
        forward_calibration,
        "read_rapm_stints",
        lambda season, analytical_dir: _stints(season),
    )
    priors = {
        "2020-21": pd.DataFrame(
            {
                "player_id": list(range(1, 6)) + list(range(11, 16)),
                "prior_rapm": [2.0] * 5 + [-1.0] * 5,
                "prior_source_season": ["2019-20"] * 10,
            }
        )
    }

    result = build_team_season_inputs(priors)

    home = result.loc[result["team_id"].eq(1)].iloc[0]
    away = result.loc[result["team_id"].eq(2)].iloc[0]
    assert home.team_prior_rapm_raw == pytest.approx(2.0)
    assert away.team_prior_rapm_raw == pytest.approx(-1.0)
    assert home.team_prior_rapm == pytest.approx(1.0)
    assert away.team_prior_rapm == pytest.approx(-1.0)
    assert home.win_pct == pytest.approx(1.0)
    assert away.win_pct == pytest.approx(0.0)
    assert home.prior_exposure_fraction == pytest.approx(1.0)


def test_missing_player_prior_receives_zero_cold_start(monkeypatch) -> None:
    monkeypatch.setattr(
        forward_calibration,
        "read_rapm_stints",
        lambda season, analytical_dir: _stints(season),
    )
    priors = {
        "2020-21": pd.DataFrame(
            {
                "player_id": list(range(1, 6)) + list(range(11, 15)),
                "prior_rapm": [2.0] * 5 + [-1.0] * 4,
                "prior_source_season": ["2019-20"] * 9,
            }
        )
    }

    result = build_team_season_inputs(priors)

    away = result.loc[result["team_id"].eq(2)].iloc[0]
    assert away.team_prior_rapm_raw == pytest.approx(-0.8)
    assert away.team_prior_rapm == pytest.approx(-0.9838699101)
    assert away.prior_exposure_fraction == pytest.approx(0.8)


def test_fit_win_calibration_recovers_known_mapping() -> None:
    ratings = np.array([-2.0, -1.0, 1.0, 2.0])
    training = pd.DataFrame(
        {
            "season": ["2020-21"] * 4,
            "team_id": [1, 2, 3, 4],
            "team_tricode": ["A", "B", "C", "D"],
            "team_prior_rapm": ratings,
            "games": [72] * 4,
            "wins": [0, 0, 0, 0],
            "losses": [72] * 4,
            "win_pct": 0.5 + 0.05 * ratings,
        }
    )

    model = fit_win_calibration(training)

    assert model.intercept == pytest.approx(0.5)
    assert model.prior_rapm_slope == pytest.approx(0.05)
    assert model.predict(np.array([-20.0, 20.0])).tolist() == [0.0, 1.0]


def test_tied_regular_season_game_is_rejected() -> None:
    stints = _stints("2020-21").assign(home_margin=0.0)

    with pytest.raises(ValueError, match="tied final margin"):
        _team_outcomes(stints)


def test_zero_second_stints_do_not_create_player_exposure() -> None:
    stints = pd.concat(
        [_stints("2020-21"), _stints("2020-21").iloc[[0]].assign(duration_seconds=0.0)],
        ignore_index=True,
    )

    exposure = _player_team_exposure(stints)

    assert exposure.loc[exposure["team_id"].eq(1), "player_seconds"].unique().tolist() == [300.0]


def _stints(season: str) -> pd.DataFrame:
    rows = []
    for stint_index, duration in enumerate((100.0, 200.0)):
        rows.append(
            {
                "season": season,
                "game_id": "0022000001",
                "game_date": "2020-12-22",
                "game_time_utc": "2020-12-23T00:00:00+00:00",
                "stint_index": stint_index,
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_tricode": "HOM",
                "away_team_tricode": "AWY",
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [11, 12, 13, 14, 15],
                "duration_seconds": duration,
                "home_margin": 5.0 if stint_index == 0 else 3.0,
            }
        )
    return pd.DataFrame(rows)
