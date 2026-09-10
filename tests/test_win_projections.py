from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.web_api.market_win_totals import BETMGM_WIN_TOTALS_2026_27
from nba_lineup_model.web_api.win_projections import (
    _completed_team_win_totals,
    fit_win_probability_scale,
    score_season_win_forecasts,
    score_win_probabilities,
)


def test_betmgm_win_total_snapshot_covers_every_nba_team() -> None:
    assert len(BETMGM_WIN_TOTALS_2026_27) == 30
    assert BETMGM_WIN_TOTALS_2026_27["OKC"] == pytest.approx(62.5)
    assert BETMGM_WIN_TOTALS_2026_27["SAC"] == pytest.approx(21.5)


def test_win_probability_scale_is_positive_for_aligned_edges() -> None:
    beta = fit_win_probability_scale(
        pd.Series([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    assert beta > 0.0
    metrics = score_win_probabilities(
        pd.Series([-2.0, 2.0]), pd.Series([0.0, 1.0]), beta=beta
    )
    assert metrics["brier"] < 0.25
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_completed_team_win_totals_and_scores() -> None:
    games = pd.DataFrame(
        {
            "home_team_tricode": ["AAA", "BBB"],
            "away_team_tricode": ["BBB", "AAA"],
            "edge": [0.0, 0.0],
            "home_win": [1.0, 0.0],
        }
    )
    totals = _completed_team_win_totals(games, beta=0.1)

    assert totals.set_index("team")["projected_wins"].to_dict() == {
        "AAA": pytest.approx(1.0),
        "BBB": pytest.approx(1.0),
    }
    assert score_season_win_forecasts(totals, prediction_column="projected_wins") == {
        "mae": pytest.approx(1.0),
        "rmse": pytest.approx(1.0),
        "bias": pytest.approx(0.0),
    }
