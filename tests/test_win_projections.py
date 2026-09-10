from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.web_api.market_win_totals import BETMGM_WIN_TOTALS_2026_27
from nba_lineup_model.web_api.win_projections import (
    _completed_team_win_totals,
    _team_win_totals,
    fit_win_probability_scale,
    score_season_win_forecasts,
    score_win_probabilities,
    simulate_baseline_win_total_intervals,
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


def test_baseline_win_total_intervals_are_deterministic_and_ordered() -> None:
    games = pd.DataFrame(
        {
            "home_team_tricode": ["AAA"] * 40 + ["BBB"] * 40,
            "away_team_tricode": ["BBB"] * 40 + ["AAA"] * 40,
            "home_win_probability": [0.5] * 80,
        }
    )
    strength = pd.Series({"AAA": 0.0, "BBB": 0.0})

    first = simulate_baseline_win_total_intervals(
        games, team_strength=strength, beta=0.1, home_court=0.0, trials=1_000, seed=7
    )
    second = simulate_baseline_win_total_intervals(
        games, team_strength=strength, beta=0.1, home_court=0.0, trials=1_000, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["win_total_p1"].le(first["win_total_p5"]).all()
    assert first["win_total_p5"].le(first["win_total_p50"]).all()
    assert first["win_total_p50"].le(first["win_total_p95"]).all()
    assert first["win_total_p95"].le(first["win_total_p99"]).all()


def test_unassigned_cup_placeholder_preserves_the_league_win_total() -> None:
    games = pd.DataFrame(
        {
            "home_team_tricode": ["AAA"] * 40 + ["BBB"] * 40,
            "away_team_tricode": ["BBB"] * 40 + ["AAA"] * 40,
            "home_win_probability": [0.6] * 40 + [0.4] * 40,
        }
    )

    totals = _team_win_totals(
        games,
        team_strength=pd.Series({"AAA": 2.0, "BBB": -2.0}),
        beta=0.1,
        home_court=2.0,
    )

    assert totals["scheduled_wins"].sum() == pytest.approx(80.0)
    assert totals["unassigned_wins"].sum() == pytest.approx(2.0)
    assert totals["projected_wins"].sum() == pytest.approx(82.0)
