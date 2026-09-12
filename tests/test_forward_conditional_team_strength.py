"""Tests for the incumbent-only team-strength cold-start feature."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.rotation import forward_conditional_team_strength


def test_incumbent_team_strength_excludes_rookies_from_its_weight(monkeypatch) -> None:
    summary = pd.DataFrame(
        {
            "season": ["2019-20", "2020-21", "2020-21", "2020-21"],
            "season_start_year": [2019, 2020, 2020, 2020],
            "player_id": [99, 1, 2, 3],
        }
    )
    opening = pd.DataFrame(
        {
            "team_id": [1, 1, 1],
            "team": ["ONE", "ONE", "ONE"],
            "player_id": [1, 2, 3],
            "player_name": ["Veteran", "Reserve", "Rookie"],
        }
    )
    conditional = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "has_prior_minutes_state": [True, True, False],
            "is_rookie": [False, False, True],
            "predicted_minutes_per_available_game": [30.0, 20.0, 35.0],
        }
    )
    availability = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "predicted_available_share": [0.8, 0.5, 0.9],
        }
    )
    monkeypatch.setattr(
        forward_conditional_team_strength,
        "read_preseason_roster_candidates",
        lambda *_args, **_kwargs: opening,
    )
    monkeypatch.setattr(
        forward_conditional_team_strength,
        "build_preseason_nail_forecast_ratings",
        lambda *_args, **_kwargs: pd.Series({1: 2.0, 2: -2.0, 3: 20.0}),
    )
    monkeypatch.setattr(
        forward_conditional_team_strength,
        "predict_conditional_minutes_season",
        lambda *_args, **_kwargs: (conditional, None),
    )
    monkeypatch.setattr(
        forward_conditional_team_strength,
        "predict_availability_season",
        lambda *_args, **_kwargs: (availability, None, {}),
    )

    augmented, diagnostic = (
        forward_conditional_team_strength.build_incumbent_team_strength_summary(
            summary,
            panel=pd.DataFrame(),
            seasons=("2020-21",),
        )
    )

    expected = (0.8 * 30.0 * 2.0 + 0.5 * 20.0 * -2.0) / (0.8 * 30.0 + 0.5 * 20.0)
    values = augmented.loc[augmented["season"].eq("2020-21"), "incumbent_team_nail"]
    assert values.tolist() == pytest.approx([expected] * 3)
    assert diagnostic.loc[0, "incumbent_player_count"] == 2
