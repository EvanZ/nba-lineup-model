"""Tests for the recursive aging player-prior inputs."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.modeling.aging import prepare_aging_prior_features
from nba_lineup_model.modeling.forward_aging_player_prior import (
    _aging_transition_history,
    _target_returning_features,
    build_aging_exposure_gated_priors,
)
from nba_lineup_model.modeling.prior_rapm import ForwardLaggedRapmSeason


def _result(season: str, rating: float) -> ForwardLaggedRapmSeason:
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=1.0,
        cv_results=pd.DataFrame(),
        player_estimates=pd.DataFrame(
            {
                "season": [season],
                "player_id": [1],
                "rapm": [rating],
                "prior_rapm": [0.0],
                "rapm_adjustment_from_prior": [rating],
                "prior_available": [False],
                "selected_lambda": [1.0],
            }
        ),
        player_priors=pd.DataFrame(),
    )


def _panel() -> pd.DataFrame:
    rows = []
    for year, age in ((1996, 20), (1997, 21), (1998, 22)):
        season = f"{year}-{str(year + 1)[-2:]}"
        rows.append(
            {
                "season": season,
                "player_id": 1,
                "player_name": "Example Player",
                "age": age,
                "nba_experience_years": age - 20,
                "is_rookie": age == 20,
                "draft_year": 1996,
                "draft_number": 5,
                "height_inches": 78.0,
                "weight_pounds": 220.0,
                "is_undrafted": False,
                "rapm_seconds": 1000.0,
                "rapm_exposure_eligible": True,
            }
        )
    return pd.DataFrame(rows)


def test_recursive_aging_transition_uses_completed_model_state() -> None:
    transitions = _aging_transition_history(
        _panel(),
        [_result("1996-97", 1.0), _result("1997-98", 2.5), _result("1998-99", 3.0)],
        [
            pd.DataFrame({"player_id": [1], "on_court_possessions": [100.0]}),
            pd.DataFrame({"player_id": [1], "on_court_possessions": [200.0]}),
            pd.DataFrame({"player_id": [1], "on_court_possessions": [300.0]}),
        ],
    )

    first = transitions.loc[transitions["target_season"].eq("1997-98")].iloc[0]
    assert first["prior_rapm"] == 1.0
    assert first["target_rapm"] == 2.5
    assert first["prior_rapm_possessions"] == 100.0
    assert first["target_rapm_possessions"] == 200.0


def test_aging_prior_features_do_not_require_target_outcomes() -> None:
    features = prepare_aging_prior_features(
        pd.DataFrame(
            {
                "target_season": ["2025-26"],
                "player_id": [1],
                "target_age": [27.0],
                "target_nba_experience_years": [6],
                "is_rookie": [False],
                "has_prior_season": [True],
                "prior_rapm": [2.0],
                "prior_rapm_possessions": [4000.0],
            }
        )
    )

    assert features["prior_rapm_filled"].item() == 2.0
    assert features["target_age"].item() == 27.0
    assert "target_rapm" not in features


def test_target_returning_features_renames_the_lagged_rapm_prior() -> None:
    target = _target_returning_features(
        _panel(),
        season="1998-99",
        returning=pd.DataFrame({"player_id": [1], "lagged_rapm_prior": [2.25]}),
        latest_exposure=pd.DataFrame({"player_id": [1], "on_court_possessions": [300.0]}),
    )

    assert target["prior_rapm"].item() == 2.25


def test_aging_prior_is_enabled_with_completed_recursive_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seasons = ("1996-97", "1997-98", "1998-99", "1999-00", "2000-01")
    player_ids = list(range(1, 9))
    panel = pd.DataFrame(
        [
            {
                "season": season,
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "age": 20 + season_index + player_id % 2,
                "nba_experience_years": season_index,
                "is_rookie": season_index == 0,
                "draft_year": 1996,
                "draft_number": player_id,
                "height_inches": 75.0 + player_id,
                "weight_pounds": 190.0 + 5 * player_id,
                "is_undrafted": False,
                "rapm_seconds": 1000.0,
                "rapm_exposure_eligible": True,
            }
            for season_index, season in enumerate(seasons)
            for player_id in player_ids
        ]
    )
    results = [
        ForwardLaggedRapmSeason(
            season=season,
            selected_lambda=1.0,
            cv_results=pd.DataFrame(),
            player_estimates=pd.DataFrame(
                {
                    "season": season,
                    "player_id": player_ids,
                    "rapm": [0.1 * player_id + season_index for player_id in player_ids],
                    "prior_rapm": [0.0] * len(player_ids),
                    "rapm_adjustment_from_prior": [0.0] * len(player_ids),
                    "prior_available": [True] * len(player_ids),
                    "selected_lambda": [1.0] * len(player_ids),
                }
            ),
            player_priors=pd.DataFrame(),
        )
        for season_index, season in enumerate(seasons[:-1])
    ]
    exposure = [
        pd.DataFrame({"player_id": player_ids, "on_court_possessions": [500.0] * len(player_ids)})
        for _ in results
    ]
    monkeypatch.setattr(
        "nba_lineup_model.modeling.forward_aging_player_prior._cold_start_priors",
        lambda **_: (pd.DataFrame(columns=["player_id", "lagged_rapm_prior"]), {}),
    )

    priors, metadata = build_aging_exposure_gated_priors(
        season="2000-01",
        panel=panel,
        completed_results=results,
        exposure_history=exposure,
        replacement_tokens=[{}],
    )

    assert metadata["aging_enabled"] is True
    assert len(priors) == len(player_ids)
