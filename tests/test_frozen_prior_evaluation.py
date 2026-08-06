from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.frozen_prior_evaluation import (
    PythagoreanWinModel,
    _team_win_evaluation,
    _validate_external_frozen_priors,
    draft_cold_start_prior_frame,
    exposure_gated_cold_start_prior_frame,
    fit_pythagorean_win_model,
    score_frozen_possessions,
    score_possession_cohort,
)


def test_draft_cold_start_prior_replaces_only_zero_prior_first_year_players():
    lagged = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "prior_rapm_mean": [4.0, 0.0, 0.0],
            "prior_available": [True, False, False],
        }
    )
    rankings = pd.DataFrame({"player_id": [2], "draft_prior": [-0.4]})

    output = draft_cold_start_prior_frame(lagged, rankings).set_index("player_id")

    assert output.loc[1, "prior_rapm_mean"] == 4.0
    assert output.loc[2, "prior_rapm_mean"] == -0.4
    assert output.loc[3, "prior_rapm_mean"] == 0.0
    assert output["prior_branch"].to_dict() == {
        1: "lagged_rapm",
        2: "draft_cold_start",
        3: "zero_cold_start",
    }

    with pytest.raises(ValueError, match="overlap"):
        draft_cold_start_prior_frame(lagged, pd.DataFrame({"player_id": [1], "draft_prior": [1.0]}))


def test_exposure_gated_cold_start_prior_replaces_only_zero_prior_first_year_players():
    lagged = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "prior_rapm_mean": [4.0, 0.0, 0.0],
            "prior_available": [True, False, False],
        }
    )
    rankings = pd.DataFrame({"player_id": [2], "blended_cold_start_prior": [-2.1]})

    output = exposure_gated_cold_start_prior_frame(lagged, rankings).set_index("player_id")

    assert output.loc[1, "prior_rapm_mean"] == 4.0
    assert output.loc[2, "prior_rapm_mean"] == -2.1
    assert output.loc[3, "prior_rapm_mean"] == 0.0
    assert output["prior_branch"].to_dict() == {
        1: "lagged_rapm",
        2: "exposure_gated_cold_start",
        3: "zero_cold_start",
    }

    with pytest.raises(ValueError, match="overlap"):
        exposure_gated_cold_start_prior_frame(
            lagged,
            pd.DataFrame({"player_id": [1], "blended_cold_start_prior": [-2.1]}),
        )


def test_frozen_possession_predictions_do_not_depend_on_target_outcomes():
    possessions = _possessions()
    priors = pd.DataFrame(
        {
            "player_id": range(1, 13),
            "prior_rapm_mean": np.arange(1, 13, dtype=float),
        }
    )
    original = score_frozen_possessions(
        possessions,
        priors,
        source_mean=1.1,
        source_home_intercept=2.0,
        cohort="regular_season",
    )
    changed_outcomes = possessions.copy()
    changed_outcomes["target_offense_margin"] += 100.0
    changed = score_frozen_possessions(
        changed_outcomes,
        priors,
        source_mean=1.1,
        source_home_intercept=2.0,
        cohort="regular_season",
    )

    assert original["prediction_offense_margin"].tolist() == (
        changed["prediction_offense_margin"].tolist()
    )
    assert original["prediction_offense_margin"].tolist() == pytest.approx(
        [0.985, 1.24]
    )
    assert original["unknown_player_exposures"].tolist() == [0, 0]


def test_frozen_metrics_keep_regular_and_playoff_cohorts_explicit():
    predictions = score_frozen_possessions(
        _possessions(),
        pd.DataFrame(
            {
                "player_id": range(1, 13),
                "prior_rapm_mean": np.zeros(12),
            }
        ),
        source_mean=1.0,
        source_home_intercept=0.0,
        cohort="playoffs",
    )

    metrics = score_possession_cohort(predictions, source_mean=1.0)

    assert metrics["cohort"].item() == "playoffs"
    assert metrics["game_count"].item() == 2
    assert metrics["possession_count"].item() == 2


def test_external_frozen_priors_are_label_free_and_normalized_to_scorer_contract():
    priors = pd.DataFrame({"player_id": [2, 1], "prior_rapm_mean": [1.5, -0.5]})

    output = _validate_external_frozen_priors(
        priors,
        target_season="2025-26",
        source_season="2024-25",
    )

    assert output["player_id"].tolist() == [1, 2]
    assert output["prior_available"].tolist() == [True, True]
    assert output["source_season"].tolist() == ["2024-25", "2024-25"]

    with pytest.raises(ValueError, match="target outcomes"):
        _validate_external_frozen_priors(
            priors.assign(target_rapm=0.0),
            target_season="2025-26",
            source_season="2024-25",
        )


def test_team_win_totals_use_pythagorean_mapping_and_keep_raw_game_winners():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "home_team_id": [1, 2, 1],
            "away_team_id": [2, 1, 2],
            "home_team_tricode": ["AAA", "BBB", "AAA"],
            "away_team_tricode": ["BBB", "AAA", "BBB"],
            "actual_home_margin": [3.0, -2.0, -1.0],
            "predicted_home_margin": [1.0, 2.0, -3.0],
            "actual_home_win": [True, False, False],
            "predicted_home_win": [True, True, False],
            "predicted_tie": [False, False, False],
        }
    )

    team_net_ratings = pd.DataFrame(
        {
            "team_id": [1, 2],
            "actual_net_rating": [1.0, -1.0],
            "predicted_net_rating": [2.0, -2.0],
        }
    )
    pythagorean_model = PythagoreanWinModel(
        intercept=0.5,
        net_rating_slope=0.03,
        training_seasons=("2023-24", "2024-25"),
        training_team_season_count=60,
        historical_win_total_rmse=3.0,
    )

    predictions, metrics = _team_win_evaluation(
        games,
        team_net_ratings,
        pythagorean_model,
    )
    teams = predictions.set_index("team_tricode")

    assert teams.loc["AAA", "wins"] == 2
    assert teams.loc["AAA", "predicted_game_winner_count"] == 1
    assert teams.loc["AAA", "pythagorean_wins"] == pytest.approx(1.68)
    assert teams.loc["BBB", "wins"] == 1
    assert teams.loc["BBB", "predicted_game_winner_count"] == 2
    assert teams.loc["BBB", "pythagorean_wins"] == pytest.approx(1.32)
    assert teams.loc["BBB", "actual_net_rating"] == -1.0
    assert metrics["pythagorean_league_win_total"].item() == pytest.approx(3.0)
    assert metrics["raw_game_winner_league_win_total"].item() == 3
    assert metrics["prediction_rule"].item().startswith("pythagorean wins")


def test_pythagorean_win_model_fits_weighted_historical_team_seasons():
    team_seasons = pd.DataFrame(
        {
            "season": ["2023-24"] * 3 + ["2024-25"] * 3,
            "games": [82] * 6,
            "wins": [16.4, 41.0, 65.6, 16.4, 41.0, 65.6],
            "win_pct": [0.2, 0.5, 0.8, 0.2, 0.5, 0.8],
            "net_rating": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
        }
    )

    model = fit_pythagorean_win_model(team_seasons)

    assert model.intercept == pytest.approx(0.5)
    assert model.net_rating_slope == pytest.approx(0.03)
    assert model.training_seasons == ("2023-24", "2024-25")
    assert model.training_team_season_count == 6
    assert model.historical_win_total_rmse == pytest.approx(0.0, abs=1e-12)


def _possessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "season_type": ["regular", "regular"],
            "game_id": ["g1", "g2"],
            "possession_id": ["g1:1", "g2:1"],
            "home_offense_sign": [1.0, -1.0],
            "offense_player_ids": [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]],
            "defense_player_ids": [[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]],
            "target_offense_margin": [2.0, 0.0],
        }
    )
