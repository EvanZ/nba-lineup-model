from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.catboost import (
    CatBoostTrainingConfig,
    catboost_predictions,
    categorical_player_state_matrix,
    fit_catboost_experiment,
)
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig


def test_categorical_player_states_are_permutation_invariant() -> None:
    possessions = _possessions(game_count=1, possessions_per_game=2)
    possessions.at[1, "offense_player_ids"] = [5, 4, 3, 2, 1]
    possessions.at[1, "defense_player_ids"] = [10, 9, 8, 7, 6]
    possessions.at[1, "home_offense_sign"] = 1.0
    player_columns = {player_id: player_id for player_id in range(1, 11)}

    features = categorical_player_state_matrix(possessions, player_columns)

    assert features.shape == (2, 11)
    np.testing.assert_array_equal(features[0], features[1])
    np.testing.assert_array_equal(features[0, :5], np.ones(5, dtype=np.int8))
    np.testing.assert_array_equal(features[0, 5:10], np.full(5, 2, dtype=np.int8))
    assert features[0, -1] == 1


def test_catboost_experiment_selects_trees_and_records_defaults() -> None:
    possessions = _possessions(game_count=10, possessions_per_game=4)

    experiment = fit_catboost_experiment(
        possessions,
        split_config=ChronologicalSplitConfig(
            cv_folds=1,
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
        training_config=CatBoostTrainingConfig(max_iterations=12),
    )

    assert 1 <= experiment.selected_tree_count <= 12
    assert experiment.best_iteration + 1 == experiment.selected_tree_count
    assert experiment.resolved_learning_rate > 0
    assert len(experiment.fold_metrics) == 1
    assert len(experiment.test_predictions) == 8
    assert set(experiment.test_metrics["model"]) == {"mean", "catboost"}
    assert len(experiment.feature_importance) == 11
    assert experiment.resolved_parameters["requested"]["iterations"] == 12
    assert experiment.resolved_parameters["selection"]["depth"] >= 1
    assert (
        experiment.resolved_parameters["test_refit"]["learning_rate"]
        == experiment.resolved_learning_rate
    )


def test_catboost_prediction_counts_unknown_player_exposures() -> None:
    possessions = _possessions(game_count=10, possessions_per_game=4)
    experiment = fit_catboost_experiment(
        possessions,
        split_config=ChronologicalSplitConfig(
            cv_folds=1,
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
        training_config=CatBoostTrainingConfig(max_iterations=5),
    )
    inference = possessions.iloc[:1].copy()
    inference.at[inference.index[0], "offense_player_ids"] = [1, 2, 3, 4, 99]

    predictions, unknown = catboost_predictions(
        experiment.full_model,
        inference,
        experiment.player_columns,
    )

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    assert unknown == 1


def _possessions(
    *,
    game_count: int,
    possessions_per_game: int,
) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for game_index in range(game_count):
        game_time = start + timedelta(days=game_index)
        for possession_index in range(possessions_per_game):
            home_offense = possession_index % 2 == 0
            offense = [1, 2, 3, 4, 5] if home_offense else [6, 7, 8, 9, 10]
            defense = [6, 7, 8, 9, 10] if home_offense else [1, 2, 3, 4, 5]
            offense_points = float((game_index + possession_index) % 4)
            if home_offense and possession_index == 0 and game_index % 2:
                offense_points += 1.0
            rows.append(
                {
                    "season": "2025-26",
                    "season_type": "regular",
                    "game_id": f"g{game_index:02d}",
                    "game_date": date.fromisoformat(game_time.date().isoformat()),
                    "game_time_utc": game_time,
                    "possession_id": f"g{game_index:02d}:{possession_index}",
                    "possession_index": possession_index,
                    "period": 1,
                    "offense_team_id": 100 if home_offense else 200,
                    "defense_team_id": 200 if home_offense else 100,
                    "offense_team_tricode": "HOM" if home_offense else "AWY",
                    "defense_team_tricode": "AWY" if home_offense else "HOM",
                    "offense_player_ids": offense,
                    "defense_player_ids": defense,
                    "home_offense": home_offense,
                    "home_offense_sign": 1.0 if home_offense else -1.0,
                    "target_offense_margin": offense_points,
                    "target_home_margin": (
                        offense_points if home_offense else -offense_points
                    ),
                }
            )
    return pd.DataFrame(rows)
