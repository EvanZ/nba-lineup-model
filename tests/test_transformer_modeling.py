from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import torch

from nba_lineup_model.modeling.neural import NeuralTrainingConfig
from nba_lineup_model.modeling.residual_data import fit_rapm_base_predictions
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.modeling.transformer import (
    RapmTransformerArchitectureConfig,
    fit_rapm_transformer_experiment,
    frozen_rapm_predictions,
)
from nba_lineup_model.models.neural import RapmTransformerResidualModel


def test_rapm_base_evaluation_predictions_exclude_evaluation_outcomes() -> None:
    possessions, stints = _modeling_rows()
    split = ChronologicalSplitConfig(
        cv_folds=1,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    original = fit_rapm_base_predictions(
        possessions,
        stints,
        split_config=split,
        regularization=0.1,
    )
    validation_games = original.split_plan.folds[0].validation_game_ids
    perturbed_stints = stints.copy()
    perturbed_stints.loc[
        perturbed_stints["game_id"].isin(validation_games),
        "target_home_net_rating",
    ] += 1_000.0
    perturbed = fit_rapm_base_predictions(
        possessions,
        perturbed_stints,
        split_config=split,
        regularization=0.1,
    )

    original_validation = original.predictions.loc[
        original.predictions["stage"].eq("cv_0")
        & original.predictions["role"].eq("validation"),
        "prediction_rapm",
    ]
    perturbed_validation = perturbed.predictions.loc[
        perturbed.predictions["stage"].eq("cv_0")
        & perturbed.predictions["role"].eq("validation"),
        "prediction_rapm",
    ]
    np.testing.assert_allclose(
        original_validation,
        perturbed_validation,
        rtol=0,
        atol=1e-12,
    )
    assert original.predictions.loc[
        original.predictions["role"].ne("train"),
        "base_is_out_of_sample",
    ].all()
    assert not original.predictions.loc[
        original.predictions["role"].eq("train"),
        "base_is_out_of_sample",
    ].any()


def test_transformer_starts_at_rapm_and_is_lineup_permutation_invariant() -> None:
    torch.manual_seed(7)
    model = RapmTransformerResidualModel(
        10,
        d_model=8,
        attention_heads=2,
        transformer_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )
    model.eval()
    offense = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    defense = torch.tensor([[6, 7, 8, 9, 10], [10, 9, 8, 7, 6]])
    signs = torch.tensor([1.0, 1.0])
    base = torch.tensor([1.25, 1.25])

    prediction = model(offense, defense, signs, base)
    _, residual = model.components(offense, defense, signs, base)

    torch.testing.assert_close(prediction, base, rtol=0, atol=0)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
    torch.testing.assert_close(prediction[0], prediction[1], rtol=0, atol=1e-7)


def test_miniature_rapm_transformer_experiment_tracks_fixed_seeds(
    tmp_path,
) -> None:
    possessions, stints = _modeling_rows()
    split = ChronologicalSplitConfig(
        cv_folds=1,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    base = fit_rapm_base_predictions(
        possessions,
        stints,
        split_config=split,
        regularization=0.1,
    )

    experiment = fit_rapm_transformer_experiment(
        possessions,
        base.predictions,
        checkpoint_dir=tmp_path,
        split_plan=base.split_plan,
        training_config=NeuralTrainingConfig(
            batch_size=32,
            max_epochs=1,
            early_stopping_patience=0,
            learning_rates=(0.001,),
            weight_decays=(0.0,),
        ),
        architecture_config=RapmTransformerArchitectureConfig(
            d_model=8,
            attention_heads=2,
            transformer_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        ),
        refit_seeds=(17, 18, 19),
    )

    assert experiment.refit_seeds == (17, 18, 19)
    assert experiment.leaderboard_seed == 17
    assert experiment.selected_epochs == 1
    assert len(experiment.seed_metrics) == 3
    assert set(experiment.test_metrics["model"]) == {
        "mean",
        "ridge_rapm",
        "rapm_transformer",
    }
    np.testing.assert_allclose(
        experiment.test_predictions["prediction_rapm"]
        + experiment.test_predictions["prediction_transformer_residual"],
        experiment.test_predictions["prediction_rapm_transformer"],
        rtol=0,
        atol=1e-7,
    )
    assert experiment.parameter_count > 0
    assert (tmp_path / "selection_model.ckpt").is_file()
    assert (tmp_path / "test_model.ckpt").is_file()
    assert (tmp_path / "model.ckpt").is_file()
    assert (tmp_path / "test_model_seed_18.ckpt").is_file()
    assert (tmp_path / "model_seed_19.ckpt").is_file()


def test_frozen_rapm_predictions_count_unknown_exposures() -> None:
    possessions, _ = _modeling_rows()
    inference = possessions.iloc[:1].copy()
    inference.at[inference.index[0], "offense_player_ids"] = [1, 2, 3, 4, 99]

    predictions, unknown = frozen_rapm_predictions(
        inference,
        {player_id: float(player_id) for player_id in range(1, 11)},
        intercept_home_net_rating=2.0,
        mean_offense_margin=1.0,
    )

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
    assert unknown == 1


def _modeling_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    possession_rows = []
    stint_rows = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for game_index in range(10):
        game_time = start + timedelta(days=game_index)
        game_id = f"g{game_index:02d}"
        home_players = [1, 2, 3, 4, 5]
        away_players = [6, 7, 8, 9, 10]
        home_margin = float((game_index % 5) - 2)
        stint_rows.append(
            {
                "game_id": game_id,
                "game_date": date.fromisoformat(game_time.date().isoformat()),
                "game_time_utc": game_time,
                "home_player_ids": home_players,
                "away_player_ids": away_players,
                "target_home_net_rating": 25.0 * home_margin,
                "possessions": 4.0,
            }
        )
        for possession_index in range(4):
            home_offense = possession_index % 2 == 0
            offense = home_players if home_offense else away_players
            defense = away_players if home_offense else home_players
            offense_points = float((game_index + possession_index) % 4)
            target_home = offense_points if home_offense else -offense_points
            possession_rows.append(
                {
                    "season": "2025-26",
                    "season_type": "regular",
                    "game_id": game_id,
                    "game_date": date.fromisoformat(game_time.date().isoformat()),
                    "game_time_utc": game_time,
                    "possession_id": f"{game_id}:{possession_index}",
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
                    "target_home_margin": target_home,
                }
            )
    return pd.DataFrame(possession_rows), pd.DataFrame(stint_rows)
