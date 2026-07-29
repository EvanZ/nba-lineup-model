from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import torch

from nba_lineup_model.modeling.deep_sets import (
    DeepSetsArchitectureConfig,
    fit_deep_sets_experiment,
)
from nba_lineup_model.modeling.neural import (
    NeuralTrainingConfig,
    fit_additive_neural_experiment,
)
from nba_lineup_model.modeling.neural_data import (
    PossessionTensorDataset,
    neural_possessions_frame,
    player_vocabulary,
)
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.models.neural import AdditivePlayerModel, DeepSetsPlayerModel


def test_neural_possessions_exclude_multi_segment_and_orient_lineups() -> None:
    segments = _source_segments()

    result = neural_possessions_frame(segments)

    assert result["possession_id"].tolist() == ["g1:0000", "g1:0002"]
    assert result["offense_player_ids"].tolist() == [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]
    assert result["defense_player_ids"].tolist() == [
        [6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5],
    ]
    assert result["home_offense_sign"].tolist() == [1.0, -1.0]
    assert result["target_offense_margin"].tolist() == [2, 3]
    assert result["target_home_margin"].tolist() == [2, -3]


def test_tensor_dataset_reserves_zero_and_encodes_fixed_lineups() -> None:
    possessions = neural_possessions_frame(_source_segments())
    columns = player_vocabulary(possessions)

    dataset = PossessionTensorDataset(possessions, columns)

    assert min(columns.values()) == 1
    assert dataset.offense_player_indices.shape == (2, 5)
    assert dataset.defense_player_indices.shape == (2, 5)
    assert not dataset.offense_player_indices.eq(0).any()
    assert dataset[0]["target"].item() == pytest.approx(2.0)


def test_additive_model_is_permutation_invariant_and_role_swap_symmetric() -> None:
    model = AdditivePlayerModel(player_count=10)
    with torch.no_grad():
        model.player_effects.weight[:, 0] = torch.arange(11, dtype=torch.float32) / 100.0
        model.intercept.fill_(1.1)
        model.home_offense_effect.fill_(0.02)
        model.player_effects.weight[0, 0] = 0.0
    offense = torch.tensor([[1, 2, 3, 4, 5]])
    defense = torch.tensor([[6, 7, 8, 9, 10]])

    home_prediction = model(offense, defense, torch.tensor([1.0]))
    shuffled_prediction = model(
        torch.tensor([[5, 3, 1, 4, 2]]),
        torch.tensor([[9, 6, 10, 8, 7]]),
        torch.tensor([1.0]),
    )
    away_prediction = model(defense, offense, torch.tensor([-1.0]))

    assert shuffled_prediction.item() == pytest.approx(home_prediction.item())
    assert (home_prediction + away_prediction).item() == pytest.approx(2.2)
    assert model.centered_player_values().mean().item() == pytest.approx(0.0, abs=1e-7)


def test_deep_sets_concatenates_roles_and_is_permutation_invariant() -> None:
    torch.manual_seed(4)
    model = DeepSetsPlayerModel(player_count=10)
    with torch.no_grad():
        model.lineup_mlp[-1].weight.fill_(0.1)
    offense = torch.tensor([[1, 2, 3, 4, 5]])
    defense = torch.tensor([[6, 7, 8, 9, 10]])
    sign = torch.tensor([1.0])

    prediction = model(offense, defense, sign)
    shuffled = model(
        torch.tensor([[5, 3, 1, 4, 2]]),
        torch.tensor([[9, 6, 10, 8, 7]]),
        sign,
    )
    additive, nonlinear = model.components(offense, defense, sign)

    assert model.player_mlp[0].in_features == 40
    assert model.lineup_mlp[0].in_features == 129
    assert shuffled.item() == pytest.approx(prediction.item(), abs=1e-7)
    assert prediction.item() == pytest.approx(
        additive.item() + nonlinear.item(),
        abs=1e-7,
    )


def test_additive_neural_experiment_writes_checkpoints_and_rankings(
    tmp_path,
) -> None:
    possessions = _modeling_possessions(game_count=20)
    split = ChronologicalSplitConfig(
        cv_folds=2,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    training = NeuralTrainingConfig(
        random_seed=7,
        batch_size=32,
        max_epochs=2,
        early_stopping_patience=1,
        learning_rates=(0.001, 0.01),
        weight_decays=(0.0, 0.01),
        accelerator="cpu",
    )

    experiment = fit_additive_neural_experiment(
        possessions,
        checkpoint_dir=tmp_path,
        split_config=split,
        training_config=training,
        minimum_ranking_possessions=1.0,
    )

    assert experiment.selected_epochs in {1, 2}
    assert experiment.selected_learning_rate in training.learning_rates
    assert experiment.selected_weight_decay in training.weight_decays
    assert experiment.resolved_accelerator == "cpu"
    assert len(experiment.hyperparameter_trials) == 8
    assert len(experiment.hyperparameter_summary) == 4
    assert experiment.hyperparameter_summary["selected"].sum() == 1
    assert experiment.hyperparameter_summary["rank"].tolist() == [1, 2, 3, 4]
    assert set(experiment.test_metrics["model"]) == {"mean", "additive_neural"}
    assert experiment.test_predictions["game_id"].nunique() == 3
    assert len(experiment.player_rankings) == 20
    assert experiment.player_rankings["rank"].tolist() == list(range(1, 21))
    assert set(experiment.training_history["stage"]) == {
        "hyperparameter_search",
        "test_refit",
        "all_season_refit",
    }
    assert (tmp_path / "selection_model.ckpt").is_file()
    assert (tmp_path / "test_model.ckpt").is_file()
    assert (tmp_path / "model.ckpt").is_file()


def test_deep_sets_experiment_tracks_fixed_seeds_and_interactions(tmp_path) -> None:
    possessions = _modeling_possessions(game_count=20)
    split = ChronologicalSplitConfig(
        cv_folds=2,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    training = NeuralTrainingConfig(
        random_seed=7,
        batch_size=32,
        max_epochs=1,
        early_stopping_patience=0,
        learning_rates=(0.001,),
        weight_decays=(0.01,),
        accelerator="cpu",
    )
    architecture = DeepSetsArchitectureConfig(
        player_embedding_dim=4,
        role_embedding_dim=2,
        player_hidden_dim=8,
        pooled_dim=8,
        lineup_hidden_dims=(16, 8),
    )

    experiment = fit_deep_sets_experiment(
        possessions,
        checkpoint_dir=tmp_path,
        split_config=split,
        training_config=training,
        architecture_config=architecture,
        refit_seeds=(7, 8, 9),
        minimum_ranking_possessions=1.0,
    )

    assert experiment.refit_seeds == (7, 8, 9)
    assert experiment.leaderboard_seed == 7
    assert experiment.parameter_count > 0
    assert experiment.seed_metrics["seed"].tolist() == [7, 8, 9]
    assert set(experiment.test_metrics["model"]) == {"mean", "deep_sets"}
    assert experiment.test_predictions["leaderboard_seed"].unique().tolist() == [7]
    assert (
        experiment.test_predictions["prediction_deep_sets"]
        - experiment.test_predictions["prediction_additive_path"]
    ).to_numpy() == pytest.approx(
        experiment.test_predictions["prediction_nonlinear_residual"].to_numpy(),
        abs=1e-6,
    )
    assert not experiment.lineup_interactions.empty
    assert len(experiment.additive_player_components) == 20
    assert (tmp_path / "selection_model.ckpt").is_file()
    assert (tmp_path / "test_model.ckpt").is_file()
    assert (tmp_path / "test_model_seed_8.ckpt").is_file()
    assert (tmp_path / "test_model_seed_9.ckpt").is_file()
    assert (tmp_path / "model.ckpt").is_file()
    assert (tmp_path / "model_seed_8.ckpt").is_file()
    assert (tmp_path / "model_seed_9.ckpt").is_file()


def _source_segments() -> pd.DataFrame:
    shared = {
        "season": "2025-26",
        "season_type": "regular",
        "game_id": "g1",
        "game_date": datetime(2025, 10, 1).date(),
        "game_time_utc": datetime(2025, 10, 1, tzinfo=UTC),
        "period": 1,
        "catalog_home_team_id": 100,
        "catalog_away_team_id": 200,
        "catalog_home_team_tricode": "HOM",
        "catalog_away_team_tricode": "AWY",
        "quality_status": "pass",
        "source_build_run_id": "run",
        "processing_code_version": "sha256:" + "a" * 64,
        "play_by_play_sha256": "b" * 64,
        "boxscore_sha256": "c" * 64,
        "home_player_ids": [1, 2, 3, 4, 5],
        "away_player_ids": [6, 7, 8, 9, 10],
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "possession_id": "g1:0000",
                "possession_index": 0,
                "offense_team_id": 100,
                "defense_team_id": 200,
                "points_home": 2,
                "points_away": 0,
                "offense_points": 2,
            },
            {
                **shared,
                "possession_id": "g1:0001",
                "possession_index": 1,
                "offense_team_id": 200,
                "defense_team_id": 100,
                "points_home": 0,
                "points_away": 1,
                "offense_points": 1,
            },
            {
                **shared,
                "possession_id": "g1:0001",
                "possession_index": 1,
                "offense_team_id": 200,
                "defense_team_id": 100,
                "points_home": 0,
                "points_away": 1,
                "offense_points": 1,
            },
            {
                **shared,
                "possession_id": "g1:0002",
                "possession_index": 2,
                "offense_team_id": 200,
                "defense_team_id": 100,
                "points_home": 0,
                "points_away": 3,
                "offense_points": 3,
            },
        ]
    )


def _modeling_possessions(game_count: int) -> pd.DataFrame:
    start = datetime(2025, 10, 1, tzinfo=UTC)
    team_players = {
        100: [1, 2, 3, 4, 5],
        200: [6, 7, 8, 9, 10],
        300: [11, 12, 13, 14, 15],
        400: [16, 17, 18, 19, 20],
    }
    team_codes = {100: "AAA", 200: "BBB", 300: "CCC", 400: "DDD"}
    rows = []
    for game_index in range(game_count):
        game_time = start + timedelta(days=game_index)
        home_team = (100, 200, 300, 400)[game_index % 4]
        away_team = (200, 300, 400, 100)[game_index % 4]
        for possession_index in range(4):
            home_offense = possession_index % 2 == 0
            offense_team = home_team if home_offense else away_team
            defense_team = away_team if home_offense else home_team
            target = int(offense_team > defense_team)
            rows.append(
                {
                    "game_id": f"{game_index:010d}",
                    "game_date": game_time.date(),
                    "game_time_utc": game_time,
                    "possession_id": f"{game_index:010d}:{possession_index:04d}",
                    "possession_index": possession_index,
                    "period": 1,
                    "offense_team_id": offense_team,
                    "defense_team_id": defense_team,
                    "offense_team_tricode": team_codes[offense_team],
                    "defense_team_tricode": team_codes[defense_team],
                    "offense_player_ids": team_players[offense_team],
                    "defense_player_ids": team_players[defense_team],
                    "home_offense": home_offense,
                    "home_offense_sign": 1.0 if home_offense else -1.0,
                    "offense_points": target,
                    "defense_points": 0,
                    "target_offense_margin": target,
                    "target_home_margin": target if home_offense else -target,
                }
            )
    return pd.DataFrame(rows)
