from __future__ import annotations

import pandas as pd
import pytest
import torch

from nba_lineup_model.modeling.profile_deep_sets import (
    LazyProfilePossessionTensorDataset,
    ProfilePossessionDataModule,
    ProfilePossessionTensorDataset,
    fit_profile_feature_scaler,
    player_ids_in_games,
    profile_data_module_factory,
    profile_token_lookups,
    profile_token_matrix,
    profile_token_tables,
)
from nba_lineup_model.modeling.profile_token_mart import TOKEN_FEATURE_COLUMNS
from nba_lineup_model.models.neural import ProfileDeepSetsPlayerModel


def _tokens() -> pd.DataFrame:
    rows = []
    for player_id in range(1, 21):
        row = {"target_season": "2025-26", "player_id": player_id}
        row.update(
            {
                feature: float(player_id * (index + 1))
                for index, feature in enumerate(TOKEN_FEATURE_COLUMNS)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _possessions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "game_id": ["g1", "g2"],
            "offense_player_ids": [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]],
            "defense_player_ids": [[6, 7, 8, 9, 10], [16, 17, 18, 19, 20]],
            "home_offense_sign": [1.0, -1.0],
            "target_offense_margin": [2.0, -1.0],
        }
    )


def test_profile_scaler_uses_only_players_in_training_games() -> None:
    possessions = _possessions()
    tokens = _tokens()

    player_ids = player_ids_in_games(possessions, ("g1",))
    scaler = fit_profile_feature_scaler(tokens, player_ids)

    assert player_ids == tuple(range(1, 11))
    assert scaler.player_count == 10
    assert scaler.means[0] == pytest.approx(5.5)
    assert scaler.means[-1] == pytest.approx(5.5 * len(TOKEN_FEATURE_COLUMNS))


def test_profile_tensor_dataset_aligns_profiles_to_lineup_vocabularies() -> None:
    possessions = _possessions()
    tokens = _tokens()
    player_columns = {player_id: player_id for player_id in range(1, 21)}
    scaler = fit_profile_feature_scaler(tokens, tuple(range(1, 6)))
    matrix = profile_token_matrix(tokens, player_columns, scaler)

    dataset = ProfilePossessionTensorDataset(
        possessions,
        player_columns,
        profile_token_lookups(tokens, scaler, ("2025-26",)),
    )

    assert dataset[0]["offense_profiles"].shape == (5, len(TOKEN_FEATURE_COLUMNS))
    assert dataset[0]["offense_profiles"][0, 0].item() == pytest.approx(-1.4142135)
    assert dataset[1]["defense_profiles"].shape == (5, len(TOKEN_FEATURE_COLUMNS))
    assert matrix[0].sum() == 0.0


def test_profile_data_module_uses_train_window_scaling_for_validation() -> None:
    possessions = _possessions()
    tokens = _tokens()
    player_columns = {player_id: player_id for player_id in range(1, 11)}
    module = ProfilePossessionDataModule(
        possessions,
        player_columns,
        tokens,
        train_game_ids=("g1",),
        validation_game_ids=("g2",),
        batch_size=2,
    )

    module.setup("fit")

    assert module.scaler is not None
    assert module.scaler.player_count == 10
    assert module.validation_dataset is not None
    batch = next(iter(module.val_dataloader()))
    assert batch["offense_profiles"].shape == (1, 5, len(TOKEN_FEATURE_COLUMNS))
    assert batch["offense_profiles"][0, 0, 0].item() == pytest.approx(1.9148542)


def test_lazy_profile_batch_vectorizes_lookup_and_preserves_unseen_profile() -> None:
    possessions = _possessions()
    tokens = _tokens()
    scaler = fit_profile_feature_scaler(tokens, tuple(range(1, 6)))
    dataset = LazyProfilePossessionTensorDataset(
        possessions,
        {player_id: player_id for player_id in range(1, 6)},
        profile_token_tables(tokens, scaler, ("2025-26",)),
    )

    batch = dataset.batch([0, 1])

    assert batch["offense_profiles"].shape == (2, 5, len(TOKEN_FEATURE_COLUMNS))
    assert batch["offense_profiles"][0, 0, 0].item() == pytest.approx(-1.4142135)
    # Player 11 is absent from the identity vocabulary but retains its side profile.
    assert batch["offense_player_indices"][1, 0].item() == 0
    assert batch["offense_profiles"][1, 0].abs().sum().item() > 0.0


def test_profile_data_module_factory_preserves_fold_boundaries() -> None:
    possessions = _possessions()
    factory = profile_data_module_factory(
        possessions,
        {player_id: player_id for player_id in range(1, 11)},
        _tokens(),
        batch_size=2,
        num_workers=0,
    )

    module = factory(("g1",), ("g2",), (), 23)
    module.setup("fit")

    assert module.random_seed == 23
    assert module.train_game_ids == ("g1",)
    assert module.validation_game_ids == ("g2",)


def test_profile_deep_sets_is_permutation_invariant() -> None:
    torch.manual_seed(4)
    model = ProfileDeepSetsPlayerModel(
        player_count=10,
        profile_feature_count=len(TOKEN_FEATURE_COLUMNS),
        player_embedding_dim=4,
        role_embedding_dim=2,
        profile_hidden_dim=8,
        player_hidden_dim=8,
        pooled_dim=8,
        lineup_hidden_dims=(16, 8),
    )
    with torch.no_grad():
        model.lineup_mlp[-1].weight.fill_(0.1)
    offense = torch.tensor([[1, 2, 3, 4, 5]])
    defense = torch.tensor([[6, 7, 8, 9, 10]])
    profiles = torch.arange(
        10 * len(TOKEN_FEATURE_COLUMNS), dtype=torch.float32
    ).reshape(10, len(TOKEN_FEATURE_COLUMNS))
    offense_profiles = profiles[:5].unsqueeze(0)
    defense_profiles = profiles[5:].unsqueeze(0)
    sign = torch.tensor([1.0])

    prediction = model(offense, defense, offense_profiles, defense_profiles, sign)
    shuffled = model(
        torch.tensor([[5, 3, 1, 4, 2]]),
        torch.tensor([[9, 6, 10, 8, 7]]),
        offense_profiles[:, [4, 2, 0, 3, 1]],
        defense_profiles[:, [3, 0, 4, 2, 1]],
        sign,
    )

    assert shuffled.item() == pytest.approx(prediction.item(), abs=1e-7)
