from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nba_lineup_model.modeling.nail_token_residual_backtest import (
    NAIL_TOKEN_FEATURE_COLUMNS,
    NailResidualStintDataset,
    build_frozen_residual_stints,
    fit_nail_token_scaler,
    nail_token_tables,
)
from nba_lineup_model.models.nail_token_residual import NailTokenResidualModel


def _tokens() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_offset, season in enumerate(("2023-24", "2024-25")):
        for player_id in range(1, 21):
            row: dict[str, object] = {
                "target_season": season,
                "player_id": player_id,
            }
            row.update(
                {
                    column: float(player_id + season_offset + index)
                    for index, column in enumerate(NAIL_TOKEN_FEATURE_COLUMNS)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _residual_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2023-24", "2024-25"],
            "home_player_ids": [tuple(range(1, 6)), tuple(range(11, 16))],
            "away_player_ids": [tuple(range(6, 11)), tuple(range(16, 21))],
            "target_residual_net_rating": [2.0, -3.0],
            "possessions": [100.0, 50.0],
        }
    )


@pytest.mark.parametrize("architecture", ["token_mlp", "set_attention"])
def test_nail_token_residual_is_permutation_invariant_and_antisymmetric(
    architecture: str,
) -> None:
    torch.manual_seed(7)
    model = NailTokenResidualModel(
        len(NAIL_TOKEN_FEATURE_COLUMNS),
        architecture=architecture,  # type: ignore[arg-type]
        hidden_dim=16,
        attention_heads=4,
        attention_layers=1,
        feedforward_dim=32,
    )
    with torch.no_grad():
        model.side_scorer.side_head[-1].weight.fill_(0.1)  # type: ignore[index,union-attr]
    profiles = torch.arange(80, dtype=torch.float32).reshape(
        10, len(NAIL_TOKEN_FEATURE_COLUMNS)
    )
    home = profiles[:5].unsqueeze(0)
    away = profiles[5:].unsqueeze(0)

    prediction = model(home, away)
    shuffled = model(home[:, [4, 0, 3, 1, 2]], away[:, [2, 4, 1, 0, 3]])
    reversed_prediction = model(away, home)

    assert shuffled.item() == pytest.approx(prediction.item(), abs=1e-6)
    assert reversed_prediction.item() == pytest.approx(-prediction.item(), abs=1e-6)


def test_nail_token_dataset_keeps_tokens_compact_and_gathers_batches() -> None:
    rows = _residual_rows()
    tokens = _tokens()
    scaler = fit_nail_token_scaler(tokens, rows)
    dataset = NailResidualStintDataset(
        rows,
        nail_token_tables(tokens, scaler, ("2023-24", "2024-25")),
    )

    batch = dataset.batch([0, 1])

    assert batch["home_profiles"].shape == (2, 5, len(NAIL_TOKEN_FEATURE_COLUMNS))
    assert batch["away_profiles"].shape == batch["home_profiles"].shape
    assert batch["target_residual"].tolist() == [2.0, -3.0]
    assert scaler.player_season_count == 20


def test_build_frozen_residual_stints_uses_prior_state_and_possession_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stints = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "home_player_ids": [tuple(range(1, 6)), tuple(range(1, 6))],
            "away_player_ids": [tuple(range(6, 11)), tuple(range(6, 11))],
            "possessions": [2.0, 3.0],
            "home_margin": [1.0, -1.0],
            "target_home_net_rating": [50.0, -100.0 / 3.0],
        }
    )

    class ContextModel:
        def predict_lineups(self, home: object, away: object, profiles: object) -> np.ndarray:
            del away, profiles
            return np.full(len(home), 2.0)  # type: ignore[arg-type]

    class State:
        priors = pd.DataFrame(
            {
                "season": "2024-25",
                "player_id": range(1, 11),
                "prior_rapm": [1.0] * 5 + [0.0] * 5,
            }
        )
        coefficients = pd.DataFrame(
            {
                "season": "2023-24",
                "player_id": range(1, 11),
                "rapm": 0.0,
            }
        )
        context_models = {"2023-24": ContextModel()}

    monkeypatch.setattr(
        "nba_lineup_model.modeling.nail_token_residual_backtest.read_rapm_stints",
        lambda *args, **kwargs: stints.copy(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.nail_token_residual_backtest._recover_home_intercept",
        lambda *args, **kwargs: 1.0,
    )

    result = build_frozen_residual_stints(
        "2024-25",
        state=State(),
        profiles=pd.DataFrame({"player_id": range(1, 11)}),
    )

    # Baseline is +5 player edge, +1 home court, +2 context = +8 NetRtg.
    expected = (100.0 * (1.0 - 1.0) - 8.0 * 5.0) / 5.0
    assert len(result) == 1
    assert result["target_residual_net_rating"].iat[0] == pytest.approx(expected)
    assert result["possessions"].iat[0] == 5.0
    assert result["stint_count"].iat[0] == 2
