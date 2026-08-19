"""Permutation-invariant residual models over lagged NAIL player profiles."""

from __future__ import annotations

from typing import Literal

import lightning as L
import torch
from torch import nn

TokenResidualArchitecture = Literal["token_mlp", "set_attention"]


class TokenMlpSideScorer(nn.Module):
    """Deep Sets side score without communication between player tokens."""

    def __init__(self, feature_count: int, *, hidden_dim: int = 32) -> None:
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.side_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.side_head[-1].weight)
        nn.init.zeros_(self.side_head[-1].bias)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        return self.side_head(self.token_mlp(profiles).mean(dim=1)).squeeze(1)


class SetAttentionSideScorer(nn.Module):
    """Shared within-unit self-attention followed by invariant mean pooling."""

    def __init__(
        self,
        feature_count: int,
        *,
        model_dim: int = 32,
        attention_heads: int = 4,
        attention_layers: int = 2,
        feedforward_dim: int = 64,
    ) -> None:
        super().__init__()
        if model_dim % attention_heads:
            raise ValueError("Set Attention model dimension must divide attention heads")
        self.input_projection = nn.Linear(feature_count, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=attention_layers)
        self.side_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        nn.init.zeros_(self.side_head[-1].weight)
        nn.init.zeros_(self.side_head[-1].bias)

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(profiles))
        return self.side_head(encoded.mean(dim=1)).squeeze(1)


class NailTokenResidualModel(nn.Module):
    """Antisymmetric home-minus-away residual over two five-player sets."""

    def __init__(
        self,
        feature_count: int,
        *,
        architecture: TokenResidualArchitecture,
        hidden_dim: int = 32,
        attention_heads: int = 4,
        attention_layers: int = 2,
        feedforward_dim: int = 64,
    ) -> None:
        super().__init__()
        if feature_count < 1 or hidden_dim < 1:
            raise ValueError("NAIL token residual dimensions must be positive")
        if architecture == "token_mlp":
            self.side_scorer: nn.Module = TokenMlpSideScorer(
                feature_count,
                hidden_dim=hidden_dim,
            )
        elif architecture == "set_attention":
            self.side_scorer = SetAttentionSideScorer(
                feature_count,
                model_dim=hidden_dim,
                attention_heads=attention_heads,
                attention_layers=attention_layers,
                feedforward_dim=feedforward_dim,
            )
        else:
            raise ValueError(f"Unknown NAIL token residual architecture: {architecture}")
        self.feature_count = feature_count
        self.architecture = architecture

    def forward(
        self,
        home_profiles: torch.Tensor,
        away_profiles: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_profiles(home_profiles, away_profiles)
        return self.side_scorer(home_profiles) - self.side_scorer(away_profiles)

    def _validate_profiles(
        self,
        home_profiles: torch.Tensor,
        away_profiles: torch.Tensor,
    ) -> None:
        expected_tail = (5, self.feature_count)
        if home_profiles.ndim != 3 or tuple(home_profiles.shape[1:]) != expected_tail:
            raise ValueError("Home profiles must have shape [batch, 5, features]")
        if away_profiles.shape != home_profiles.shape:
            raise ValueError("Away profiles must match home profiles")


class NailTokenResidualModule(L.LightningModule):
    """Lightning wrapper with possession-weighted stint residual loss."""

    def __init__(
        self,
        feature_count: int,
        *,
        architecture: TokenResidualArchitecture,
        hidden_dim: int = 32,
        attention_heads: int = 4,
        attention_layers: int = 2,
        feedforward_dim: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Invalid NAIL token residual optimizer configuration")
        self.save_hyperparameters()
        self.model = NailTokenResidualModel(
            feature_count,
            architecture=architecture,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            feedforward_dim=feedforward_dim,
        )

    def forward(
        self,
        home_profiles: torch.Tensor,
        away_profiles: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(home_profiles, away_profiles)

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        prediction = self(batch["home_profiles"], batch["away_profiles"])
        weights = batch["possessions"]
        loss = torch.sum(weights * torch.square(prediction - batch["target_residual"]))
        loss = loss / torch.sum(weights)
        self.log(
            "train_weighted_mse",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(weights),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
