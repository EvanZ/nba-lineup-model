from __future__ import annotations

import lightning as L
import torch
from torch import nn


class AdditivePlayerModel(nn.Module):
    """Signed scalar player embeddings with a centered home-offense effect."""

    def __init__(self, player_count: int) -> None:
        super().__init__()
        if player_count < 1:
            raise ValueError("Additive model requires at least one player")
        self.player_effects = nn.Embedding(player_count + 1, 1, padding_idx=0)
        self.intercept = nn.Parameter(torch.zeros(()))
        self.home_offense_effect = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.player_effects.weight)

    def forward(
        self,
        offense_player_indices: torch.Tensor,
        defense_player_indices: torch.Tensor,
        home_offense_sign: torch.Tensor,
    ) -> torch.Tensor:
        if offense_player_indices.ndim != 2 or offense_player_indices.shape[1] != 5:
            raise ValueError("Offense player indices must have shape [batch, 5]")
        if defense_player_indices.shape != offense_player_indices.shape:
            raise ValueError("Defense player indices must match offense shape")
        if home_offense_sign.ndim != 1:
            raise ValueError("Home-offense signs must have shape [batch]")
        offense = self.player_effects(offense_player_indices).squeeze(-1).sum(dim=1)
        defense = self.player_effects(defense_player_indices).squeeze(-1).sum(dim=1)
        return (
            self.intercept
            + self.home_offense_effect * home_offense_sign
            + offense
            - defense
        )

    def centered_player_values(self) -> torch.Tensor:
        """Return identifiable player values, excluding the reserved unknown row."""

        values = self.player_effects.weight[1:, 0]
        return values - values.mean()


class AdditiveRapmModule(L.LightningModule):
    """Lightning training wrapper for the additive player model."""

    def __init__(
        self,
        player_count: int,
        *,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")
        self.save_hyperparameters()
        self.model = AdditivePlayerModel(player_count)
        self.loss = nn.MSELoss()

    def forward(
        self,
        offense_player_indices: torch.Tensor,
        defense_player_indices: torch.Tensor,
        home_offense_sign: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            offense_player_indices,
            defense_player_indices,
            home_offense_sign,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        loss = self._batch_loss(batch)
        self.log(
            "train_mse",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch["target"]),
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        loss = self._batch_loss(batch)
        self.log(
            "val_mse",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["target"]),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.player_effects.parameters(),
                    "weight_decay": float(self.hparams.weight_decay),
                },
                {
                    "params": (
                        self.model.intercept,
                        self.model.home_offense_effect,
                    ),
                    "weight_decay": 0.0,
                },
            ],
            lr=float(self.hparams.learning_rate),
        )

    def _batch_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = self(
            batch["offense_player_indices"],
            batch["defense_player_indices"],
            batch["home_offense_sign"],
        )
        return self.loss(prediction, batch["target"])


class DeepSetsPlayerModel(nn.Module):
    """Permutation-invariant nonlinear lineups with an additive skip path."""

    def __init__(
        self,
        player_count: int,
        *,
        player_embedding_dim: int = 32,
        role_embedding_dim: int = 8,
        pooled_dim: int = 64,
        player_hidden_dim: int = 64,
        lineup_hidden_dims: tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        if player_count < 1:
            raise ValueError("Deep Sets model requires at least one player")
        dimensions = (
            player_embedding_dim,
            role_embedding_dim,
            pooled_dim,
            player_hidden_dim,
            *lineup_hidden_dims,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("Deep Sets dimensions must be positive")
        self.player_effects = nn.Embedding(player_count + 1, 1, padding_idx=0)
        self.player_embeddings = nn.Embedding(
            player_count + 1,
            player_embedding_dim,
            padding_idx=0,
        )
        self.role_embeddings = nn.Embedding(2, role_embedding_dim)
        self.player_mlp = nn.Sequential(
            nn.Linear(player_embedding_dim + role_embedding_dim, player_hidden_dim),
            nn.GELU(),
            nn.Linear(player_hidden_dim, pooled_dim),
            nn.GELU(),
        )
        self.lineup_mlp = nn.Sequential(
            nn.Linear(2 * pooled_dim + 1, lineup_hidden_dims[0]),
            nn.GELU(),
            nn.Linear(lineup_hidden_dims[0], lineup_hidden_dims[1]),
            nn.GELU(),
            nn.Linear(lineup_hidden_dims[1], 1),
        )
        self.intercept = nn.Parameter(torch.zeros(()))
        self.home_offense_effect = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.player_effects.weight)
        nn.init.normal_(self.player_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.player_embeddings.weight[0].zero_()
        final_layer = self.lineup_mlp[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Deep Sets output layer must be linear")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        offense_player_indices: torch.Tensor,
        defense_player_indices: torch.Tensor,
        home_offense_sign: torch.Tensor,
    ) -> torch.Tensor:
        additive, nonlinear = self.components(
            offense_player_indices,
            defense_player_indices,
            home_offense_sign,
        )
        return additive + nonlinear

    def components(
        self,
        offense_player_indices: torch.Tensor,
        defense_player_indices: torch.Tensor,
        home_offense_sign: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if offense_player_indices.ndim != 2 or offense_player_indices.shape[1] != 5:
            raise ValueError("Offense player indices must have shape [batch, 5]")
        if defense_player_indices.shape != offense_player_indices.shape:
            raise ValueError("Defense player indices must match offense shape")
        if home_offense_sign.ndim != 1:
            raise ValueError("Home-offense signs must have shape [batch]")
        offense_tokens = self._role_tokens(offense_player_indices, role_index=0)
        defense_tokens = self._role_tokens(defense_player_indices, role_index=1)
        offense_pool = self.player_mlp(offense_tokens).sum(dim=1)
        defense_pool = self.player_mlp(defense_tokens).sum(dim=1)
        nonlinear = self.lineup_mlp(
            torch.cat(
                [
                    offense_pool,
                    defense_pool,
                    home_offense_sign.unsqueeze(1),
                ],
                dim=1,
            )
        ).squeeze(1)
        offense_effect = (
            self.player_effects(offense_player_indices).squeeze(-1).sum(dim=1)
        )
        defense_effect = (
            self.player_effects(defense_player_indices).squeeze(-1).sum(dim=1)
        )
        additive = (
            self.intercept
            + self.home_offense_effect * home_offense_sign
            + offense_effect
            - defense_effect
        )
        return additive, nonlinear

    def centered_player_values(self) -> torch.Tensor:
        """Return additive-path values, excluding the reserved unknown row."""

        values = self.player_effects.weight[1:, 0]
        return values - values.mean()

    def _role_tokens(
        self,
        player_indices: torch.Tensor,
        *,
        role_index: int,
    ) -> torch.Tensor:
        player_values = self.player_embeddings(player_indices)
        roles = self.role_embeddings.weight[role_index].view(1, 1, -1)
        return torch.cat(
            [player_values, roles.expand(player_values.shape[0], 5, -1)],
            dim=2,
        )


class DeepSetsRapmModule(L.LightningModule):
    """Lightning wrapper for the additive-plus-Deep-Sets possession model."""

    def __init__(
        self,
        player_count: int,
        *,
        player_embedding_dim: int = 32,
        role_embedding_dim: int = 8,
        pooled_dim: int = 64,
        player_hidden_dim: int = 64,
        lineup_hidden_dims: tuple[int, int] = (128, 64),
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")
        self.save_hyperparameters()
        self.model = DeepSetsPlayerModel(
            player_count,
            player_embedding_dim=player_embedding_dim,
            role_embedding_dim=role_embedding_dim,
            pooled_dim=pooled_dim,
            player_hidden_dim=player_hidden_dim,
            lineup_hidden_dims=lineup_hidden_dims,
        )
        self.loss = nn.MSELoss()

    def forward(
        self,
        offense_player_indices: torch.Tensor,
        defense_player_indices: torch.Tensor,
        home_offense_sign: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            offense_player_indices,
            defense_player_indices,
            home_offense_sign,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        loss = self._batch_loss(batch)
        self.log(
            "train_mse",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch["target"]),
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        loss = self._batch_loss(batch)
        self.log(
            "val_mse",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["target"]),
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        excluded = {
            id(self.model.intercept),
            id(self.model.home_offense_effect),
        }
        regularized = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in excluded
        ]
        return torch.optim.AdamW(
            [
                {
                    "params": regularized,
                    "weight_decay": float(self.hparams.weight_decay),
                },
                {
                    "params": (
                        self.model.intercept,
                        self.model.home_offense_effect,
                    ),
                    "weight_decay": 0.0,
                },
            ],
            lr=float(self.hparams.learning_rate),
        )

    def _batch_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        prediction = self(
            batch["offense_player_indices"],
            batch["defense_player_indices"],
            batch["home_offense_sign"],
        )
        return self.loss(prediction, batch["target"])
