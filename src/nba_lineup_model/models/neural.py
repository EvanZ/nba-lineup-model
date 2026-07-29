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
