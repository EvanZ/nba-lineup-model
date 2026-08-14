from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.neural_data import (
    PossessionTensorDataset,
    build_neural_possession_dataset,
    neural_code_fingerprint,
    player_vocabulary,
    read_neural_possessions,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    ChronologicalFold,
    ChronologicalSplitConfig,
    NeuralPossessionManifest,
    NeuralRapmRunManifest,
)
from nba_lineup_model.modeling.train import (
    GameFold,
    GameSplitPlan,
    chronological_game_splits,
)
from nba_lineup_model.models.neural import AdditiveRapmModule

DEFAULT_RANDOM_SEED = 17
DEFAULT_BATCH_SIZE = 2_048
DEFAULT_MAX_EPOCHS = 30
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003)
DEFAULT_WEIGHT_DECAYS = (0.0, 0.001, 0.01, 0.1, 1.0)
NeuralModuleFactory = Callable[[int, float, float, float], L.LightningModule]
NeuralDataModuleFactory = Callable[
    [tuple[str, ...], tuple[str, ...], tuple[str, ...], int],
    L.LightningDataModule,
]


@dataclass(frozen=True)
class NeuralTrainingConfig:
    """Hyperparameters and runtime controls for the first neural baseline."""

    random_seed: int = DEFAULT_RANDOM_SEED
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    learning_rates: tuple[float, ...] = DEFAULT_LEARNING_RATES
    weight_decays: tuple[float, ...] = DEFAULT_WEIGHT_DECAYS
    accelerator: Literal["cpu", "mps", "auto"] = "cpu"
    num_workers: int = 0

    def validate(self) -> None:
        if self.random_seed < 0:
            raise ValueError("Random seed cannot be negative")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.max_epochs < 1:
            raise ValueError("Maximum epochs must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("Early-stopping patience cannot be negative")
        if not self.learning_rates:
            raise ValueError("At least one learning rate is required")
        if not self.weight_decays:
            raise ValueError("At least one weight decay is required")
        if any(not np.isfinite(value) or value <= 0 for value in self.learning_rates):
            raise ValueError("Learning rates must be finite and positive")
        if any(not np.isfinite(value) or value < 0 for value in self.weight_decays):
            raise ValueError("Weight decays must be finite and nonnegative")
        if len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("Learning-rate candidates must be unique")
        if len(set(self.weight_decays)) != len(self.weight_decays):
            raise ValueError("Weight-decay candidates must be unique")
        if self.accelerator not in {"cpu", "mps", "auto"}:
            raise ValueError("Accelerator must be cpu, mps, or auto")
        if self.num_workers < 0:
            raise ValueError("Data-loader workers cannot be negative")


@dataclass(frozen=True)
class AdditiveNeuralExperiment:
    """In-memory summaries plus checkpoints written in a temporary run directory."""

    split_plan: GameSplitPlan
    player_columns: dict[int, int]
    selected_epochs: int
    selected_learning_rate: float
    selected_weight_decay: float
    resolved_accelerator: str
    training_history: pd.DataFrame
    hyperparameter_trials: pd.DataFrame
    hyperparameter_summary: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    player_rankings: pd.DataFrame
    game_splits: pd.DataFrame
    model_parameters: dict[str, Any]


@dataclass(frozen=True)
class _HyperparameterSearch:
    selected_learning_rate: float
    selected_weight_decay: float
    selected_epochs: int
    resolved_accelerator: str
    training_history: pd.DataFrame
    trials: pd.DataFrame
    summary: pd.DataFrame


class PossessionDataModule(L.LightningDataModule):
    """Lightning data module backed by contiguous possession tensors."""

    def __init__(
        self,
        possessions: pd.DataFrame,
        player_columns: dict[int, int],
        *,
        train_game_ids: tuple[str, ...],
        validation_game_ids: tuple[str, ...] = (),
        test_game_ids: tuple[str, ...] = (),
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = 0,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        super().__init__()
        self.possessions = possessions
        self.player_columns = player_columns
        self.train_game_ids = train_game_ids
        self.validation_game_ids = validation_game_ids
        self.test_game_ids = test_game_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.train_dataset: PossessionTensorDataset | None = None
        self.validation_dataset: PossessionTensorDataset | None = None
        self.test_dataset: PossessionTensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        game_ids = self.possessions["game_id"].astype(str)
        self.train_dataset = PossessionTensorDataset(
            self.possessions.loc[game_ids.isin(self.train_game_ids)],
            self.player_columns,
        )
        self.validation_dataset = self._optional_dataset(
            game_ids.isin(self.validation_game_ids)
        )
        self.test_dataset = self._optional_dataset(game_ids.isin(self.test_game_ids))

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Data module has not been set up")
        generator = torch.Generator().manual_seed(self.random_seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, torch.Tensor]] | list[DataLoader[dict[str, torch.Tensor]]]:
        if self.validation_dataset is None:
            return []
        return self._evaluation_loader(self.validation_dataset)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, torch.Tensor]] | list[DataLoader[dict[str, torch.Tensor]]]:
        if self.test_dataset is None:
            return []
        return self._evaluation_loader(self.test_dataset)

    def _optional_dataset(self, mask: pd.Series) -> PossessionTensorDataset | None:
        frame = self.possessions.loc[mask]
        if frame.empty:
            return None
        return PossessionTensorDataset(frame, self.player_columns)

    def _evaluation_loader(
        self,
        dataset: PossessionTensorDataset,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class _MetricHistory(Callback):
    def __init__(self, stage: str) -> None:
        super().__init__()
        self.stage = stage
        self.rows: list[dict[str, Any]] = []

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        del pl_module
        self._record(trainer)

    def _record(self, trainer: L.Trainer) -> None:
        metrics = trainer.callback_metrics
        row: dict[str, Any] = {
            "stage": self.stage,
            "epoch": trainer.current_epoch + 1,
            "train_mse": _optional_metric(metrics.get("train_mse")),
            "validation_mse": _optional_metric(metrics.get("val_mse")),
        }
        self.rows.append(row)


def train_additive_neural_rapm(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    split_config: ChronologicalSplitConfig | None = None,
    training_config: NeuralTrainingConfig | None = None,
    minimum_ranking_possessions: float = 500.0,
    enable_progress_bar: bool = True,
) -> tuple[NeuralRapmRunManifest, Path]:
    """Build neural possessions, train additive RAPM, and persist an atomic run."""

    split = split_config or ChronologicalSplitConfig()
    config = training_config or NeuralTrainingConfig()
    config.validate()
    if minimum_ranking_possessions < 0:
        raise ValueError("Minimum ranking possessions cannot be negative")
    dataset_manifest = build_neural_possession_dataset(
        season,
        curated_dir=curated_dir,
        analytical_dir=analytical_dir,
    )
    possessions = read_neural_possessions(season, analytical_dir)
    bios_path = Path(curated_dir) / "player_seasons" / season / "regular" / "part-00000.parquet"
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    return _write_experiment(
        season,
        dataset_manifest,
        possessions,
        player_bios,
        split,
        config,
        minimum_ranking_possessions,
        analytical_dir,
        artifacts_dir,
        enable_progress_bar,
    )


def fit_additive_neural_experiment(
    possessions: pd.DataFrame,
    *,
    checkpoint_dir: Path | str,
    split_config: ChronologicalSplitConfig | None = None,
    training_config: NeuralTrainingConfig | None = None,
    minimum_ranking_possessions: float = 500.0,
    player_bios: pd.DataFrame | None = None,
    enable_progress_bar: bool = False,
) -> AdditiveNeuralExperiment:
    """Select epochs, evaluate the final test, and refit the full season."""

    split = split_config or ChronologicalSplitConfig()
    config = training_config or NeuralTrainingConfig()
    config.validate()
    _validate_possessions(possessions)
    if minimum_ranking_possessions < 0:
        raise ValueError("Minimum ranking possessions cannot be negative")
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_plan = chronological_game_splits(possessions, split)
    player_columns = player_vocabulary(possessions)
    search = _search_hyperparameters(
        possessions,
        player_columns,
        split_plan.folds,
        config,
        output_dir,
        _new_additive_module,
        enable_progress_bar,
    )

    test_model, test_history, _ = _fit_fixed_epoch_model(
        possessions,
        player_columns,
        split_plan.final_train_game_ids,
        search.selected_epochs,
        config,
        output_dir / "test_model.ckpt",
        learning_rate=search.selected_learning_rate,
        weight_decay=search.selected_weight_decay,
        module_factory=_new_additive_module,
        stage="test_refit",
        seed_offset=1,
        enable_progress_bar=enable_progress_bar,
    )
    test_rows = possessions.loc[
        possessions["game_id"].astype(str).isin(split_plan.final_test_game_ids)
    ].copy()
    train_rows = possessions.loc[
        possessions["game_id"].astype(str).isin(split_plan.final_train_game_ids)
    ]
    additive_predictions = _predict(
        test_model,
        test_rows,
        player_columns,
        config,
    )
    mean_prediction = float(train_rows["target_offense_margin"].mean())
    mean_predictions = np.full(len(test_rows), mean_prediction)
    test_metrics = _test_metrics(test_rows, mean_predictions, additive_predictions)
    test_predictions = _test_predictions(
        test_rows,
        mean_predictions,
        additive_predictions,
    )

    all_season_ids = tuple(
        split_plan.ordered_games["game_id"].astype(str).tolist()
    )
    all_season_model, all_season_history, _ = _fit_fixed_epoch_model(
        possessions,
        player_columns,
        all_season_ids,
        search.selected_epochs,
        config,
        output_dir / "model.ckpt",
        learning_rate=search.selected_learning_rate,
        weight_decay=search.selected_weight_decay,
        module_factory=_new_additive_module,
        stage="all_season_refit",
        seed_offset=2,
        enable_progress_bar=enable_progress_bar,
    )
    rankings = _player_rankings(
        possessions,
        player_columns,
        all_season_model,
        minimum_ranking_possessions,
        player_bios,
    )
    history = pd.concat(
        [search.training_history, test_history, all_season_history],
        ignore_index=True,
    )
    all_values = all_season_model.model.centered_player_values().detach().cpu().numpy()
    model_parameters = {
        "architecture": "additive signed scalar player embeddings",
        "equation": (
            "intercept + home_offense_effect * home_offense_sign "
            "+ sum(offense_player_value) - sum(defense_player_value)"
        ),
        "target": "offense points minus defense points on one possession",
        "lineup_policy": "exclude every possession with more than one lineup segment",
        "unknown_player_index": 0,
        "player_net_rating_conversion": (
            "centered embedding value times 200 for two role-swapped possessions"
        ),
        "hyperparameter_selection_metric": "validation_possession_weighted_mse",
        "hyperparameter_selection_folds": len(split_plan.folds),
        "learning_rate_candidates": list(config.learning_rates),
        "weight_decay_candidates": list(config.weight_decays),
        "selected_learning_rate": search.selected_learning_rate,
        "selected_weight_decay": search.selected_weight_decay,
        "epoch_selection_fold": split_plan.folds[-1].fold,
        "selected_epochs": search.selected_epochs,
        "mean_test_baseline": mean_prediction,
        "all_season_intercept_points_per_possession": float(
            all_season_model.model.intercept.detach().cpu()
        ),
        "all_season_home_offense_effect_points_per_possession": float(
            all_season_model.model.home_offense_effect.detach().cpu()
        ),
        "all_season_home_court_net_rating": float(
            200.0 * all_season_model.model.home_offense_effect.detach().cpu()
        ),
        "all_season_player_embedding_mean_after_centering": float(all_values.mean()),
        "final_train_unseen_player_count": _unseen_player_count(train_rows, player_columns),
    }
    return AdditiveNeuralExperiment(
        split_plan=split_plan,
        player_columns=player_columns,
        selected_epochs=search.selected_epochs,
        selected_learning_rate=search.selected_learning_rate,
        selected_weight_decay=search.selected_weight_decay,
        resolved_accelerator=search.resolved_accelerator,
        training_history=history,
        hyperparameter_trials=search.trials,
        hyperparameter_summary=search.summary,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
        player_rankings=rankings,
        game_splits=_game_splits_frame(split_plan),
        model_parameters=model_parameters,
    )


def validate_neural_rapm_run(run_dir: Path | str) -> NeuralRapmRunManifest:
    """Require every recorded neural artifact to match its manifest."""

    root = Path(run_dir)
    manifest = NeuralRapmRunManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    actual = {
        path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    expected = {artifact.filename for artifact in manifest.artifacts}
    if actual != expected:
        raise ValueError("Neural model run files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Neural artifact byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Neural artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None and len(pd.read_parquet(path)) != artifact.row_count:
            raise ValueError(f"Neural artifact rows changed: {artifact.filename}")
    return manifest


def _search_hyperparameters(
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    folds: tuple[GameFold, ...],
    config: NeuralTrainingConfig,
    output_dir: Path,
    module_factory: NeuralModuleFactory,
    enable_progress_bar: bool,
    data_module_factory: NeuralDataModuleFactory | None = None,
) -> _HyperparameterSearch:
    search_dir = output_dir / "hyperparameter_checkpoints"
    search_dir.mkdir()
    game_ids = possessions["game_id"].astype(str)
    trial_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    accelerators: set[str] = set()
    candidates = tuple(product(config.learning_rates, config.weight_decays))
    for candidate_index, (learning_rate, weight_decay) in enumerate(
        candidates,
        start=1,
    ):
        for fold in folds:
            checkpoint_path = (
                search_dir
                / f"candidate-{candidate_index:03d}-fold-{fold.fold}"
                / "model.ckpt"
            )
            history, selected_epochs, resolved_accelerator = _fit_selection_model(
                possessions,
                player_columns,
                fold.train_game_ids,
                fold.validation_game_ids,
                config,
                checkpoint_path,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                module_factory=module_factory,
                enable_progress_bar=enable_progress_bar,
                data_module_factory=data_module_factory,
            )
            accelerators.add(resolved_accelerator)
            selected_row = history.loc[history["epoch"].eq(selected_epochs)]
            if len(selected_row) != 1:
                raise ValueError("Selected epoch is missing from training history")
            validation_mse = float(selected_row["validation_mse"].iloc[0])
            validation_possession_count = int(
                game_ids.isin(fold.validation_game_ids).sum()
            )
            trial_rows.append(
                {
                    "candidate_index": candidate_index,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "fold": fold.fold,
                    "train_game_count": len(fold.train_game_ids),
                    "validation_game_count": len(fold.validation_game_ids),
                    "validation_possession_count": validation_possession_count,
                    "selected_epochs": selected_epochs,
                    "validation_mse": validation_mse,
                }
            )
            history.insert(1, "candidate_index", candidate_index)
            history.insert(2, "fold", fold.fold)
            history.insert(3, "learning_rate", learning_rate)
            history.insert(4, "weight_decay", weight_decay)
            history_frames.append(history)
    if len(accelerators) != 1:
        raise ValueError("Hyperparameter trials resolved to different accelerators")

    trials = pd.DataFrame(trial_rows).sort_values(
        ["candidate_index", "fold"],
        kind="stable",
    )
    summaries: list[dict[str, Any]] = []
    latest_fold = max(fold.fold for fold in folds)
    for candidate_index, candidate_trials in trials.groupby(
        "candidate_index",
        sort=True,
    ):
        weights = candidate_trials["validation_possession_count"].to_numpy(
            dtype=float
        )
        losses = candidate_trials["validation_mse"].to_numpy(dtype=float)
        latest = candidate_trials.loc[candidate_trials["fold"].eq(latest_fold)].iloc[0]
        summaries.append(
            {
                "candidate_index": int(candidate_index),
                "learning_rate": float(candidate_trials["learning_rate"].iloc[0]),
                "weight_decay": float(candidate_trials["weight_decay"].iloc[0]),
                "fold_count": len(candidate_trials),
                "validation_possession_count": int(weights.sum()),
                "weighted_validation_mse": float(np.average(losses, weights=weights)),
                "mean_validation_mse": float(losses.mean()),
                "validation_mse_std": float(losses.std(ddof=0)),
                "latest_fold_validation_mse": float(latest["validation_mse"]),
                "latest_fold_selected_epochs": int(latest["selected_epochs"]),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["weighted_validation_mse", "candidate_index"],
        kind="stable",
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary["selected"] = summary["rank"].eq(1)
    winner = summary.iloc[0]
    selected_candidate = int(winner["candidate_index"])
    selected_epochs = int(winner["latest_fold_selected_epochs"])
    selected_checkpoint = (
        search_dir
        / f"candidate-{selected_candidate:03d}-fold-{latest_fold}"
        / "model.ckpt"
    )
    shutil.copy2(selected_checkpoint, output_dir / "selection_model.ckpt")
    shutil.rmtree(search_dir)
    return _HyperparameterSearch(
        selected_learning_rate=float(winner["learning_rate"]),
        selected_weight_decay=float(winner["weight_decay"]),
        selected_epochs=selected_epochs,
        resolved_accelerator=next(iter(accelerators)),
        training_history=pd.concat(history_frames, ignore_index=True),
        trials=trials.reset_index(drop=True),
        summary=summary,
    )


def _fit_selection_model(
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    train_game_ids: tuple[str, ...],
    validation_game_ids: tuple[str, ...],
    config: NeuralTrainingConfig,
    checkpoint_path: Path,
    *,
    learning_rate: float,
    weight_decay: float,
    module_factory: NeuralModuleFactory,
    enable_progress_bar: bool,
    data_module_factory: NeuralDataModuleFactory | None = None,
) -> tuple[pd.DataFrame, int, str]:
    L.seed_everything(config.random_seed, workers=True, verbose=False)
    data = (
        data_module_factory(train_game_ids, validation_game_ids, (), config.random_seed)
        if data_module_factory is not None
        else PossessionDataModule(
            possessions,
            player_columns,
            train_game_ids=train_game_ids,
            validation_game_ids=validation_game_ids,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_seed=config.random_seed,
        )
    )
    train_mean = float(
        possessions.loc[
            possessions["game_id"].astype(str).isin(train_game_ids),
            "target_offense_margin",
        ].mean()
    )
    module = module_factory(
        len(player_columns),
        learning_rate,
        weight_decay,
        train_mean,
    )
    history = _MetricHistory("hyperparameter_search")
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_path.parent,
        filename=checkpoint_path.stem,
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_mse",
        mode="min",
        patience=config.early_stopping_patience,
        min_delta=1e-6,
        check_finite=True,
    )
    trainer = _trainer(
        config,
        max_epochs=config.max_epochs,
        callbacks=[history, checkpoint, early_stopping],
        enable_progress_bar=enable_progress_bar,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=data)
    checkpoint_data = torch.load(
        checkpoint.best_model_path,
        map_location="cpu",
        weights_only=False,
    )
    selected_epochs = int(checkpoint_data["epoch"]) + 1
    return (
        pd.DataFrame(history.rows),
        selected_epochs,
        trainer.strategy.root_device.type,
    )


def _fit_fixed_epoch_model(
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    train_game_ids: tuple[str, ...],
    epochs: int,
    config: NeuralTrainingConfig,
    checkpoint_path: Path,
    *,
    learning_rate: float,
    weight_decay: float,
    module_factory: NeuralModuleFactory,
    stage: str,
    seed_offset: int,
    enable_progress_bar: bool,
    data_module_factory: NeuralDataModuleFactory | None = None,
) -> tuple[L.LightningModule, pd.DataFrame, str]:
    seed = config.random_seed + seed_offset
    L.seed_everything(seed, workers=True, verbose=False)
    data = (
        data_module_factory(train_game_ids, (), (), seed)
        if data_module_factory is not None
        else PossessionDataModule(
            possessions,
            player_columns,
            train_game_ids=train_game_ids,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_seed=seed,
        )
    )
    train_mean = float(
        possessions.loc[
            possessions["game_id"].astype(str).isin(train_game_ids),
            "target_offense_margin",
        ].mean()
    )
    module = module_factory(
        len(player_columns),
        learning_rate,
        weight_decay,
        train_mean,
    )
    history = _MetricHistory(stage)
    trainer = _trainer(
        config,
        max_epochs=epochs,
        callbacks=[history],
        enable_progress_bar=enable_progress_bar,
        enable_checkpointing=False,
    )
    data.setup("fit")
    trainer.fit(module, train_dataloaders=data.train_dataloader())
    trainer.save_checkpoint(checkpoint_path, weights_only=False)
    return module, pd.DataFrame(history.rows), trainer.strategy.root_device.type


def _new_additive_module(
    player_count: int,
    learning_rate: float,
    weight_decay: float,
    target_mean: float,
) -> L.LightningModule:
    module = AdditiveRapmModule(
        player_count,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    with torch.no_grad():
        module.model.intercept.fill_(target_mean)
    return module


def _trainer(
    config: NeuralTrainingConfig,
    *,
    max_epochs: int,
    callbacks: list[Callback],
    enable_progress_bar: bool,
    enable_checkpointing: bool,
) -> L.Trainer:
    return L.Trainer(
        accelerator=config.accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=False,
        deterministic=True,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )


def _predict(
    module: L.LightningModule,
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    config: NeuralTrainingConfig,
) -> np.ndarray:
    dataset = PossessionTensorDataset(possessions, player_columns)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    module.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            prediction = module(
                batch["offense_player_indices"].to(module.device),
                batch["defense_player_indices"].to(module.device),
                batch["home_offense_sign"].to(module.device),
            )
            predictions.append(prediction.detach().cpu().numpy())
    return np.concatenate(predictions)


def _test_metrics(
    test_rows: pd.DataFrame,
    mean_predictions: np.ndarray,
    additive_predictions: np.ndarray,
) -> pd.DataFrame:
    actual = test_rows["target_offense_margin"].to_numpy(dtype=float)
    rows = [
        _metric_row(test_rows, actual, predictions, model=model)
        for model, predictions in (
            ("mean", mean_predictions),
            ("additive_neural", additive_predictions),
        )
    ]
    metrics = pd.DataFrame(rows)
    mean_mse = float(metrics.loc[metrics["model"].eq("mean"), "mse"].iloc[0])
    metrics["skill_vs_mean"] = metrics["mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    return metrics


def _metric_row(
    possessions: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    model: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "test_game_count": int(possessions["game_id"].nunique()),
        "test_possession_count": len(possessions),
        "mse": mean_squared_error(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "game_margin_rmse": possession_game_margin_rmse(
            possessions["game_id"],
            actual,
            predicted,
            possessions["home_offense_sign"].to_numpy(dtype=float),
        ),
    }


def _test_predictions(
    test_rows: pd.DataFrame,
    mean_predictions: np.ndarray,
    additive_predictions: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "game_id",
        "game_time_utc",
        "possession_id",
        "possession_index",
        "period",
        "offense_team_id",
        "defense_team_id",
        "home_offense",
        "home_offense_sign",
        "target_offense_margin",
        "target_home_margin",
    ]
    output = test_rows.loc[:, columns].reset_index(drop=True)
    output["prediction_mean"] = mean_predictions
    output["prediction_additive_neural"] = additive_predictions
    output["predicted_home_margin_mean"] = (
        mean_predictions * output["home_offense_sign"]
    )
    output["predicted_home_margin_additive_neural"] = (
        additive_predictions * output["home_offense_sign"]
    )
    return output


def _player_rankings(
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    module: AdditiveRapmModule,
    minimum_possessions: float,
    player_bios: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for side, sign in (("offense", 1.0), ("defense", -1.0)):
        side_frame = possessions.loc[
            :,
            [
                f"{side}_player_ids",
                f"{side}_team_id",
                f"{side}_team_tricode",
                "target_offense_margin",
            ],
        ].explode(f"{side}_player_ids", ignore_index=True)
        side_frame = side_frame.rename(
            columns={
                f"{side}_player_ids": "player_id",
                f"{side}_team_id": "team_id",
                f"{side}_team_tricode": "team_tricode",
            }
        )
        side_frame["player_margin"] = sign * side_frame["target_offense_margin"]
        side_frame["side"] = side
        rows.append(side_frame)
    exposure = pd.concat(rows, ignore_index=True)
    exposure["player_id"] = exposure["player_id"].astype("int64")
    totals = exposure.groupby("player_id", as_index=False).agg(
        possessions=("player_id", "size"),
        point_margin=("player_margin", "sum"),
        offense_possessions=("side", lambda values: int(values.eq("offense").sum())),
        defense_possessions=("side", lambda values: int(values.eq("defense").sum())),
    )
    totals["raw_on_court_net_rating"] = (
        100.0 * totals["point_margin"] / totals["possessions"]
    )
    team_exposure = (
        exposure.groupby(["player_id", "team_id", "team_tricode"], as_index=False)
        .size()
        .rename(columns={"size": "primary_team_possessions"})
        .sort_values(
            ["player_id", "primary_team_possessions", "team_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("player_id")
        .rename(
            columns={
                "team_id": "primary_team_id",
                "team_tricode": "primary_team_tricode",
            }
        )
    )
    identifiers = [
        player_id
        for player_id, _ in sorted(player_columns.items(), key=lambda item: item[1])
    ]
    values = module.model.centered_player_values().detach().cpu().numpy()
    rankings = pd.DataFrame(
        {
            "player_id": identifiers,
            "embedding_value_points_per_possession": values,
            "neural_rapm": 200.0 * values,
        }
    )
    rankings = rankings.merge(totals, on="player_id", how="left", validate="one_to_one")
    rankings = rankings.merge(
        team_exposure,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    name_map: dict[int, str] = {}
    if player_bios is not None and {"player_id", "player_name"} <= set(player_bios.columns):
        name_map = {
            int(row.player_id): str(row.player_name)
            for row in player_bios.loc[:, ["player_id", "player_name"]].itertuples(
                index=False
            )
        }
    rankings["player_name"] = (
        rankings["player_id"].map(name_map).fillna(rankings["player_id"].astype(str))
    )
    rankings = rankings.sort_values(
        ["neural_rapm", "possessions", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    rankings["rank"] = np.arange(1, len(rankings) + 1)
    rankings["exposure_eligible"] = rankings["possessions"].ge(minimum_possessions)
    eligible = rankings.loc[rankings["exposure_eligible"]].sort_values(
        ["neural_rapm", "possessions", "player_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    rankings["eligible_rank"] = pd.Series(
        np.arange(1, len(eligible) + 1),
        index=eligible.index,
        dtype="Int64",
    )
    return rankings.loc[
        :,
        [
            "rank",
            "eligible_rank",
            "player_id",
            "player_name",
            "primary_team_id",
            "primary_team_tricode",
            "neural_rapm",
            "embedding_value_points_per_possession",
            "raw_on_court_net_rating",
            "possessions",
            "offense_possessions",
            "defense_possessions",
            "point_margin",
            "primary_team_possessions",
            "exposure_eligible",
        ],
    ]


def _game_splits_frame(split_plan: GameSplitPlan) -> pd.DataFrame:
    game_metadata = split_plan.ordered_games.set_index("game_id")
    rows: list[dict[str, Any]] = []
    for fold in split_plan.folds:
        for role, identifiers in (
            ("train", fold.train_game_ids),
            ("validation", fold.validation_game_ids),
        ):
            rows.extend(
                _split_row(game_metadata, f"cv_{fold.fold}", role, game_id)
                for game_id in identifiers
            )
    for role, identifiers in (
        ("train", split_plan.final_train_game_ids),
        ("test", split_plan.final_test_game_ids),
    ):
        rows.extend(
            _split_row(game_metadata, "final", role, game_id)
            for game_id in identifiers
        )
    return pd.DataFrame(rows).sort_values(
        ["split", "game_time_utc", "game_id"],
        kind="stable",
    ).reset_index(drop=True)


def _split_row(
    game_metadata: pd.DataFrame,
    split: str,
    role: str,
    game_id: str,
) -> dict[str, Any]:
    metadata = game_metadata.loc[game_id]
    return {
        "split": split,
        "role": role,
        "game_id": game_id,
        "game_date": metadata["game_date"],
        "game_time_utc": metadata["game_time_utc"],
    }


def _write_experiment(
    season: str,
    dataset_manifest: NeuralPossessionManifest,
    possessions: pd.DataFrame,
    player_bios: pd.DataFrame | None,
    split_config: ChronologicalSplitConfig,
    training_config: NeuralTrainingConfig,
    minimum_ranking_possessions: float,
    analytical_dir: Path | str,
    artifacts_dir: Path | str,
    enable_progress_bar: bool,
) -> tuple[NeuralRapmRunManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"neural-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(artifacts_dir) / "neural_rapm" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        experiment = fit_additive_neural_experiment(
            possessions,
            checkpoint_dir=temporary_dir,
            split_config=split_config,
            training_config=training_config,
            minimum_ranking_possessions=minimum_ranking_possessions,
            player_bios=player_bios,
            enable_progress_bar=enable_progress_bar,
        )
        parquet_outputs = {
            "training_history.parquet": experiment.training_history,
            "hyperparameter_trials.parquet": experiment.hyperparameter_trials,
            "hyperparameter_summary.parquet": experiment.hyperparameter_summary,
            "test_metrics.parquet": experiment.test_metrics,
            "test_predictions.parquet": experiment.test_predictions,
            "player_rankings.parquet": experiment.player_rankings,
            "game_splits.parquet": experiment.game_splits,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        json_outputs = {
            "player_columns.json": {
                str(player_id): column
                for player_id, column in experiment.player_columns.items()
            },
            "model_parameters.json": experiment.model_parameters,
        }
        for filename, payload in json_outputs.items():
            (temporary_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=(
                    len(parquet_outputs[path.name]) if path.name in parquet_outputs else None
                ),
                byte_count=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in sorted(temporary_dir.iterdir())
            if path.is_file()
        )
        folds = tuple(
            ChronologicalFold(
                fold=fold.fold,
                train_game_count=len(fold.train_game_ids),
                validation_game_count=len(fold.validation_game_ids),
                train_first_game_id=fold.train_game_ids[0],
                train_last_game_id=fold.train_game_ids[-1],
                validation_first_game_id=fold.validation_game_ids[0],
                validation_last_game_id=fold.validation_game_ids[-1],
            )
            for fold in experiment.split_plan.folds
        )
        selection_fold = experiment.split_plan.folds[-1]
        dataset_dir = (
            Path(analytical_dir) / "neural_possessions" / season / "regular"
        )
        manifest = NeuralRapmRunManifest(
            schema_version=2,
            run_id=run_id,
            created_at=now,
            season=season,
            architecture="additive",
            neural_code_version=neural_code_fingerprint(),
            dataset_manifest_sha256=_sha256_file(dataset_dir / "_manifest.json"),
            dataset_part_sha256=dataset_manifest.part_sha256,
            possession_count=len(possessions),
            game_count=int(possessions["game_id"].nunique()),
            player_count=len(experiment.player_columns),
            split_config=split_config,
            folds=folds,
            selection_train_game_count=len(selection_fold.train_game_ids),
            selection_validation_game_count=len(selection_fold.validation_game_ids),
            final_train_game_count=len(experiment.split_plan.final_train_game_ids),
            final_test_game_count=len(experiment.split_plan.final_test_game_ids),
            random_seed=training_config.random_seed,
            batch_size=training_config.batch_size,
            max_epochs=training_config.max_epochs,
            early_stopping_patience=training_config.early_stopping_patience,
            selected_epochs=experiment.selected_epochs,
            learning_rate=experiment.selected_learning_rate,
            weight_decay=experiment.selected_weight_decay,
            learning_rate_candidates=training_config.learning_rates,
            weight_decay_candidates=training_config.weight_decays,
            hyperparameter_selection_metric="validation_possession_weighted_mse",
            requested_accelerator=training_config.accelerator,
            resolved_accelerator=experiment.resolved_accelerator,
            target="offense_points_minus_defense_points",
            minimum_ranking_possessions=minimum_ranking_possessions,
            torch_version=torch.__version__,
            lightning_version=L.__version__,
            artifacts=artifacts,
        )
        (temporary_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        temporary_dir.replace(run_dir)
        validate_neural_rapm_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _validate_possessions(possessions: pd.DataFrame) -> None:
    required = {
        "game_id",
        "game_date",
        "game_time_utc",
        "possession_id",
        "offense_team_id",
        "defense_team_id",
        "offense_team_tricode",
        "defense_team_tricode",
        "offense_player_ids",
        "defense_player_ids",
        "home_offense_sign",
        "target_offense_margin",
        "target_home_margin",
    }
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(f"Neural possessions missing columns: {sorted(missing)}")
    if possessions.empty:
        raise ValueError("Neural possessions cannot be empty")
    if possessions.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Neural possession keys must be unique")
    if not possessions["home_offense_sign"].isin((-1.0, 1.0)).all():
        raise ValueError("Home-offense signs must be negative or positive one")


def _unseen_player_count(
    train_rows: pd.DataFrame,
    player_columns: dict[int, int],
) -> int:
    seen = set().union(
        *train_rows["offense_player_ids"],
        *train_rows["defense_player_ids"],
    )
    return len(set(player_columns) - seen)


def _optional_metric(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the regular-season additive neural RAPM baseline."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rates",
        action="append",
        type=float,
        help=(
            "Candidate learning rate; repeat to override the default grid "
            f"{DEFAULT_LEARNING_RATES}"
        ),
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decays",
        action="append",
        type=float,
        help=(
            "Candidate AdamW weight decay; repeat to override the default grid "
            f"{DEFAULT_WEIGHT_DECAYS}"
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--minimum-ranking-possessions", type=float, default=500.0)
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable Lightning progress bars",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_config = ChronologicalSplitConfig(
        cv_folds=args.cv_folds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    training_config = NeuralTrainingConfig(
        random_seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        learning_rates=(
            tuple(args.learning_rates)
            if args.learning_rates is not None
            else DEFAULT_LEARNING_RATES
        ),
        weight_decays=(
            tuple(args.weight_decays)
            if args.weight_decays is not None
            else DEFAULT_WEIGHT_DECAYS
        ),
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )
    manifest, run_dir = train_additive_neural_rapm(
        args.season,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        split_config=split_config,
        training_config=training_config,
        minimum_ranking_possessions=args.minimum_ranking_possessions,
        enable_progress_bar=not args.no_progress_bar,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = (
        f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    )
    print(
        f"{manifest.season} additive neural RAPM: "
        f"possessions={manifest.possession_count}, games={manifest.game_count}, "
        f"players={manifest.player_count}, lr={manifest.learning_rate:g}, "
        f"weight_decay={manifest.weight_decay:g}, "
        f"epochs={manifest.selected_epochs}, "
        f"accelerator={manifest.resolved_accelerator}; "
        f"run={run_dir}{tracking_text}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
