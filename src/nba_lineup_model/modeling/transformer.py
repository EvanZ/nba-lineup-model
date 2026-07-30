from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.neural import (
    DEFAULT_RANDOM_SEED,
    NeuralTrainingConfig,
    _game_splits_frame,
    _MetricHistory,
    _trainer,
    _validate_possessions,
)
from nba_lineup_model.modeling.neural_data import (
    PossessionTensorDataset,
    player_vocabulary,
    read_neural_possessions,
    validate_neural_possession_partition,
)
from nba_lineup_model.modeling.residual_data import (
    build_rapm_base_prediction_dataset,
    read_rapm_base_predictions,
    read_rapm_base_state,
    validate_rapm_base_prediction_partition,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    ChronologicalFold,
    RapmBasePredictionManifest,
    RapmTransformerRunManifest,
)
from nba_lineup_model.modeling.train import (
    GameSplitPlan,
    chronological_game_splits,
)
from nba_lineup_model.models.neural import RapmTransformerModule

DEFAULT_TRANSFORMER_BATCH_SIZE = 8_192
DEFAULT_TRANSFORMER_MAX_EPOCHS = 10
DEFAULT_TRANSFORMER_EARLY_STOPPING_PATIENCE = 3
DEFAULT_TRANSFORMER_LEARNING_RATES = (0.0003, 0.001)
DEFAULT_TRANSFORMER_WEIGHT_DECAYS = (0.0, 0.01)


@dataclass(frozen=True)
class RapmTransformerArchitectureConfig:
    """Fixed architecture for the first attention-based lineup residual."""

    d_model: int = 32
    attention_heads: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1

    def validate(self) -> None:
        dimensions = (
            self.d_model,
            self.attention_heads,
            self.transformer_layers,
            self.feedforward_dim,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("Transformer dimensions must be positive")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("Transformer width must be divisible by attention heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Transformer dropout must be in [0, 1)")


@dataclass(frozen=True)
class RapmTransformerExperiment:
    """In-memory outputs ready for an immutable Transformer model run."""

    split_plan: GameSplitPlan
    player_columns: dict[int, int]
    selected_epochs: int
    selected_learning_rate: float
    selected_weight_decay: float
    resolved_accelerator: str
    refit_seeds: tuple[int, ...]
    leaderboard_seed: int
    parameter_count: int
    training_history: pd.DataFrame
    hyperparameter_trials: pd.DataFrame
    hyperparameter_summary: pd.DataFrame
    test_metrics: pd.DataFrame
    seed_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    seed_predictions: pd.DataFrame
    lineup_residuals: pd.DataFrame
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


class RapmResidualTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Possession tensors augmented with one frozen RAPM base prediction."""

    def __init__(
        self,
        possessions: pd.DataFrame,
        player_columns: dict[int, int],
    ) -> None:
        if "base_prediction" not in possessions:
            raise ValueError("Transformer rows require base_prediction")
        self.possessions = PossessionTensorDataset(possessions, player_columns)
        self.base_prediction = torch.as_tensor(
            possessions["base_prediction"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.possessions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        output = dict(self.possessions[index])
        output["base_prediction"] = self.base_prediction[index]
        return output


class RapmResidualDataModule(L.LightningDataModule):
    """Lightning loaders for one stage of the RAPM residual mart."""

    def __init__(
        self,
        stage_rows: pd.DataFrame,
        player_columns: dict[int, int],
        *,
        validation_role: Literal["validation", "test"] | None = None,
        batch_size: int = DEFAULT_TRANSFORMER_BATCH_SIZE,
        num_workers: int = 0,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        super().__init__()
        self.stage_rows = stage_rows
        self.player_columns = player_columns
        self.validation_role = validation_role
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.train_dataset: RapmResidualTensorDataset | None = None
        self.validation_dataset: RapmResidualTensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        train = self.stage_rows.loc[self.stage_rows["base_role"].eq("train")]
        if train.empty:
            raise ValueError("Transformer stage has no training rows")
        self.train_dataset = RapmResidualTensorDataset(
            train,
            self.player_columns,
        )
        if self.validation_role is None:
            self.validation_dataset = None
        else:
            validation = self.stage_rows.loc[
                self.stage_rows["base_role"].eq(self.validation_role)
            ]
            if validation.empty:
                raise ValueError("Transformer stage has no validation rows")
            self.validation_dataset = RapmResidualTensorDataset(
                validation,
                self.player_columns,
            )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Transformer data module has not been set up")
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
    ) -> DataLoader[dict[str, torch.Tensor]] | list[DataLoader]:
        if self.validation_dataset is None:
            return []
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def train_rapm_transformer(
    season: str,
    *,
    source_rapm_run_id: str | None = None,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    training_config: NeuralTrainingConfig | None = None,
    architecture_config: RapmTransformerArchitectureConfig | None = None,
    refit_seeds: tuple[int, ...] | None = None,
    enable_progress_bar: bool = True,
) -> tuple[RapmTransformerRunManifest, Path]:
    """Build frozen RAPM predictions, train the residual, and persist a run."""

    training = training_config or default_transformer_training_config()
    architecture = architecture_config or RapmTransformerArchitectureConfig()
    seeds = refit_seeds or (
        training.random_seed,
        training.random_seed + 1,
        training.random_seed + 2,
    )
    base_manifest = build_rapm_base_prediction_dataset(
        season,
        source_rapm_run_id=source_rapm_run_id,
        curated_dir=curated_dir,
        analytical_dir=analytical_dir,
        artifacts_dir=artifacts_dir,
    )
    possessions = read_neural_possessions(season, analytical_dir)
    base_predictions = read_rapm_base_predictions(season, analytical_dir)
    coefficients, stage_parameters = read_rapm_base_state(
        season,
        analytical_dir,
    )
    return _write_experiment(
        season,
        possessions,
        base_predictions,
        coefficients,
        stage_parameters,
        base_manifest,
        training,
        architecture,
        seeds,
        analytical_dir,
        artifacts_dir,
        enable_progress_bar,
    )


def fit_rapm_transformer_experiment(
    possessions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    *,
    checkpoint_dir: Path | str,
    split_plan: GameSplitPlan | None = None,
    training_config: NeuralTrainingConfig | None = None,
    architecture_config: RapmTransformerArchitectureConfig | None = None,
    refit_seeds: tuple[int, ...] | None = None,
    enable_progress_bar: bool = False,
) -> RapmTransformerExperiment:
    """Select optimization settings, evaluate fixed seeds, and refit all games."""

    training = training_config or default_transformer_training_config()
    architecture = architecture_config or RapmTransformerArchitectureConfig()
    training.validate()
    architecture.validate()
    _validate_possessions(possessions)
    seeds = refit_seeds or (
        training.random_seed,
        training.random_seed + 1,
        training.random_seed + 2,
    )
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Transformer requires at least three unique nonnegative seeds")
    plan = split_plan or chronological_game_splits(
        possessions,
        _split_config_from_base_predictions(base_predictions),
    )
    _validate_base_prediction_input(possessions, base_predictions, plan)
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    player_columns = player_vocabulary(possessions)
    stage_frames = {
        stage: _stage_training_frame(possessions, base_predictions, stage)
        for stage in {
            *(f"cv_{fold.fold}" for fold in plan.folds),
            "final",
            "all_season",
        }
    }
    search = _search_hyperparameters(
        stage_frames,
        player_columns,
        plan,
        training,
        architecture,
        output_dir,
        enable_progress_bar,
    )

    final_rows = stage_frames["final"]
    test_rows = final_rows.loc[final_rows["base_role"].eq("test")].copy()
    final_train_rows = final_rows.loc[final_rows["base_role"].eq("train")]
    mean_prediction = float(final_train_rows["target_offense_margin"].mean())
    mean_predictions = np.full(len(test_rows), mean_prediction)
    histories = [search.training_history]
    seed_metric_rows: list[dict[str, Any]] = []
    seed_prediction_frames: list[pd.DataFrame] = []
    canonical_predictions: np.ndarray | None = None
    canonical_base: np.ndarray | None = None
    canonical_residual: np.ndarray | None = None
    canonical_full_model: RapmTransformerModule | None = None
    accelerators = {search.resolved_accelerator}
    leaderboard_seed = seeds[0]

    for seed in seeds:
        test_checkpoint = (
            output_dir / "test_model.ckpt"
            if seed == leaderboard_seed
            else output_dir / f"test_model_seed_{seed}.ckpt"
        )
        test_model, test_history, resolved = _fit_fixed_epoch_model(
            final_rows,
            player_columns,
            search.selected_epochs,
            training,
            architecture,
            test_checkpoint,
            learning_rate=search.selected_learning_rate,
            weight_decay=search.selected_weight_decay,
            stage="test_refit",
            seed=seed,
            enable_progress_bar=enable_progress_bar,
        )
        test_history.insert(1, "refit_seed", seed)
        histories.append(test_history)
        accelerators.add(resolved)
        predictions, base, residual = _predict_components(
            test_model,
            test_rows,
            player_columns,
            training,
        )
        seed_metric_rows.append(
            _metric_row(test_rows, predictions, model="rapm_transformer", seed=seed)
        )
        seed_prediction_frames.append(
            pd.DataFrame(
                {
                    "game_id": test_rows["game_id"].astype(str).to_numpy(),
                    "possession_id": test_rows["possession_id"].astype(str).to_numpy(),
                    "seed": seed,
                    "prediction_rapm": base,
                    "prediction_transformer_residual": residual,
                    "prediction_rapm_transformer": predictions,
                }
            )
        )
        if seed == leaderboard_seed:
            canonical_predictions = predictions
            canonical_base = base
            canonical_residual = residual

        full_checkpoint = (
            output_dir / "model.ckpt"
            if seed == leaderboard_seed
            else output_dir / f"model_seed_{seed}.ckpt"
        )
        full_model, full_history, resolved = _fit_fixed_epoch_model(
            stage_frames["all_season"],
            player_columns,
            search.selected_epochs,
            training,
            architecture,
            full_checkpoint,
            learning_rate=search.selected_learning_rate,
            weight_decay=search.selected_weight_decay,
            stage="all_season_refit",
            seed=seed,
            enable_progress_bar=enable_progress_bar,
        )
        full_history.insert(1, "refit_seed", seed)
        histories.append(full_history)
        accelerators.add(resolved)
        if seed == leaderboard_seed:
            canonical_full_model = full_model

    if len(accelerators) != 1:
        raise ValueError("Transformer stages resolved to different accelerators")
    if (
        canonical_predictions is None
        or canonical_base is None
        or canonical_residual is None
        or canonical_full_model is None
    ):
        raise RuntimeError("Canonical Transformer refit was not produced")

    mean_row = _metric_row(test_rows, mean_predictions, model="mean", seed=None)
    rapm_row = _metric_row(test_rows, canonical_base, model="ridge_rapm", seed=None)
    transformer_row = next(
        row for row in seed_metric_rows if row["seed"] == leaderboard_seed
    )
    test_metrics = pd.DataFrame([mean_row, rapm_row, transformer_row])
    _add_skill_columns(test_metrics)
    seed_metrics = pd.DataFrame(seed_metric_rows)
    _add_skill_columns(
        seed_metrics,
        mean_mse=float(mean_row["mse"]),
        mean_game_mse=float(mean_row["game_margin_rmse"]) ** 2,
        rapm_mse=float(rapm_row["mse"]),
        rapm_game_mse=float(rapm_row["game_margin_rmse"]) ** 2,
    )
    test_predictions = _test_predictions(
        test_rows,
        mean_predictions,
        canonical_base,
        canonical_residual,
        canonical_predictions,
        leaderboard_seed,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in canonical_full_model.parameters()
        if parameter.requires_grad
    )
    model_parameters = {
        "architecture": "frozen RAPM plus position-free Transformer residual",
        "equation": "prediction_rapm + transformer_residual",
        "token_count": 13,
        "tokens": [
            "state",
            "offense_marker",
            "five_offense_players",
            "defense_marker",
            "five_defense_players",
        ],
        "positional_encoding": "none",
        "player_role_combination": "addition",
        "context_features": ["home_offense_sign"],
        "d_model": architecture.d_model,
        "attention_heads": architecture.attention_heads,
        "transformer_layers": architecture.transformer_layers,
        "feedforward_dim": architecture.feedforward_dim,
        "dropout": architecture.dropout,
        "parameter_count": parameter_count,
        "learning_rate_candidates": list(training.learning_rates),
        "weight_decay_candidates": list(training.weight_decays),
        "selected_learning_rate": search.selected_learning_rate,
        "selected_weight_decay": search.selected_weight_decay,
        "selected_epochs": search.selected_epochs,
        "hyperparameter_selection_metric": "validation_possession_weighted_mse",
        "refit_seeds": list(seeds),
        "leaderboard_seed": leaderboard_seed,
        "base_training_rows": "same-stage fitted RAPM predictions",
        "base_evaluation_rows": "strictly earlier-game RAPM predictions",
        "target": "offense points minus defense points on one possession",
        "lineup_policy": "exclude every possession with more than one lineup segment",
    }
    return RapmTransformerExperiment(
        split_plan=plan,
        player_columns=player_columns,
        selected_epochs=search.selected_epochs,
        selected_learning_rate=search.selected_learning_rate,
        selected_weight_decay=search.selected_weight_decay,
        resolved_accelerator=next(iter(accelerators)),
        refit_seeds=seeds,
        leaderboard_seed=leaderboard_seed,
        parameter_count=parameter_count,
        training_history=pd.concat(histories, ignore_index=True),
        hyperparameter_trials=search.trials,
        hyperparameter_summary=search.summary,
        test_metrics=test_metrics,
        seed_metrics=seed_metrics,
        test_predictions=test_predictions,
        seed_predictions=pd.concat(seed_prediction_frames, ignore_index=True),
        lineup_residuals=_lineup_residuals(test_predictions, test_rows),
        game_splits=_game_splits_frame(plan),
        model_parameters=model_parameters,
    )


def validate_rapm_transformer_run(
    run_dir: Path | str,
) -> RapmTransformerRunManifest:
    """Require every Transformer artifact to match its run manifest."""

    root = Path(run_dir)
    manifest = RapmTransformerRunManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != expected:
        raise ValueError("RAPM Transformer files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"Transformer byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Transformer artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None and len(pd.read_parquet(path)) != artifact.row_count:
            raise ValueError(f"Transformer artifact rows changed: {artifact.filename}")
    return manifest


def frozen_rapm_predictions(
    possessions: pd.DataFrame,
    coefficients: dict[int, float],
    *,
    intercept_home_net_rating: float,
    mean_offense_margin: float,
) -> tuple[np.ndarray, int]:
    """Apply a stored RAPM state in offense orientation."""

    signed_effect = np.empty(len(possessions), dtype=float)
    unknown_exposures = 0
    for index, (offense, defense) in enumerate(
        zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            strict=True,
        )
    ):
        unknown_exposures += sum(
            int(player_id) not in coefficients for player_id in offense
        )
        unknown_exposures += sum(
            int(player_id) not in coefficients for player_id in defense
        )
        signed_effect[index] = sum(
            coefficients.get(int(player_id), 0.0) for player_id in offense
        ) - sum(
            coefficients.get(int(player_id), 0.0) for player_id in defense
        )
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    return (
        mean_offense_margin
        + signed_effect / 200.0
        + signs * intercept_home_net_rating / 200.0,
        unknown_exposures,
    )


def rapm_transformer_predictions(
    module: RapmTransformerModule,
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    base_predictions: np.ndarray,
    *,
    batch_size: int = DEFAULT_TRANSFORMER_BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted Transformer and return total and residual predictions."""

    frame = possessions.copy()
    frame["base_prediction"] = np.asarray(base_predictions, dtype=float)
    config = NeuralTrainingConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rates=(float(module.hparams.learning_rate),),
        weight_decays=(float(module.hparams.weight_decay),),
    )
    total, base, residual = _predict_components(
        module,
        frame,
        player_columns,
        config,
    )
    if not np.allclose(base, base_predictions, rtol=0, atol=1e-7):
        raise ValueError("Transformer changed its frozen RAPM base predictions")
    return total, residual


def transformer_code_fingerprint(
    source_paths: tuple[Path | str, ...] | None = None,
) -> str:
    """Hash Transformer-owned model, training, and evaluation sources."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        paths = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "neural.py",
            package_root / "modeling" / "neural_data.py",
            package_root / "modeling" / "residual_data.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "transformer.py",
            package_root / "models" / "neural.py",
        )
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one Transformer source path is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"Transformer source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def default_transformer_training_config() -> NeuralTrainingConfig:
    """Return the CPU-conscious first Transformer search budget."""

    return NeuralTrainingConfig(
        batch_size=DEFAULT_TRANSFORMER_BATCH_SIZE,
        max_epochs=DEFAULT_TRANSFORMER_MAX_EPOCHS,
        early_stopping_patience=DEFAULT_TRANSFORMER_EARLY_STOPPING_PATIENCE,
        learning_rates=DEFAULT_TRANSFORMER_LEARNING_RATES,
        weight_decays=DEFAULT_TRANSFORMER_WEIGHT_DECAYS,
    )


def _stage_training_frame(
    possessions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    base = base_predictions.loc[
        base_predictions["stage"].eq(stage),
        [
            "game_id",
            "possession_id",
            "role",
            "base_is_out_of_sample",
            "prediction_rapm",
            "residual_target",
            "target_offense_margin",
        ],
    ].rename(
        columns={
            "role": "base_role",
            "prediction_rapm": "base_prediction",
            "target_offense_margin": "base_target_offense_margin",
        }
    )
    frame = possessions.merge(
        base,
        on=["game_id", "possession_id"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(frame) != len(base):
        raise ValueError(f"Transformer stage does not cover every base row: {stage}")
    if not np.array_equal(
        frame["target_offense_margin"].to_numpy(dtype=float),
        frame["base_target_offense_margin"].to_numpy(dtype=float),
    ):
        raise ValueError(f"Transformer stage target does not match base mart: {stage}")
    return frame.drop(columns="base_target_offense_margin")


def _search_hyperparameters(
    stage_frames: dict[str, pd.DataFrame],
    player_columns: dict[int, int],
    split_plan: GameSplitPlan,
    training: NeuralTrainingConfig,
    architecture: RapmTransformerArchitectureConfig,
    output_dir: Path,
    enable_progress_bar: bool,
) -> _HyperparameterSearch:
    search_dir = output_dir / "hyperparameter_checkpoints"
    search_dir.mkdir()
    trial_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    accelerators: set[str] = set()
    candidates = tuple(product(training.learning_rates, training.weight_decays))
    for candidate_index, (learning_rate, weight_decay) in enumerate(
        candidates,
        start=1,
    ):
        for fold in split_plan.folds:
            stage = f"cv_{fold.fold}"
            checkpoint = (
                search_dir
                / f"candidate-{candidate_index:03d}-fold-{fold.fold}"
                / "model.ckpt"
            )
            history, selected_epochs, resolved = _fit_selection_model(
                stage_frames[stage],
                player_columns,
                training,
                architecture,
                checkpoint,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                enable_progress_bar=enable_progress_bar,
            )
            accelerators.add(resolved)
            selected = history.loc[history["epoch"].eq(selected_epochs)]
            if len(selected) != 1:
                raise ValueError("Transformer selected epoch is missing from history")
            validation_count = int(
                stage_frames[stage]["base_role"].eq("validation").sum()
            )
            trial_rows.append(
                {
                    "candidate_index": candidate_index,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "fold": fold.fold,
                    "train_game_count": len(fold.train_game_ids),
                    "validation_game_count": len(fold.validation_game_ids),
                    "validation_possession_count": validation_count,
                    "selected_epochs": selected_epochs,
                    "validation_mse": float(selected["validation_mse"].iloc[0]),
                }
            )
            history.insert(1, "candidate_index", candidate_index)
            history.insert(2, "fold", fold.fold)
            history.insert(3, "learning_rate", learning_rate)
            history.insert(4, "weight_decay", weight_decay)
            histories.append(history)
    if len(accelerators) != 1:
        raise ValueError("Transformer search stages resolved to different accelerators")
    trials = pd.DataFrame(trial_rows).sort_values(
        ["candidate_index", "fold"],
        kind="stable",
    )
    latest_fold = max(fold.fold for fold in split_plan.folds)
    summaries = []
    for candidate_index, candidate in trials.groupby("candidate_index", sort=True):
        weights = candidate["validation_possession_count"].to_numpy(dtype=float)
        losses = candidate["validation_mse"].to_numpy(dtype=float)
        latest = candidate.loc[candidate["fold"].eq(latest_fold)].iloc[0]
        summaries.append(
            {
                "candidate_index": int(candidate_index),
                "learning_rate": float(candidate["learning_rate"].iloc[0]),
                "weight_decay": float(candidate["weight_decay"].iloc[0]),
                "fold_count": len(candidate),
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
        training_history=pd.concat(histories, ignore_index=True),
        trials=trials.reset_index(drop=True),
        summary=summary,
    )


def _fit_selection_model(
    stage_rows: pd.DataFrame,
    player_columns: dict[int, int],
    training: NeuralTrainingConfig,
    architecture: RapmTransformerArchitectureConfig,
    checkpoint_path: Path,
    *,
    learning_rate: float,
    weight_decay: float,
    enable_progress_bar: bool,
) -> tuple[pd.DataFrame, int, str]:
    L.seed_everything(training.random_seed, workers=True, verbose=False)
    data = RapmResidualDataModule(
        stage_rows,
        player_columns,
        validation_role="validation",
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        random_seed=training.random_seed,
    )
    module = _new_transformer_module(
        len(player_columns),
        learning_rate,
        weight_decay,
        architecture,
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
        patience=training.early_stopping_patience,
        min_delta=1e-6,
        check_finite=True,
    )
    trainer = _trainer(
        training,
        max_epochs=training.max_epochs,
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
    return (
        pd.DataFrame(history.rows),
        int(checkpoint_data["epoch"]) + 1,
        trainer.strategy.root_device.type,
    )


def _fit_fixed_epoch_model(
    stage_rows: pd.DataFrame,
    player_columns: dict[int, int],
    epochs: int,
    training: NeuralTrainingConfig,
    architecture: RapmTransformerArchitectureConfig,
    checkpoint_path: Path,
    *,
    learning_rate: float,
    weight_decay: float,
    stage: str,
    seed: int,
    enable_progress_bar: bool,
) -> tuple[RapmTransformerModule, pd.DataFrame, str]:
    L.seed_everything(seed, workers=True, verbose=False)
    data = RapmResidualDataModule(
        stage_rows,
        player_columns,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        random_seed=seed,
    )
    data.setup("fit")
    module = _new_transformer_module(
        len(player_columns),
        learning_rate,
        weight_decay,
        architecture,
    )
    history = _MetricHistory(stage)
    trainer = _trainer(
        training,
        max_epochs=epochs,
        callbacks=[history],
        enable_progress_bar=enable_progress_bar,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=data.train_dataloader())
    trainer.save_checkpoint(checkpoint_path, weights_only=False)
    return module, pd.DataFrame(history.rows), trainer.strategy.root_device.type


def _new_transformer_module(
    player_count: int,
    learning_rate: float,
    weight_decay: float,
    architecture: RapmTransformerArchitectureConfig,
) -> RapmTransformerModule:
    return RapmTransformerModule(
        player_count,
        d_model=architecture.d_model,
        attention_heads=architecture.attention_heads,
        transformer_layers=architecture.transformer_layers,
        feedforward_dim=architecture.feedforward_dim,
        dropout=architecture.dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _predict_components(
    module: RapmTransformerModule,
    rows: pd.DataFrame,
    player_columns: dict[int, int],
    training: NeuralTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = RapmResidualTensorDataset(rows, player_columns)
    loader = DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
    )
    totals: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    module.eval()
    with torch.inference_mode():
        for batch in loader:
            base, residual = module.model.components(
                batch["offense_player_indices"].to(module.device),
                batch["defense_player_indices"].to(module.device),
                batch["home_offense_sign"].to(module.device),
                batch["base_prediction"].to(module.device),
            )
            totals.append((base + residual).detach().cpu().numpy())
            bases.append(base.detach().cpu().numpy())
            residuals.append(residual.detach().cpu().numpy())
    return (
        np.concatenate(totals),
        np.concatenate(bases),
        np.concatenate(residuals),
    )


def _metric_row(
    rows: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    seed: int | None,
) -> dict[str, Any]:
    actual = rows["target_offense_margin"].to_numpy(dtype=float)
    return {
        "model": model,
        "seed": seed,
        "test_game_count": int(rows["game_id"].nunique()),
        "test_possession_count": len(rows),
        "mse": mean_squared_error(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "game_margin_rmse": possession_game_margin_rmse(
            rows["game_id"],
            actual,
            predicted,
            rows["home_offense_sign"].to_numpy(dtype=float),
        ),
    }


def _add_skill_columns(
    metrics: pd.DataFrame,
    *,
    mean_mse: float | None = None,
    mean_game_mse: float | None = None,
    rapm_mse: float | None = None,
    rapm_game_mse: float | None = None,
) -> None:
    if mean_mse is None:
        mean_row = metrics.loc[metrics["model"].eq("mean")].iloc[0]
        mean_mse = float(mean_row["mse"])
        mean_game_mse = float(mean_row["game_margin_rmse"]) ** 2
    if rapm_mse is None:
        rapm_row = metrics.loc[metrics["model"].eq("ridge_rapm")].iloc[0]
        rapm_mse = float(rapm_row["mse"])
        rapm_game_mse = float(rapm_row["game_margin_rmse"]) ** 2
    if mean_game_mse is None or rapm_game_mse is None:
        raise ValueError("Skill references are incomplete")
    metrics["possession_skill_vs_mean"] = metrics["mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    metrics["game_margin_skill_vs_mean"] = metrics["game_margin_rmse"].map(
        lambda value: skill_score(float(value) ** 2, mean_game_mse)
    )
    metrics["possession_skill_vs_rapm"] = metrics["mse"].map(
        lambda value: skill_score(float(value), rapm_mse)
    )
    metrics["game_margin_skill_vs_rapm"] = metrics["game_margin_rmse"].map(
        lambda value: skill_score(float(value) ** 2, rapm_game_mse)
    )


def _test_predictions(
    rows: pd.DataFrame,
    mean: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
    total: np.ndarray,
    seed: int,
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
    output = rows.loc[:, columns].reset_index(drop=True)
    output["leaderboard_seed"] = seed
    output["prediction_mean"] = mean
    output["prediction_rapm"] = base
    output["prediction_transformer_residual"] = residual
    output["prediction_rapm_transformer"] = total
    output["predicted_home_margin_mean"] = mean * output["home_offense_sign"]
    output["predicted_home_margin_rapm"] = base * output["home_offense_sign"]
    output["predicted_home_margin_rapm_transformer"] = (
        total * output["home_offense_sign"]
    )
    return output


def _lineup_residuals(
    predictions: pd.DataFrame,
    test_rows: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.copy()
    frame["offense_lineup"] = test_rows["offense_player_ids"].map(
        lambda values: "-".join(str(value) for value in sorted(values))
    ).to_numpy()
    frame["defense_lineup"] = test_rows["defense_player_ids"].map(
        lambda values: "-".join(str(value) for value in sorted(values))
    ).to_numpy()
    frame["absolute_transformer_residual"] = frame[
        "prediction_transformer_residual"
    ].abs()
    grouped = frame.groupby(
        ["offense_lineup", "defense_lineup"],
        as_index=False,
        sort=False,
    ).agg(
        game_count=("game_id", "nunique"),
        possession_count=("possession_id", "size"),
        actual_offense_margin=("target_offense_margin", "mean"),
        rapm_prediction=("prediction_rapm", "mean"),
        transformer_prediction=("prediction_rapm_transformer", "mean"),
        transformer_residual=("prediction_transformer_residual", "mean"),
        mean_absolute_transformer_residual=(
            "absolute_transformer_residual",
            "mean",
        ),
    )
    return grouped.sort_values(
        ["mean_absolute_transformer_residual", "possession_count"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def _validate_base_prediction_input(
    possessions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    split_plan: GameSplitPlan,
) -> None:
    expected_stages = {
        *(f"cv_{fold.fold}" for fold in split_plan.folds),
        "final",
        "all_season",
    }
    if set(base_predictions["stage"].astype(str)) != expected_stages:
        raise ValueError("Transformer base predictions have unexpected stages")
    if base_predictions.duplicated(["stage", "game_id", "possession_id"]).any():
        raise ValueError("Transformer base prediction keys must be unique")
    if not base_predictions["base_is_out_of_sample"].eq(
        base_predictions["role"].ne("train")
    ).all():
        raise ValueError("Transformer base sample flags do not match roles")
    all_season = base_predictions.loc[base_predictions["stage"].eq("all_season")]
    if len(all_season) != len(possessions):
        raise ValueError("Transformer all-season base rows do not cover possessions")
    if set(all_season["game_id"].astype(str)) != set(
        split_plan.ordered_games["game_id"].astype(str)
    ):
        raise ValueError("Transformer base rows have different games")


def _split_config_from_base_predictions(
    base_predictions: pd.DataFrame,
):
    from nba_lineup_model.modeling.schema import ChronologicalSplitConfig

    cv_folds = len(
        {
            stage
            for stage in base_predictions["stage"].astype(str)
            if stage.startswith("cv_")
        }
    )
    return ChronologicalSplitConfig(cv_folds=cv_folds)


def _write_experiment(
    season: str,
    possessions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    stage_parameters: pd.DataFrame,
    base_manifest: RapmBasePredictionManifest,
    training: NeuralTrainingConfig,
    architecture: RapmTransformerArchitectureConfig,
    refit_seeds: tuple[int, ...],
    analytical_dir: Path | str,
    artifacts_dir: Path | str,
    enable_progress_bar: bool,
) -> tuple[RapmTransformerRunManifest, Path]:
    now = datetime.now(UTC)
    run_id = (
        f"rapm-transformer-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    season_dir = Path(artifacts_dir) / "rapm_transformer" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        split_plan = chronological_game_splits(
            possessions,
            base_manifest.split_config,
        )
        experiment = fit_rapm_transformer_experiment(
            possessions,
            base_predictions,
            checkpoint_dir=temporary_dir,
            split_plan=split_plan,
            training_config=training,
            architecture_config=architecture,
            refit_seeds=refit_seeds,
            enable_progress_bar=enable_progress_bar,
        )
        all_season_coefficients = coefficients.loc[
            coefficients["stage"].eq("all_season"),
            ["player_id", "rapm"],
        ].reset_index(drop=True)
        all_season_state = stage_parameters.loc[
            stage_parameters["stage"].eq("all_season")
        ]
        if len(all_season_state) != 1:
            raise ValueError("Transformer requires one all-season RAPM state")
        state = all_season_state.iloc[0]
        parquet_outputs = {
            "training_history.parquet": experiment.training_history,
            "hyperparameter_trials.parquet": experiment.hyperparameter_trials,
            "hyperparameter_summary.parquet": experiment.hyperparameter_summary,
            "test_metrics.parquet": experiment.test_metrics,
            "seed_metrics.parquet": experiment.seed_metrics,
            "test_predictions.parquet": experiment.test_predictions,
            "seed_predictions.parquet": experiment.seed_predictions,
            "lineup_residuals.parquet": experiment.lineup_residuals,
            "game_splits.parquet": experiment.game_splits,
            "rapm_player_coefficients.parquet": all_season_coefficients,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        json_outputs = {
            "player_columns.json": {
                str(player_id): column
                for player_id, column in experiment.player_columns.items()
            },
            "model_parameters.json": experiment.model_parameters,
            "rapm_state.json": {
                "source_rapm_run_id": base_manifest.source_rapm_run_id,
                "selected_rapm_lambda": base_manifest.selected_rapm_lambda,
                "intercept_home_net_rating": float(
                    state["intercept_home_net_rating"]
                ),
                "mean_offense_margin": float(state["mean_offense_margin"]),
                "train_game_count": int(state["train_game_count"]),
            },
        }
        for filename, payload in json_outputs.items():
            (temporary_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        artifacts = tuple(
            ArtifactRecord(
                filename=path.name,
                row_count=(
                    len(parquet_outputs[path.name])
                    if path.name in parquet_outputs
                    else None
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
        neural_dir = (
            Path(analytical_dir) / "neural_possessions" / season / "regular"
        )
        base_dir = (
            Path(analytical_dir) / "rapm_base_predictions" / season / "regular"
        )
        neural_manifest = validate_neural_possession_partition(neural_dir)
        validate_rapm_base_prediction_partition(base_dir)
        manifest = RapmTransformerRunManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            transformer_code_version=transformer_code_fingerprint(),
            source_rapm_run_id=base_manifest.source_rapm_run_id,
            source_rapm_manifest_sha256=(
                base_manifest.source_rapm_manifest_sha256
            ),
            selected_rapm_lambda=base_manifest.selected_rapm_lambda,
            dataset_manifest_sha256=_sha256_file(neural_dir / "_manifest.json"),
            dataset_part_sha256=neural_manifest.part_sha256,
            base_predictions_manifest_sha256=_sha256_file(
                base_dir / "_manifest.json"
            ),
            base_predictions_part_sha256=next(
                artifact.sha256
                for artifact in base_manifest.artifacts
                if artifact.filename == "part-00000.parquet"
            ),
            possession_count=len(possessions),
            game_count=int(possessions["game_id"].nunique()),
            player_count=len(experiment.player_columns),
            split_config=base_manifest.split_config,
            folds=folds,
            selection_train_game_count=len(selection_fold.train_game_ids),
            selection_validation_game_count=len(
                selection_fold.validation_game_ids
            ),
            final_train_game_count=len(experiment.split_plan.final_train_game_ids),
            final_test_game_count=len(experiment.split_plan.final_test_game_ids),
            random_seed=training.random_seed,
            batch_size=training.batch_size,
            max_epochs=training.max_epochs,
            early_stopping_patience=training.early_stopping_patience,
            selected_epochs=experiment.selected_epochs,
            learning_rate=experiment.selected_learning_rate,
            weight_decay=experiment.selected_weight_decay,
            learning_rate_candidates=training.learning_rates,
            weight_decay_candidates=training.weight_decays,
            hyperparameter_selection_metric=(
                "validation_possession_weighted_mse"
            ),
            d_model=architecture.d_model,
            attention_heads=architecture.attention_heads,
            transformer_layers=architecture.transformer_layers,
            feedforward_dim=architecture.feedforward_dim,
            dropout=architecture.dropout,
            parameter_count=experiment.parameter_count,
            refit_seeds=experiment.refit_seeds,
            leaderboard_seed=experiment.leaderboard_seed,
            requested_accelerator=training.accelerator,
            resolved_accelerator=experiment.resolved_accelerator,
            target="offense_points_minus_defense_points",
            torch_version=torch.__version__,
            lightning_version=L.__version__,
            artifacts=artifacts,
        )
        (temporary_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        temporary_dir.replace(run_dir)
        validate_rapm_transformer_run(run_dir)
        latest = season_dir / "latest.json"
        temporary_latest = latest.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest)
        return manifest, run_dir
    except (Exception, KeyboardInterrupt):
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the frozen-RAPM plus Transformer residual model."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--rapm-run-id")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRANSFORMER_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_TRANSFORMER_MAX_EPOCHS,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_TRANSFORMER_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rates",
        action="append",
        type=float,
        help="Candidate learning rate; repeat to override the default grid",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decays",
        action="append",
        type=float,
        help="Candidate AdamW weight decay; repeat to override the default grid",
    )
    parser.add_argument("--accelerator", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable Lightning progress bars",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    training = NeuralTrainingConfig(
        random_seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        learning_rates=(
            tuple(args.learning_rates)
            if args.learning_rates is not None
            else DEFAULT_TRANSFORMER_LEARNING_RATES
        ),
        weight_decays=(
            tuple(args.weight_decays)
            if args.weight_decays is not None
            else DEFAULT_TRANSFORMER_WEIGHT_DECAYS
        ),
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )
    architecture = RapmTransformerArchitectureConfig(
        d_model=args.d_model,
        attention_heads=args.attention_heads,
        transformer_layers=args.transformer_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    manifest, run_dir = train_rapm_transformer(
        args.season,
        source_rapm_run_id=args.rapm_run_id,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        training_config=training,
        architecture_config=architecture,
        enable_progress_bar=not args.no_progress_bar,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = (
        f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    )
    print(
        f"{manifest.season} RAPM+Transformer: "
        f"possessions={manifest.possession_count}, games={manifest.game_count}, "
        f"players={manifest.player_count}, parameters={manifest.parameter_count}, "
        f"learning_rate={manifest.learning_rate:g}, "
        f"weight_decay={manifest.weight_decay:g}, "
        f"epochs={manifest.selected_epochs}; run={run_dir}{tracking_text}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
