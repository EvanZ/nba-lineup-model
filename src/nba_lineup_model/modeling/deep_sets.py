from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.neural import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_RANDOM_SEED,
    NeuralTrainingConfig,
    _fit_fixed_epoch_model,
    _game_splits_frame,
    _player_rankings,
    _search_hyperparameters,
    _validate_possessions,
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
from nba_lineup_model.modeling.train import GameSplitPlan, chronological_game_splits
from nba_lineup_model.models.neural import DeepSetsRapmModule

DEFAULT_DEEP_SETS_BATCH_SIZE = 8_192
DEFAULT_DEEP_SETS_MAX_EPOCHS = 15
DEFAULT_DEEP_SETS_LEARNING_RATES = (0.0003, 0.001, 0.003)
DEFAULT_DEEP_SETS_WEIGHT_DECAYS = (0.0, 0.001, 0.01, 0.1)


@dataclass(frozen=True)
class DeepSetsArchitectureConfig:
    """Fixed architecture for the first nonlinear lineup-composition model."""

    player_embedding_dim: int = 32
    role_embedding_dim: int = 8
    player_hidden_dim: int = 64
    pooled_dim: int = 64
    lineup_hidden_dims: tuple[int, int] = (128, 64)

    def validate(self) -> None:
        dimensions = (
            self.player_embedding_dim,
            self.role_embedding_dim,
            self.player_hidden_dim,
            self.pooled_dim,
            *self.lineup_hidden_dims,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("Deep Sets dimensions must be positive")


@dataclass(frozen=True)
class DeepSetsExperiment:
    """In-memory Deep Sets outputs ready for atomic persistence."""

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
    lineup_interactions: pd.DataFrame
    additive_player_components: pd.DataFrame
    game_splits: pd.DataFrame
    model_parameters: dict[str, Any]


def train_deep_sets(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    split_config: ChronologicalSplitConfig | None = None,
    training_config: NeuralTrainingConfig | None = None,
    architecture_config: DeepSetsArchitectureConfig | None = None,
    refit_seeds: tuple[int, ...] | None = None,
    minimum_ranking_possessions: float = 500.0,
    enable_progress_bar: bool = True,
) -> tuple[NeuralRapmRunManifest, Path]:
    """Tune, evaluate, and refit the regular-season Deep Sets model."""

    split = split_config or ChronologicalSplitConfig()
    training = training_config or default_deep_sets_training_config()
    architecture = architecture_config or DeepSetsArchitectureConfig()
    seeds = refit_seeds or (
        training.random_seed,
        training.random_seed + 1,
        training.random_seed + 2,
    )
    dataset_manifest = build_neural_possession_dataset(
        season,
        curated_dir=curated_dir,
        analytical_dir=analytical_dir,
    )
    possessions = read_neural_possessions(season, analytical_dir)
    bios_path = (
        Path(curated_dir)
        / "player_seasons"
        / season
        / "regular"
        / "part-00000.parquet"
    )
    player_bios = pd.read_parquet(bios_path) if bios_path.is_file() else None
    return _write_experiment(
        season,
        dataset_manifest,
        possessions,
        player_bios,
        split,
        training,
        architecture,
        seeds,
        minimum_ranking_possessions,
        analytical_dir,
        artifacts_dir,
        enable_progress_bar,
    )


def fit_deep_sets_experiment(
    possessions: pd.DataFrame,
    *,
    checkpoint_dir: Path | str,
    split_config: ChronologicalSplitConfig | None = None,
    training_config: NeuralTrainingConfig | None = None,
    architecture_config: DeepSetsArchitectureConfig | None = None,
    refit_seeds: tuple[int, ...] | None = None,
    minimum_ranking_possessions: float = 500.0,
    player_bios: pd.DataFrame | None = None,
    enable_progress_bar: bool = False,
) -> DeepSetsExperiment:
    """Select optimization settings, evaluate fixed seeds, and refit all games."""

    split = split_config or ChronologicalSplitConfig()
    training = training_config or default_deep_sets_training_config()
    architecture = architecture_config or DeepSetsArchitectureConfig()
    training.validate()
    architecture.validate()
    _validate_possessions(possessions)
    if minimum_ranking_possessions < 0:
        raise ValueError("Minimum ranking possessions cannot be negative")
    seeds = refit_seeds or (
        training.random_seed,
        training.random_seed + 1,
        training.random_seed + 2,
    )
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Deep Sets requires at least three unique nonnegative seeds")
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module_factory = _deep_sets_module_factory(architecture)

    split_plan = chronological_game_splits(possessions, split)
    player_columns = player_vocabulary(possessions)
    search = _search_hyperparameters(
        possessions,
        player_columns,
        split_plan.folds,
        training,
        output_dir,
        module_factory,
        enable_progress_bar,
    )
    game_ids = possessions["game_id"].astype(str)
    test_rows = possessions.loc[
        game_ids.isin(split_plan.final_test_game_ids)
    ].copy()
    train_rows = possessions.loc[
        game_ids.isin(split_plan.final_train_game_ids)
    ]
    mean_prediction = float(train_rows["target_offense_margin"].mean())
    mean_predictions = np.full(len(test_rows), mean_prediction)
    all_season_ids = tuple(split_plan.ordered_games["game_id"].astype(str))

    histories = [search.training_history]
    seed_metric_rows: list[dict[str, Any]] = []
    seed_prediction_frames: list[pd.DataFrame] = []
    canonical_test_model: DeepSetsRapmModule | None = None
    canonical_full_model: DeepSetsRapmModule | None = None
    canonical_predictions: np.ndarray | None = None
    canonical_additive: np.ndarray | None = None
    canonical_nonlinear: np.ndarray | None = None
    accelerators = {search.resolved_accelerator}
    leaderboard_seed = seeds[0]

    for seed in seeds:
        test_checkpoint = (
            output_dir / "test_model.ckpt"
            if seed == leaderboard_seed
            else output_dir / f"test_model_seed_{seed}.ckpt"
        )
        raw_test_model, test_history, resolved = _fit_fixed_epoch_model(
            possessions,
            player_columns,
            split_plan.final_train_game_ids,
            search.selected_epochs,
            training,
            test_checkpoint,
            learning_rate=search.selected_learning_rate,
            weight_decay=search.selected_weight_decay,
            module_factory=module_factory,
            stage="test_refit",
            seed_offset=seed - training.random_seed,
            enable_progress_bar=enable_progress_bar,
        )
        test_model = _require_deep_sets_module(raw_test_model)
        test_history.insert(1, "refit_seed", seed)
        histories.append(test_history)
        accelerators.add(resolved)
        predictions, additive, nonlinear = _predict_components(
            test_model,
            test_rows,
            player_columns,
            training,
        )
        seed_metric_rows.append(
            _metric_row(
                test_rows,
                predictions,
                model="deep_sets",
                seed=seed,
            )
        )
        seed_prediction_frames.append(
            pd.DataFrame(
                {
                    "game_id": test_rows["game_id"].astype(str).to_numpy(),
                    "possession_id": test_rows["possession_id"].astype(str).to_numpy(),
                    "seed": seed,
                    "prediction_deep_sets": predictions,
                    "prediction_additive_path": additive,
                    "prediction_nonlinear_residual": nonlinear,
                }
            )
        )
        if seed == leaderboard_seed:
            canonical_test_model = test_model
            canonical_predictions = predictions
            canonical_additive = additive
            canonical_nonlinear = nonlinear

        full_checkpoint = (
            output_dir / "model.ckpt"
            if seed == leaderboard_seed
            else output_dir / f"model_seed_{seed}.ckpt"
        )
        raw_full_model, full_history, resolved = _fit_fixed_epoch_model(
            possessions,
            player_columns,
            all_season_ids,
            search.selected_epochs,
            training,
            full_checkpoint,
            learning_rate=search.selected_learning_rate,
            weight_decay=search.selected_weight_decay,
            module_factory=module_factory,
            stage="all_season_refit",
            seed_offset=seed - training.random_seed,
            enable_progress_bar=enable_progress_bar,
        )
        full_model = _require_deep_sets_module(raw_full_model)
        full_history.insert(1, "refit_seed", seed)
        histories.append(full_history)
        accelerators.add(resolved)
        if seed == leaderboard_seed:
            canonical_full_model = full_model

    if len(accelerators) != 1:
        raise ValueError("Deep Sets stages resolved to different accelerators")
    if (
        canonical_test_model is None
        or canonical_full_model is None
        or canonical_predictions is None
        or canonical_additive is None
        or canonical_nonlinear is None
    ):
        raise RuntimeError("Canonical Deep Sets refit was not produced")

    mean_row = _metric_row(test_rows, mean_predictions, model="mean", seed=None)
    canonical_row = next(
        row for row in seed_metric_rows if row["seed"] == leaderboard_seed
    )
    mean_mse = float(mean_row["mse"])
    mean_game_mse = float(mean_row["game_margin_rmse"]) ** 2
    test_metrics = pd.DataFrame([mean_row, canonical_row])
    test_metrics["possession_skill_vs_mean"] = test_metrics["mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    test_metrics["game_margin_skill_vs_mean"] = test_metrics[
        "game_margin_rmse"
    ].map(lambda value: skill_score(float(value) ** 2, mean_game_mse))
    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_metrics["possession_skill_vs_mean"] = seed_metrics["mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    seed_metrics["game_margin_skill_vs_mean"] = seed_metrics[
        "game_margin_rmse"
    ].map(lambda value: skill_score(float(value) ** 2, mean_game_mse))
    test_predictions = _test_predictions(
        test_rows,
        mean_predictions,
        canonical_predictions,
        canonical_additive,
        canonical_nonlinear,
        leaderboard_seed,
    )
    lineup_interactions = _lineup_interactions(test_predictions, test_rows)
    additive_components = _player_rankings(
        possessions,
        player_columns,
        canonical_full_model,
        minimum_ranking_possessions,
        player_bios,
    ).rename(
        columns={
            "neural_rapm": "additive_component_net_rating",
            "embedding_value_points_per_possession": (
                "additive_component_points_per_possession"
            ),
        }
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in canonical_full_model.parameters()
        if parameter.requires_grad
    )
    model_parameters = {
        "architecture": "additive skip plus Deep Sets residual",
        "equation": (
            "intercept + home effect + signed additive player values + "
            "rho(concat(sum(phi(concat(player, offense role))), "
            "sum(phi(concat(player, defense role))), home_offense_sign))"
        ),
        "pooling": "sum separately within offense and defense",
        "player_embedding_dim": architecture.player_embedding_dim,
        "role_embedding_dim": architecture.role_embedding_dim,
        "player_hidden_dim": architecture.player_hidden_dim,
        "pooled_dim": architecture.pooled_dim,
        "lineup_hidden_dims": list(architecture.lineup_hidden_dims),
        "lineup_input_dim": 2 * architecture.pooled_dim + 1,
        "parameter_count": parameter_count,
        "learning_rate_candidates": list(training.learning_rates),
        "weight_decay_candidates": list(training.weight_decays),
        "selected_learning_rate": search.selected_learning_rate,
        "selected_weight_decay": search.selected_weight_decay,
        "selected_epochs": search.selected_epochs,
        "hyperparameter_selection_metric": "validation_possession_weighted_mse",
        "refit_seeds": list(seeds),
        "leaderboard_seed": leaderboard_seed,
        "target": "offense points minus defense points on one possession",
        "lineup_policy": "exclude every possession with more than one lineup segment",
        "context_features": ["home_offense_sign"],
    }
    return DeepSetsExperiment(
        split_plan=split_plan,
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
        lineup_interactions=lineup_interactions,
        additive_player_components=additive_components,
        game_splits=_game_splits_frame(split_plan),
        model_parameters=model_parameters,
    )


def validate_deep_sets_run(run_dir: Path | str) -> NeuralRapmRunManifest:
    """Validate a Deep Sets run and require its architecture contract."""

    from nba_lineup_model.modeling.neural import validate_neural_rapm_run

    manifest = validate_neural_rapm_run(run_dir)
    if manifest.architecture != "deep_sets":
        raise ValueError("Neural run is not a Deep Sets model")
    return manifest


def _deep_sets_module_factory(architecture: DeepSetsArchitectureConfig):
    def build(
        player_count: int,
        learning_rate: float,
        weight_decay: float,
        target_mean: float,
    ) -> L.LightningModule:
        module = DeepSetsRapmModule(
            player_count,
            player_embedding_dim=architecture.player_embedding_dim,
            role_embedding_dim=architecture.role_embedding_dim,
            player_hidden_dim=architecture.player_hidden_dim,
            pooled_dim=architecture.pooled_dim,
            lineup_hidden_dims=architecture.lineup_hidden_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        with torch.no_grad():
            module.model.intercept.fill_(target_mean)
        return module

    return build


def _require_deep_sets_module(module: L.LightningModule) -> DeepSetsRapmModule:
    if not isinstance(module, DeepSetsRapmModule):
        raise TypeError("Expected a Deep Sets Lightning module")
    return module


def _predict_components(
    module: DeepSetsRapmModule,
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
    config: NeuralTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = PossessionTensorDataset(possessions, player_columns)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    total_values: list[np.ndarray] = []
    additive_values: list[np.ndarray] = []
    nonlinear_values: list[np.ndarray] = []
    module.eval()
    with torch.inference_mode():
        for batch in loader:
            offense = batch["offense_player_indices"].to(module.device)
            defense = batch["defense_player_indices"].to(module.device)
            sign = batch["home_offense_sign"].to(module.device)
            additive, nonlinear = module.model.components(offense, defense, sign)
            total_values.append((additive + nonlinear).detach().cpu().numpy())
            additive_values.append(additive.detach().cpu().numpy())
            nonlinear_values.append(nonlinear.detach().cpu().numpy())
    return (
        np.concatenate(total_values),
        np.concatenate(additive_values),
        np.concatenate(nonlinear_values),
    )


def _metric_row(
    possessions: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    seed: int | None,
) -> dict[str, Any]:
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    return {
        "model": model,
        "seed": seed,
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
    predictions: np.ndarray,
    additive: np.ndarray,
    nonlinear: np.ndarray,
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
    output = test_rows.loc[:, columns].reset_index(drop=True)
    output["leaderboard_seed"] = seed
    output["prediction_mean"] = mean_predictions
    output["prediction_deep_sets"] = predictions
    output["prediction_additive_path"] = additive
    output["prediction_nonlinear_residual"] = nonlinear
    output["predicted_home_margin_mean"] = (
        mean_predictions * output["home_offense_sign"]
    )
    output["predicted_home_margin_deep_sets"] = (
        predictions * output["home_offense_sign"]
    )
    return output


def _lineup_interactions(
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
    frame["absolute_nonlinear_residual"] = frame[
        "prediction_nonlinear_residual"
    ].abs()
    grouped = frame.groupby(
        ["offense_lineup", "defense_lineup"],
        as_index=False,
        sort=False,
    ).agg(
        game_count=("game_id", "nunique"),
        possession_count=("possession_id", "size"),
        actual_offense_margin=("target_offense_margin", "mean"),
        predicted_offense_margin=("prediction_deep_sets", "mean"),
        additive_path=("prediction_additive_path", "mean"),
        nonlinear_residual=("prediction_nonlinear_residual", "mean"),
        mean_absolute_nonlinear_residual=(
            "absolute_nonlinear_residual",
            "mean",
        ),
    )
    return grouped.sort_values(
        ["mean_absolute_nonlinear_residual", "possession_count"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def _write_experiment(
    season: str,
    dataset_manifest: NeuralPossessionManifest,
    possessions: pd.DataFrame,
    player_bios: pd.DataFrame | None,
    split_config: ChronologicalSplitConfig,
    training_config: NeuralTrainingConfig,
    architecture_config: DeepSetsArchitectureConfig,
    refit_seeds: tuple[int, ...],
    minimum_ranking_possessions: float,
    analytical_dir: Path | str,
    artifacts_dir: Path | str,
    enable_progress_bar: bool,
) -> tuple[NeuralRapmRunManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"deep-sets-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(artifacts_dir) / "deep_sets" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        experiment = fit_deep_sets_experiment(
            possessions,
            checkpoint_dir=temporary_dir,
            split_config=split_config,
            training_config=training_config,
            architecture_config=architecture_config,
            refit_seeds=refit_seeds,
            minimum_ranking_possessions=minimum_ranking_possessions,
            player_bios=player_bios,
            enable_progress_bar=enable_progress_bar,
        )
        parquet_outputs = {
            "training_history.parquet": experiment.training_history,
            "hyperparameter_trials.parquet": experiment.hyperparameter_trials,
            "hyperparameter_summary.parquet": experiment.hyperparameter_summary,
            "test_metrics.parquet": experiment.test_metrics,
            "seed_metrics.parquet": experiment.seed_metrics,
            "test_predictions.parquet": experiment.test_predictions,
            "seed_predictions.parquet": experiment.seed_predictions,
            "lineup_interactions.parquet": experiment.lineup_interactions,
            "additive_player_components.parquet": (
                experiment.additive_player_components
            ),
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
        dataset_dir = (
            Path(analytical_dir) / "neural_possessions" / season / "regular"
        )
        manifest = NeuralRapmRunManifest(
            schema_version=3,
            run_id=run_id,
            created_at=now,
            season=season,
            architecture="deep_sets",
            neural_code_version=neural_code_fingerprint(),
            dataset_manifest_sha256=_sha256_file(dataset_dir / "_manifest.json"),
            dataset_part_sha256=dataset_manifest.part_sha256,
            possession_count=len(possessions),
            game_count=int(possessions["game_id"].nunique()),
            player_count=len(experiment.player_columns),
            split_config=split_config,
            folds=folds,
            selection_train_game_count=len(selection_fold.train_game_ids),
            selection_validation_game_count=len(
                selection_fold.validation_game_ids
            ),
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
            player_embedding_dim=architecture_config.player_embedding_dim,
            role_embedding_dim=architecture_config.role_embedding_dim,
            player_hidden_dim=architecture_config.player_hidden_dim,
            pooled_dim=architecture_config.pooled_dim,
            lineup_hidden_dims=architecture_config.lineup_hidden_dims,
            parameter_count=experiment.parameter_count,
            refit_seeds=experiment.refit_seeds,
            leaderboard_seed=experiment.leaderboard_seed,
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
        validate_deep_sets_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except (Exception, KeyboardInterrupt):
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def default_deep_sets_training_config() -> NeuralTrainingConfig:
    """Return the CPU-conscious default search budget for Deep Sets."""

    return NeuralTrainingConfig(
        batch_size=DEFAULT_DEEP_SETS_BATCH_SIZE,
        max_epochs=DEFAULT_DEEP_SETS_MAX_EPOCHS,
        learning_rates=DEFAULT_DEEP_SETS_LEARNING_RATES,
        weight_decays=DEFAULT_DEEP_SETS_WEIGHT_DECAYS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the regular-season Deep Sets lineup model."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_DEEP_SETS_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_DEEP_SETS_MAX_EPOCHS,
    )
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
        help="Candidate learning rate; repeat to override the default grid",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decays",
        action="append",
        type=float,
        help="Candidate AdamW weight decay; repeat to override the default grid",
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
            else DEFAULT_DEEP_SETS_LEARNING_RATES
        ),
        weight_decays=(
            tuple(args.weight_decays)
            if args.weight_decays is not None
            else DEFAULT_DEEP_SETS_WEIGHT_DECAYS
        ),
        accelerator=args.accelerator,
        num_workers=args.num_workers,
    )
    manifest, run_dir = train_deep_sets(
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
        f"{manifest.season} Deep Sets: possessions={manifest.possession_count}, "
        f"games={manifest.game_count}, players={manifest.player_count}, "
        f"parameters={manifest.parameter_count}, lr={manifest.learning_rate:g}, "
        f"weight_decay={manifest.weight_decay:g}, epochs={manifest.selected_epochs}, "
        f"leaderboard_seed={manifest.leaderboard_seed}; "
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
