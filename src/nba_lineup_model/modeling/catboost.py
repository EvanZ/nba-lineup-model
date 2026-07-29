from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from catboost import __version__ as catboost_version

from nba_lineup_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    possession_game_margin_rmse,
    rmse,
    skill_score,
)
from nba_lineup_model.modeling.neural import (
    DEFAULT_RANDOM_SEED,
    _game_splits_frame,
    _validate_possessions,
)
from nba_lineup_model.modeling.neural_data import (
    build_neural_possession_dataset,
    player_vocabulary,
    read_neural_possessions,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    CatBoostRunManifest,
    ChronologicalFold,
    ChronologicalSplitConfig,
    NeuralPossessionManifest,
)
from nba_lineup_model.modeling.train import GameSplitPlan, chronological_game_splits

DEFAULT_CATBOOST_MAX_ITERATIONS = 1_000
CATBOOST_PLAYER_STATE_VALUES = ("absent", "offense", "defense")


@dataclass(frozen=True)
class CatBoostTrainingConfig:
    """Explicit controls for the defaults-first CatBoost exemplar."""

    max_iterations: int = DEFAULT_CATBOOST_MAX_ITERATIONS
    random_seed: int = DEFAULT_RANDOM_SEED
    one_hot_max_size: int = 3
    has_time: bool = True
    use_best_model: bool = True

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("CatBoost iteration ceiling must be positive")
        if self.random_seed < 0:
            raise ValueError("CatBoost seed must be nonnegative")
        if self.one_hot_max_size != len(CATBOOST_PLAYER_STATE_VALUES):
            raise ValueError("CatBoost one-hot boundary must match player states")
        if not self.has_time:
            raise ValueError("The first CatBoost model requires chronological order")
        if not self.use_best_model:
            raise ValueError("The first CatBoost model requires best-model truncation")


@dataclass(frozen=True)
class CatBoostExperiment:
    """In-memory CatBoost outputs ready for atomic persistence."""

    split_plan: GameSplitPlan
    player_columns: dict[int, int]
    selection_model: CatBoostRegressor
    test_model: CatBoostRegressor
    full_model: CatBoostRegressor
    max_iterations: int
    best_iteration: int
    selected_tree_count: int
    resolved_learning_rate: float
    fold_metrics: pd.DataFrame
    training_history: pd.DataFrame
    test_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    game_splits: pd.DataFrame
    resolved_parameters: dict[str, Any]
    model_parameters: dict[str, Any]


def train_catboost_lineup_model(
    season: str,
    *,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    split_config: ChronologicalSplitConfig | None = None,
    training_config: CatBoostTrainingConfig | None = None,
) -> tuple[CatBoostRunManifest, Path]:
    """Train, evaluate, and persist the categorical player-state model."""

    split = split_config or ChronologicalSplitConfig()
    training = training_config or CatBoostTrainingConfig()
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
        analytical_dir,
        artifacts_dir,
    )


def fit_catboost_experiment(
    possessions: pd.DataFrame,
    *,
    split_config: ChronologicalSplitConfig | None = None,
    training_config: CatBoostTrainingConfig | None = None,
    player_bios: pd.DataFrame | None = None,
) -> CatBoostExperiment:
    """Fit the defaults-first CatBoost protocol on an in-memory dataset."""

    split = split_config or ChronologicalSplitConfig()
    training = training_config or CatBoostTrainingConfig()
    training.validate()
    _validate_possessions(possessions)
    ordered = _chronological_possessions(possessions)
    split_plan = chronological_game_splits(ordered, split)
    player_columns = player_vocabulary(ordered)
    features = categorical_player_state_matrix(ordered, player_columns)
    targets = ordered["target_offense_margin"].to_numpy(dtype=np.float32)
    feature_names = catboost_feature_names(player_columns)
    full_pool = Pool(
        data=features,
        label=targets,
        cat_features=list(range(features.shape[1])),
        feature_names=feature_names,
    )
    game_ids = ordered["game_id"].astype(str).to_numpy()
    fold_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    resolved_folds: dict[str, Any] = {}
    selection_model: CatBoostRegressor | None = None

    for fold in split_plan.folds:
        train_indices = np.flatnonzero(np.isin(game_ids, fold.train_game_ids))
        validation_indices = np.flatnonzero(
            np.isin(game_ids, fold.validation_game_ids)
        )
        model = _selection_model(training)
        train_pool = full_pool.slice(train_indices.tolist())
        validation_pool = full_pool.slice(validation_indices.tolist())
        model.fit(
            train_pool,
            eval_set=validation_pool,
            verbose=False,
        )
        predictions = np.asarray(model.predict(validation_pool), dtype=float)
        validation_rows = ordered.iloc[validation_indices]
        best_iteration = int(model.get_best_iteration())
        tree_count = int(model.tree_count_)
        if best_iteration < 0 or tree_count != best_iteration + 1:
            raise ValueError("CatBoost best iteration does not match saved trees")
        params = _json_safe(model.get_all_params())
        resolved_folds[str(fold.fold)] = params
        fold_rows.append(
            {
                "fold": fold.fold,
                "train_game_count": len(fold.train_game_ids),
                "validation_game_count": len(fold.validation_game_ids),
                "train_possession_count": len(train_indices),
                "validation_possession_count": len(validation_indices),
                "max_iterations": training.max_iterations,
                "best_iteration": best_iteration,
                "tree_count": tree_count,
                "resolved_learning_rate": float(params["learning_rate"]),
                **_prediction_metrics(validation_rows, predictions),
            }
        )
        histories.append(_training_history(model, fold.fold))
        if fold.fold == split_plan.folds[-1].fold:
            selection_model = model

    if selection_model is None:
        raise RuntimeError("CatBoost selection model was not fitted")
    best_iteration = int(selection_model.get_best_iteration())
    selected_tree_count = int(selection_model.tree_count_)
    selection_parameters = _json_safe(selection_model.get_all_params())
    resolved_learning_rate = float(selection_parameters["learning_rate"])

    final_train_indices = np.flatnonzero(
        np.isin(game_ids, split_plan.final_train_game_ids)
    )
    final_test_indices = np.flatnonzero(
        np.isin(game_ids, split_plan.final_test_game_ids)
    )
    test_model = _refit_model(
        training,
        iterations=selected_tree_count,
        learning_rate=resolved_learning_rate,
    )
    test_model.fit(full_pool.slice(final_train_indices.tolist()), verbose=False)
    test_predictions = np.asarray(
        test_model.predict(full_pool.slice(final_test_indices.tolist())),
        dtype=float,
    )
    test_rows = ordered.iloc[final_test_indices].copy()
    final_train_rows = ordered.iloc[final_train_indices]
    mean_prediction = float(final_train_rows["target_offense_margin"].mean())
    mean_predictions = np.full(len(test_rows), mean_prediction)
    mean_metrics = _metric_row(test_rows, mean_predictions, model="mean")
    catboost_metrics = _metric_row(test_rows, test_predictions, model="catboost")
    mean_mse = float(mean_metrics["mse"])
    mean_game_mse = float(mean_metrics["game_margin_rmse"]) ** 2
    test_metrics = pd.DataFrame([mean_metrics, catboost_metrics])
    test_metrics["possession_skill_vs_mean"] = test_metrics["mse"].map(
        lambda value: skill_score(float(value), mean_mse)
    )
    test_metrics["game_margin_skill_vs_mean"] = test_metrics[
        "game_margin_rmse"
    ].map(lambda value: skill_score(float(value) ** 2, mean_game_mse))

    full_model = _refit_model(
        training,
        iterations=selected_tree_count,
        learning_rate=resolved_learning_rate,
    )
    full_model.fit(full_pool, verbose=False)
    resolved_parameters = {
        "requested": {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": training.max_iterations,
            "one_hot_max_size": training.one_hot_max_size,
            "has_time": training.has_time,
            "use_best_model": training.use_best_model,
            "random_seed": training.random_seed,
            "allow_writing_files": False,
        },
        "folds": resolved_folds,
        "selection": selection_parameters,
        "test_refit": _json_safe(test_model.get_all_params()),
        "all_season_refit": _json_safe(full_model.get_all_params()),
    }
    model_parameters = {
        "architecture": "one categorical absent/offense/defense feature per player",
        "player_state_values": list(CATBOOST_PLAYER_STATE_VALUES),
        "context_features": ["home_offense"],
        "feature_count": features.shape[1],
        "player_feature_count": len(player_columns),
        "max_iterations": training.max_iterations,
        "best_iteration_zero_based": best_iteration,
        "selected_tree_count": selected_tree_count,
        "resolved_learning_rate": resolved_learning_rate,
        "iteration_selection": "latest chronological validation fold",
        "hyperparameter_search": "none; CatBoost defaults",
        "target": "offense points minus defense points on one possession",
        "lineup_policy": "exclude every possession with more than one lineup segment",
    }
    return CatBoostExperiment(
        split_plan=split_plan,
        player_columns=player_columns,
        selection_model=selection_model,
        test_model=test_model,
        full_model=full_model,
        max_iterations=training.max_iterations,
        best_iteration=best_iteration,
        selected_tree_count=selected_tree_count,
        resolved_learning_rate=resolved_learning_rate,
        fold_metrics=pd.DataFrame(fold_rows),
        training_history=pd.concat(histories, ignore_index=True),
        test_metrics=test_metrics,
        test_predictions=_test_predictions(
            test_rows,
            mean_predictions,
            test_predictions,
        ),
        feature_importance=_feature_importance(
            full_model,
            player_columns,
            ordered,
            player_bios,
        ),
        game_splits=_game_splits_frame(split_plan),
        resolved_parameters=resolved_parameters,
        model_parameters=model_parameters,
    )


def categorical_player_state_matrix(
    possessions: pd.DataFrame,
    player_columns: Mapping[int, int],
) -> np.ndarray:
    """Encode each player as absent, offense, or defense plus home offense."""

    if possessions.empty:
        raise ValueError("CatBoost possession features cannot be empty")
    mapping = {
        int(player_id): int(column) - 1
        for player_id, column in player_columns.items()
    }
    if not mapping or min(mapping.values()) != 0:
        raise ValueError("Player columns must be contiguous and one-based")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Player columns must be contiguous and one-based")
    features = np.zeros((len(possessions), len(mapping) + 1), dtype=np.int8)
    for row_index, (offense, defense) in enumerate(
        zip(
            possessions["offense_player_ids"],
            possessions["defense_player_ids"],
            strict=True,
        )
    ):
        for player_id in offense:
            column = mapping.get(int(player_id))
            if column is not None:
                features[row_index, column] = 1
        for player_id in defense:
            column = mapping.get(int(player_id))
            if column is not None:
                features[row_index, column] = 2
    signs = possessions["home_offense_sign"].to_numpy(dtype=float)
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ValueError("Home-offense signs must be negative or positive one")
    features[:, -1] = (signs > 0).astype(np.int8)
    return features


def catboost_feature_names(player_columns: Mapping[int, int]) -> list[str]:
    """Return feature names in encoded column order."""

    players = sorted(
        ((int(column), int(player_id)) for player_id, column in player_columns.items())
    )
    return [f"player_{player_id}" for _, player_id in players] + ["home_offense"]


def catboost_predictions(
    model: CatBoostRegressor,
    possessions: pd.DataFrame,
    player_columns: Mapping[int, int],
) -> tuple[np.ndarray, int]:
    """Predict possessions and count player exposures absent from training."""

    features = categorical_player_state_matrix(possessions, player_columns)
    pool = Pool(
        features,
        cat_features=list(range(features.shape[1])),
        feature_names=catboost_feature_names(player_columns),
    )
    known = {int(player_id) for player_id in player_columns}
    exposures = pd.concat(
        [
            possessions["offense_player_ids"].explode(),
            possessions["defense_player_ids"].explode(),
        ],
        ignore_index=True,
    ).astype(int)
    unknown_count = int((~exposures.isin(known)).sum())
    return np.asarray(model.predict(pool), dtype=float), unknown_count


def validate_catboost_run(run_dir: Path | str) -> CatBoostRunManifest:
    """Require every recorded CatBoost artifact to match its manifest."""

    root = Path(run_dir)
    manifest = CatBoostRunManifest.model_validate_json(
        (root / "manifest.json").read_text()
    )
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    expected = {artifact.filename for artifact in manifest.artifacts}
    if actual != expected:
        raise ValueError("CatBoost run files do not match the manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(f"CatBoost artifact byte count changed: {artifact.filename}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"CatBoost artifact hash changed: {artifact.filename}")
        if artifact.row_count is not None and len(pd.read_parquet(path)) != artifact.row_count:
            raise ValueError(f"CatBoost artifact rows changed: {artifact.filename}")
    return manifest


def catboost_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash CatBoost-owned sources for reproducible runs."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        paths = (
            package_root / "evaluation" / "metrics.py",
            package_root / "modeling" / "catboost.py",
            package_root / "modeling" / "neural_data.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "train.py",
        )
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one CatBoost source path is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"CatBoost source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _selection_model(config: CatBoostTrainingConfig) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=config.max_iterations,
        one_hot_max_size=config.one_hot_max_size,
        has_time=config.has_time,
        use_best_model=config.use_best_model,
        random_seed=config.random_seed,
        allow_writing_files=False,
    )


def _refit_model(
    config: CatBoostTrainingConfig,
    *,
    iterations: int,
    learning_rate: float,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        one_hot_max_size=config.one_hot_max_size,
        has_time=config.has_time,
        use_best_model=False,
        random_seed=config.random_seed,
        allow_writing_files=False,
    )


def _chronological_possessions(possessions: pd.DataFrame) -> pd.DataFrame:
    return possessions.sort_values(
        ["game_date", "game_time_utc", "game_id", "possession_index"],
        kind="stable",
    ).reset_index(drop=True)


def _training_history(model: CatBoostRegressor, fold: int) -> pd.DataFrame:
    results = model.get_evals_result()
    learn = results.get("learn", {}).get("RMSE", [])
    validation = results.get("validation", {}).get("RMSE", [])
    if not learn or len(learn) != len(validation):
        raise ValueError("CatBoost did not return aligned RMSE histories")
    return pd.DataFrame(
        {
            "fold": fold,
            "iteration": np.arange(len(learn)),
            "learn_rmse": learn,
            "validation_rmse": validation,
        }
    )


def _prediction_metrics(
    possessions: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, float]:
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    return {
        "validation_mse": mean_squared_error(actual, predictions),
        "validation_rmse": rmse(actual, predictions),
        "validation_mae": mean_absolute_error(actual, predictions),
        "validation_game_margin_rmse": possession_game_margin_rmse(
            possessions["game_id"],
            actual,
            predictions,
            possessions["home_offense_sign"].to_numpy(dtype=float),
        ),
    }


def _metric_row(
    possessions: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model: str,
) -> dict[str, Any]:
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    return {
        "model": model,
        "test_game_count": int(possessions["game_id"].nunique()),
        "test_possession_count": len(possessions),
        "mse": mean_squared_error(actual, predictions),
        "rmse": rmse(actual, predictions),
        "mae": mean_absolute_error(actual, predictions),
        "game_margin_rmse": possession_game_margin_rmse(
            possessions["game_id"],
            actual,
            predictions,
            possessions["home_offense_sign"].to_numpy(dtype=float),
        ),
    }


def _test_predictions(
    test_rows: pd.DataFrame,
    mean_predictions: np.ndarray,
    predictions: np.ndarray,
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
    output["prediction_catboost"] = predictions
    output["predicted_home_margin_mean"] = (
        mean_predictions * output["home_offense_sign"]
    )
    output["predicted_home_margin_catboost"] = (
        predictions * output["home_offense_sign"]
    )
    return output


def _feature_importance(
    model: CatBoostRegressor,
    player_columns: Mapping[int, int],
    possessions: pd.DataFrame,
    player_bios: pd.DataFrame | None,
) -> pd.DataFrame:
    names = catboost_feature_names(player_columns)
    values = np.asarray(model.get_feature_importance(), dtype=float)
    if len(values) != len(names):
        raise ValueError("CatBoost feature importance does not match features")
    player_by_name = {
        f"player_{int(player_id)}": int(player_id)
        for player_id in player_columns
    }
    name_map: dict[int, str] = {}
    if player_bios is not None and {"player_id", "player_name"} <= set(player_bios):
        name_map = {
            int(row.player_id): str(row.player_name)
            for row in player_bios.loc[:, ["player_id", "player_name"]].itertuples(
                index=False
            )
        }
    offense_counts = possessions["offense_player_ids"].explode().value_counts()
    defense_counts = possessions["defense_player_ids"].explode().value_counts()
    frame = pd.DataFrame(
        {
            "feature_name": names,
            "feature_importance": values,
        }
    )
    frame["feature_type"] = np.where(
        frame["feature_name"].eq("home_offense"),
        "context",
        "player_state",
    )
    frame["player_id"] = frame["feature_name"].map(player_by_name).astype("Int64")
    frame["player_name"] = frame["player_id"].map(name_map)
    frame["offense_possessions"] = (
        frame["player_id"].map(offense_counts).fillna(0).astype(int)
    )
    frame["defense_possessions"] = (
        frame["player_id"].map(defense_counts).fillna(0).astype(int)
    )
    frame = frame.sort_values(
        ["feature_importance", "feature_name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _write_experiment(
    season: str,
    dataset_manifest: NeuralPossessionManifest,
    possessions: pd.DataFrame,
    player_bios: pd.DataFrame | None,
    split_config: ChronologicalSplitConfig,
    training_config: CatBoostTrainingConfig,
    analytical_dir: Path | str,
    artifacts_dir: Path | str,
) -> tuple[CatBoostRunManifest, Path]:
    now = datetime.now(UTC)
    run_id = f"catboost-{season}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    season_dir = Path(artifacts_dir) / "catboost" / season
    run_dir = season_dir / run_id
    temporary_dir = season_dir / f".{run_id}.tmp"
    season_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()
    try:
        experiment = fit_catboost_experiment(
            possessions,
            split_config=split_config,
            training_config=training_config,
            player_bios=player_bios,
        )
        parquet_outputs = {
            "fold_metrics.parquet": experiment.fold_metrics,
            "training_history.parquet": experiment.training_history,
            "test_metrics.parquet": experiment.test_metrics,
            "test_predictions.parquet": experiment.test_predictions,
            "feature_importance.parquet": experiment.feature_importance,
            "game_splits.parquet": experiment.game_splits,
        }
        for filename, frame in parquet_outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        json_outputs = {
            "player_columns.json": {
                str(player_id): column
                for player_id, column in experiment.player_columns.items()
            },
            "resolved_parameters.json": experiment.resolved_parameters,
            "model_parameters.json": experiment.model_parameters,
        }
        for filename, payload in json_outputs.items():
            (temporary_dir / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        experiment.selection_model.save_model(temporary_dir / "selection_model.cbm")
        experiment.test_model.save_model(temporary_dir / "test_model.cbm")
        experiment.full_model.save_model(temporary_dir / "model.cbm")
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
        manifest = CatBoostRunManifest(
            run_id=run_id,
            created_at=now,
            season=season,
            catboost_code_version=catboost_code_fingerprint(),
            dataset_manifest_sha256=_sha256_file(dataset_dir / "_manifest.json"),
            dataset_part_sha256=dataset_manifest.part_sha256,
            possession_count=len(possessions),
            game_count=int(possessions["game_id"].nunique()),
            player_count=len(experiment.player_columns),
            feature_count=len(experiment.player_columns) + 1,
            split_config=split_config,
            folds=folds,
            selection_train_game_count=len(selection_fold.train_game_ids),
            selection_validation_game_count=len(
                selection_fold.validation_game_ids
            ),
            final_train_game_count=len(experiment.split_plan.final_train_game_ids),
            final_test_game_count=len(experiment.split_plan.final_test_game_ids),
            max_iterations=experiment.max_iterations,
            best_iteration=experiment.best_iteration,
            selected_tree_count=experiment.selected_tree_count,
            one_hot_max_size=training_config.one_hot_max_size,
            has_time=training_config.has_time,
            use_best_model=training_config.use_best_model,
            random_seed=training_config.random_seed,
            resolved_learning_rate=experiment.resolved_learning_rate,
            target="offense_points_minus_defense_points",
            catboost_version=catboost_version,
            artifacts=artifacts,
        )
        (temporary_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        temporary_dir.replace(run_dir)
        validate_catboost_run(run_dir)
        latest_path = season_dir / "latest.json"
        temporary_latest = latest_path.with_suffix(".json.tmp")
        temporary_latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary_latest.replace(latest_path)
        return manifest, run_dir
    except (Exception, KeyboardInterrupt):
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the categorical player-state CatBoost lineup model."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--analytical-dir", default="data/analytical")
    parser.add_argument("--artifacts-dir", default="artifacts/models")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_CATBOOST_MAX_ITERATIONS,
        help="Maximum trees built on each chronological selection fold",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_config = ChronologicalSplitConfig(
        cv_folds=args.cv_folds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    training_config = CatBoostTrainingConfig(
        max_iterations=args.iterations,
        random_seed=args.seed,
    )
    manifest, run_dir = train_catboost_lineup_model(
        args.season,
        curated_dir=args.curated_dir,
        analytical_dir=args.analytical_dir,
        artifacts_dir=args.artifacts_dir,
        split_config=split_config,
        training_config=training_config,
    )
    from nba_lineup_model.tracking import track_completed_run

    tracking = track_completed_run(run_dir)
    tracking_text = (
        f"; mlflow_run_id={tracking.mlflow_run_id}" if tracking is not None else ""
    )
    print(
        f"{manifest.season} CatBoost: possessions={manifest.possession_count}, "
        f"games={manifest.game_count}, players={manifest.player_count}, "
        f"features={manifest.feature_count}, max_iterations={manifest.max_iterations}, "
        f"best_iteration={manifest.best_iteration}, "
        f"trees={manifest.selected_tree_count}, "
        f"learning_rate={manifest.resolved_learning_rate:g}; "
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
