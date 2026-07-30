from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.neural_data import (
    build_neural_possession_dataset,
    read_neural_possessions,
    validate_neural_possession_partition,
)
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    BaselineRunManifest,
    ChronologicalFold,
    ChronologicalSplitConfig,
    RapmBasePredictionManifest,
)
from nba_lineup_model.modeling.stints import (
    build_rapm_stint_dataset,
    read_rapm_stints,
    validate_rapm_stint_partition,
)
from nba_lineup_model.modeling.train import (
    GameSplitPlan,
    chronological_game_splits,
    validate_baseline_run,
)
from nba_lineup_model.models.baselines import (
    RidgeLineupModel,
    entity_vocabulary,
    signed_entity_matrix,
    vocabulary_mapping,
)

BASE_PREDICTION_COLUMNS = (
    "schema_version",
    "season",
    "season_type",
    "stage",
    "role",
    "base_is_out_of_sample",
    "base_train_game_count",
    "game_id",
    "game_date",
    "game_time_utc",
    "possession_id",
    "possession_index",
    "home_offense_sign",
    "target_offense_margin",
    "target_home_margin",
    "rapm_home_net_rating",
    "prediction_rapm",
    "prediction_rapm_home_margin",
    "residual_target",
)


@dataclass(frozen=True)
class RapmBasePredictions:
    """Stage-aware RAPM predictions and the states that produced them."""

    split_plan: GameSplitPlan
    player_columns: dict[int, int]
    predictions: pd.DataFrame
    player_coefficients: pd.DataFrame
    stage_parameters: pd.DataFrame


def build_rapm_base_prediction_dataset(
    season: str,
    *,
    source_rapm_run_id: str | None = None,
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
) -> RapmBasePredictionManifest:
    """Build leakage-safe stage predictions from the canonical RAPM baseline."""

    source_dir, source_manifest = _resolve_source_rapm_run(
        season,
        source_rapm_run_id,
        artifacts_dir,
    )
    rapm_dir = Path(analytical_dir) / "rapm_stints" / season / "regular"
    if rapm_dir.is_dir():
        rapm_manifest = validate_rapm_stint_partition(rapm_dir)
    else:
        rapm_manifest = build_rapm_stint_dataset(
            season,
            curated_dir=curated_dir,
            analytical_dir=analytical_dir,
        )
    neural_dir = (
        Path(analytical_dir) / "neural_possessions" / season / "regular"
    )
    if neural_dir.is_dir():
        neural_manifest = validate_neural_possession_partition(neural_dir)
    else:
        neural_manifest = build_neural_possession_dataset(
            season,
            curated_dir=curated_dir,
            analytical_dir=analytical_dir,
        )
    if (
        rapm_manifest.included_stint_count != source_manifest.stint_count
        or rapm_manifest.source_game_count != source_manifest.game_count
        or rapm_manifest.player_count != source_manifest.player_count
    ):
        raise ValueError("Current RAPM stint structure does not match the source run")
    stints = read_rapm_stints(season, analytical_dir)
    possessions = read_neural_possessions(season, analytical_dir)
    result = fit_rapm_base_predictions(
        possessions,
        stints,
        split_config=source_manifest.split_config,
        regularization=source_manifest.selected_rapm_lambda,
    )
    _require_source_splits(source_dir, result.split_plan)
    return _write_dataset(
        season,
        result,
        source_dir,
        source_manifest,
        rapm_manifest.part_sha256,
        neural_manifest.part_sha256,
        analytical_dir,
    )


def fit_rapm_base_predictions(
    possessions: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    split_config: ChronologicalSplitConfig | None = None,
    regularization: float,
) -> RapmBasePredictions:
    """Fit one frozen RAPM state per chronological model stage."""

    if regularization < 0:
        raise ValueError("RAPM regularization must be nonnegative")
    split = split_config or ChronologicalSplitConfig()
    if possessions.empty or stints.empty:
        raise ValueError("RAPM base predictions require possessions and stints")
    if possessions.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("RAPM base prediction possession keys must be unique")
    possession_split = chronological_game_splits(possessions, split)
    stint_split = chronological_game_splits(stints, split)
    _require_matching_split_plans(possession_split, stint_split)

    player_ids = entity_vocabulary(
        stints,
        "home_player_ids",
        "away_player_ids",
        multiple=True,
    )
    player_columns = vocabulary_mapping(player_ids)
    stint_matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    possession_matrix = _possession_home_player_matrix(
        possessions,
        player_columns,
    )
    stint_target = stints["target_home_net_rating"].to_numpy(dtype=float)
    stint_weights = stints["possessions"].to_numpy(dtype=float)
    stint_game_ids = stints["game_id"].astype(str).to_numpy()
    possession_game_ids = possessions["game_id"].astype(str).to_numpy()

    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for stage, roles in _stage_definitions(possession_split):
        train_game_ids = roles["train"]
        prediction_game_ids = tuple(
            game_id for identifiers in roles.values() for game_id in identifiers
        )
        train_stint_mask = np.isin(stint_game_ids, train_game_ids)
        prediction_mask = np.isin(possession_game_ids, prediction_game_ids)
        train_possession_mask = np.isin(possession_game_ids, train_game_ids)
        if not train_stint_mask.any() or not prediction_mask.any():
            raise ValueError(f"RAPM base stage is empty: {stage}")

        model = RidgeLineupModel(regularization).fit(
            stint_matrix[train_stint_mask],
            stint_target[train_stint_mask],
            stint_weights[train_stint_mask],
        )
        stage_possessions = possessions.loc[prediction_mask].copy()
        home_net_rating = model.predict(possession_matrix[prediction_mask])
        mean_offense_margin = float(
            possessions.loc[
                train_possession_mask,
                "target_offense_margin",
            ].mean()
        )
        signs = stage_possessions["home_offense_sign"].to_numpy(dtype=float)
        base_prediction = mean_offense_margin + signs * home_net_rating / 200.0
        role_by_game = {
            game_id: role
            for role, identifiers in roles.items()
            for game_id in identifiers
        }
        stage_roles = stage_possessions["game_id"].astype(str).map(role_by_game)
        if stage_roles.isna().any():
            raise ValueError(f"RAPM base stage has unmapped game roles: {stage}")
        output = stage_possessions.loc[
            :,
            [
                "season",
                "season_type",
                "game_id",
                "game_date",
                "game_time_utc",
                "possession_id",
                "possession_index",
                "home_offense_sign",
                "target_offense_margin",
                "target_home_margin",
            ],
        ].copy()
        output.insert(0, "schema_version", 1)
        output.insert(3, "stage", stage)
        output.insert(4, "role", stage_roles.to_numpy())
        output.insert(5, "base_is_out_of_sample", output["role"].ne("train"))
        output.insert(6, "base_train_game_count", len(train_game_ids))
        output["rapm_home_net_rating"] = home_net_rating
        output["prediction_rapm"] = base_prediction
        output["prediction_rapm_home_margin"] = base_prediction * signs
        output["residual_target"] = (
            output["target_offense_margin"].to_numpy(dtype=float)
            - base_prediction
        )
        prediction_frames.append(output.loc[:, BASE_PREDICTION_COLUMNS])
        coefficient_frames.append(
            pd.DataFrame(
                {
                    "stage": stage,
                    "player_id": player_ids,
                    "rapm": model.coef_,
                }
            )
        )
        train_games = possession_split.ordered_games.loc[
            possession_split.ordered_games["game_id"].astype(str).isin(
                train_game_ids
            )
        ]
        parameter_rows.append(
            {
                "stage": stage,
                "selected_rapm_lambda": regularization,
                "sklearn_alpha": model.sklearn_alpha,
                "intercept_home_net_rating": model.intercept_,
                "mean_offense_margin": mean_offense_margin,
                "train_game_count": len(train_game_ids),
                "prediction_game_count": len(prediction_game_ids),
                "train_stint_count": int(train_stint_mask.sum()),
                "train_possession_count": int(train_possession_mask.sum()),
                "prediction_possession_count": int(prediction_mask.sum()),
                "first_train_game_id": train_game_ids[0],
                "last_train_game_id": train_game_ids[-1],
                "last_train_game_time_utc": train_games["game_time_utc"].max(),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    _validate_prediction_frames(
        predictions,
        coefficients,
        parameters,
        possession_split,
        len(player_ids),
    )
    return RapmBasePredictions(
        split_plan=possession_split,
        player_columns=player_columns,
        predictions=predictions,
        player_coefficients=coefficients,
        stage_parameters=parameters,
    )


def read_rapm_base_predictions(
    season: str,
    analytical_dir: Path | str = Path("data/analytical"),
) -> pd.DataFrame:
    """Read and validate the stage-specific RAPM prediction rows."""

    root = Path(analytical_dir) / "rapm_base_predictions" / season / "regular"
    validate_rapm_base_prediction_partition(root)
    return pd.read_parquet(root / "part-00000.parquet")


def read_rapm_base_state(
    season: str,
    analytical_dir: Path | str = Path("data/analytical"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read fitted coefficients and stage parameters for base prediction."""

    root = Path(analytical_dir) / "rapm_base_predictions" / season / "regular"
    validate_rapm_base_prediction_partition(root)
    return (
        pd.read_parquet(root / "rapm_player_coefficients.parquet"),
        pd.read_parquet(root / "stage_parameters.parquet"),
    )


def validate_rapm_base_prediction_partition(
    partition_dir: Path | str,
) -> RapmBasePredictionManifest:
    """Require exact files, hashes, roles, and chronological boundaries."""

    root = Path(partition_dir)
    manifest = RapmBasePredictionManifest.model_validate_json(
        (root / "_manifest.json").read_text()
    )
    expected = {artifact.filename for artifact in manifest.artifacts}
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "_manifest.json"
    }
    if actual != expected:
        raise ValueError("RAPM base-prediction files do not match the manifest")
    frames: dict[str, pd.DataFrame] = {}
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if path.stat().st_size != artifact.byte_count:
            raise ValueError(
                f"RAPM base-prediction byte count changed: {artifact.filename}"
            )
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(
                f"RAPM base-prediction hash changed: {artifact.filename}"
            )
        if artifact.row_count is not None:
            frame = pd.read_parquet(path)
            frames[path.name] = frame
            if len(frame) != artifact.row_count:
                raise ValueError(
                    f"RAPM base-prediction rows changed: {artifact.filename}"
                )
    predictions = frames["part-00000.parquet"]
    coefficients = frames["rapm_player_coefficients.parquet"]
    parameters = frames["stage_parameters.parquet"]
    missing = set(BASE_PREDICTION_COLUMNS) - set(predictions)
    if missing:
        raise ValueError(f"RAPM base predictions missing columns: {sorted(missing)}")
    if predictions.duplicated(["stage", "game_id", "possession_id"]).any():
        raise ValueError("RAPM base prediction keys must be unique within stage")
    if not predictions["base_is_out_of_sample"].eq(
        predictions["role"].ne("train")
    ).all():
        raise ValueError("RAPM base out-of-sample flags do not match row roles")
    residual = (
        predictions["target_offense_margin"].to_numpy(dtype=float)
        - predictions["prediction_rapm"].to_numpy(dtype=float)
    )
    if not np.allclose(
        residual,
        predictions["residual_target"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("RAPM base residual targets changed")
    stages = set(parameters["stage"].astype(str))
    if stages != set(predictions["stage"].astype(str)):
        raise ValueError("RAPM base prediction and parameter stages do not match")
    if stages != set(coefficients["stage"].astype(str)):
        raise ValueError("RAPM base coefficient and parameter stages do not match")
    if coefficients.duplicated(["stage", "player_id"]).any():
        raise ValueError("RAPM base coefficient keys must be unique")
    _validate_chronology(predictions, parameters)
    return manifest


def rapm_base_code_fingerprint(
    source_paths: Sequence[Path | str] | None = None,
) -> str:
    """Hash source files that define the stage-aware base predictions."""

    if source_paths is None:
        package_root = Path(__file__).parents[1]
        paths = (
            package_root / "modeling" / "residual_data.py",
            package_root / "modeling" / "schema.py",
            package_root / "modeling" / "train.py",
            package_root / "models" / "baselines.py",
        )
    else:
        paths = tuple(Path(path) for path in source_paths)
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("At least one RAPM base-prediction source is required")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise ValueError(f"RAPM base-prediction source does not exist: {path}")
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _stage_definitions(
    split_plan: GameSplitPlan,
) -> tuple[tuple[str, dict[str, tuple[str, ...]]], ...]:
    stages: list[tuple[str, dict[str, tuple[str, ...]]]] = [
        (
            f"cv_{fold.fold}",
            {
                "train": fold.train_game_ids,
                "validation": fold.validation_game_ids,
            },
        )
        for fold in split_plan.folds
    ]
    stages.extend(
        [
            (
                "final",
                {
                    "train": split_plan.final_train_game_ids,
                    "test": split_plan.final_test_game_ids,
                },
            ),
            (
                "all_season",
                {
                    "train": tuple(
                        split_plan.ordered_games["game_id"].astype(str)
                    ),
                },
            ),
        ]
    )
    return tuple(stages)


def _possession_home_player_matrix(
    possessions: pd.DataFrame,
    player_columns: dict[int, int],
):
    home_player_ids = []
    away_player_ids = []
    for offense, defense, sign in zip(
        possessions["offense_player_ids"],
        possessions["defense_player_ids"],
        possessions["home_offense_sign"],
        strict=True,
    ):
        if float(sign) == 1.0:
            home_player_ids.append(offense)
            away_player_ids.append(defense)
        elif float(sign) == -1.0:
            home_player_ids.append(defense)
            away_player_ids.append(offense)
        else:
            raise ValueError("RAPM base home-offense signs must be positive or negative one")
    frame = pd.DataFrame(
        {
            "home_player_ids": home_player_ids,
            "away_player_ids": away_player_ids,
        }
    )
    return signed_entity_matrix(
        frame,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )


def _require_matching_split_plans(
    possessions: GameSplitPlan,
    stints: GameSplitPlan,
) -> None:
    if tuple(possessions.ordered_games["game_id"].astype(str)) != tuple(
        stints.ordered_games["game_id"].astype(str)
    ):
        raise ValueError("RAPM stints and neural possessions have different games")
    if possessions.final_train_game_ids != stints.final_train_game_ids:
        raise ValueError("RAPM stints and neural possessions have different final train games")
    if possessions.final_test_game_ids != stints.final_test_game_ids:
        raise ValueError("RAPM stints and neural possessions have different final test games")
    for possession_fold, stint_fold in zip(
        possessions.folds,
        stints.folds,
        strict=True,
    ):
        if (
            possession_fold.train_game_ids != stint_fold.train_game_ids
            or possession_fold.validation_game_ids
            != stint_fold.validation_game_ids
        ):
            raise ValueError("RAPM stints and neural possessions have different CV folds")


def _require_source_splits(source_dir: Path, split_plan: GameSplitPlan) -> None:
    source = pd.read_parquet(source_dir / "game_splits.parquet")
    expected: dict[tuple[str, str], tuple[str, ...]] = {}
    for fold in split_plan.folds:
        expected[(f"cv_{fold.fold}", "train")] = fold.train_game_ids
        expected[(f"cv_{fold.fold}", "validation")] = fold.validation_game_ids
    expected[("final", "train")] = split_plan.final_train_game_ids
    expected[("final", "test")] = split_plan.final_test_game_ids
    for (split, role), identifiers in expected.items():
        actual = tuple(
            source.loc[
                source["split"].eq(split) & source["role"].eq(role),
                "game_id",
            ].astype(str)
        )
        if actual != identifiers:
            raise ValueError(f"Source RAPM split does not match: {split}/{role}")


def _validate_prediction_frames(
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    parameters: pd.DataFrame,
    split_plan: GameSplitPlan,
    player_count: int,
) -> None:
    if predictions.duplicated(["stage", "game_id", "possession_id"]).any():
        raise ValueError("RAPM base prediction keys must be unique within stage")
    if coefficients.duplicated(["stage", "player_id"]).any():
        raise ValueError("RAPM base coefficient keys must be unique")
    expected_stages = {
        *(f"cv_{fold.fold}" for fold in split_plan.folds),
        "final",
        "all_season",
    }
    if set(predictions["stage"]) != expected_stages:
        raise ValueError("RAPM base prediction stages are incomplete")
    if set(parameters["stage"]) != expected_stages:
        raise ValueError("RAPM base stage parameters are incomplete")
    if len(coefficients) != len(expected_stages) * player_count:
        raise ValueError("RAPM base coefficients do not cover every stage and player")
    if not predictions["base_is_out_of_sample"].eq(
        predictions["role"].ne("train")
    ).all():
        raise ValueError("RAPM base sample flags do not match roles")
    _validate_chronology(predictions, parameters)


def _validate_chronology(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    parameter_lookup = parameters.set_index("stage")
    out_of_sample = predictions.loc[predictions["base_is_out_of_sample"]]
    for stage, rows in out_of_sample.groupby("stage", sort=False):
        cutoff = pd.Timestamp(parameter_lookup.loc[stage, "last_train_game_time_utc"])
        if not rows["game_time_utc"].gt(cutoff).all():
            raise ValueError(
                f"RAPM base evaluation rows are not after their train cutoff: {stage}"
            )


def _resolve_source_rapm_run(
    season: str,
    run_id: str | None,
    artifacts_dir: Path | str,
) -> tuple[Path, BaselineRunManifest]:
    season_dir = Path(artifacts_dir) / "rapm" / season
    if run_id is None:
        latest = season_dir / "latest.json"
        if not latest.is_file():
            raise ValueError(f"No source RAPM run exists under {season_dir}")
        run_id = str(json.loads(latest.read_text())["run_id"])
    run_dir = season_dir / run_id
    if not run_dir.is_dir():
        raise ValueError(f"Source RAPM run does not exist: {run_dir}")
    manifest = validate_baseline_run(run_dir)
    if manifest.season != season:
        raise ValueError("Source RAPM run season does not match")
    return run_dir, manifest


def _write_dataset(
    season: str,
    result: RapmBasePredictions,
    source_dir: Path,
    source_manifest: BaselineRunManifest,
    rapm_part_sha256: str,
    neural_part_sha256: str,
    analytical_dir: Path | str,
) -> RapmBasePredictionManifest:
    target_dir = Path(analytical_dir) / "rapm_base_predictions" / season / "regular"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = target_dir.parent / f".{target_dir.name}.tmp-{uuid4().hex}"
    backup_dir = target_dir.parent / f".{target_dir.name}.bak-{uuid4().hex}"
    temporary_dir.mkdir()
    try:
        outputs = {
            "part-00000.parquet": result.predictions,
            "rapm_player_coefficients.parquet": result.player_coefficients,
            "stage_parameters.parquet": result.stage_parameters,
        }
        for filename, frame in outputs.items():
            frame.to_parquet(temporary_dir / filename, index=False)
        artifacts = tuple(
            ArtifactRecord(
                filename=filename,
                row_count=len(frame),
                byte_count=(temporary_dir / filename).stat().st_size,
                sha256=_sha256_file(temporary_dir / filename),
            )
            for filename, frame in outputs.items()
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
            for fold in result.split_plan.folds
        )
        rapm_dataset_dir = Path(analytical_dir) / "rapm_stints" / season / "regular"
        neural_dataset_dir = (
            Path(analytical_dir) / "neural_possessions" / season / "regular"
        )
        manifest = RapmBasePredictionManifest(
            season=season,
            created_at=datetime.now(UTC),
            builder_code_version=rapm_base_code_fingerprint(),
            source_rapm_run_id=source_manifest.run_id,
            source_rapm_manifest_sha256=_sha256_file(
                source_dir / "manifest.json"
            ),
            rapm_stints_manifest_sha256=_sha256_file(
                rapm_dataset_dir / "_manifest.json"
            ),
            rapm_stints_part_sha256=rapm_part_sha256,
            neural_possessions_manifest_sha256=_sha256_file(
                neural_dataset_dir / "_manifest.json"
            ),
            neural_possessions_part_sha256=neural_part_sha256,
            selected_rapm_lambda=source_manifest.selected_rapm_lambda,
            possession_count=len(
                result.predictions.loc[
                    result.predictions["stage"].eq("all_season")
                ]
            ),
            game_count=len(result.split_plan.ordered_games),
            player_count=len(result.player_columns),
            split_config=source_manifest.split_config,
            folds=folds,
            stage_count=len(result.stage_parameters),
            prediction_row_count=len(result.predictions),
            in_sample_prediction_count=int(
                (~result.predictions["base_is_out_of_sample"]).sum()
            ),
            out_of_sample_prediction_count=int(
                result.predictions["base_is_out_of_sample"].sum()
            ),
            player_coefficient_row_count=len(result.player_coefficients),
            artifacts=artifacts,
        )
        (temporary_dir / "_manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
        validate_rapm_base_prediction_partition(temporary_dir)
        if target_dir.exists():
            target_dir.replace(backup_dir)
        try:
            temporary_dir.replace(target_dir)
        except Exception:
            if backup_dir.exists() and not target_dir.exists():
                backup_dir.replace(target_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        validate_rapm_base_prediction_partition(target_dir)
        return manifest
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
