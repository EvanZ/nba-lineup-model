"""Validate and inventory every numerical artifact in an NBA GESTALT release."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_profiles import PROFILE_COLUMNS
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    PRESEASON_PREVIEW_SEASON,
    _published_profile_padding_contract,
    exposure_cohort_path,
    historical_profiles_path,
    historical_realized_profiles_path,
    lineup_rankings_path,
    player_context_exposure_path,
    player_team_splits_path,
    preseason_rankings_path,
    published_player_ratings_path,
)

DEFAULT_RELEASE_MANIFEST_DIR = Path("artifacts/web/releases")
EXPECTED_CONTEXT_ALPHA = 10_000.0
NUMERICAL_TOLERANCE = 1e-9


class ReleaseValidationError(ValueError):
    """Raised when a release mixes model runs or contains stale numerical data."""


def release_manifest_path(model_artifact: str, run_id: str) -> Path:
    """Return the immutable validation-manifest path for one release candidate."""

    return DEFAULT_RELEASE_MANIFEST_DIR / model_artifact / run_id / "bundle_manifest.json"


def validate_release_bundle(
    *,
    run_id: str | None = None,
    season: str = DISPLAY_SEASON,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Validate all model-dependent web data and optionally persist its manifest."""

    season_root = Path(artifacts_dir) / MODEL_ARTIFACT / season
    selected_run_id = run_id or _latest_run_id(season_root)
    run_dir = season_root / selected_run_id
    metadata_path = _require_file(run_dir / "metadata.json")
    metadata = json.loads(metadata_path.read_text())
    _validate_model_contract(metadata, season=season)
    padding_contract = _published_profile_padding_contract(metadata)

    models_path = _require_file(run_dir / "season_context_models.joblib")
    models = joblib.load(models_path)
    if not isinstance(models, dict) or not models:
        raise ReleaseValidationError("The release has no completed season context models")
    model_seasons = sorted(str(value) for value in models)
    if model_seasons[-1] != season:
        raise ReleaseValidationError(
            f"The final context season is {model_seasons[-1]}, expected {season}"
        )
    season_model_contracts: list[dict[str, Any]] = []
    for model_season, model in sorted(models.items()):
        configured = getattr(model, "configured_regularization", None)
        if configured is None:
            pipeline = getattr(model, "pipeline", None)
            ridge = getattr(pipeline, "named_steps", {}).get("ridge")
            configured = getattr(ridge, "alpha", None)
        if configured is None or not np.isclose(
            float(configured),
            EXPECTED_CONTEXT_ALPHA,
            rtol=0.0,
            atol=NUMERICAL_TOLERANCE,
        ):
            raise ReleaseValidationError(
                f"The {model_season} context model does not use the published "
                f"regularization {EXPECTED_CONTEXT_ALPHA:g}"
            )
        season_model_contracts.append(
            {
                "season": str(model_season),
                "feature_set": getattr(model, "feature_set", None),
                "context_alpha": float(configured),
                "pipeline_steps": list(getattr(model.pipeline, "named_steps", {})),
            }
        )

    row_counts: dict[Path, int] = {}
    lineup_files = _validate_lineup_rankings(
        run_id=selected_run_id,
        seasons=model_seasons,
        row_counts=row_counts,
    )
    ratings_path = published_player_ratings_path(MODEL_ARTIFACT, selected_run_id)
    ratings = _read_parquet(ratings_path, row_counts=row_counts)
    _validate_season_player_table(
        ratings,
        label="published player ratings",
        required_seasons=model_seasons,
        finite_columns=("rapm", "prior_rapm", "rapm_adjustment_from_prior"),
    )

    team_splits_path = player_team_splits_path(MODEL_ARTIFACT, selected_run_id)
    team_splits = _read_parquet(team_splits_path, row_counts=row_counts)
    _validate_season_player_table(
        team_splits,
        label="player team splits",
        required_seasons=model_seasons,
        allow_multiple_rows=True,
    )

    context_exposure_path = player_context_exposure_path(MODEL_ARTIFACT, selected_run_id)
    context_exposure = _read_parquet(context_exposure_path, row_counts=row_counts)
    _validate_season_player_table(
        context_exposure,
        label="player context exposure",
        required_seasons=model_seasons,
        finite_columns=("observed_context_exposure",),
    )

    cohort_path = exposure_cohort_path(MODEL_ARTIFACT, selected_run_id)
    cohort = _read_parquet(cohort_path, row_counts=row_counts)
    _validate_season_player_table(
        cohort,
        label="exposure cohort",
        required_seasons=model_seasons,
        finite_columns=("on_court_possessions", "exposure_share"),
    )

    historical_path = historical_profiles_path(MODEL_ARTIFACT, selected_run_id)
    historical_profiles = _read_parquet(historical_path, row_counts=row_counts)
    historical_profile_columns = tuple(
        column for column in PROFILE_COLUMNS if column in historical_profiles.columns
    )
    missing_profile_columns = set(PROFILE_COLUMNS) - set(historical_profile_columns)
    legacy_usage_contract = (
        "usage_percentage_pseudo_possessions"
        not in metadata.get("profile_padding_contract", {})
    )
    if missing_profile_columns and not (
        legacy_usage_contract and missing_profile_columns == {"usage_pct"}
    ):
        raise ReleaseValidationError(
            "historical player profiles lack required columns "
            + ", ".join(sorted(missing_profile_columns))
        )
    _validate_season_player_table(
        historical_profiles,
        label="historical player profiles",
        required_seasons=model_seasons,
        finite_columns=historical_profile_columns,
    )
    _validate_target_profiles(
        historical_profiles,
        target_profiles_path=run_dir / "target_player_profiles.parquet",
        target_season=season,
        row_counts=row_counts,
        profile_columns=historical_profile_columns,
    )
    realized_path = historical_realized_profiles_path(MODEL_ARTIFACT, selected_run_id)
    realized_profiles = _read_parquet(realized_path, row_counts=row_counts)
    _validate_season_player_table(
        realized_profiles,
        label="historical realized player profiles",
        required_seasons=model_seasons,
        finite_columns=PROFILE_COLUMNS,
    )

    preseason_path = preseason_rankings_path(
        MODEL_ARTIFACT, selected_run_id, PRESEASON_PREVIEW_SEASON
    )
    preseason = _read_parquet(preseason_path, row_counts=row_counts)
    _validate_season_player_table(
        preseason,
        label="preseason player rankings",
        required_seasons=[PRESEASON_PREVIEW_SEASON],
        finite_columns=("rapm", "prior_rating"),
    )
    preseason_metadata_path = _require_file(preseason_path.with_suffix(".json"))
    preseason_metadata = json.loads(preseason_metadata_path.read_text())
    if (
        preseason_metadata.get("model_artifact") != MODEL_ARTIFACT
        or preseason_metadata.get("run_id") != selected_run_id
    ):
        raise ReleaseValidationError("Preseason rankings belong to a different model run")

    model_files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    numerical_files = [
        ratings_path,
        *lineup_files,
        team_splits_path,
        context_exposure_path,
        cohort_path,
        historical_path,
        realized_path,
        preseason_path,
        preseason_metadata_path,
    ]
    manifest = {
        "schema_version": 1,
        "validated_at": datetime.now(UTC).isoformat(),
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_artifact": MODEL_ARTIFACT,
        "run_id": selected_run_id,
        "display_season": season,
        "available_lab_seasons": model_seasons,
        "context_alpha": EXPECTED_CONTEXT_ALPHA,
        "profile_padding_contract": padding_contract.metadata(),
        "response_curve_contract": "linear_context_no_spline_curve_artifacts",
        "season_context_models": season_model_contracts,
        "model_files": [_file_record(path, row_counts=row_counts) for path in model_files],
        "numerical_cache_files": [
            _file_record(path, row_counts=row_counts) for path in numerical_files
        ],
    }
    if write_manifest:
        output = release_manifest_path(MODEL_ARTIFACT, selected_run_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _latest_run_id(season_root: Path) -> str:
    latest = _require_file(season_root / "latest.json")
    return str(json.loads(latest.read_text())["run_id"])


def _validate_model_contract(metadata: dict[str, Any], *, season: str) -> None:
    if metadata.get("model") != MODEL_NAME:
        raise ReleaseValidationError("The release artifact has an unexpected model identity")
    if str(metadata.get("target_season")) != season:
        raise ReleaseValidationError("The release artifact has an unexpected target season")
    if not np.isclose(
        float(metadata.get("context_alpha", np.nan)),
        EXPECTED_CONTEXT_ALPHA,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        raise ReleaseValidationError(
            f"The release context alpha is not the published {EXPECTED_CONTEXT_ALPHA:g}"
        )


def _validate_lineup_rankings(
    *,
    run_id: str,
    seasons: list[str],
    row_counts: dict[Path, int],
) -> list[Path]:
    root = lineup_rankings_path(MODEL_ARTIFACT, run_id, seasons[0]).parent
    actual = {path.stem for path in root.glob("*.parquet") if path.stem != "player_ratings"}
    expected = set(seasons)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReleaseValidationError(
            f"Observed-lineup cache season mismatch; missing={missing}, extra={extra}"
        )
    files: list[Path] = []
    for season in seasons:
        path = lineup_rankings_path(MODEL_ARTIFACT, run_id, season)
        frame = _read_parquet(path, row_counts=row_counts)
        required = {
            "team_id",
            "lineup_key",
            "player_edge",
            "composition_edge",
            "matchup_bonus",
            "context_edge",
            "gestalt_score",
        }
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ReleaseValidationError(f"{season} lineup cache lacks {missing_columns}")
        if frame.empty or frame.duplicated(["team_id", "lineup_key"]).any():
            raise ReleaseValidationError(f"{season} lineup cache is empty or has duplicate units")
        _require_finite(frame, required - {"team_id", "lineup_key"}, label=f"{season} lineups")
        _require_close(
            frame["context_edge"],
            frame["composition_edge"] + frame["matchup_bonus"],
            label=f"{season} context edge",
        )
        _require_close(
            frame["gestalt_score"],
            frame["player_edge"] + frame["context_edge"],
            label=f"{season} GESTALT score",
        )
        files.append(path)
    return files


def _validate_season_player_table(
    frame: pd.DataFrame,
    *,
    label: str,
    required_seasons: list[str],
    finite_columns: tuple[str, ...] = (),
    allow_multiple_rows: bool = False,
) -> None:
    required = {"season", "player_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ReleaseValidationError(f"{label} lacks {missing}")
    if frame.empty:
        raise ReleaseValidationError(f"{label} is empty")
    actual_seasons = set(frame["season"].astype(str))
    missing_seasons = sorted(set(required_seasons) - actual_seasons)
    if missing_seasons:
        raise ReleaseValidationError(f"{label} is missing seasons {missing_seasons}")
    if not allow_multiple_rows and frame.duplicated(["season", "player_id"]).any():
        raise ReleaseValidationError(f"{label} has duplicate season-player rows")
    _require_finite(frame, finite_columns, label=label)


def _validate_target_profiles(
    historical_profiles: pd.DataFrame,
    *,
    target_profiles_path: Path,
    target_season: str,
    row_counts: dict[Path, int],
    profile_columns: tuple[str, ...] = PROFILE_COLUMNS,
) -> None:
    target = _read_parquet(target_profiles_path, row_counts=row_counts).set_index("player_id")
    cached = historical_profiles.loc[
        historical_profiles["season"].astype(str).eq(target_season)
    ].set_index("player_id")
    if set(target.index.astype(int)) != set(cached.index.astype(int)):
        raise ReleaseValidationError(
            "The completed-season historical profile cache has a different player pool"
        )
    cached = cached.loc[target.index]
    for column in profile_columns:
        _require_close(
            cached[column],
            target[column],
            label=f"completed-season {column} profile",
        )
    for column in ("profile_imputed", "profile_replacement_weight"):
        if column in target and column in cached:
            _require_close(
                cached[column],
                target[column],
                label=f"completed-season {column} profile",
            )


def _read_parquet(path: Path, *, row_counts: dict[Path, int]) -> pd.DataFrame:
    _require_file(path)
    frame = pd.read_parquet(path)
    row_counts[path] = len(frame)
    return frame


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise ReleaseValidationError(f"Required release artifact is missing: {path}")
    return path


def _require_finite(frame: pd.DataFrame, columns: Any, *, label: str) -> None:
    for column in columns:
        if column not in frame:
            raise ReleaseValidationError(f"{label} lacks numerical column {column}")
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ReleaseValidationError(f"{label} contains non-finite {column} values")


def _require_close(left: Any, right: Any, *, label: str) -> None:
    if not np.allclose(
        np.asarray(left, dtype=float),
        np.asarray(right, dtype=float),
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
        equal_nan=False,
    ):
        raise ReleaseValidationError(f"{label} does not match its release identity")


def _file_record(path: Path, *, row_counts: dict[Path, int]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path in row_counts:
        record["rows"] = row_counts[path]
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Validate the selected public web release and write its immutable manifest."""

    parser = argparse.ArgumentParser(description="Validate an NBA GESTALT release bundle")
    parser.add_argument("--season", default=DISPLAY_SEASON)
    parser.add_argument("--run-id")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    manifest = validate_release_bundle(
        run_id=args.run_id,
        season=str(args.season),
        write_manifest=not args.no_write,
    )
    output = release_manifest_path(MODEL_ARTIFACT, str(manifest["run_id"]))
    print(
        f"Validated {manifest['model_display_name']} release {manifest['run_id']} "
        f"across {len(manifest['available_lab_seasons'])} seasons"
    )
    if not args.no_write:
        print(f"Wrote release manifest: {output}")


if __name__ == "__main__":
    main()
