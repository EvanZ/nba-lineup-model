from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run

DEFAULT_TRACKING_ROOT = Path("artifacts/mlflow")
TRACKING_ENABLED_ENV = "NBA_MLFLOW_TRACKING_ENABLED"
TRACKING_ROOT_ENV = "NBA_MLFLOW_ROOT"

_MODEL_RUN_PREFIXES = (
    "baseline-",
    "bayesian-",
    "catboost-",
    "deep-sets-",
    "neural-",
    "rapm-transformer-",
)
_METRIC_FILES = (
    "test_metrics.parquet",
    "metrics.parquet",
    "seed_metrics.parquet",
    "comparison_metrics.parquet",
    "predictive_calibration.parquet",
    "comparisons.parquet",
)
_IDENTITY_COLUMNS = (
    "cohort",
    "model",
    "comparison",
    "metric",
    "seed",
    "fold",
    "nominal_coverage",
    "allocation_policy",
)
_PARAMETER_EXCLUSIONS = {"artifacts", "folds", "created_at", "run_id"}
_PARAMETER_VALUE_LIMIT = 1_000


@dataclass(frozen=True)
class TrackingResult:
    """MLflow identity for one immutable project run."""

    project_run_id: str
    mlflow_run_id: str
    experiment_name: str
    created: bool


def tracking_enabled() -> bool:
    """Return whether completed CLI runs should be indexed automatically."""

    value = os.getenv(TRACKING_ENABLED_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def resolve_tracking_root(root: Path | str | None = None) -> Path:
    """Resolve and create the local MLflow storage root."""

    configured = root
    if configured is None:
        configured = os.getenv(TRACKING_ROOT_ENV, str(DEFAULT_TRACKING_ROOT))
    path = Path(configured).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "artifacts").mkdir(exist_ok=True)
    return path


def resolve_tracking_uri(
    *,
    tracking_root: Path | str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Use an explicit/server URI or the project-local SQLite database."""

    configured = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    root = resolve_tracking_root(tracking_root)
    return f"sqlite:///{root / 'mlflow.db'}"


def track_completed_run(
    run_dir: Path | str,
    *,
    tracking_root: Path | str | None = None,
    tracking_uri: str | None = None,
) -> TrackingResult | None:
    """Index a completed run unless automatic MLflow tracking is disabled."""

    if not tracking_enabled():
        return None
    return track_immutable_run(
        run_dir,
        tracking_root=tracking_root,
        tracking_uri=tracking_uri,
    )


def track_immutable_run(
    run_dir: Path | str,
    *,
    tracking_root: Path | str | None = None,
    tracking_uri: str | None = None,
) -> TrackingResult:
    """Index one immutable artifact run in MLflow, idempotently."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"Run directory does not contain manifest.json: {root}")
    manifest = json.loads(manifest_path.read_text())
    project_run_id = _required_text(manifest, "run_id")
    season = _required_text(manifest, "season")
    if root.name != project_run_id:
        raise ValueError("Run directory name does not match manifest run_id")
    manifest_sha256 = _sha256_file(manifest_path)
    local_root = resolve_tracking_root(tracking_root)
    uri = resolve_tracking_uri(
        tracking_root=local_root,
        tracking_uri=tracking_uri,
    )
    client = MlflowClient(tracking_uri=uri)
    experiment_name = _experiment_name(project_run_id, season)
    experiment_id = _get_or_create_experiment(
        client,
        experiment_name,
        local_root,
        uri,
    )
    existing = _find_primary_run(client, experiment_id, project_run_id)
    if existing is not None:
        recorded_hash = existing.data.tags.get("project.manifest_sha256")
        if recorded_hash != manifest_sha256:
            raise ValueError(
                "Existing MLflow run has a different immutable manifest hash: "
                f"{project_run_id}"
            )
        _ensure_candidate_runs(
            client,
            experiment_id,
            existing.info.run_id,
            root,
            manifest,
        )
        return TrackingResult(
            project_run_id=project_run_id,
            mlflow_run_id=existing.info.run_id,
            experiment_name=experiment_name,
            created=False,
        )

    created_at_ms = _created_at_milliseconds(manifest)
    tags = {
        "project.run_id": project_run_id,
        "project.run_role": "primary",
        "project.run_kind": _run_kind(project_run_id, manifest),
        "project.season": season,
        "project.manifest_sha256": manifest_sha256,
        "project.run_directory": str(root),
        "project.created_at": str(manifest.get("created_at", "")),
        "mlflow.note.content": (
            "Secondary experiment index for the immutable project artifact run "
            f"`{project_run_id}`."
        ),
    }
    run = client.create_run(
        experiment_id,
        start_time=created_at_ms,
        tags=tags,
        run_name=project_run_id,
    )
    mlflow_run_id = run.info.run_id
    try:
        parameters = _manifest_parameters(manifest)
        parameters.update(_model_parameters(root))
        for key, value in sorted(parameters.items()):
            client.log_param(mlflow_run_id, key, value)
        for key, value in sorted(_run_metrics(root).items()):
            client.log_metric(
                mlflow_run_id,
                key,
                value,
                timestamp=created_at_ms,
                step=0,
            )
        client.log_artifacts(
            mlflow_run_id,
            str(root),
            artifact_path="immutable_run",
        )
        _ensure_candidate_runs(
            client,
            experiment_id,
            mlflow_run_id,
            root,
            manifest,
        )
    except Exception:
        client.set_terminated(
            mlflow_run_id,
            status="FAILED",
            end_time=created_at_ms,
        )
        raise
    client.set_terminated(
        mlflow_run_id,
        status="FINISHED",
        end_time=created_at_ms,
    )
    return TrackingResult(
        project_run_id=project_run_id,
        mlflow_run_id=mlflow_run_id,
        experiment_name=experiment_name,
        created=True,
    )


def discover_latest_runs(
    artifacts_root: Path | str = Path("artifacts"),
    *,
    season: str | None = None,
) -> tuple[Path, ...]:
    """Resolve every valid run referenced by a latest.json pointer."""

    root = Path(artifacts_root)
    runs: list[Path] = []
    for latest_path in sorted(root.rglob("latest.json")):
        if DEFAULT_TRACKING_ROOT.name in latest_path.parts:
            continue
        payload = json.loads(latest_path.read_text())
        run_id = _required_text(payload, "run_id")
        run_dir = latest_path.parent / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Latest pointer does not resolve to a run: {latest_path}")
        if season is not None:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("season") != season:
                continue
        runs.append(run_dir.resolve())
    return tuple(runs)


def discover_all_runs(
    artifacts_root: Path | str = Path("artifacts"),
    *,
    season: str | None = None,
) -> tuple[Path, ...]:
    """Discover every immutable artifact run beneath the artifact root."""

    root = Path(artifacts_root)
    runs: list[Path] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        if DEFAULT_TRACKING_ROOT.name in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text())
        if "run_id" not in manifest or "season" not in manifest:
            continue
        if season is not None and manifest["season"] != season:
            continue
        runs.append(manifest_path.parent.resolve())
    return tuple(runs)


def sync_runs(
    run_dirs: Iterable[Path | str],
    *,
    tracking_root: Path | str | None = None,
    tracking_uri: str | None = None,
) -> tuple[TrackingResult, ...]:
    """Index a sequence of immutable runs."""

    return tuple(
        track_immutable_run(
            run_dir,
            tracking_root=tracking_root,
            tracking_uri=tracking_uri,
        )
        for run_dir in run_dirs
    )


def _get_or_create_experiment(
    client: MlflowClient,
    name: str,
    tracking_root: Path,
    tracking_uri: str,
) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_location = None
    if tracking_uri.startswith(("sqlite:", "file:")):
        artifact_dir = tracking_root / "artifacts" / _slug(name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_dir.as_uri()
    return client.create_experiment(
        name,
        artifact_location=artifact_location,
        tags={"project": "nba-lineup-model"},
    )


def _find_primary_run(
    client: MlflowClient,
    experiment_id: str,
    project_run_id: str,
) -> Run | None:
    escaped = project_run_id.replace("\\", "\\\\").replace("'", "\\'")
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`project.run_id` = '{escaped}'",
        max_results=1_000,
    )
    primary = [
        run
        for run in matches
        if run.data.tags.get("project.run_role") == "primary"
    ]
    if len(primary) > 1:
        raise ValueError(f"Multiple MLflow primary runs exist for {project_run_id}")
    return primary[0] if primary else None


def _ensure_candidate_runs(
    client: MlflowClient,
    experiment_id: str,
    parent_run_id: str,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    summary_path = run_dir / "hyperparameter_summary.parquet"
    if not summary_path.is_file():
        return
    summary = pd.read_parquet(summary_path)
    trials_path = run_dir / "hyperparameter_trials.parquet"
    trials = pd.read_parquet(trials_path) if trials_path.is_file() else None
    project_run_id = _required_text(manifest, "run_id")
    created_at_ms = _created_at_milliseconds(manifest)
    existing = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
        max_results=1_000,
    )
    existing_keys = {
        run.data.tags.get("project.child_key")
        for run in existing
        if run.data.tags.get("project.child_key")
    }
    for row in summary.to_dict("records"):
        candidate_index = int(row["candidate_index"])
        child_key = f"candidate:{candidate_index}"
        if child_key in existing_keys:
            continue
        child = client.create_run(
            experiment_id,
            start_time=created_at_ms,
            run_name=f"{project_run_id}/candidate-{candidate_index}",
            tags={
                "mlflow.parentRunId": parent_run_id,
                "project.run_id": project_run_id,
                "project.run_role": "hyperparameter_candidate",
                "project.child_key": child_key,
                "project.selected": str(bool(row.get("selected", False))).lower(),
            },
        )
        child_id = child.info.run_id
        try:
            parameter_columns = {
                "candidate_index",
                "learning_rate",
                "weight_decay",
                "fold_count",
                "selected",
            }
            for key in sorted(parameter_columns):
                value = row.get(key)
                if value is not None and not _is_missing(value):
                    client.log_param(child_id, key, value)
            for key, value in sorted(row.items()):
                if key in parameter_columns or not _is_finite_number(value):
                    continue
                client.log_metric(
                    child_id,
                    _metric_key(key),
                    float(value),
                    timestamp=created_at_ms,
                    step=0,
                )
            if trials is not None:
                candidate_trials = trials.loc[
                    trials["candidate_index"] == candidate_index
                ].sort_values("fold")
                for trial in candidate_trials.to_dict("records"):
                    step = int(trial["fold"])
                    for key in ("validation_mse", "selected_epochs"):
                        value = trial.get(key)
                        if _is_finite_number(value):
                            client.log_metric(
                                child_id,
                                f"fold.{_metric_key(key)}",
                                float(value),
                                timestamp=created_at_ms,
                                step=step,
                            )
        except Exception:
            client.set_terminated(
                child_id,
                status="FAILED",
                end_time=created_at_ms,
            )
            raise
        client.set_terminated(
            child_id,
            status="FINISHED",
            end_time=created_at_ms,
        )


def _manifest_parameters(manifest: Mapping[str, Any]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    _flatten_parameters(manifest, flattened, prefix="manifest")
    return flattened


def _model_parameters(run_dir: Path) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for filename in ("model_parameters.json", "resolved_parameters.json"):
        path = run_dir / filename
        if not path.is_file():
            continue
        _flatten_parameters(
            json.loads(path.read_text()),
            flattened,
            prefix=_metric_key(path.stem),
        )
    return flattened


def _flatten_parameters(
    value: Any,
    output: dict[str, str],
    *,
    prefix: str,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _PARAMETER_EXCLUSIONS:
                continue
            _flatten_parameters(
                child,
                output,
                prefix=f"{prefix}.{_metric_key(str(key))}",
            )
        return
    if isinstance(value, (list, tuple)):
        if not all(_is_scalar(item) for item in value):
            return
        rendered = json.dumps(value, separators=(",", ":"), default=str)
    elif _is_scalar(value):
        rendered = str(value)
    else:
        return
    if len(rendered) <= _PARAMETER_VALUE_LIMIT:
        output[prefix] = rendered


def _run_metrics(run_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for filename in _METRIC_FILES:
        path = run_dir / filename
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        stem = _metric_key(path.stem)
        for row_index, row in enumerate(frame.to_dict("records")):
            identities = [
                f"{_metric_key(column)}_{_metric_key(str(row[column]))}"
                for column in _IDENTITY_COLUMNS
                if column in row and not _is_missing(row[column])
            ]
            prefix = ".".join([stem, *identities])
            for column, value in row.items():
                if column in _IDENTITY_COLUMNS or not _is_finite_number(value):
                    continue
                key = f"{prefix}.{_metric_key(column)}"
                if key in metrics:
                    key = f"{key}.row_{row_index}"
                metrics[key] = float(value)
    selected_path = run_dir / "hyperparameter_summary.parquet"
    if selected_path.is_file():
        selected = pd.read_parquet(selected_path)
        if "selected" in selected.columns:
            selected = selected.loc[selected["selected"]]
        if len(selected) == 1:
            for column, value in selected.iloc[0].items():
                if _is_finite_number(value):
                    metrics[f"selection.{_metric_key(column)}"] = float(value)
    return metrics


def _experiment_name(project_run_id: str, season: str) -> str:
    group = (
        "models"
        if project_run_id.startswith(_MODEL_RUN_PREFIXES)
        else "reports"
    )
    return f"nba-lineup-model-{season}-{group}"


def _run_kind(project_run_id: str, manifest: Mapping[str, Any]) -> str:
    if project_run_id.startswith("baseline-"):
        return "ridge_rapm"
    if project_run_id.startswith("bayesian-"):
        return "bayesian_rapm"
    if project_run_id.startswith("neural-"):
        return str(manifest.get("architecture", "additive_neural"))
    if project_run_id.startswith("deep-sets-"):
        return "deep_sets"
    if project_run_id.startswith("catboost-"):
        return "catboost"
    if project_run_id.startswith("rapm-transformer-"):
        return "rapm_transformer"
    if project_run_id.startswith("evaluation-"):
        return "model_evaluation"
    if project_run_id.startswith("diagnostics-"):
        return "rapm_diagnostics"
    return "artifact_run"


def _created_at_milliseconds(manifest: Mapping[str, Any]) -> int:
    raw = _required_text(manifest, "created_at")
    timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return int(timestamp.timestamp() * 1_000)


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest requires a non-empty {key}")
    return value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value))


def _metric_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "value"


def _slug(value: str) -> str:
    return _metric_key(value).replace(".", "-").lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index immutable model and report runs in MLflow."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Explicit immutable run directories; defaults to latest pointers",
    )
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--tracking-root")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--season")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync every discovered run instead of only latest pointers",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_dirs and args.all:
        raise SystemExit("Explicit run directories cannot be combined with --all")
    if args.run_dirs:
        run_dirs = tuple(Path(path) for path in args.run_dirs)
    elif args.all:
        run_dirs = discover_all_runs(args.artifacts_root, season=args.season)
    else:
        run_dirs = discover_latest_runs(args.artifacts_root, season=args.season)
    if not run_dirs:
        raise SystemExit("No immutable runs were discovered")
    results = sync_runs(
        run_dirs,
        tracking_root=args.tracking_root,
        tracking_uri=args.tracking_uri,
    )
    for result in results:
        action = "created" if result.created else "already indexed"
        print(
            f"{result.project_run_id}: {action}; "
            f"experiment={result.experiment_name}; "
            f"mlflow_run_id={result.mlflow_run_id}"
        )


if __name__ == "__main__":
    main()
