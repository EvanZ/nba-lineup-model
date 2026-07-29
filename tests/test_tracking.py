from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from nba_lineup_model.tracking import (
    discover_all_runs,
    discover_latest_runs,
    track_immutable_run,
)


def test_track_immutable_run_logs_and_deduplicates(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    tracking_root = tmp_path / "mlflow"

    first = track_immutable_run(run_dir, tracking_root=tracking_root)
    second = track_immutable_run(run_dir, tracking_root=tracking_root)

    assert first.created is True
    assert second.created is False
    assert second.mlflow_run_id == first.mlflow_run_id
    client = MlflowClient(
        tracking_uri=f"sqlite:///{tracking_root.resolve() / 'mlflow.db'}"
    )
    experiment = client.get_experiment_by_name(
        "nba-lineup-model-2025-26-models"
    )
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    primary = [
        run
        for run in runs
        if run.data.tags.get("project.run_role") == "primary"
    ]
    children = [
        run
        for run in runs
        if run.data.tags.get("project.run_role") == "hyperparameter_candidate"
    ]
    assert len(primary) == 1
    assert len(children) == 2
    run = primary[0]
    assert run.info.status == "FINISHED"
    assert run.data.tags["project.run_kind"] == "catboost"
    assert run.data.params["manifest.iterations"] == "1000"
    assert run.data.params["resolved_parameters.depth"] == "6"
    assert run.data.metrics["test_metrics.model_catboost.rmse"] == pytest.approx(
        1.19
    )
    assert run.data.metrics["selection.weighted_validation_mse"] == pytest.approx(
        1.41
    )
    artifacts = {
        artifact.path
        for artifact in client.list_artifacts(run.info.run_id, "immutable_run")
    }
    assert "immutable_run/manifest.json" in artifacts
    assert "immutable_run/test_metrics.parquet" in artifacts

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["iterations"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different immutable manifest hash"):
        track_immutable_run(run_dir, tracking_root=tracking_root)


def test_discover_runs_resolves_latest_and_all(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    latest_path = run_dir.parent / "latest.json"
    latest_path.write_text(json.dumps({"run_id": run_dir.name}))

    assert discover_latest_runs(tmp_path / "artifacts") == (run_dir.resolve(),)
    assert discover_all_runs(tmp_path / "artifacts") == (run_dir.resolve(),)
    assert discover_latest_runs(
        tmp_path / "artifacts",
        season="2024-25",
    ) == ()


def _write_run(tmp_path: Path) -> Path:
    run_id = "catboost-2025-26-20260729T120000Z-1234abcd"
    run_dir = (
        tmp_path
        / "artifacts"
        / "models"
        / "catboost"
        / "2025-26"
        / run_id
    )
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": "2026-07-29T12:00:00Z",
        "season": "2025-26",
        "architecture": "categorical_player_state",
        "iterations": 1000,
        "one_hot_max_size": 3,
        "split_config": {
            "cv_folds": 3,
            "validation_fraction": 0.1,
            "test_fraction": 0.15,
        },
        "artifacts": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "resolved_parameters.json").write_text(
        json.dumps({"depth": 6, "learning_rate": 0.08})
    )
    pd.DataFrame(
        [
            {
                "model": "catboost",
                "test_game_count": 186,
                "rmse": 1.19,
                "mae": 1.14,
            }
        ]
    ).to_parquet(run_dir / "test_metrics.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_index": 1,
                "learning_rate": 0.08,
                "weight_decay": 0.0,
                "fold_count": 3,
                "validation_possession_count": 100,
                "weighted_validation_mse": 1.41,
                "rank": 1,
                "selected": True,
            },
            {
                "candidate_index": 2,
                "learning_rate": 0.03,
                "weight_decay": 0.0,
                "fold_count": 3,
                "validation_possession_count": 100,
                "weighted_validation_mse": 1.42,
                "rank": 2,
                "selected": False,
            },
        ]
    ).to_parquet(run_dir / "hyperparameter_summary.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_index": candidate,
                "fold": fold,
                "validation_mse": 1.4 + candidate / 100 + fold / 1_000,
                "selected_epochs": 100 + fold,
            }
            for candidate in (1, 2)
            for fold in range(3)
        ]
    ).to_parquet(run_dir / "hyperparameter_trials.parquet", index=False)
    return run_dir
