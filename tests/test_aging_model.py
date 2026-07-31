from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient

from nba_lineup_model.modeling.aging import (
    expanding_target_season_folds,
    run_aging_experiment,
    train_forward_aging_model,
    validate_aging_model_run,
)
from nba_lineup_model.modeling.player_history import (
    PlayerSeasonPanelManifest,
    PlayerSeasonPanelSource,
)
from nba_lineup_model.modeling.schema import ArtifactRecord
from nba_lineup_model.tracking import track_immutable_run


def test_expanding_aging_folds_are_strictly_forward():
    transitions = synthetic_transitions()

    training_seasons, holdout, folds = expanding_target_season_folds(transitions)

    assert training_seasons == ("2020-21", "2021-22", "2022-23")
    assert holdout == "2023-24"
    assert [(fold.train_target_seasons, fold.validation_target_season) for fold in folds] == [
        (("2020-21",), "2021-22"),
        (("2020-21", "2021-22"), "2022-23"),
    ]


def test_holdout_outcomes_cannot_change_aging_priors():
    transitions = synthetic_transitions()
    original = run_aging_experiment(
        transitions,
        regularization_grid=(0.001, 0.1, 1.0),
        age_spline_knots=4,
    )
    perturbed = transitions.copy()
    holdout = perturbed["target_season"].eq("2023-24")
    perturbed.loc[holdout, "target_rapm"] += 100.0
    changed = run_aging_experiment(
        perturbed,
        regularization_grid=(0.001, 0.1, 1.0),
        age_spline_knots=4,
    )

    assert changed.selected_regularization == original.selected_regularization
    assert changed.player_priors["aging_prior_mean"].tolist() == (
        original.player_priors["aging_prior_mean"].tolist()
    )
    assert changed.feature_coefficients["coefficient"].tolist() == (
        original.feature_coefficients["coefficient"].tolist()
    )
    assert not changed.holdout_predictions["target_rapm"].equals(
        original.holdout_predictions["target_rapm"]
    )


def test_aging_priors_preserve_cold_starts_and_exclude_target_outcomes():
    experiment = run_aging_experiment(
        synthetic_transitions(),
        regularization_grid=(0.001, 0.1),
        age_spline_knots=4,
    )

    priors = experiment.player_priors
    forbidden = {
        "target_rapm",
        "target_rapm_possessions",
        "target_rapm_seconds",
        "target_rapm_exposure_eligible",
    }
    assert not forbidden & set(priors)
    assert priors["aging_prior_mean"].map(np.isfinite).all()
    assert priors["aging_prior_error_scale"].gt(0).all()
    cold = priors.loc[~priors["has_prior_season"].astype(bool)]
    assert not cold.empty
    assert cold["prior_rapm"].isna().all()
    all_metrics = experiment.holdout_metrics.loc[
        experiment.holdout_metrics["cohort"].eq("all")
    ].set_index("model")
    assert (
        all_metrics.loc["aging", "weighted_rmse"] < all_metrics.loc["persistence", "weighted_rmse"]
    )


def test_aging_run_round_trips_with_immutable_artifacts(tmp_path: Path):
    panel_dir = write_panel_fixture(tmp_path, synthetic_transitions())

    manifest, run_dir = train_forward_aging_model(
        panel_dir=panel_dir,
        artifacts_dir=tmp_path / "artifacts" / "models",
        regularization_grid=(0.001, 0.1),
        age_spline_knots=4,
    )

    assert manifest.season == "2023-24"
    assert manifest.training_target_seasons == (
        "2020-21",
        "2021-22",
        "2022-23",
    )
    assert manifest.holdout_player_count == 24
    assert validate_aging_model_run(run_dir) == manifest
    assert (run_dir / "model.joblib").is_file()
    assert (run_dir.parent / "latest.json").is_file()

    tracking_root = tmp_path / "mlflow"
    tracking = track_immutable_run(run_dir, tracking_root=tracking_root)
    client = MlflowClient(tracking_uri=f"sqlite:///{tracking_root.resolve() / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("nba-lineup-model-2023-24-models")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    primary = next(run for run in runs if run.data.tags.get("project.run_role") == "primary")
    children = [
        run for run in runs if run.data.tags.get("project.run_role") == "hyperparameter_candidate"
    ]
    assert tracking.created is True
    assert primary.data.tags["project.run_kind"] == "forward_aging"
    assert len(children) == 2


def synthetic_transitions() -> pd.DataFrame:
    target_seasons = ("2020-21", "2021-22", "2022-23", "2023-24")
    rows: list[dict[str, object]] = []
    for season_index, target_season in enumerate(target_seasons):
        target_year = int(target_season[:4])
        prior_season = f"{target_year - 1}-{str(target_year)[-2:]}"
        for player_id in range(1, 25):
            cold_start = (player_id + season_index) % 7 == 0
            age = 20.0 + player_id % 13 + season_index
            prior_rapm = np.nan if cold_start else -2.5 + player_id * 0.22 + season_index * 0.08
            age_change = 0.18 * (27.0 - age) - 0.015 * (age - 27.0) ** 2
            target_rapm = (
                (0.0 if cold_start else 0.78 * float(prior_rapm))
                + age_change
                + ((player_id % 5) - 2) * 0.04
            )
            target_possessions = float(250 + player_id * 20 + season_index * 15)
            rows.append(
                {
                    "schema_version": 1,
                    "target_season": target_season,
                    "prior_season": prior_season,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "has_prior_season": not cold_start,
                    "target_age": age,
                    "target_nba_experience_years": (0 if cold_start else max(1, int(age - 21))),
                    "listed_position": "G" if player_id % 2 else "F",
                    "height_inches": 74 + player_id % 10,
                    "weight_pounds": 180 + player_id * 2,
                    "draft_year": target_year if cold_start else target_year - 3,
                    "draft_round": 1,
                    "draft_number": player_id,
                    "is_undrafted": False,
                    "is_rookie": cold_start,
                    "target_rapm": target_rapm,
                    "target_rapm_possessions": target_possessions,
                    "target_rapm_seconds": target_possessions * 14.0,
                    "target_rapm_exposure_eligible": target_possessions >= 500,
                    "prior_age": age - 1,
                    "prior_nba_experience_years": (np.nan if cold_start else max(0, int(age - 22))),
                    "prior_rapm": prior_rapm,
                    "prior_rapm_possessions": (np.nan if cold_start else target_possessions - 30),
                }
            )
    return pd.DataFrame(rows)


def write_panel_fixture(
    tmp_path: Path,
    transitions: pd.DataFrame,
) -> Path:
    panel_dir = tmp_path / "analytical" / "player_season_panel"
    panel_dir.mkdir(parents=True)
    seasons = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24")
    panel = pd.DataFrame(
        [
            {
                "season": season,
                "player_id": player_id,
            }
            for season in seasons
            for player_id in range(1, 25)
        ]
    )
    outputs = {
        "player_seasons.parquet": panel,
        "transitions.parquet": transitions,
    }
    for filename, frame in outputs.items():
        frame.to_parquet(panel_dir / filename, index=False)
    artifacts = tuple(
        ArtifactRecord(
            filename=filename,
            row_count=len(frame),
            byte_count=(panel_dir / filename).stat().st_size,
            sha256=sha256_file(panel_dir / filename),
        )
        for filename, frame in outputs.items()
    )
    source_hash = "0" * 64
    sources = tuple(
        PlayerSeasonPanelSource(
            season=season,
            rapm_run_id=f"rapm-{season}",
            rapm_manifest_sha256=source_hash,
            curated_players_manifest_sha256=source_hash,
            player_bios_manifest_sha256=source_hash,
            player_count=24,
            curated_player_row_count=24,
        )
        for season in seasons
    )
    manifest = PlayerSeasonPanelManifest(
        created_at=datetime.now(UTC),
        builder_code_version=f"sha256:{source_hash}",
        seasons=seasons,
        source_count=len(sources),
        player_season_row_count=len(panel),
        transition_row_count=len(transitions),
        cold_start_transition_count=int((~transitions["has_prior_season"].astype(bool)).sum()),
        prior_feature_columns=("rapm", "rapm_possessions"),
        sources=sources,
        artifacts=artifacts,
    )
    (panel_dir / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return panel_dir


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
