from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.evaluation.metrics import possession_game_margin_rmse
from nba_lineup_model.modeling.leaderboard import (
    MODEL_ORDER,
    paired_game_cluster_bootstrap,
    render_evaluation_page,
    score_prediction_cohort,
)
from nba_lineup_model.modeling.neural_data import neural_possessions_frame
from nba_lineup_model.modeling.schema import (
    ArtifactRecord,
    ModelEvaluationManifest,
)


def test_possession_game_margin_rmse_aggregates_in_home_frame() -> None:
    actual = np.array([2.0, 0.0, 0.0, 3.0])
    predicted = np.array([1.0, 0.0, 0.0, 2.0])
    signs = np.array([1.0, -1.0, 1.0, -1.0])

    result = possession_game_margin_rmse(
        np.array(["g1", "g1", "g2", "g2"]),
        actual,
        predicted,
        signs,
    )

    assert result == pytest.approx(1.0)


def test_score_prediction_cohort_uses_identical_rows_for_every_model() -> None:
    possessions = _possessions()
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    predictions = {
        "ridge_rapm": np.array([1.0, 0.0, 0.0, 2.0]),
        "bayesian_rapm": np.array([1.0, 0.0, 0.0, 2.0]),
        "additive_neural": actual.copy(),
        "deep_sets": actual.copy(),
        "catboost": actual.copy(),
        "rapm_transformer": actual.copy(),
    }

    metrics, prediction_rows = score_prediction_cohort(
        possessions,
        predictions,
        cohort="playoffs",
        training_window="regular",
        mean_prediction=1.0,
    )

    assert tuple(metrics["model"]) == MODEL_ORDER
    assert len(prediction_rows) == len(possessions) * len(MODEL_ORDER)
    assert metrics.loc[
        metrics["model"].eq("additive_neural"), "possession_rmse"
    ].item() == pytest.approx(0.0)
    assert metrics.loc[
        metrics["model"].eq("additive_neural"),
        "eligible_possession_game_margin_rmse",
    ].item() == pytest.approx(0.0)
    assert metrics.loc[
        metrics["model"].eq("ridge_rapm"), "possession_rmse"
    ].item() > 0


def test_paired_game_cluster_bootstrap_preserves_paired_games() -> None:
    possessions = _possessions()
    result = paired_game_cluster_bootstrap(
        possessions,
        np.array([1.0, 0.0, 0.0, 2.0]),
        possessions["target_offense_margin"].to_numpy(dtype=float),
        cohort="playoffs",
        draws=200,
        random_seed=9,
    )

    assert len(result) == 2
    assert result["difference"].lt(0).all()
    assert result["ci_upper"].lt(0).all()
    assert result["candidate_model"].eq("deep_sets").all()
    assert result["reference_model"].eq("additive_neural").all()
    assert result["probability_candidate_better"].eq(1.0).all()


def test_generated_evaluation_page_defines_metrics_and_bolds_winner(
    tmp_path,
) -> None:
    possessions = _possessions()
    actual = possessions["target_offense_margin"].to_numpy(dtype=float)
    metrics, _ = score_prediction_cohort(
        possessions,
        {
            "ridge_rapm": np.array([1.0, 0.0, 0.0, 2.0]),
            "bayesian_rapm": np.array([1.0, 0.0, 0.0, 2.0]),
            "additive_neural": actual.copy(),
            "deep_sets": actual.copy(),
            "catboost": actual.copy(),
            "rapm_transformer": actual.copy(),
        },
        cohort="regular_holdout",
        training_window="regular",
        mean_prediction=1.0,
    )
    playoff_metrics = metrics.assign(cohort="playoffs")
    all_metrics = pd.concat([metrics, playoff_metrics], ignore_index=True)
    cohorts = pd.DataFrame(
        [
            _cohort_row("regular_holdout"),
            _cohort_row("playoffs"),
        ]
    )
    manifest = _manifest()

    path = render_evaluation_page(
        manifest,
        all_metrics,
        cohorts,
        tmp_path / "evaluation.md",
    )
    content = path.read_text()

    assert "\\operatorname{RMSE}_{poss}" in content
    assert "eligible-possession game-margin RMSE" in content
    assert "\\operatorname{Skill}_{game}" in content
    assert "Possession skill vs mean" in content
    assert "Game-margin skill vs mean" in content
    assert "| One-year additive neural | **0.000000**" in content
    assert "\f" not in content


def test_neural_possession_frame_accepts_playoff_partition() -> None:
    segments = _playoff_segments()

    result = neural_possessions_frame(segments)

    assert result["season_type"].tolist() == ["playoffs"]
    assert result["target_offense_margin"].tolist() == [2]


def _possessions() -> pd.DataFrame:
    game_time = datetime(2026, 4, 18, tzinfo=UTC)
    return pd.DataFrame(
        {
            "season": ["2025-26"] * 4,
            "season_type": ["playoffs"] * 4,
            "game_id": ["g1", "g1", "g2", "g2"],
            "game_date": [game_time.date()] * 4,
            "game_time_utc": [game_time] * 4,
            "possession_id": ["g1:0", "g1:1", "g2:0", "g2:1"],
            "possession_index": [0, 1, 0, 1],
            "period": [1] * 4,
            "offense_team_id": [100, 200, 300, 400],
            "defense_team_id": [200, 100, 400, 300],
            "home_offense_sign": [1.0, -1.0, 1.0, -1.0],
            "target_offense_margin": [2, 0, 0, 3],
            "target_home_margin": [2, 0, 0, -3],
        }
    )


def _cohort_row(cohort: str) -> dict[str, object]:
    return {
        "cohort": cohort,
        "season": "2025-26",
        "season_type": "regular" if cohort == "regular_holdout" else "playoffs",
        "game_count": 2,
        "source_possession_count": 5,
        "eligible_possession_count": 4,
        "excluded_multi_lineup_possession_count": 1,
        "eligible_fraction": 0.8,
        "first_game_date": date(2026, 4, 18),
        "last_game_date": date(2026, 4, 19),
        "training_game_count": 10,
    }


def _manifest() -> ModelEvaluationManifest:
    digest = "a" * 64
    return ModelEvaluationManifest(
        schema_version=3,
        run_id="evaluation-2025-26-test",
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        season="2025-26",
        evaluation_code_version="sha256:" + digest,
        ridge_run_id="ridge",
        ridge_manifest_sha256=digest,
        bayesian_run_id="bayesian",
        bayesian_manifest_sha256=digest,
        neural_run_id="neural",
        neural_manifest_sha256=digest,
        catboost_run_id="catboost",
        catboost_manifest_sha256=digest,
        catboost_max_iterations=100,
        catboost_best_iteration=6,
        catboost_selected_tree_count=7,
        catboost_resolved_learning_rate=0.1,
        rapm_transformer_run_id="rapm-transformer",
        rapm_transformer_manifest_sha256=digest,
        rapm_transformer_source_rapm_run_id="ridge",
        rapm_transformer_learning_rate=0.001,
        rapm_transformer_weight_decay=0.01,
        rapm_transformer_selected_epochs=3,
        rapm_transformer_leaderboard_seed=17,
        regular_segments_manifest_sha256=digest,
        regular_lineup_stints_manifest_sha256=digest,
        playoff_segments_manifest_sha256=digest,
        models=MODEL_ORDER,
        regular_holdout_game_count=2,
        regular_holdout_possession_count=4,
        playoff_game_count=2,
        playoff_possession_count=4,
        artifacts=(
            ArtifactRecord(
                filename="metrics.parquet",
                row_count=6,
                byte_count=1,
                sha256=digest,
            ),
        ),
    )


def _playoff_segments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2025-26",
                "season_type": "playoffs",
                "game_id": "g1",
                "game_date": date(2026, 4, 18),
                "game_time_utc": datetime(2026, 4, 18, tzinfo=UTC),
                "possession_id": "g1:0",
                "possession_index": 0,
                "period": 1,
                "offense_team_id": 100,
                "defense_team_id": 200,
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [6, 7, 8, 9, 10],
                "points_home": 2,
                "points_away": 0,
                "offense_points": 2,
                "catalog_home_team_id": 100,
                "catalog_away_team_id": 200,
                "catalog_home_team_tricode": "HOM",
                "catalog_away_team_tricode": "AWY",
                "quality_status": "pass",
                "source_build_run_id": "run",
                "processing_code_version": "sha256:" + "a" * 64,
                "play_by_play_sha256": "b" * 64,
                "boxscore_sha256": "c" * 64,
            }
        ]
    )
