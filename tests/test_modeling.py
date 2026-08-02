from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.evaluation.metrics import (
    game_margin_rmse,
    mean_squared_error,
    skill_score,
)
from nba_lineup_model.modeling.schema import ChronologicalSplitConfig
from nba_lineup_model.modeling.stints import (
    build_rapm_stints_from_curated_games,
    rapm_stints_frame,
)
from nba_lineup_model.modeling.train import (
    chronological_game_splits,
    fit_baseline_experiment,
)
from nba_lineup_model.models.baselines import (
    FittedMeanModel,
    RidgeLineupModel,
    signed_entity_matrix,
    vocabulary_mapping,
)
from nba_lineup_model.season.layout import CuratedDatasetLayout, CuratedPartition


def test_rapm_stints_conserve_points_and_split_possession_exposure() -> None:
    stints = _source_stints()
    segments = _source_segments()

    result = rapm_stints_frame(stints, segments)

    assert len(result) == 2
    assert result["points_home"].sum() == 2
    assert result["points_away"].sum() == 2
    assert result["home_offensive_possessions"].sum() == pytest.approx(1.0)
    assert result["away_offensive_possessions"].sum() == pytest.approx(1.0)
    assert result["possessions"].tolist() == pytest.approx([0.25, 0.75])
    assert result["target_home_net_rating"].tolist() == pytest.approx([400.0, -400.0 / 3.0])


def test_curated_stint_builder_excludes_whole_nonconserving_game(
    tmp_path: Path,
) -> None:
    """A historical score mismatch removes the game, never one malformed stint."""

    curated = tmp_path / "curated"
    layout = CuratedDatasetLayout(curated)
    lineup_dir = layout.partition_dir(
        CuratedPartition(table="lineup_stints", season="2025-26", season_type="regular")
    )
    segment_dir = layout.partition_dir(
        CuratedPartition(
            table="possession_segments",
            season="2025-26",
            season_type="regular",
        )
    )
    lineup_dir.mkdir(parents=True)
    segment_dir.mkdir(parents=True)
    bad_stints = _source_stints()
    bad_segments = _source_segments()
    bad_stints.loc[0, "points_home"] = 0
    good_stints = _source_stints().assign(game_id="0022500002")
    good_segments = _source_segments().assign(game_id="0022500002")
    pd.concat([bad_stints, good_stints], ignore_index=True).to_parquet(
        lineup_dir / "part-00000.parquet",
        index=False,
    )
    pd.concat([bad_segments, good_segments], ignore_index=True).to_parquet(
        segment_dir / "part-00000.parquet",
        index=False,
    )

    result, excluded = build_rapm_stints_from_curated_games(
        "2025-26",
        curated_dir=curated,
    )

    assert excluded == ("0022500001",)
    assert set(result["game_id"]) == {"0022500002"}


def test_signed_sparse_player_encoding() -> None:
    frame = pd.DataFrame(
        {
            "home": [[10, 11], [11, 12]],
            "away": [[12, 13], [10, 13]],
        }
    )
    mapping = vocabulary_mapping((10, 11, 12, 13))

    matrix = signed_entity_matrix(
        frame,
        "home",
        "away",
        mapping,
        multiple=True,
    )

    assert matrix.shape == (2, 4)
    assert matrix.nnz == 8
    assert matrix.toarray().tolist() == [
        [1.0, 1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0],
    ]


def test_weighted_mean_and_sparse_ridge_models() -> None:
    features = pd.DataFrame(
        {
            "home": [1, 1, 2, 2],
            "away": [2, 2, 1, 1],
        }
    )
    mapping = vocabulary_mapping((1, 2))
    matrix = signed_entity_matrix(
        features,
        "home",
        "away",
        mapping,
        multiple=False,
    )
    target = np.array([10.0, 12.0, -10.0, -12.0])
    weights = np.ones(4)

    mean = FittedMeanModel.fit(target, weights)
    ridge = RidgeLineupModel(0.0001).fit(matrix, target, weights)

    assert mean.mean == pytest.approx(0.0)
    assert ridge.predict(matrix)[0] > 9.0
    assert ridge.coef_[0] > ridge.coef_[1]
    assert ridge.sklearn_alpha == pytest.approx(0.0004)


def test_chronological_splits_are_grouped_and_ordered() -> None:
    stints = _modeling_stints(game_count=20)
    config = ChronologicalSplitConfig(
        cv_folds=2,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    plan = chronological_game_splits(stints, config)

    assert len(plan.folds) == 2
    assert len(plan.final_test_game_ids) == 3
    for fold in plan.folds:
        assert set(fold.train_game_ids).isdisjoint(fold.validation_game_ids)
        assert max(fold.train_game_ids) < min(fold.validation_game_ids)
    assert set(plan.final_train_game_ids).isdisjoint(plan.final_test_game_ids)
    assert max(plan.final_train_game_ids) < min(plan.final_test_game_ids)


def test_chronological_splits_keep_game_dates_together() -> None:
    stints = _modeling_stints(game_count=20)
    stints.loc[stints["game_id"].isin(["0000000016", "0000000017"]), "game_date"] = date(
        2025,
        10,
        17,
    )
    config = ChronologicalSplitConfig(
        cv_folds=2,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    plan = chronological_game_splits(stints, config)
    game_dates = stints.drop_duplicates("game_id").set_index("game_id")["game_date"]

    assert set(game_dates.loc[list(plan.final_train_game_ids)]).isdisjoint(
        set(game_dates.loc[list(plan.final_test_game_ids)])
    )


def test_baseline_experiment_produces_rankings_and_test_comparisons() -> None:
    stints = _modeling_stints(game_count=20)
    config = ChronologicalSplitConfig(
        cv_folds=2,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    experiment = fit_baseline_experiment(
        stints,
        lambda_grid=(0.0001, 0.01, 1.0),
        split_config=config,
        minimum_ranking_possessions=1.0,
    )

    assert set(experiment.test_metrics["model"]) == {"mean", "team", "rapm"}
    assert experiment.selected_team_lambda in {0.0001, 0.01, 1.0}
    assert experiment.selected_rapm_lambda in {0.0001, 0.01, 1.0}
    assert len(experiment.player_rankings) == 20
    assert len(experiment.team_ratings) == 4
    assert experiment.player_rankings["rank"].tolist() == list(range(1, 21))
    assert experiment.test_predictions["game_id"].nunique() == 3
    assert len(experiment.cv_results) == 14


def test_weighted_metrics_and_game_aggregation() -> None:
    actual = np.array([100.0, -100.0])
    predicted = np.array([50.0, -50.0])
    possessions = np.array([2.0, 2.0])

    assert mean_squared_error(actual, predicted, possessions) == pytest.approx(2500.0)
    assert game_margin_rmse(
        np.array(["a", "a"]),
        actual,
        predicted,
        possessions,
    ) == pytest.approx(0.0)
    assert skill_score(75.0, 100.0) == pytest.approx(0.25)


def _source_stints() -> pd.DataFrame:
    shared = {
        "season": "2025-26",
        "season_type": "regular",
        "game_id": "0022500001",
        "game_date": date(2025, 10, 21),
        "game_time_utc": datetime(2025, 10, 21, 23, 30, tzinfo=UTC),
        "period": 1,
        "catalog_home_team_id": 100,
        "catalog_away_team_id": 200,
        "catalog_home_team_tricode": "HOM",
        "catalog_away_team_tricode": "AWY",
        "quality_status": "pass",
        "quality_issue_codes_json": "[]",
        "source_build_run_id": "run",
        "processing_code_version": "sha256:" + "a" * 64,
        "play_by_play_sha256": "b" * 64,
        "boxscore_sha256": "c" * 64,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "stint_index": 0,
                "start_event_index": 0,
                "end_event_index": 4,
                "start_elapsed_game_seconds": 0.0,
                "end_elapsed_game_seconds": 20.0,
                "duration_seconds": 20.0,
                "points_home": 1,
                "points_away": 0,
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [6, 7, 8, 9, 10],
            },
            {
                **shared,
                "stint_index": 1,
                "start_event_index": 5,
                "end_event_index": 10,
                "start_elapsed_game_seconds": 20.0,
                "end_elapsed_game_seconds": 40.0,
                "duration_seconds": 20.0,
                "points_home": 1,
                "points_away": 2,
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [6, 7, 8, 9, 11],
            },
        ]
    )


def _source_segments() -> pd.DataFrame:
    shared = {
        "game_id": "0022500001",
        "catalog_home_team_id": 100,
        "catalog_away_team_id": 200,
        "home_player_ids": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "possession_id": "0022500001:0000",
                "start_event_index": 1,
                "end_event_index": 4,
                "offense_team_id": 100,
                "away_player_ids": [6, 7, 8, 9, 10],
                "points_home": 1,
                "points_away": 0,
            },
            {
                **shared,
                "possession_id": "0022500001:0000",
                "start_event_index": 5,
                "end_event_index": 6,
                "offense_team_id": 100,
                "away_player_ids": [6, 7, 8, 9, 11],
                "points_home": 1,
                "points_away": 0,
            },
            {
                **shared,
                "possession_id": "0022500001:0001",
                "start_event_index": 7,
                "end_event_index": 10,
                "offense_team_id": 200,
                "away_player_ids": [6, 7, 8, 9, 11],
                "points_home": 0,
                "points_away": 2,
            },
        ]
    )


def _modeling_stints(game_count: int) -> pd.DataFrame:
    start = datetime(2025, 10, 1, tzinfo=UTC)
    team_players = {
        100: [1, 2, 3, 4, 5],
        200: [6, 7, 8, 9, 10],
        300: [11, 12, 13, 14, 15],
        400: [16, 17, 18, 19, 20],
    }
    team_codes = {100: "AAA", 200: "BBB", 300: "CCC", 400: "DDD"}
    rows = []
    for game_index in range(game_count):
        home_team = (100, 200, 300, 400)[game_index % 4]
        away_team = (200, 300, 400, 100)[game_index % 4]
        home_strength = home_team / 100.0
        away_strength = away_team / 100.0
        possessions = 10.0
        margin = int(round(home_strength - away_strength))
        rows.append(
            {
                "game_id": f"{game_index:010d}",
                "game_date": (start + timedelta(days=game_index)).date(),
                "game_time_utc": start + timedelta(days=game_index),
                "stint_index": 0,
                "home_team_id": home_team,
                "away_team_id": away_team,
                "home_team_tricode": team_codes[home_team],
                "away_team_tricode": team_codes[away_team],
                "home_player_ids": team_players[home_team],
                "away_player_ids": team_players[away_team],
                "duration_seconds": 300.0,
                "home_margin": margin,
                "possessions": possessions,
                "target_home_net_rating": 100.0 * margin / possessions,
            }
        )
    return pd.DataFrame(rows)
