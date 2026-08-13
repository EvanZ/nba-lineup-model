from __future__ import annotations

import pandas as pd

import nba_lineup_model.modeling.forward_box_score_hpm as box_hpm
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN, ForwardLaggedRapmSeason


def test_residual_regularization_uses_only_earlier_target_seasons() -> None:
    selection = box_hpm.select_box_score_residual_regularization(
        _residual_rows(), regularization_grid=(0.01, 0.1)
    )

    assert selection.training_target_seasons == ("2020-21", "2021-22", "2022-23")
    assert selection.validation_fold_count == 2
    assert selection.summary["selected"].sum() == 1


def test_completed_residual_rows_subtract_completed_hpm_prior() -> None:
    results = [_result("2020-21", 2.0), _result("2021-22", 3.0), _result("2022-23", 4.0)]
    exposures = [
        pd.DataFrame({"player_id": [1, 2], "on_court_possessions": [100.0, 200.0]})
        for _ in results
    ]
    priors = {
        season: pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [1.0, 1.5]})
        for season in ("2020-21", "2021-22", "2022-23")
    }

    actual = box_hpm._completed_residual_rows(
        _residual_rows(),
        results=results,
        exposures=exposures,
        base_priors_by_season=priors,
    )

    assert actual["target_rapm"].tolist() == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def test_wrapper_uses_box_score_builder_and_bounded_context(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"season": ["2020-21"], "player_id": [1]}).to_parquet(panel_path, index=False)
    features = _residual_rows()

    monkeypatch.setattr(
        box_hpm,
        "build_box_score_prior_features",
        lambda _: (features, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        box_hpm,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    box_hpm.train_forward_box_score_residual_hpm(player_season_panel_path=panel_path)

    assert captured["model_name"] == box_hpm.MODEL_NAME
    assert isinstance(captured["player_prior_builder"], box_hpm.ForwardBoxScoreResidualPriorBuilder)


def test_interaction_features_are_exact_lagged_products() -> None:
    frame = _residual_rows().iloc[:1].copy()
    frame.loc[:, "prior_fga_per_100_on_court_possessions"] = 20.0
    frame.loc[:, "prior_fta_per_100_on_court_possessions"] = 5.0
    frame.loc[:, "prior_turnovers_per_100_on_court_possessions"] = 4.0
    frame.loc[:, "prior_assists_per_100_on_court_possessions"] = 8.0
    frame.loc[:, "prior_three_pa_per_100_on_court_possessions"] = 10.0
    frame.loc[:, "prior_stabilized_effective_field_goal_percentage"] = 0.6
    frame.loc[:, "prior_offensive_rebounds_per_100_on_court_possessions"] = 3.0
    frame.loc[:, "prior_defensive_rebounds_per_100_on_court_possessions"] = 9.0
    frame.loc[:, "prior_steals_per_100_on_court_possessions"] = 2.0
    frame.loc[:, "prior_blocks_per_100_on_court_possessions"] = 1.5

    actual = box_hpm.add_box_score_interaction_features(frame).iloc[0]

    usage = 20.0 + 0.44 * 5.0 + 4.0
    assert actual["prior_usage_assists_interaction"] == usage * 8.0
    assert actual["prior_usage_turnovers_interaction"] == usage * 4.0
    assert actual["prior_three_point_volume_efficiency_interaction"] == 6.0
    assert actual["prior_field_goal_free_throw_interaction"] == 100.0
    assert actual["prior_offensive_defensive_rebounds_interaction"] == 27.0
    assert actual["prior_steals_blocks_interaction"] == 3.0


def test_interaction_wrapper_uses_augmented_feature_set(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"season": ["2020-21"], "player_id": [1]}).to_parquet(panel_path, index=False)

    monkeypatch.setattr(
        box_hpm,
        "build_box_score_prior_features",
        lambda _: (_residual_rows(), pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        box_hpm,
        "train_forward_portable_matchup_contextual_rapm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    box_hpm.train_forward_box_score_interaction_hpm(player_season_panel_path=panel_path)

    builder = captured["player_prior_builder"]
    assert captured["model_name"] == "forward_box_score_interaction_hpm"
    assert isinstance(builder, box_hpm.ForwardBoxScoreResidualPriorBuilder)
    assert builder.feature_columns == box_hpm.BOX_SCORE_INTERACTION_RESIDUAL_FEATURE_COLUMNS


def _result(season: str, start: float) -> ForwardLaggedRapmSeason:
    return ForwardLaggedRapmSeason(
        season=season,
        selected_lambda=0.1,
        cv_results=pd.DataFrame(),
        player_estimates=pd.DataFrame(
            {"season": [season, season], "player_id": [1, 2], "rapm": [start, start + 1.0]}
        ),
        player_priors=pd.DataFrame({"player_id": [1, 2], PRIOR_MEAN_COLUMN: [0.0, 0.0]}),
    )


def _residual_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_index, season in enumerate(("2020-21", "2021-22", "2022-23")):
        for player_id in (1, 2):
            row: dict[str, object] = {
                "target_season": season,
                "prior_source_season": "2019-20",
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "target_rapm": 0.0,
                "target_rapm_possessions": 100.0 + player_id,
                "prior_exposure_cohort": "established",
                "has_prior_season": True,
                "prior_rapm_available": True,
                "prior_boxscore_features_available": True,
                "prior_rapm": 0.0,
            }
            for feature_index, column in enumerate(box_hpm.BOX_SCORE_RESIDUAL_FEATURE_COLUMNS):
                row[column] = float(season_index + player_id + feature_index / 10.0)
            rows.append(row)
    return pd.DataFrame(rows)
