from __future__ import annotations

import json

import pandas as pd

from nba_lineup_model.modeling.forward_nail_context_regularization import (
    context_lambda_slug,
    model_name_for_context_lambda,
    model_name_for_raw_context_alpha,
    raw_context_alpha_slug,
)
from nba_lineup_model.modeling.nail_context_regularization_study import (
    MODEL_NAME,
    _latest_compatible_metrics,
    _selection_summary,
)


def test_context_lambda_slug_is_stable() -> None:
    assert context_lambda_slug(0.00275) == "0p00275"
    assert model_name_for_context_lambda(0.00275).endswith("0p00275")
    assert raw_context_alpha_slug(5_000.0) == "5000"
    assert model_name_for_raw_context_alpha(5_000.0).endswith("5000")


def test_one_standard_error_rule_selects_strongest_eligible_penalty() -> None:
    rows: list[dict[str, object]] = []
    for model, context_lambda, mse_values in (
        ("minimum", 0.01, (99.0, 100.0, 101.0)),
        ("stronger", 0.02, (98.5, 100.5, 102.5)),
        ("too_strong", 0.04, (101.0, 102.0, 103.0)),
    ):
        for season, mse in zip(("2020-21", "2021-22", "2022-23"), mse_values, strict=True):
            rows.append(
                {
                    "model": model,
                    "label": model,
                    "season": season,
                    "regularization_contract": "mean_weighted_loss",
                    "context_lambda": context_lambda,
                    "game_count": 1,
                    "full_game_margin_mse": mse,
                    "full_game_margin_rmse": mse**0.5,
                    "full_game_margin_mae": mse**0.5,
                    "game_winner_accuracy": 0.5,
                }
            )

    summary = _selection_summary(pd.DataFrame(rows)).set_index("model")

    assert bool(summary.loc["minimum", "exact_mse_minimum"])
    assert bool(summary.loc["stronger", "within_one_standard_error"])
    assert bool(summary.loc["stronger", "selected_by_one_se_rule"])
    assert not bool(summary.loc["too_strong", "within_one_standard_error"])
    assert summary.loc["stronger", "paired_mse_delta"] == 0.5
    assert summary.loc["stronger", "paired_mse_delta_standard_error"] > 0.5


def test_raw_alpha_control_is_not_selection_eligible() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["raw", "raw", "normalized", "normalized"],
            "label": ["raw", "raw", "normalized", "normalized"],
            "season": ["2021-22", "2022-23", "2021-22", "2022-23"],
            "regularization_contract": [
                "fixed_raw_alpha",
                "fixed_raw_alpha",
                "mean_weighted_loss",
                "mean_weighted_loss",
            ],
            "context_lambda": [10.0, 10.0, 1.0, 1.0],
            "full_game_margin_mse": [1.0, 1.0, 2.0, 2.0],
            "full_game_margin_mae": [1.0, 1.0, 2.0**0.5, 2.0**0.5],
            "game_winner_accuracy": [1.0, 1.0, 0.0, 0.0],
        }
    )

    summary = _selection_summary(metrics).set_index("model")

    assert not bool(summary.loc["raw", "exact_mse_minimum"])
    assert not bool(summary.loc["raw", "within_one_standard_error"])
    assert not bool(summary.loc["raw", "selected_by_one_se_rule"])
    assert bool(summary.loc["normalized", "selected_by_one_se_rule"])


def test_latest_compatible_metrics_reuses_only_matching_validation_range(tmp_path) -> None:
    seasons = ("2020-21", "2021-22")
    run_id = "study-run"
    root = tmp_path / MODEL_NAME / "2020-21_to_2021-22"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    expected = pd.DataFrame({"model": ["candidate"], "season": ["2020-21"]})
    expected.to_parquet(run_dir / "season_metrics.parquet", index=False)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "validation_seasons": list(seasons),
                "replay_run_dirs": ["replay-a", "replay-b"],
            }
        )
    )
    (root / "latest.json").write_text(json.dumps({"run_id": run_id}))

    actual, replay_dirs = _latest_compatible_metrics(
        tmp_path,
        validation_seasons=seasons,
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert replay_dirs == ("replay-a", "replay-b")
    mismatched, _ = _latest_compatible_metrics(
        tmp_path,
        validation_seasons=("2020-21", "2022-23"),
    )
    assert mismatched.empty
