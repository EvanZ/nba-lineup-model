from __future__ import annotations

import pandas as pd

from nba_lineup_model.modeling.forward_nail_context_regularization import (
    model_name_for_raw_context_alpha,
)
from nba_lineup_model.modeling.forward_nail_profile_padding import PUBLISHED_MODEL_NAME
from nba_lineup_model.modeling.nail_fixed_context_regularization_study import (
    _selection_summary,
    candidate_model_name,
)


def test_published_alpha_uses_incumbent_artifact_name() -> None:
    assert candidate_model_name(10_000.0) == PUBLISHED_MODEL_NAME
    assert candidate_model_name(5_000.0) == model_name_for_raw_context_alpha(5_000.0)


def test_fixed_alpha_selection_uses_paired_one_se_rule() -> None:
    rows: list[dict[str, object]] = []
    seasons = ("2020-21", "2021-22", "2022-23")
    for model, raw_alpha, mse_values in (
        ("minimum", 5_000.0, (99.0, 100.0, 101.0)),
        ("stronger", 10_000.0, (98.5, 100.5, 102.5)),
        ("too_strong", 20_000.0, (101.0, 102.0, 103.0)),
    ):
        for season, mse in zip(seasons, mse_values, strict=True):
            rows.append(
                {
                    "model": model,
                    "label": model,
                    "season": season,
                    "raw_context_alpha": raw_alpha,
                    "full_game_margin_mse": mse,
                    "game_winner_accuracy": 0.5,
                }
            )

    summary = _selection_summary(pd.DataFrame(rows)).set_index("model")

    assert bool(summary.loc["minimum", "exact_mse_minimum"])
    assert bool(summary.loc["stronger", "within_one_standard_error"])
    assert bool(summary.loc["stronger", "selected_by_one_se_rule"])
    assert not bool(summary.loc["too_strong", "within_one_standard_error"])
