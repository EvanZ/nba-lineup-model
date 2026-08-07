from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.modeling import student_t_talent_lambda_sensitivity
from nba_lineup_model.modeling.student_t import (
    fit_student_t_coefficient_prior_ridge,
    fit_student_t_prior_centered_ridge,
)
from nba_lineup_model.modeling.student_t_talent_forward_rapm import (
    MODEL_NAME,
    render_student_t_talent_rankings_page,
)
from nba_lineup_model.models.baselines import PriorCenteredRidgeLineupModel


def test_student_t_irls_downweights_an_extreme_observation() -> None:
    features = sparse.csr_matrix((5, 1))
    target = np.array([0.0, 0.0, 0.0, 0.0, 20.0])
    fit = fit_student_t_prior_centered_ridge(
        features,
        target,
        np.ones(5),
        np.zeros(1),
        regularization=0.01,
    )

    assert fit.model.intercept_ < 2.0


def test_student_t_coefficient_prior_relaxes_shrinkage_for_an_extreme_player() -> None:
    features = sparse.csr_matrix([[0.0], [0.0], [0.0], [0.0], [1.0]])
    target = np.array([0.0, 0.0, 0.0, 0.0, 20.0])
    gaussian = PriorCenteredRidgeLineupModel(1.0).fit(features, target, np.ones(5), np.zeros(1))
    student_t = fit_student_t_coefficient_prior_ridge(
        features,
        target,
        np.ones(5),
        np.zeros(1),
        regularization=1.0,
        degrees_of_freedom=3.0,
        prior_scale=3.0,
    )

    assert student_t.converged
    assert student_t.model.adjustment_[0] > gaussian.adjustment_[0]


def test_student_t_talent_rankings_renderer_writes_sortable_table(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(f'{{"model": "{MODEL_NAME}"}}')
    pd.DataFrame(
        {
            "rank": [1],
            "player_name": ["Example Player"],
            "listed_position": ["G"],
            "rapm": [4.25],
            "prior_rapm": [2.0],
            "rapm_adjustment_from_prior": [2.25],
            "rapm_possessions": [1234.0],
        }
    ).to_parquet(run / "next_season_top_100_returning_rankings.parquet", index=False)

    page = render_student_t_talent_rankings_page(run, page_path=tmp_path / "rankings.md")

    text = page.read_text()
    assert "# 2026-27 Student-t Talent-Prior Rankings" in text
    assert "| 1 | +4.25 | Example Player | G | +2.00 | +2.25 | 1,234 |" in text


def test_student_t_lambda_sensitivity_renderer_writes_summary(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(
        '{"model": "'
        + student_t_talent_lambda_sensitivity.MODEL_NAME
        + '", "source_run": "source", '
        '"source_lambda": 0.03, "alternative_lambda": 0.1}'
    )
    (run / "summary.json").write_text(
        '{"player_count": 1, "pearson_correlation": 1.0, "spearman_correlation": 1.0, '
        '"mean_absolute_rating_difference": 0.5, "maximum_absolute_rating_difference": 0.5, '
        '"mean_absolute_rank_difference": 1.0, "maximum_absolute_rank_difference": 1}'
    )
    pd.DataFrame(
        {
            "player_name": ["Example Player"],
            "listed_position": ["G"],
            "rapm_possessions": [1234.0],
            "rank_lambda_0_10": [1],
            "rapm_lambda_0_10": [4.0],
            "rank_lambda_0_03": [2],
            "rapm_lambda_0_03": [3.5],
            "rapm_difference": [0.5],
            "rank_difference": [-1],
        }
    ).to_parquet(run / "ranking_comparison.parquet", index=False)

    page = student_t_talent_lambda_sensitivity.render_student_t_talent_lambda_sensitivity_page(
        run, page_path=tmp_path / "sensitivity.md"
    )

    text = page.read_text()
    assert "# Student-t Talent-Prior Lambda Sensitivity" in text
    assert "| 1 | Example Player | G | +4.00 | +3.50 | +0.50 | 2 | -1 | 1,234 |" in text
