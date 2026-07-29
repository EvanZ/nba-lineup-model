from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.modeling.bayesian_case_study import (
    BayesianCaseStudySource,
    _write_interval_chart,
    _write_rank_probability_chart,
    prepare_case_study_players,
    render_bayesian_case_study_markdown,
)


def test_prepare_case_study_players_compares_matching_uncertainty() -> None:
    posterior = _posterior_rankings()
    bootstrap = _bootstrap_summary()

    eligible, top = prepare_case_study_players(
        posterior,
        bootstrap,
        top_n=3,
    )

    assert len(eligible) == 3
    assert top["player_name"].tolist() == ["Alpha", "Beta", "Gamma"]
    assert top["posterior_interval_width_90"].tolist() == [4.0, 4.0, 4.0]
    assert top["bootstrap_interval_width_90"].tolist() == [2.0, 2.0, 2.0]
    assert top["posterior_to_bootstrap_width_ratio"].tolist() == [2.0, 2.0, 2.0]
    assert top["top_25_probability_gap"].tolist() == pytest.approx(
        [-0.10, -0.15, -0.20]
    )


def test_bayesian_case_study_markdown_records_results_and_provenance() -> None:
    eligible, top = prepare_case_study_players(
        _posterior_rankings(),
        _bootstrap_summary(),
        top_n=3,
    )
    source = BayesianCaseStudySource(
        season="2025-26",
        bayesian_run_id="bayesian-run",
        source_model_run_id="ridge-run",
        diagnostics_run_id="diagnostics-run",
        bayesian_manifest_sha256="a" * 64,
        diagnostics_manifest_sha256="b" * 64,
        generator_code_sha256="c" * 64,
        selected_lambda=0.03,
        posterior_draws=4000,
        minimum_ranking_possessions=500.0,
        player_count=582,
        eligible_player_count=3,
        game_count=1230,
        stint_count=39918,
    )

    markdown = render_bayesian_case_study_markdown(
        source,
        eligible,
        top,
        _comparison(),
        _calibration(),
        interval_reference="../assets/intervals.svg",
        rank_reference="../assets/ranks.svg",
    )

    assert "# What Bayesian RAPM Adds to the Same Ridge Ranking" in markdown
    assert "bayesian-run" in markdown
    assert "ridge-run" in markdown
    assert "diagnostics-run" in markdown
    assert "Alpha" in markdown
    assert "2.00 times" in markdown
    assert "../assets/intervals.svg" in markdown
    assert "../assets/ranks.svg" in markdown


def test_bayesian_case_study_charts_are_deterministic_svg(tmp_path: Path) -> None:
    _, top = prepare_case_study_players(
        _posterior_rankings(),
        _bootstrap_summary(),
        top_n=3,
    )
    first_intervals = tmp_path / "intervals-1.svg"
    second_intervals = tmp_path / "intervals-2.svg"
    first_ranks = tmp_path / "ranks-1.svg"
    second_ranks = tmp_path / "ranks-2.svg"

    _write_interval_chart(top, first_intervals)
    _write_interval_chart(top, second_intervals)
    _write_rank_probability_chart(top, first_ranks)
    _write_rank_probability_chart(top, second_ranks)

    assert first_intervals.read_bytes() == second_intervals.read_bytes()
    assert first_ranks.read_bytes() == second_ranks.read_bytes()
    assert b"<svg" in first_intervals.read_bytes()
    assert b"<svg" in first_ranks.read_bytes()


def _posterior_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Alpha", "Beta", "Gamma"],
            "eligible_rank": [1, 2, 3],
            "exposure_eligible": [True, True, True],
            "ridge_rapm": [6.0, 5.0, 4.0],
            "posterior_lower": [4.0, 3.0, 2.0],
            "posterior_upper": [8.0, 7.0, 6.0],
            "probability_positive": [1.0, 0.99, 0.95],
            "posterior_top_25_probability": [0.70, 0.60, 0.50],
            "posterior_rank_p05": [1.0, 1.0, 1.0],
            "posterior_rank_median": [1.0, 2.0, 3.0],
            "posterior_rank_p95": [3.0, 3.0, 3.0],
            "possessions": [3000.0, 2500.0, 2000.0],
        }
    )


def _bootstrap_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "bootstrap_p05": [5.0, 4.0, 3.0],
            "bootstrap_p95": [7.0, 6.0, 5.0],
            "positive_probability": [1.0, 1.0, 0.99],
            "top_25_probability": [0.80, 0.75, 0.70],
            "median_eligible_rank": [1.0, 2.0, 3.0],
        }
    )


def _comparison() -> pd.Series:
    return pd.Series(
        {
            "max_absolute_coefficient_difference": 2e-6,
            "coefficient_correlation": 1.0,
            "eligible_rank_spearman": 1.0,
            "top_25_overlap": 3,
            "max_absolute_test_prediction_difference": 5e-6,
            "ridge_game_margin_rmse": 15.8,
            "posterior_mean_game_margin_rmse": 15.8,
        }
    )


def _calibration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nominal_coverage": [0.50, 0.80, 0.90, 0.95],
            "unweighted_coverage": [0.51, 0.80, 0.91, 0.95],
            "possession_weighted_coverage": [0.52, 0.82, 0.92, 0.96],
            "possession_weighted_mean_interval_width": [130.0, 247.0, 317.0, 378.0],
        }
    )
