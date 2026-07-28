from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.case_study import (
    CaseStudySource,
    CaseStudyThresholds,
    _write_bootstrap_chart,
    _write_sensitivity_chart,
    classify_top_rankings,
    render_case_study_markdown,
)


def test_classify_top_rankings_uses_transparent_review_bands() -> None:
    frame = _diagnostic_frame(names=("Stable", "Qualified", "Multiple warnings", "Low bootstrap"))
    frame.loc[0, "top_25_probability"] = 0.80
    frame.loc[1, "top_25_probability"] = 0.60
    frame.loc[1, "chronological_eligible_rank_range"] = 50
    frame.loc[2, "top_25_probability"] = 0.60
    frame.loc[2, "chronological_eligible_rank_range"] = 50
    frame.loc[2, "lambda_eligible_rank_range"] = 50
    frame.loc[3, "top_25_probability"] = 0.40
    thresholds = CaseStudyThresholds(top_n=4)

    result = classify_top_rankings(frame, thresholds)

    assert result["review_band"].tolist() == [
        "stable core",
        "qualified",
        "fragile",
        "fragile",
    ]
    assert result["structural_warning_count"].tolist() == [0, 1, 2, 0]
    assert result.loc[2, "structural_warnings"] == "chronology, lambda"


def test_case_study_markdown_records_provenance_and_player_evidence() -> None:
    top = _profile_frame()
    source = CaseStudySource(
        season="2025-26",
        diagnostics_run_id="diagnostics-run",
        source_model_run_id="model-run",
        manifest_sha256="a" * 64,
        generator_code_sha256="b" * 64,
        selected_lambda=0.03,
        bootstrap_samples=200,
        player_count=582,
        eligible_player_count=452,
        game_count=1230,
        stint_count=39918,
    )

    markdown = render_case_study_markdown(
        source,
        top,
        _lambda_summary(),
        _allocation_metrics(),
        CaseStudyThresholds(top_n=7),
        bootstrap_reference="../assets/bootstrap.svg",
        sensitivity_reference="../assets/sensitivity.svg",
    )

    assert "# What a One-Season RAPM Can Establish" in markdown
    assert "diagnostics-run" in markdown
    assert "b" * 64 in markdown
    assert "Victor Wembanyama" in markdown
    assert "Neemias Queta" in markdown
    assert "Stable core" in markdown
    assert "Fragile" in markdown
    assert "../assets/bootstrap.svg" in markdown
    assert "five basketball allocation" not in markdown


def test_case_study_charts_are_deterministic_svg(
    tmp_path: Path,
) -> None:
    top = _profile_frame()
    first_bootstrap = tmp_path / "bootstrap-1.svg"
    second_bootstrap = tmp_path / "bootstrap-2.svg"
    first_sensitivity = tmp_path / "sensitivity-1.svg"
    second_sensitivity = tmp_path / "sensitivity-2.svg"

    _write_bootstrap_chart(top, first_bootstrap)
    _write_bootstrap_chart(top, second_bootstrap)
    _write_sensitivity_chart(
        top,
        first_sensitivity,
        CaseStudyThresholds(top_n=7),
    )
    _write_sensitivity_chart(
        top,
        second_sensitivity,
        CaseStudyThresholds(top_n=7),
    )

    assert first_bootstrap.read_bytes() == second_bootstrap.read_bytes()
    assert first_sensitivity.read_bytes() == second_sensitivity.read_bytes()
    assert b"<svg" in first_bootstrap.read_bytes()
    assert b"<svg" in first_sensitivity.read_bytes()


def _diagnostic_frame(names: tuple[str, ...]) -> pd.DataFrame:
    size = len(names)
    return pd.DataFrame(
        {
            "eligible_rank": range(1, size + 1),
            "exposure_eligible": True,
            "player_name": names,
            "primary_team_tricode": "TST",
            "rapm": [5.0 - index / 10 for index in range(size)],
            "possessions": 1000.0,
            "raw_on_court_net_rating": 6.0,
            "top_25_probability": 0.80,
            "chronological_eligible_rank_range": 10.0,
            "lambda_eligible_rank_range": 10.0,
            "max_allocation_absolute_eligible_rank_change": 10.0,
            "max_delete_game_absolute_coefficient_change": 0.20,
            "most_common_teammate_share": 0.50,
            "most_common_teammate_name": "Teammate",
            "bootstrap_p05": 2.0,
            "bootstrap_p95": 7.0,
        }
    )


def _profile_frame() -> pd.DataFrame:
    names = (
        "Victor Wembanyama",
        "Shai Gilgeous-Alexander",
        "Kawhi Leonard",
        "Neemias Queta",
        "Jimmy Butler III",
        "Ausar Thompson",
        "Brandon Miller",
    )
    frame = _diagnostic_frame(names)
    frame["review_band"] = (
        "stable core",
        "stable core",
        "qualified",
        "qualified",
        "fragile",
        "fragile",
        "fragile",
    )
    frame["structural_warning_count"] = (0, 0, 1, 1, 2, 2, 1)
    frame["structural_warnings"] = (
        "none",
        "none",
        "single-game influence",
        "teammate concentration",
        "lambda, allocation",
        "chronology, allocation",
        "chronology",
    )
    frame.loc[frame["player_name"].eq("Neemias Queta"), "most_common_teammate_name"] = (
        "Derrick White"
    )
    frame.loc[frame["player_name"].eq("Neemias Queta"), "most_common_teammate_share"] = 0.82
    return frame


def _lambda_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regularization": [0.01, 0.03, 0.10],
            "coefficient_correlation": [0.97, 1.0, 0.96],
            "rank_spearman": [0.97, 1.0, 0.95],
            "top_25_overlap": [19, 25, 21],
            "is_selected": [False, True, False],
        }
    )


def _allocation_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "allocation_policy": ["equal_segments", "terminal_lineup"],
            "model": ["rapm", "rapm"],
            "test_possessions": [18561.5, 18561.5],
            "game_margin_rmse": [15.81, 15.80],
            "skill_vs_mean": [0.013, 0.012],
        }
    )
