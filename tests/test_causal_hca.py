from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.modeling.causal_hca import (
    CANDIDATE_ID,
    augment_game_contribution_with_home_court,
    select_home_court_prior,
    solve_joint_home_court_state,
)
from nba_lineup_model.modeling.in_season_nail import build_game_contribution


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022600001", "0022600001"],
            "home_player_ids": [(1,), (2,)],
            "away_player_ids": [(2,), (1,)],
            "target_home_net_rating": [10.0, 2.0],
            "possessions": [60.0, 40.0],
        }
    )


def test_joint_hca_contribution_matches_explicit_augmented_design() -> None:
    stints = _stints()
    prior = {1: 1.5, 2: -0.5}
    neutral_context = np.asarray([0.25, -0.75])
    base = build_game_contribution(
        stints,
        season="2026-27",
        game_id="0022600001",
        game_time_utc=datetime(2026, 10, 20, 23, tzinfo=UTC),
        finalized_at=datetime(2026, 10, 21, 3, tzinfo=UTC),
        source_sha256="a" * 64,
        source_uri="artifact://daily-data/test#0022600001",
        core_model_version="test-core",
        context_model_version="test-neutral-context",
        player_ids=(1, 2),
        preseason_nail=prior,
        fixed_context_offset=neutral_context,
    )
    joint = augment_game_contribution_with_home_court(
        base,
        stints,
        preseason_nail=prior,
        fixed_neutral_context_offset=neutral_context,
    )

    player_design = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    design = np.column_stack([player_design, np.ones(2)])
    weights = np.diag([60.0, 40.0])
    residual = (
        stints["target_home_net_rating"].to_numpy(dtype=float)
        - neutral_context
        - player_design @ np.asarray([1.5, -0.5])
    )
    np.testing.assert_allclose(joint.xtwx.toarray(), design.T @ weights @ design)
    np.testing.assert_allclose(joint.xtwy, design.T @ weights @ residual)
    np.testing.assert_allclose(joint.exposure, [100.0, 100.0, 100.0])


def test_joint_hca_solve_uses_centered_hca_prior_and_player_zero_prior() -> None:
    gram = sparse.csr_matrix(
        np.asarray(
            [
                [100.0, -100.0, 20.0],
                [-100.0, 100.0, -20.0],
                [20.0, -20.0, 100.0],
            ]
        )
    )
    score = np.asarray([50.0, -50.0, 200.0])
    solved = solve_joint_home_court_state(
        gram,
        score,
        player_count=2,
        player_prior_equivalent_possessions=6000.0,
        home_court_prior_mean=1.7,
        home_court_prior_equivalent_possessions=40000.0,
    )

    precision = gram.toarray() + np.diag([6000.0, 6000.0, 40000.0])
    expected = np.linalg.solve(precision, score + np.asarray([0.0, 0.0, 68000.0]))
    np.testing.assert_allclose(solved.player_adjustments, expected[:2])
    assert solved.home_court == expected[-1]
    assert solved.home_court_prior_weight == 40000.0 / 40100.0
    assert solved.home_court_evidence_weight == 100.0 / 40100.0


def test_prior_selection_is_forward_only_and_translates_variance_to_possessions() -> None:
    values = [
        4.2,
        4.0,
        3.8,
        3.6,
        3.4,
        3.2,
        3.0,
        2.8,
        2.6,
        2.4,
        2.2,
        2.0,
        1.8,
        1.6,
    ]
    seasons = [f"{2000 + index}-{str(2001 + index)[-2:]}" for index in range(len(values))]
    context = pd.DataFrame({"season": seasons, "context_home_intercept": values})
    residuals = pd.DataFrame({"residual_offense_margin": [1.0, -1.0, 0.5, -0.5]})

    selection = select_home_court_prior(
        context,
        residuals,
        target_season="2014-15",
        minimum_training_seasons=8,
    )

    assert selection.candidate_id == CANDIDATE_ID
    assert selection.source_through_season == "2013-14"
    assert selection.forward_validation_rows == 6
    metrics = json.loads(selection.method_metrics_json)
    assert selection.transition_method in metrics
    assert metrics[selection.transition_method]["rmse"] == min(
        row["rmse"] for row in metrics.values()
    )
    assert np.isfinite(selection.prior_mean)
    assert selection.observation_variance_per_100 == 6250.0
    assert selection.prior_equivalent_possessions == (
        selection.observation_variance_per_100 / selection.prior_variance
    )
    assert "last_season" in selection.method_metrics_json


def test_prior_selection_rejects_target_season_rows_by_date() -> None:
    seasons = [f"{2000 + index}-{str(2001 + index)[-2:]}" for index in range(12)]
    context = pd.DataFrame(
        {
            "season": seasons + ["2026-27"],
            "context_home_intercept": [3.0 - index * 0.1 for index in range(12)] + [999.0],
        }
    )
    residuals = pd.DataFrame({"residual_offense_margin": [1.0, -1.0]})

    selection = select_home_court_prior(
        context,
        residuals,
        target_season="2026-27",
        minimum_training_seasons=8,
    )

    assert selection.source_through_season == seasons[-1]
    assert selection.prior_mean < 10.0
