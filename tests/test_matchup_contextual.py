from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING,
    contextual_feature_columns,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_bounded_hierarchical_matchup_contextual_model,
    fit_linear_ridge_matchup_contextual_model,
    fit_matchup_contextual_model,
    isolated_feature_component,
)


def test_matchup_context_is_antisymmetric_and_reference_identified() -> None:
    columns = side_context_feature_columns()
    values = np.arange(32 * len(columns), dtype=float).reshape(32, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = np.linspace(-3.0, 3.0, len(home))
    weights = np.linspace(1.0, 2.0, len(home))
    model = fit_matchup_contextual_model(home, away, target, weights, alpha=10.0)

    total = model.predict_side_pairs(home, away)
    reverse = model.predict_side_pairs(away, home)
    np.testing.assert_allclose(total, -reverse, atol=1e-12)

    components = model.decompose_side_pairs(home, away)
    np.testing.assert_allclose(
        components["total_context_net_rating"],
        components["home_portable_context_net_rating"]
        - components["away_portable_context_net_rating"]
        + components["matchup_context_net_rating"],
        atol=1e-12,
    )


def test_linear_ridge_context_has_no_matchup_remainder() -> None:
    columns = side_context_feature_columns()
    values = np.arange(32 * len(columns), dtype=float).reshape(32, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    model = fit_linear_ridge_matchup_contextual_model(
        home,
        away,
        np.linspace(-3.0, 3.0, len(home)),
        np.ones(len(home)),
        alpha=10.0,
    )

    components = model.decompose_side_pairs(home, away)
    np.testing.assert_allclose(components["matchup_context_net_rating"], 0.0, atol=1e-12)

    candidate = home.iloc[[0]].copy()
    reference = model.reference_features
    candidate_repeated = pd.concat([candidate] * len(reference), ignore_index=True)
    centered = model.decompose_side_pairs(candidate_repeated, reference)
    np.testing.assert_allclose(
        np.average(
            centered["matchup_context_net_rating"],
            weights=model.reference_weights,
        ),
        0.0,
        atol=1e-10,
    )


def test_isolated_feature_component_matches_pipeline_response() -> None:
    side_columns = side_context_feature_columns()
    values = np.arange(32 * len(side_columns), dtype=float).reshape(32, len(side_columns))
    home = pd.DataFrame(values, columns=side_columns)
    away = pd.DataFrame(values[::-1], columns=side_columns)
    model = fit_matchup_contextual_model(
        home,
        away,
        np.linspace(-3.0, 3.0, len(home)),
        np.ones(len(home)),
        alpha=10.0,
    )
    feature_index = 4
    feature_values = np.array([-8.0, -1.0, 0.0, 2.0, 7.0])
    relative = pd.DataFrame(
        0.0, index=range(len(feature_values)), columns=contextual_feature_columns()
    )
    relative.iloc[:, feature_index] = feature_values
    expected = 0.5 * (
        model.pipeline.predict(relative) - model.pipeline.predict(-relative)
    )
    np.testing.assert_allclose(
        isolated_feature_component(model, feature_index, feature_values), expected, atol=1e-12
    )


def test_temporal_pspline_prior_pulls_current_response_toward_previous_state() -> None:
    """A high temporal penalty carries a completed feature function forward."""

    columns = side_context_feature_columns()
    values = np.arange(64 * len(columns), dtype=float).reshape(64, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = home[columns[0]].to_numpy(dtype=float) - away[columns[0]].to_numpy(dtype=float)
    weights = np.ones(len(home))
    previous = fit_matchup_contextual_model(home, away, target, weights, alpha=1.0)
    unpooled = fit_matchup_contextual_model(home, away, -target, weights, alpha=1.0)
    pooled = fit_matchup_contextual_model(
        home,
        away,
        -target,
        weights,
        alpha=1.0,
        curvature_alpha=10.0,
        temporal_alpha=1_000_000.0,
        previous_model=previous,
    )
    grid = np.linspace(-100.0, 100.0, 31)
    previous_response = isolated_feature_component(previous, 0, grid)
    unpooled_response = isolated_feature_component(unpooled, 0, grid)
    pooled_response = isolated_feature_component(pooled, 0, grid)
    assert np.mean(np.abs(pooled_response - previous_response)) < np.mean(
        np.abs(unpooled_response - previous_response)
    )


def test_temporal_pspline_prior_requires_completed_state() -> None:
    """The first contextual season cannot invent a temporal predecessor."""

    columns = side_context_feature_columns()
    values = np.arange(16 * len(columns), dtype=float).reshape(16, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    with np.testing.assert_raises_regex(ValueError, "requires a previous model"):
        fit_matchup_contextual_model(
            home,
            away,
            np.ones(len(home)),
            np.ones(len(home)),
            alpha=1.0,
            temporal_alpha=1.0,
        )


def test_bounded_portable_matchup_context_caps_support_and_preserves_antisymmetry() -> None:
    columns = side_context_feature_columns()
    values = np.arange(64 * len(columns), dtype=float).reshape(64, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = home[columns[0]].to_numpy(dtype=float) - away[columns[0]].to_numpy(dtype=float)
    model = fit_bounded_hierarchical_matchup_contextual_model(
        home,
        away,
        target,
        np.ones(len(home)),
        alpha=10.0,
        curvature_alpha=1.0,
    )

    extreme_home = home.iloc[[0]].copy() - 1_000.0
    extreme_away = away.iloc[[0]].copy() + 1_000.0
    clipped_home = extreme_home.clip(model.side_lower, model.side_upper, axis=1)
    clipped_away = extreme_away.clip(model.side_lower, model.side_upper, axis=1)
    np.testing.assert_allclose(
        model.predict_side_pairs(extreme_home, extreme_away),
        model.predict_side_pairs(clipped_home, clipped_away),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.predict_side_pairs(extreme_home, extreme_away),
        -model.predict_side_pairs(extreme_away, extreme_home),
        atol=1e-12,
    )


def test_depth_aware_feature_contract_survives_bounded_temporal_update() -> None:
    """V2 state keeps its schema through the rolling hierarchical prior."""

    feature_set = CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING
    columns = side_context_feature_columns(feature_set)
    values = np.arange(64 * len(columns), dtype=float).reshape(64, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = home[columns[0]].to_numpy(dtype=float) - away[columns[0]].to_numpy(dtype=float)
    previous = fit_bounded_hierarchical_matchup_contextual_model(
        home,
        away,
        target,
        np.ones(len(home)),
        alpha=10.0,
        curvature_alpha=1.0,
        feature_set=feature_set,
    )
    current = fit_bounded_hierarchical_matchup_contextual_model(
        home,
        away,
        -target,
        np.ones(len(home)),
        alpha=10.0,
        curvature_alpha=1.0,
        temporal_alpha=1.0,
        previous_model=previous,
        feature_set=feature_set,
    )

    assert current.feature_set == feature_set
    assert tuple(current.side_lower.index) == columns
