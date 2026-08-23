from __future__ import annotations

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V13,
    CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
    CONTEXT_FEATURE_SET_V2_DEPTH_AWARE_SHOOTING,
    LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES,
    contextual_feature_columns,
    side_context_feature_columns,
)
from nba_lineup_model.modeling.matchup_contextual import (
    fit_bounded_hierarchical_matchup_contextual_model,
    fit_kalman_filtered_linear_ridge_matchup_contextual_model,
    fit_linear_ridge_matchup_contextual_model,
    fit_matchup_contextual_model,
    fit_mean_reverting_linear_ridge_matchup_contextual_model,
    fit_normalized_linear_ridge_matchup_contextual_model,
    isolated_feature_component,
)


def test_kalman_linear_ridge_carries_only_additive_coefficients() -> None:
    """The Kalman state must persist only the twelve additive coefficients."""

    columns = side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V13)
    generator = np.random.default_rng(7)
    home = pd.DataFrame(generator.normal(size=(96, len(columns))), columns=columns)
    away = pd.DataFrame(generator.normal(size=(96, len(columns))), columns=columns)
    additive_target = home["assists_per_100"].to_numpy(dtype=float) - away[
        "assists_per_100"
    ].to_numpy(dtype=float)
    previous = fit_kalman_filtered_linear_ridge_matchup_contextual_model(
        home,
        away,
        additive_target,
        np.ones(len(home)),
        alpha=1.0,
        process_variance_multiplier=1.0,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V13,
    )
    changed = fit_kalman_filtered_linear_ridge_matchup_contextual_model(
        home,
        away,
        -additive_target,
        np.ones(len(home)),
        alpha=1.0,
        process_variance_multiplier=1.0,
        previous_model=previous,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V13,
    )
    unfiltered = fit_linear_ridge_matchup_contextual_model(
        home,
        away,
        -additive_target,
        np.ones(len(home)),
        alpha=1.0,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V13,
    )
    scale = changed.pipeline.named_steps["scale"]
    ridge = changed.pipeline.named_steps["ridge"]
    weights = pd.Series(ridge.coef_, index=scale.feature_names_in_)
    assert previous.additive_kalman_mean_raw is not None
    assert previous.additive_kalman_covariance_raw is not None
    assert previous.additive_kalman_mean_raw.shape == (12,)
    assert previous.additive_kalman_covariance_raw.shape == (12, 12)
    assert changed.additive_kalman_process_multiplier == 1.0
    assert changed.additive_state_precision is not None
    assert changed.additive_state_source_season == "previous_completed_season"
    unfiltered_weights = pd.Series(
        unfiltered.pipeline.named_steps["ridge"].coef_,
        index=unfiltered.pipeline.named_steps["scale"].feature_names_in_,
    )
    previous_weights = pd.Series(
        previous.pipeline.named_steps["ridge"].coef_,
        index=previous.pipeline.named_steps["scale"].feature_names_in_,
    )
    feature = "home_minus_away_assists_per_100"
    # The second-season posterior must move toward its evidence without fully
    # discarding the first-season additive posterior.
    assert unfiltered_weights[feature] < weights[feature] < previous_weights[feature]


def test_dynamic_additive_state_is_feature_specific_and_forward_only() -> None:
    """The v1.3.2 state carries ten raw coordinates and a prior-only history."""

    features = tuple(LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES)
    columns = side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE)
    generator = np.random.default_rng(19)
    home = pd.DataFrame(generator.normal(size=(128, len(columns))), columns=columns)
    away = pd.DataFrame(generator.normal(size=(128, len(columns))), columns=columns)
    target = home["assists_per_100"].to_numpy(dtype=float) - away[
        "assists_per_100"
    ].to_numpy(dtype=float)
    stable = frozenset(set(features) - {"unassisted_three_makes_per_100"})
    regime = frozenset({"unassisted_three_makes_per_100"})
    first = fit_mean_reverting_linear_ridge_matchup_contextual_model(
        home,
        away,
        target,
        np.ones(len(home)),
        alpha=1.0,
        additive_features=features,
        stable_features=stable,
        regime_features=regime,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
    )
    second = fit_mean_reverting_linear_ridge_matchup_contextual_model(
        home,
        away,
        -target,
        np.ones(len(home)),
        alpha=1.0,
        additive_features=features,
        stable_features=stable,
        regime_features=regime,
        previous_model=first,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
    )

    assert first.additive_dynamic_history_raw is not None
    assert first.additive_dynamic_history_raw.shape == (1, 10)
    assert second.additive_dynamic_history_raw is not None
    assert second.additive_dynamic_history_raw.shape == (2, 10)
    assert second.additive_dynamic_feature_names == features
    assert second.additive_dynamic_long_run_mean_raw is not None
    assert second.additive_dynamic_mean_reversion is not None
    assert second.additive_dynamic_process_variance_raw is not None
    assert second.additive_dynamic_zero_gated_features == ()
    assert np.all(second.additive_dynamic_process_variance_raw > 0)


def test_dynamic_additive_state_zero_gate_constrains_a_feature() -> None:
    features = tuple(LINEAR_NAIL_V131_PRUNED_BASKETBALL_ADDITIVE_FEATURES)
    columns = side_context_feature_columns(CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE)
    generator = np.random.default_rng(23)
    home = pd.DataFrame(generator.normal(size=(128, len(columns))), columns=columns)
    away = pd.DataFrame(generator.normal(size=(128, len(columns))), columns=columns)
    target = 100.0 * (
        home["assists_per_100"].to_numpy(dtype=float)
        - away["assists_per_100"].to_numpy(dtype=float)
    )
    model = fit_mean_reverting_linear_ridge_matchup_contextual_model(
        home,
        away,
        target,
        np.ones(len(home)),
        alpha=1.0,
        additive_features=features,
        stable_features=frozenset(set(features) - {"assists_per_100"}),
        regime_features=frozenset(),
        zero_gated_features=frozenset({"assists_per_100"}),
        feature_set=CONTEXT_FEATURE_SET_NAIL_V131_PRUNED_ADDITIVE,
    )

    assert model.additive_kalman_mean_raw is not None
    assert abs(model.additive_kalman_mean_raw[features.index("assists_per_100")]) < 1e-7


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


def test_normalized_linear_ridge_is_invariant_to_weight_scale() -> None:
    columns = side_context_feature_columns()
    values = np.arange(32 * len(columns), dtype=float).reshape(32, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = np.linspace(-3.0, 3.0, len(home))
    weights = np.linspace(1.0, 3.0, len(home))

    baseline = fit_normalized_linear_ridge_matchup_contextual_model(
        home, away, target, weights, alpha=0.05
    )
    rescaled = fit_normalized_linear_ridge_matchup_contextual_model(
        home, away, target, 17.0 * weights, alpha=0.05
    )

    np.testing.assert_allclose(
        baseline.predict_side_pairs(home, away),
        rescaled.predict_side_pairs(home, away),
        atol=1e-12,
    )
    assert baseline.regularization_contract == "mean_weighted_loss"
    assert rescaled.effective_ridge_alpha == 17.0 * baseline.effective_ridge_alpha


def test_normalized_linear_ridge_matches_equivalent_raw_alpha() -> None:
    columns = side_context_feature_columns()
    values = np.arange(32 * len(columns), dtype=float).reshape(32, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = np.linspace(-3.0, 3.0, len(home))
    weights = np.linspace(1.0, 3.0, len(home))
    normalized = fit_normalized_linear_ridge_matchup_contextual_model(
        home, away, target, weights, alpha=0.05
    )
    raw = fit_linear_ridge_matchup_contextual_model(
        home,
        away,
        target,
        weights,
        alpha=normalized.effective_ridge_alpha,
    )

    np.testing.assert_allclose(
        normalized.predict_side_pairs(home, away),
        raw.predict_side_pairs(home, away),
        atol=1e-12,
    )


def test_published_raw_alpha_matches_its_exact_normalized_lambda() -> None:
    columns = side_context_feature_columns()
    values = np.arange(32 * len(columns), dtype=float).reshape(32, len(columns))
    home = pd.DataFrame(values, columns=columns)
    away = pd.DataFrame(values[::-1], columns=columns)
    target = np.linspace(-3.0, 3.0, len(home))
    weights = np.linspace(1.0, 3.0, len(home))
    raw_alpha = 10_000.0
    augmented_weight_sum = 2.0 * weights.sum()

    raw = fit_linear_ridge_matchup_contextual_model(
        home, away, target, weights, alpha=raw_alpha
    )
    normalized = fit_normalized_linear_ridge_matchup_contextual_model(
        home,
        away,
        target,
        weights,
        alpha=raw_alpha / augmented_weight_sum,
    )

    assert normalized.effective_ridge_alpha == raw_alpha
    np.testing.assert_allclose(
        normalized.predict_side_pairs(home, away),
        raw.predict_side_pairs(home, away),
        atol=1e-12,
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
