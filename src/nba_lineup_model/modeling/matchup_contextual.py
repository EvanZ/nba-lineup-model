"""Antisymmetric lineup context with portable-unit and matchup components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_V1,
    LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES,
    contextual_feature_columns,
    lineup_side_context_features,
    side_context_feature_columns,
)

if TYPE_CHECKING:
    from nba_lineup_model.modeling.rebound_opportunity import ReboundOpportunityModel
    from nba_lineup_model.modeling.usage_allocation import UsageAllocationModel

REFERENCE_CHUNK_SIZE: Final = 256


@dataclass(frozen=True)
class MatchupContextualModel:
    """Frozen total context and its reference-anchored h/q decomposition.

    The total context is antisymmetric by construction. The stored reference
    distribution identifies a portable side score h(A) and centers the matchup
    residual q(A, B) against the completed season's unit field.
    """

    pipeline: Pipeline
    reference_features: pd.DataFrame
    reference_weights: np.ndarray
    feature_set: str = field(default=CONTEXT_FEATURE_SET_V1, kw_only=True)
    rebound_model: ReboundOpportunityModel | None = field(default=None, kw_only=True)
    usage_model: UsageAllocationModel | None = field(default=None, kw_only=True)
    regularization_contract: str = field(default="weighted_sum_loss", kw_only=True)
    configured_regularization: float | None = field(default=None, kw_only=True)
    effective_ridge_alpha: float | None = field(default=None, kw_only=True)
    block_penalties: dict[str, float | None] | None = field(default=None, kw_only=True)
    training_weight_sum: float | None = field(default=None, kw_only=True)
    additive_state_precision: float | None = field(default=None, kw_only=True)
    additive_state_source_season: str | None = field(default=None, kw_only=True)
    additive_kalman_mean_raw: np.ndarray | None = field(default=None, kw_only=True)
    additive_kalman_covariance_raw: np.ndarray | None = field(default=None, kw_only=True)
    additive_kalman_process_multiplier: float | None = field(default=None, kw_only=True)
    additive_kalman_observation_variance: float | None = field(default=None, kw_only=True)
    additive_dynamic_feature_names: tuple[str, ...] | None = field(default=None, kw_only=True)
    additive_dynamic_history_raw: np.ndarray | None = field(default=None, kw_only=True)
    additive_dynamic_long_run_mean_raw: np.ndarray | None = field(default=None, kw_only=True)
    additive_dynamic_mean_reversion: np.ndarray | None = field(default=None, kw_only=True)
    additive_dynamic_process_variance_raw: np.ndarray | None = field(default=None, kw_only=True)
    additive_dynamic_zero_gated_features: tuple[str, ...] | None = field(
        default=None, kw_only=True
    )

    def predict_lineups(
        self,
        home_lineups: list[list[int]] | list[tuple[int, ...]],
        away_lineups: list[list[int]] | list[tuple[int, ...]],
        profiles: pd.DataFrame,
    ) -> np.ndarray:
        """Return total context for home minus away units."""

        home = lineup_side_context_features(
            home_lineups,
            profiles,
            feature_set=self.feature_set,
            rebound_model=getattr(self, "rebound_model", None),
            usage_model=getattr(self, "usage_model", None),
        )
        away = lineup_side_context_features(
            away_lineups,
            profiles,
            feature_set=self.feature_set,
            rebound_model=getattr(self, "rebound_model", None),
            usage_model=getattr(self, "usage_model", None),
        )
        return self.predict_side_pairs(home, away)

    def predict_side_pairs(self, home: pd.DataFrame, away: pd.DataFrame) -> np.ndarray:
        """Return the antisymmetric total context for aligned side features."""

        relative = _relative_features(home, away, self.feature_set)
        forward = self.pipeline.predict(relative)
        reverse = self.pipeline.predict(-relative)
        return 0.5 * (
            np.asarray(forward, dtype=float) - np.asarray(reverse, dtype=float)
        )

    def predict_side_scores(self, side_features: pd.DataFrame) -> np.ndarray:
        """Return h(x): expected total context against the frozen reference field."""

        side = _validated_side_features(side_features, self.feature_set)
        reference = self.reference_features.loc[:, side_context_feature_columns(self.feature_set)]
        weights = self.reference_weights
        output = np.empty(len(side), dtype=float)
        for start in range(0, len(side), REFERENCE_CHUNK_SIZE):
            chunk = side.iloc[start : start + REFERENCE_CHUNK_SIZE].reset_index(drop=True)
            count = len(chunk)
            home = pd.DataFrame(
                np.repeat(chunk.to_numpy(dtype=float), len(reference), axis=0),
                columns=side_context_feature_columns(self.feature_set),
            )
            away = pd.DataFrame(
                np.tile(reference.to_numpy(dtype=float), (count, 1)),
                columns=side_context_feature_columns(self.feature_set),
            )
            total = self.predict_side_pairs(home, away).reshape(count, len(reference))
            output[start : start + count] = total @ weights
        return output

    def decompose_side_pairs(self, home: pd.DataFrame, away: pd.DataFrame) -> pd.DataFrame:
        """Return C(A,B), h(A), h(B), and the centered q(A,B) residual."""

        home = _validated_side_features(home, self.feature_set)
        away = _validated_side_features(away, self.feature_set)
        total = self.predict_side_pairs(home, away)
        home_score = self.predict_side_scores(home)
        away_score = self.predict_side_scores(away)
        matchup = total - home_score + away_score
        return pd.DataFrame(
            {
                "total_context_net_rating": total,
                "home_portable_context_net_rating": home_score,
                "away_portable_context_net_rating": away_score,
                "matchup_context_net_rating": matchup,
            }
        )


def fit_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> MatchupContextualModel:
    """Fit an antisymmetric relative spline with optional P-spline hierarchy.

    ``alpha`` is the usual zero-centered Ridge penalty. ``curvature_alpha``
    penalizes second differences of unscaled B-spline coefficients. When a
    completed ``previous_model`` is supplied, ``temporal_alpha`` pulls the
    current standardized coefficients toward that prior response function,
    projected onto the current season's basis. This keeps the seasonal
    information boundary forward-looking even though each season learns its
    spline knots and feature scaler from its own completed data.
    """

    if alpha <= 0 or curvature_alpha < 0 or temporal_alpha < 0:
        raise ValueError("Contextual penalties must be non-negative, with alpha positive")
    if temporal_alpha and previous_model is None:
        raise ValueError("Temporal contextual penalty requires a previous model")
    if previous_model is not None and previous_model.feature_set != feature_set:
        raise ValueError("Temporal contextual prior requires the same feature set")
    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    relative = _relative_features(home, away, feature_set)
    # Paired orientations make the fitted residual itself respect the home/away
    # sign convention; predict_side_pairs additionally enforces it exactly.
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    spline = SplineTransformer(n_knots=4, degree=2, extrapolation="linear")
    basis = spline.fit_transform(augmented_features)
    scale = StandardScaler()
    design = scale.fit_transform(basis)
    penalty_rows: list[np.ndarray] = []
    penalty_targets: list[np.ndarray] = []
    if curvature_alpha:
        penalty_rows.append(
            np.sqrt(curvature_alpha) * _second_difference_penalty(spline, scale, feature_set)
        )
        penalty_targets.append(np.zeros(penalty_rows[-1].shape[0], dtype=float))
    if temporal_alpha and previous_model is not None:
        prior = _project_previous_response(
            previous_model,
            spline,
            scale,
            augmented_features,
            feature_set,
        )
        coefficient_count = design.shape[1]
        penalty_rows.append(np.sqrt(temporal_alpha) * np.eye(coefficient_count))
        penalty_targets.append(np.sqrt(temporal_alpha) * prior)
    if penalty_rows:
        design = np.vstack([design, *penalty_rows])
        augmented_target = np.concatenate([augmented_target, *penalty_targets])
        augmented_weight = np.concatenate(
            [augmented_weight, np.ones(sum(len(rows) for rows in penalty_rows), dtype=float)]
        )
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(
        design,
        augmented_target,
        sample_weight=augmented_weight,
    )
    pipeline = Pipeline(
        [
            ("spline", spline),
            ("scale", scale),
            ("ridge", ridge),
        ]
    )
    reference_features, reference_weights = _reference_distribution(
        home, away, weights, feature_set
    )
    return MatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        feature_set=feature_set,
    )


def fit_linear_ridge_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> MatchupContextualModel:
    """Fit a standardized linear Ridge context model on side-feature differences."""

    if alpha <= 0:
        raise ValueError("Linear contextual alpha must be positive")
    if curvature_alpha or temporal_alpha:
        raise ValueError("Linear Ridge context does not use spline or temporal penalties")
    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    relative = _relative_features(home, away, feature_set)
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    scale = StandardScaler()
    design = scale.fit_transform(augmented_features)
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
        design,
        augmented_target,
        sample_weight=augmented_weight,
    )
    pipeline = Pipeline([("scale", scale), ("ridge", ridge)])
    reference_features, reference_weights = _reference_distribution(
        home, away, weights, feature_set
    )
    return MatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        feature_set=feature_set,
        regularization_contract="weighted_sum_loss",
        configured_regularization=float(alpha),
        effective_ridge_alpha=float(alpha),
        training_weight_sum=float(augmented_weight.sum()),
    )


def fit_block_penalized_linear_ridge_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    additive_alpha: float,
    nonadditive_alpha: float | None,
    additive_features: tuple[str, ...],
    nonadditive_features: tuple[str, ...],
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> MatchupContextualModel:
    """Fit linear Ridge with separate additive and non-additive penalties.

    The design is standardized first, exactly as in the shared-alpha model.  We
    then solve the diagonal-penalty problem by rescaling each standardized
    column before fitting a unit-alpha Ridge and transforming the resulting
    coefficients back into the original standardized coordinates.  This keeps
    the stored inference pipeline unchanged while yielding the objective

    ``weighted SSE + additive_alpha * ||beta_add||^2 +
    nonadditive_alpha * ||beta_nonadd||^2``. Passing ``None`` for
    ``nonadditive_alpha`` removes that block structurally.
    """

    if alpha <= 0 or additive_alpha <= 0 or (
        nonadditive_alpha is not None and nonadditive_alpha <= 0
    ):
        raise ValueError("Block-penalized linear contextual alphas must be positive")
    if curvature_alpha or temporal_alpha:
        raise ValueError("Block-penalized linear Ridge does not use spline penalties")
    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    side_columns = side_context_feature_columns(feature_set)
    declared = set(additive_features) | set(nonadditive_features)
    if declared != set(side_columns) or set(additive_features) & set(nonadditive_features):
        raise ValueError("Additive and non-additive features must partition the side contract")

    relative = _relative_features(home, away, feature_set)
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    scale = StandardScaler()
    design = scale.fit_transform(augmented_features)

    columns = contextual_feature_columns(feature_set)
    additive_indices = [
        index
        for index, column in enumerate(columns)
        if column.removeprefix("home_minus_away_") in set(additive_features)
    ]
    if nonadditive_alpha is None:
        selector = ColumnTransformer(
            [("additive", "passthrough", additive_indices)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        selected_design = selector.fit_transform(design)
        ridge = Ridge(alpha=additive_alpha, fit_intercept=False).fit(
            selected_design,
            augmented_target,
            sample_weight=augmented_weight,
        )
        pipeline = Pipeline([("scale", scale), ("select", selector), ("ridge", ridge)])
        regularization_contract = "weighted_sum_loss_additive_only_ridge"
        effective_ridge_alpha = float(additive_alpha)
    else:
        penalties = np.asarray(
            [
                additive_alpha if index in additive_indices else nonadditive_alpha
                for index in range(len(columns))
            ],
            dtype=float,
        )
        rescaling = 1.0 / np.sqrt(penalties)
        ridge = Ridge(alpha=1.0, fit_intercept=False).fit(
            design * rescaling,
            augmented_target,
            sample_weight=augmented_weight,
        )
        ridge.coef_ = np.asarray(ridge.coef_, dtype=float) * rescaling
        pipeline = Pipeline([("scale", scale), ("ridge", ridge)])
        regularization_contract = "weighted_sum_loss_block_ridge"
        effective_ridge_alpha = 1.0
    reference_features, reference_weights = _reference_distribution(
        home, away, weights, feature_set
    )
    return MatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        feature_set=feature_set,
        regularization_contract=regularization_contract,
        configured_regularization=float(alpha),
        effective_ridge_alpha=effective_ridge_alpha,
        training_weight_sum=float(augmented_weight.sum()),
        block_penalties={
            "additive": float(additive_alpha),
            "nonadditive": (
                float(nonadditive_alpha) if nonadditive_alpha is not None else None
            ),
        },
    )


def fit_kalman_filtered_linear_ridge_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    process_variance_multiplier: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> MatchupContextualModel:
    """Fit linear Ridge with a forward Kalman state for additive profiles.

    The persisted state is an additive-coefficient posterior mean and
    covariance in raw feature units. Before fitting season ``t``, the filter
    applies a diagonal random-walk process covariance and maps that prior into
    the current season's standardized coordinates. The weighted normal matrix
    supplies the season's measurement update. Non-additive terms remain
    zero-centered Ridge nuisance coefficients rather than state variables.
    """

    if alpha <= 0 or process_variance_multiplier < 0:
        raise ValueError(
            "Kalman linear Ridge requires positive alpha and non-negative process multiplier"
        )
    if curvature_alpha or temporal_alpha:
        raise ValueError("Kalman linear Ridge does not use spline penalties")
    if previous_model is not None and previous_model.feature_set != feature_set:
        raise ValueError("Forward additive state requires the same feature set")
    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    relative = _relative_features(home, away, feature_set)
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    scale = StandardScaler()
    design = scale.fit_transform(augmented_features)
    columns = list(scale.feature_names_in_)
    additive_columns = [
        f"home_minus_away_{feature}"
        for feature in LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES
    ]
    additive_indices = np.asarray([columns.index(column) for column in additive_columns])
    weighted_design = design * augmented_weight[:, None]
    base_precision = design.T @ weighted_design + alpha * np.eye(design.shape[1])
    base_information = design.T @ (augmented_weight * augmented_target)
    base_coefficients = np.linalg.solve(base_precision, base_information)
    base_residual = augmented_target - design @ base_coefficients
    observation_variance = float(
        np.sum(augmented_weight * np.square(base_residual)) / augmented_weight.sum()
    )
    observation_variance = max(observation_variance, np.finfo(float).eps)
    posterior_precision = base_precision.copy()
    posterior_information = base_information.copy()
    state_source_season: str | None = None
    state_precision_summary: float | None = None
    if previous_model is not None:
        prior_mean_raw, prior_covariance_raw = _kalman_additive_state(
            previous_model, feature_set
        )
        current_scale = np.asarray(scale.scale_, dtype=float)[additive_indices]
        process_covariance_raw = np.diag(
            process_variance_multiplier * np.diag(prior_covariance_raw)
        )
        prior_covariance_raw = prior_covariance_raw + process_covariance_raw
        prior_mean = current_scale * prior_mean_raw
        prior_covariance = (
            current_scale[:, None] * prior_covariance_raw * current_scale[None, :]
        )
        prior_precision = observation_variance * np.linalg.pinv(prior_covariance)
        posterior_precision[np.ix_(additive_indices, additive_indices)] += prior_precision
        posterior_information[additive_indices] += prior_precision @ prior_mean
        state_source_season = "previous_completed_season"
        state_precision_summary = float(np.trace(prior_precision) / len(additive_indices))
    coefficients = np.linalg.solve(posterior_precision, posterior_information)
    posterior_covariance = observation_variance * np.linalg.pinv(posterior_precision)
    raw_scale = np.asarray(scale.scale_, dtype=float)[additive_indices]
    additive_mean_raw = coefficients[additive_indices] / raw_scale
    additive_covariance_raw = posterior_covariance[np.ix_(additive_indices, additive_indices)]
    additive_covariance_raw = additive_covariance_raw / (
        raw_scale[:, None] * raw_scale[None, :]
    )
    # Fit once to populate scikit-learn's estimator metadata, then replace its
    # coefficients with the closed-form Kalman posterior solution.
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
        design,
        augmented_target,
        sample_weight=augmented_weight,
    )
    ridge.coef_ = coefficients
    pipeline = Pipeline([("scale", scale), ("ridge", ridge)])
    reference_features, reference_weights = _reference_distribution(
        home, away, weights, feature_set
    )
    return MatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        feature_set=feature_set,
        regularization_contract="weighted_sum_loss_with_kalman_additive_state",
        configured_regularization=float(alpha),
        effective_ridge_alpha=float(alpha),
        training_weight_sum=float(augmented_weight.sum()),
        additive_state_precision=state_precision_summary,
        additive_state_source_season=state_source_season,
        additive_kalman_mean_raw=additive_mean_raw,
        additive_kalman_covariance_raw=additive_covariance_raw,
        additive_kalman_process_multiplier=float(process_variance_multiplier),
        additive_kalman_observation_variance=observation_variance,
    )


def _kalman_additive_state(
    model: MatchupContextualModel,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a persisted additive posterior in raw feature coordinates."""

    if model.feature_set != feature_set:
        raise ValueError("Forward additive state requires a matching feature contract")
    mean = getattr(model, "additive_kalman_mean_raw", None)
    covariance = getattr(model, "additive_kalman_covariance_raw", None)
    expected = len(LINEAR_NAIL_V13_BASKETBALL_ADDITIVE_FEATURES)
    if mean is None or covariance is None:
        raise ValueError("Previous context model has no Kalman additive state")
    mean_values = np.asarray(mean, dtype=float)
    covariance_values = np.asarray(covariance, dtype=float)
    if mean_values.shape != (expected,) or covariance_values.shape != (expected, expected):
        raise ValueError("Previous Kalman additive state has an invalid shape")
    if (
        not np.isfinite(mean_values).all()
        or not np.isfinite(covariance_values).all()
        or (np.diag(covariance_values) <= 0).any()
    ):
        raise ValueError("Previous Kalman additive state is not a valid covariance posterior")
    return mean_values, covariance_values


def fit_mean_reverting_linear_ridge_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    additive_features: tuple[str, ...],
    stable_features: frozenset[str],
    regime_features: frozenset[str],
    zero_gated_features: frozenset[str] = frozenset(),
    mean_reversion_prior_strength: float = 6.0,
    process_variance_floor_ratio: float = 0.10,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> MatchupContextualModel:
    """Fit a forward mean-reverting empirical-Bayes additive state.

    Each raw additive coefficient follows a feature-specific AR(1) transition
    toward its running completed-season mean. Innovation variances are learned
    from prior posterior innovations and floored by prior uncertainty. Features
    in ``zero_gated_features`` are constrained to zero in every season.
    """

    if alpha <= 0 or mean_reversion_prior_strength < 0 or process_variance_floor_ratio < 0:
        raise ValueError(
            "Dynamic linear Ridge requires positive alpha and non-negative state terms"
        )
    if curvature_alpha or temporal_alpha:
        raise ValueError("Dynamic linear Ridge does not use spline penalties")
    all_classified = stable_features | regime_features | zero_gated_features
    if set(additive_features) != all_classified:
        raise ValueError("Every dynamic additive feature must have exactly one state category")
    if (stable_features & regime_features) or (stable_features & zero_gated_features) or (
        regime_features & zero_gated_features
    ):
        raise ValueError("Dynamic additive state categories must be disjoint")
    if previous_model is not None and previous_model.feature_set != feature_set:
        raise ValueError("Forward additive state requires the same feature set")

    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    relative = _relative_features(home, away, feature_set)
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    scale = StandardScaler()
    design = scale.fit_transform(augmented_features)
    columns = list(scale.feature_names_in_)
    additive_indices = np.asarray(
        [columns.index(f"home_minus_away_{feature}") for feature in additive_features]
    )
    weighted_design = design * augmented_weight[:, None]
    base_precision = design.T @ weighted_design + alpha * np.eye(design.shape[1])
    base_information = design.T @ (augmented_weight * augmented_target)
    base_coefficients = np.linalg.solve(base_precision, base_information)
    base_residual = augmented_target - design @ base_coefficients
    observation_variance = float(
        np.sum(augmented_weight * np.square(base_residual)) / augmented_weight.sum()
    )
    observation_variance = max(observation_variance, np.finfo(float).eps)

    posterior_precision = base_precision.copy()
    posterior_information = base_information.copy()
    state_source_season: str | None = None
    state_precision_summary: float | None = None
    previous_history: np.ndarray | None = None
    if previous_model is not None:
        previous_history, previous_covariance = _dynamic_additive_state(
            previous_model, feature_set, additive_features
        )
        long_run_mean, mean_reversion, process_variance = _dynamic_transition(
            previous_history,
            np.diag(previous_covariance),
            additive_features,
            stable_features,
            regime_features,
            zero_gated_features,
            mean_reversion_prior_strength=mean_reversion_prior_strength,
            process_variance_floor_ratio=process_variance_floor_ratio,
        )
        phi = np.diag(mean_reversion)
        prior_mean_raw = long_run_mean + mean_reversion * (
            previous_history[-1] - long_run_mean
        )
        prior_covariance_raw = phi @ previous_covariance @ phi + np.diag(process_variance)
        current_scale = np.asarray(scale.scale_, dtype=float)[additive_indices]
        prior_mean = current_scale * prior_mean_raw
        prior_covariance = (
            current_scale[:, None] * prior_covariance_raw * current_scale[None, :]
        )
        prior_precision = observation_variance * np.linalg.pinv(prior_covariance)
        posterior_precision[np.ix_(additive_indices, additive_indices)] += prior_precision
        posterior_information[additive_indices] += prior_precision @ prior_mean
        state_source_season = "previous_completed_season"
        state_precision_summary = float(np.trace(prior_precision) / len(additive_indices))
    else:
        long_run_mean = np.zeros(len(additive_features), dtype=float)
        mean_reversion = _dynamic_default_reversion(
            additive_features, stable_features, regime_features, zero_gated_features
        )
        process_variance = np.zeros(len(additive_features), dtype=float)

    for feature_index, feature in enumerate(additive_features):
        if feature in zero_gated_features:
            design_index = additive_indices[feature_index]
            posterior_precision[design_index, design_index] += 1e12

    coefficients = np.linalg.solve(posterior_precision, posterior_information)
    posterior_covariance = observation_variance * np.linalg.pinv(posterior_precision)
    raw_scale = np.asarray(scale.scale_, dtype=float)[additive_indices]
    additive_mean_raw = coefficients[additive_indices] / raw_scale
    additive_covariance_raw = posterior_covariance[np.ix_(additive_indices, additive_indices)]
    additive_covariance_raw = additive_covariance_raw / (
        raw_scale[:, None] * raw_scale[None, :]
    )
    if previous_model is None:
        process_variance = process_variance_floor_ratio * np.diag(additive_covariance_raw)
    history = (
        additive_mean_raw[None, :]
        if previous_history is None
        else np.vstack([previous_history, additive_mean_raw])
    )

    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
        design,
        augmented_target,
        sample_weight=augmented_weight,
    )
    ridge.coef_ = coefficients
    pipeline = Pipeline([("scale", scale), ("ridge", ridge)])
    reference_features, reference_weights = _reference_distribution(
        home, away, weights, feature_set
    )
    return MatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        feature_set=feature_set,
        regularization_contract="weighted_sum_loss_with_dynamic_additive_state",
        configured_regularization=float(alpha),
        effective_ridge_alpha=float(alpha),
        training_weight_sum=float(augmented_weight.sum()),
        additive_state_precision=state_precision_summary,
        additive_state_source_season=state_source_season,
        additive_kalman_mean_raw=additive_mean_raw,
        additive_kalman_covariance_raw=additive_covariance_raw,
        additive_kalman_observation_variance=observation_variance,
        additive_dynamic_feature_names=additive_features,
        additive_dynamic_history_raw=history,
        additive_dynamic_long_run_mean_raw=long_run_mean,
        additive_dynamic_mean_reversion=mean_reversion,
        additive_dynamic_process_variance_raw=process_variance,
        additive_dynamic_zero_gated_features=tuple(sorted(zero_gated_features)),
    )


def _dynamic_additive_state(
    model: MatchupContextualModel,
    feature_set: str,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if model.feature_set != feature_set:
        raise ValueError("Forward dynamic state requires a matching feature contract")
    if tuple(getattr(model, "additive_dynamic_feature_names", ())) != features:
        raise ValueError("Previous dynamic state has an incompatible feature contract")
    history = np.asarray(getattr(model, "additive_dynamic_history_raw", None), dtype=float)
    covariance = np.asarray(getattr(model, "additive_kalman_covariance_raw", None), dtype=float)
    expected = len(features)
    if history.ndim != 2 or history.shape[1] != expected or len(history) < 1:
        raise ValueError("Previous dynamic state has an invalid coefficient history")
    if covariance.shape != (expected, expected) or (np.diag(covariance) <= 0).any():
        raise ValueError("Previous dynamic state has an invalid covariance posterior")
    return history, covariance


def _dynamic_default_reversion(
    features: tuple[str, ...],
    stable_features: frozenset[str],
    regime_features: frozenset[str],
    zero_gated_features: frozenset[str],
) -> np.ndarray:
    return np.asarray(
        [
            0.90 if feature in stable_features else 0.60 if feature in regime_features else 0.0
            for feature in features
        ],
        dtype=float,
    )


def _dynamic_transition(
    history: np.ndarray,
    previous_variance: np.ndarray,
    features: tuple[str, ...],
    stable_features: frozenset[str],
    regime_features: frozenset[str],
    zero_gated_features: frozenset[str],
    *,
    mean_reversion_prior_strength: float,
    process_variance_floor_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_run_mean = history.mean(axis=0)
    default_reversion = _dynamic_default_reversion(
        features, stable_features, regime_features, zero_gated_features
    )
    if len(history) < 3:
        process_variance = process_variance_floor_ratio * previous_variance
        return long_run_mean, default_reversion, process_variance
    previous = history[:-1] - long_run_mean
    current = history[1:] - long_run_mean
    scale = np.maximum(np.mean(np.square(previous), axis=0), np.finfo(float).eps)
    numerator = np.sum(previous * current, axis=0) + (
        mean_reversion_prior_strength * default_reversion * scale
    )
    denominator = np.sum(np.square(previous), axis=0) + mean_reversion_prior_strength * scale
    reversion = numerator / denominator
    lower = np.asarray([-0.25 if feature in regime_features else 0.0 for feature in features])
    upper = np.asarray([0.85 if feature in regime_features else 0.98 for feature in features])
    reversion = np.clip(reversion, lower, upper)
    reversion = np.where(
        np.asarray([feature in zero_gated_features for feature in features]), 0.0, reversion
    )
    innovation = current - reversion * previous
    empirical_variance = np.mean(np.square(innovation), axis=0)
    process_variance = np.maximum(
        empirical_variance, process_variance_floor_ratio * previous_variance
    )
    return long_run_mean, reversion, process_variance


def fit_normalized_linear_ridge_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: MatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> MatchupContextualModel:
    """Fit linear context with a penalty relative to mean weighted loss.

    ``alpha`` is the dimensionless ``lambda_C`` in
    ``weighted_sse / sum(weight) + lambda_C * ||beta||^2``. The sum is over
    the signed orientation-augmented sample, where each stint appears twice.
    Scikit-learn's Ridge objective uses weighted SSE, so its fitted alpha is
    ``lambda_C * augmented_weight_sum``. This keeps regularization strength
    invariant to season length and to a common rescaling of every sample
    weight.
    """

    if alpha <= 0:
        raise ValueError("Normalized linear contextual alpha must be positive")
    if curvature_alpha or temporal_alpha:
        raise ValueError("Normalized linear Ridge context does not use extra penalties")
    del previous_model
    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    relative = _relative_features(home, away, feature_set)
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    training_weight_sum = float(augmented_weight.sum())
    effective_ridge_alpha = float(alpha * training_weight_sum)
    scale = StandardScaler()
    design = scale.fit_transform(augmented_features)
    ridge = Ridge(alpha=effective_ridge_alpha, fit_intercept=False).fit(
        design,
        augmented_target,
        sample_weight=augmented_weight,
    )
    pipeline = Pipeline([("scale", scale), ("ridge", ridge)])
    reference_features, reference_weights = _reference_distribution(
        home, away, weights, feature_set
    )
    return MatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        feature_set=feature_set,
        regularization_contract="mean_weighted_loss",
        configured_regularization=float(alpha),
        effective_ridge_alpha=effective_ridge_alpha,
        training_weight_sum=training_weight_sum,
    )


@dataclass(frozen=True)
class BoundedMatchupContextualModel(MatchupContextualModel):
    """Portable-matchup context with forward-safe feature support bounds.

    Side profiles are capped independently to their completed-season central
    support. The relative matchup term is then capped symmetrically, preserving
    exact antisymmetry when the two units are swapped.
    """

    side_lower: pd.Series
    side_upper: pd.Series
    relative_cap: pd.Series

    def predict_side_pairs(self, home: pd.DataFrame, away: pd.DataFrame) -> np.ndarray:
        home = _clip_side_features(home, self.side_lower, self.side_upper, self.feature_set)
        away = _clip_side_features(away, self.side_lower, self.side_upper, self.feature_set)
        relative = _clip_relative_features(
            _relative_features(home, away, self.feature_set), self.relative_cap, self.feature_set
        )
        forward = self.pipeline.predict(relative)
        reverse = self.pipeline.predict(-relative)
        return 0.5 * (np.asarray(forward, dtype=float) - np.asarray(reverse, dtype=float))


def fit_bounded_hierarchical_matchup_contextual_model(
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
    curvature_alpha: float = 0.0,
    temporal_alpha: float = 0.0,
    previous_model: BoundedMatchupContextualModel | None = None,
    feature_set: str = CONTEXT_FEATURE_SET_V1,
) -> BoundedMatchupContextualModel:
    """Fit bounded, portable ``h(A)-h(B)+q(A,B)`` contextual state.

    Quantile bounds are learned only from the completed season currently being
    fit. Predictions cap side profiles at their 5th--95th percentile support
    and use a symmetric central-support cap for relative matchup features.
    """

    if alpha <= 0 or curvature_alpha < 0 or temporal_alpha < 0:
        raise ValueError("Contextual penalties must be non-negative, with alpha positive")
    if temporal_alpha and previous_model is None:
        raise ValueError("Temporal contextual penalty requires a previous model")
    if previous_model is not None and previous_model.feature_set != feature_set:
        raise ValueError("Temporal contextual prior requires the same feature set")
    home = _validated_side_features(home_features, feature_set)
    away = _validated_side_features(away_features, feature_set)
    target_values = np.asarray(target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if len(home) != len(away) or len(home) != len(target_values) or len(home) != len(weights):
        raise ValueError("Contextual training inputs must have equal lengths")
    if (
        not np.isfinite(target_values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Contextual targets and weights must be finite, with positive weights")

    side_lower, side_upper = _side_support_bounds(home, away, weights, feature_set)
    bounded_home = _clip_side_features(home, side_lower, side_upper, feature_set)
    bounded_away = _clip_side_features(away, side_lower, side_upper, feature_set)
    raw_relative = _relative_features(bounded_home, bounded_away, feature_set)
    relative_cap = _symmetric_relative_caps(raw_relative, weights, feature_set)
    relative = _clip_relative_features(raw_relative, relative_cap, feature_set)
    augmented_features = pd.concat([relative, -relative], ignore_index=True)
    augmented_target = np.concatenate([target_values, -target_values])
    augmented_weight = np.concatenate([weights, weights])
    spline = SplineTransformer(n_knots=4, degree=2, extrapolation="linear")
    basis = spline.fit_transform(augmented_features)
    scale = StandardScaler()
    design = scale.fit_transform(basis)
    penalty_rows: list[np.ndarray] = []
    penalty_targets: list[np.ndarray] = []
    if curvature_alpha:
        penalty_rows.append(
            np.sqrt(curvature_alpha) * _second_difference_penalty(spline, scale, feature_set)
        )
        penalty_targets.append(np.zeros(penalty_rows[-1].shape[0], dtype=float))
    if temporal_alpha and previous_model is not None:
        prior = _project_previous_response(
            previous_model,
            spline,
            scale,
            augmented_features,
            feature_set,
        )
        penalty_rows.append(np.sqrt(temporal_alpha) * np.eye(design.shape[1]))
        penalty_targets.append(np.sqrt(temporal_alpha) * prior)
    if penalty_rows:
        design = np.vstack([design, *penalty_rows])
        augmented_target = np.concatenate([augmented_target, *penalty_targets])
        augmented_weight = np.concatenate(
            [augmented_weight, np.ones(sum(len(rows) for rows in penalty_rows), dtype=float)]
        )
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
        design,
        augmented_target,
        sample_weight=augmented_weight,
    )
    pipeline = Pipeline([("spline", spline), ("scale", scale), ("ridge", ridge)])
    reference_features, reference_weights = _reference_distribution(
        bounded_home,
        bounded_away,
        weights,
        feature_set,
    )
    return BoundedMatchupContextualModel(
        pipeline,
        reference_features,
        reference_weights,
        side_lower,
        side_upper,
        relative_cap,
        feature_set=feature_set,
    )


def model_metadata(model: MatchupContextualModel) -> dict[str, object]:
    """Return durable metadata for the seasonal context state."""

    return {
        "context_function": "C(A,B) = h(A) - h(B) + q(A,B)",
        "context_antisymmetry": "C(A,B) = -C(B,A) exactly",
        "context_reference_distribution": "completed-season possession-weighted side units",
        "context_reference_unit_count": int(len(model.reference_features)),
        "context_reference_weight_sum": float(model.reference_weights.sum()),
        "context_feature_set": model.feature_set,
        "context_regularization_contract": getattr(
            model, "regularization_contract", "weighted_sum_loss"
        ),
        "context_configured_regularization": getattr(
            model, "configured_regularization", None
        ),
        "context_effective_ridge_alpha": getattr(model, "effective_ridge_alpha", None),
        "context_block_penalties": getattr(model, "block_penalties", None),
        "context_training_weight_sum": getattr(model, "training_weight_sum", None),
        "context_additive_state_precision": getattr(model, "additive_state_precision", None),
        "context_additive_state_source_season": getattr(
            model, "additive_state_source_season", None
        ),
        "context_kalman_process_multiplier": getattr(
            model, "additive_kalman_process_multiplier", None
        ),
        "context_kalman_observation_variance": getattr(
            model, "additive_kalman_observation_variance", None
        ),
    }


def bounded_model_metadata(model: BoundedMatchupContextualModel) -> dict[str, object]:
    """Return portable-context metadata including its learned support bounds."""

    return {
        **model_metadata(model),
        "context_side_feature_bounds": "possession-weighted 5th--95th percentiles",
        "context_relative_feature_bounds": "symmetric possession-weighted central support",
        "context_side_lower": model.side_lower.to_dict(),
        "context_side_upper": model.side_upper.to_dict(),
        "context_relative_cap": model.relative_cap.to_dict(),
    }


def _second_difference_penalty(
    spline: SplineTransformer, scale: StandardScaler, feature_set: str
) -> np.ndarray:
    """Return the P-spline penalty in the standardized design coordinates."""

    feature_count = len(contextual_feature_columns(feature_set))
    coefficient_count = int(spline.n_features_out_)
    if coefficient_count % feature_count:
        raise ValueError("Spline feature layout is incompatible with P-spline penalty")
    basis_count = coefficient_count // feature_count
    if basis_count < 3:
        raise ValueError("P-spline penalty requires at least three basis functions per feature")
    difference = np.diff(np.eye(basis_count), n=2, axis=0)
    scale_values = np.asarray(scale.scale_, dtype=float)
    if len(scale_values) != coefficient_count or (scale_values <= 0).any():
        raise ValueError("Spline feature scaling is incompatible with P-spline penalty")
    rows = np.zeros((feature_count * len(difference), coefficient_count), dtype=float)
    for feature_index in range(feature_count):
        start = feature_index * basis_count
        stop = start + basis_count
        row_start = feature_index * len(difference)
        row_stop = row_start + len(difference)
        rows[row_start:row_stop, start:stop] = difference / scale_values[start:stop]
    return rows


def _project_previous_response(
    previous_model: MatchupContextualModel,
    spline: SplineTransformer,
    scale: StandardScaler,
    features: pd.DataFrame,
    feature_set: str,
) -> np.ndarray:
    """Project one completed model's component functions onto a new basis."""

    if previous_model.feature_set != feature_set:
        raise ValueError("Temporal contextual projection requires the same feature set")
    feature_count = len(contextual_feature_columns(feature_set))
    coefficient_count = int(spline.n_features_out_)
    if coefficient_count % feature_count:
        raise ValueError("Spline feature layout is incompatible with temporal contextual prior")
    basis_count = coefficient_count // feature_count
    previous_spline = previous_model.pipeline.named_steps["spline"]
    previous_scale = previous_model.pipeline.named_steps["scale"]
    previous_ridge = previous_model.pipeline.named_steps["ridge"]
    previous_coefficients = np.asarray(previous_ridge.coef_, dtype=float)
    if len(previous_coefficients) % feature_count:
        raise ValueError("Previous contextual model has an incompatible spline layout")
    previous_basis_count = len(previous_coefficients) // feature_count
    if previous_basis_count != basis_count:
        raise ValueError("Previous contextual model has a different basis width")
    current_means = np.asarray(scale.mean_, dtype=float)
    current_scales = np.asarray(scale.scale_, dtype=float)
    previous_means = np.asarray(previous_scale.mean_, dtype=float)
    previous_scales = np.asarray(previous_scale.scale_, dtype=float)
    if (current_scales <= 0).any() or (previous_scales <= 0).any():
        raise ValueError("Spline feature scaling contains a zero deviation")

    projected = np.empty(coefficient_count, dtype=float)
    for feature_index, column in enumerate(contextual_feature_columns(feature_set)):
        values = features[column].to_numpy(dtype=float)
        grid = np.quantile(values, np.linspace(0.005, 0.995, 81))
        start = feature_index * basis_count
        stop = start + basis_count
        previous_grid = grid
        if isinstance(previous_model, BoundedMatchupContextualModel):
            previous_grid = np.clip(
                grid,
                -float(previous_model.relative_cap[column]),
                float(previous_model.relative_cap[column]),
            )
        previous_basis = _feature_basis(previous_spline, feature_index, previous_grid, feature_set)
        previous_response = (
            (previous_basis - previous_means[start:stop]) / previous_scales[start:stop]
        ) @ previous_coefficients[start:stop]
        current_basis = _feature_basis(spline, feature_index, grid, feature_set)
        current_design = (current_basis - current_means[start:stop]) / current_scales[start:stop]
        projected[start:stop] = np.linalg.lstsq(current_design, previous_response, rcond=None)[0]
    return projected


def _feature_basis(
    spline: SplineTransformer, feature_index: int, values: np.ndarray, feature_set: str
) -> np.ndarray:
    """Transform one feature while preserving SplineTransformer extrapolation."""

    columns = contextual_feature_columns(feature_set)
    frame = pd.DataFrame(0.0, index=range(len(values)), columns=columns)
    frame.iloc[:, feature_index] = values
    transformed = np.asarray(spline.transform(frame), dtype=float)
    feature_count = len(columns)
    if transformed.shape[1] % feature_count:
        raise ValueError("Spline feature layout is incompatible with temporal contextual prior")
    basis_count = transformed.shape[1] // feature_count
    start = feature_index * basis_count
    return transformed[:, start : start + basis_count]


def isolated_feature_component(
    model: MatchupContextualModel, feature_index: int, values: np.ndarray
) -> np.ndarray:
    """Evaluate one exact orientation-symmetrized original-feature component.

    The contextual pipeline is additive after its per-feature spline expansion.
    This extracts one original feature's spline-Ridge contribution directly,
    without materializing the other feature columns. It is used for response
    diagnostics and does not alter the fitted total contextual prediction.
    """

    feature_count = len(contextual_feature_columns(model.feature_set))
    if not 0 <= feature_index < feature_count:
        raise ValueError(f"Contextual feature index is out of range: {feature_index}")
    spline = model.pipeline.named_steps["spline"]
    scale = model.pipeline.named_steps["scale"]
    ridge = model.pipeline.named_steps["ridge"]
    coefficients = np.asarray(ridge.coef_, dtype=float)
    if len(coefficients) % feature_count:
        raise ValueError("Spline feature layout is incompatible with contextual attribution")
    basis_count = len(coefficients) // feature_count
    start = feature_index * basis_count
    stop = start + basis_count
    mean = np.asarray(scale.mean_, dtype=float)[start:stop]
    deviation = np.asarray(scale.scale_, dtype=float)[start:stop]
    if (deviation == 0).any():
        raise ValueError("Spline feature scaling contains a zero deviation")
    input_values = np.asarray(values, dtype=float)
    if isinstance(model, BoundedMatchupContextualModel):
        column = contextual_feature_columns(model.feature_set)[feature_index]
        input_values = np.clip(
            input_values,
            -float(model.relative_cap[column]),
            float(model.relative_cap[column]),
        )
    forward_basis = _feature_basis(spline, feature_index, input_values, model.feature_set)
    reverse_basis = _feature_basis(spline, feature_index, -input_values, model.feature_set)
    forward = ((forward_basis - mean) / deviation) @ coefficients[start:stop]
    reverse = ((reverse_basis - mean) / deviation) @ coefficients[start:stop]
    return 0.5 * (forward - reverse)


def _relative_features(home: pd.DataFrame, away: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    home = _validated_side_features(home, feature_set)
    away = _validated_side_features(away, feature_set)
    if len(home) != len(away):
        raise ValueError("Contextual side-feature frames must align")
    return pd.DataFrame(
        {
            f"home_minus_away_{column}": (
                home[column].to_numpy(dtype=float) - away[column].to_numpy(dtype=float)
            )
            for column in side_context_feature_columns(feature_set)
        },
        columns=contextual_feature_columns(feature_set),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile, cumulative / cumulative[-1], values[order]))


def _side_support_bounds(
    home: pd.DataFrame, away: pd.DataFrame, weights: np.ndarray, feature_set: str
) -> tuple[pd.Series, pd.Series]:
    columns = side_context_feature_columns(feature_set)
    combined = pd.concat([home.loc[:, columns], away.loc[:, columns]], ignore_index=True)
    combined_weights = np.concatenate([weights, weights])
    lower = pd.Series(
        [
            _weighted_quantile(combined[column].to_numpy(float), combined_weights, 0.05)
            for column in columns
        ],
        index=columns,
    )
    upper = pd.Series(
        [
            _weighted_quantile(combined[column].to_numpy(float), combined_weights, 0.95)
            for column in columns
        ],
        index=columns,
    )
    return lower, upper


def _symmetric_relative_caps(
    relative: pd.DataFrame, weights: np.ndarray, feature_set: str
) -> pd.Series:
    columns = contextual_feature_columns(feature_set)
    return pd.Series(
        [
            _weighted_quantile(
                np.abs(relative[column].to_numpy(float)),
                weights,
                0.95,
            )
            for column in columns
        ],
        index=columns,
    )


def _clip_side_features(
    features: pd.DataFrame, lower: pd.Series, upper: pd.Series, feature_set: str
) -> pd.DataFrame:
    return _validated_side_features(features, feature_set).clip(lower, upper, axis=1)


def _clip_relative_features(
    features: pd.DataFrame, cap: pd.Series, feature_set: str
) -> pd.DataFrame:
    columns = contextual_feature_columns(feature_set)
    return features.loc[:, columns].clip(-cap, cap, axis=1)


def _validated_side_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    columns = side_context_feature_columns(feature_set)
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Contextual side features missing columns: {sorted(missing)}")
    output = frame.loc[:, columns].copy()
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError("Contextual side features must be finite")
    return output


def _reference_distribution(
    home: pd.DataFrame,
    away: pd.DataFrame,
    weights: np.ndarray,
    feature_set: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    sides = pd.concat([home, away], ignore_index=True)
    side_weights = np.concatenate([weights, weights])
    grouped = (
        sides.assign(_reference_weight=side_weights)
        .groupby(
            list(side_context_feature_columns(feature_set)),
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(_reference_weight=("_reference_weight", "sum"))
    )
    reference_weights = grouped.pop("_reference_weight").to_numpy(dtype=float)
    reference_weights /= reference_weights.sum()
    return grouped.reset_index(drop=True), reference_weights
