"""Antisymmetric lineup context with portable-unit and matchup components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_V1,
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
