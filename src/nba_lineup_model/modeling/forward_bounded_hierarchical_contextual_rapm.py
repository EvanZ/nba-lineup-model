"""Bounded relative-context RAPM with P-spline temporal shrinkage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_lineup_model.modeling.contextual_features import contextual_feature_columns
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_CONTEXT_ALPHA,
    DEFAULT_TARGET_SEASON,
    ForwardContextualRapmRun,
    train_forward_contextual_rapm,
)
from nba_lineup_model.modeling.forward_hierarchical_pspline_contextual_rapm import (
    DEFAULT_CONTEXT_CURVATURE_ALPHA,
    DEFAULT_CONTEXT_TEMPORAL_ALPHA,
)
from nba_lineup_model.modeling.matchup_contextual import (
    _feature_basis,
    _second_difference_penalty,
)

MODEL_NAME = "forward_bounded_hierarchical_pspline_contextual_rapm"
RUN_PREFIX = "forward-bounded-hierarchical-pspline-contextual-rapm"
ARTIFACT_NAME = "forward_bounded_hierarchical_pspline_contextual_rapm"


@dataclass(frozen=True)
class BoundedRelativeContextModel:
    """Relative spline context with fitted central-support clipping bounds."""

    pipeline: Pipeline
    lower: pd.Series
    upper: pd.Series

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        clipped = features.loc[:, contextual_feature_columns()].clip(self.lower, self.upper, axis=1)
        return np.asarray(self.pipeline.predict(clipped), dtype=float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    return float(np.interp(quantile, np.cumsum(weights[order]) / weights.sum(), values[order]))


def _project_bounded_previous_response(
    previous: BoundedRelativeContextModel,
    spline: SplineTransformer,
    scale: StandardScaler,
    features: pd.DataFrame,
) -> np.ndarray:
    """Project the prior clipped function onto the current bounded basis."""

    columns = contextual_feature_columns()
    coefficient = np.asarray(previous.pipeline.named_steps["ridge"].coef_, dtype=float)
    old_scale = previous.pipeline.named_steps["scale"]
    old_spline = previous.pipeline.named_steps["spline"]
    basis_count = len(coefficient) // len(columns)
    output = np.empty_like(coefficient)
    for index, column in enumerate(columns):
        grid = np.quantile(features[column].to_numpy(float), np.linspace(0.005, 0.995, 81))
        prior_grid = np.clip(grid, previous.lower[column], previous.upper[column])
        start, stop = index * basis_count, (index + 1) * basis_count
        old_basis = _feature_basis(old_spline, index, prior_grid)
        target = (
            (old_basis - old_scale.mean_[start:stop]) / old_scale.scale_[start:stop]
        ) @ coefficient[start:stop]
        new_basis = _feature_basis(spline, index, grid)
        design = (new_basis - scale.mean_[start:stop]) / scale.scale_[start:stop]
        output[start:stop] = np.linalg.lstsq(design, target, rcond=None)[0]
    return output


def fit_bounded_hierarchical_context(
    frame: pd.DataFrame, alpha: float, previous: object | None
) -> BoundedRelativeContextModel:
    """Fit one completed-season bounded relative context state."""

    columns = contextual_feature_columns()
    raw = frame.loc[:, columns]
    weights = frame["possessions"].to_numpy(dtype=float)
    lower = pd.Series(
        [_weighted_quantile(raw[column].to_numpy(float), weights, 0.05) for column in columns],
        index=columns,
    )
    upper = pd.Series(
        [_weighted_quantile(raw[column].to_numpy(float), weights, 0.95) for column in columns],
        index=columns,
    )
    features = raw.clip(lower, upper, axis=1)
    spline = SplineTransformer(n_knots=4, degree=2, extrapolation="linear")
    basis = spline.fit_transform(features)
    scale = StandardScaler()
    design = scale.fit_transform(basis)
    rows = [design]
    targets = [frame["target_residual_net_rating"].to_numpy(dtype=float)]
    sample_weights = [weights]
    curvature = _second_difference_penalty(spline, scale)
    rows.append(np.sqrt(DEFAULT_CONTEXT_CURVATURE_ALPHA) * curvature)
    targets.append(np.zeros(len(curvature)))
    sample_weights.append(np.ones(len(curvature)))
    if previous is not None:
        prior = _project_bounded_previous_response(previous, spline, scale, features)
        identity = np.sqrt(DEFAULT_CONTEXT_TEMPORAL_ALPHA) * np.eye(design.shape[1])
        rows.append(identity)
        targets.append(np.sqrt(DEFAULT_CONTEXT_TEMPORAL_ALPHA) * prior)
        sample_weights.append(np.ones(len(identity)))
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
        np.vstack(rows), np.concatenate(targets), sample_weight=np.concatenate(sample_weights)
    )
    return BoundedRelativeContextModel(
        Pipeline([("spline", spline), ("scale", scale), ("ridge", ridge)]), lower, upper
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bounded hierarchical contextual RAPM")
    parser.add_argument("--through-season", default=DEFAULT_TARGET_SEASON)
    args = parser.parse_args()
    run: ForwardContextualRapmRun = train_forward_contextual_rapm(
        through_season=args.through_season,
        context_alpha=DEFAULT_CONTEXT_ALPHA,
        context_fit=fit_bounded_hierarchical_context,
        model_name=MODEL_NAME,
        run_prefix=RUN_PREFIX,
        artifact_name=ARTIFACT_NAME,
    )
    print(f"Forward bounded hierarchical contextual RAPM: run={run.run_dir}")


if __name__ == "__main__":
    main()
