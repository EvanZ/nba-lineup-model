"""Causal home-court state for the current-season NAIL updater.

The production updater historically treated home-court advantage as a fixed
pooled context coefficient.  This module supplies the replacement candidate:
home court is one additional coefficient in the same possession-weighted
normal equations as the player residuals.  The player prior remains centered
on the frozen preseason NAIL forecast and the HCA prior is selected using only
completed seasons preceding the target season.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba_lineup_model.modeling.in_season_nail import GameContribution
from nba_lineup_model.models.baselines import signed_entity_matrix, vocabulary_mapping

CANDIDATE_ID = "causal_hca_replacement_candidate_v2"
TRANSITION_METHODS = ("last_season", "rolling_3", "expanding_mean", "ar1")


@dataclass(frozen=True)
class HomeCourtPriorSelection:
    """Forward-only prior and precision selected from completed history."""

    candidate_id: str
    target_season: str
    source_through_season: str
    transition_method: str
    prior_mean: float
    prior_variance: float
    observation_variance_per_100: float
    prior_equivalent_possessions: float
    forward_validation_rows: int
    forward_rmse: float
    forward_mae: float
    method_metrics_json: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class JointHomeCourtContribution:
    """One game's augmented player-plus-HCA normal-equation contribution."""

    base: GameContribution
    xtwx: sparse.csr_matrix
    xtwy: np.ndarray
    exposure: np.ndarray

    def __post_init__(self) -> None:
        player_count = len(self.base.player_ids)
        matrix = sparse.csr_matrix(self.xtwx, dtype=float)
        score = np.asarray(self.xtwy, dtype=float)
        exposure = np.asarray(self.exposure, dtype=float)
        if matrix.shape != (player_count + 1, player_count + 1):
            raise ValueError("Joint HCA matrix must add exactly one coefficient")
        if score.shape != (player_count + 1,) or exposure.shape != (player_count + 1,):
            raise ValueError("Joint HCA vectors must match the augmented matrix")
        if matrix.nnz and not np.isfinite(matrix.data).all():
            raise ValueError("Joint HCA matrix must be finite")
        if not np.isfinite(score).all() or not np.isfinite(exposure).all():
            raise ValueError("Joint HCA vectors must be finite")
        if not np.allclose(matrix.toarray(), matrix.toarray().T, atol=1e-10):
            raise ValueError("Joint HCA matrix must be symmetric")
        object.__setattr__(self, "xtwx", matrix)
        object.__setattr__(self, "xtwy", score)
        object.__setattr__(self, "exposure", exposure)


@dataclass(frozen=True)
class JointHomeCourtSolve:
    """Posterior player adjustments and the current HCA coefficient."""

    player_adjustments: np.ndarray
    player_standard_errors: np.ndarray
    home_court: float
    home_court_standard_error: float
    home_court_information: float
    home_court_prior_weight: float
    home_court_evidence_weight: float


def select_home_court_prior(
    context_metadata: pd.DataFrame,
    possession_predictions: pd.DataFrame,
    *,
    target_season: str,
    minimum_training_seasons: int = 8,
) -> HomeCourtPriorSelection:
    """Select a transition rule by expanding-window completed-season RMSE.

    No target-season outcome is accepted by this function.  The selected
    transition forecast variance is translated to normal-equation precision
    using the completed possession-level residual variance in per-100 units.
    """

    required = {"season", "context_home_intercept"}
    missing = required - set(context_metadata)
    if missing:
        raise ValueError(f"Context metadata missing columns: {sorted(missing)}")
    if "residual_offense_margin" not in possession_predictions:
        raise ValueError("Possession predictions lack residual_offense_margin")
    target_start = int(str(target_season)[:4])
    history = context_metadata.loc[:, ["season", "context_home_intercept"]].copy()
    history["season_start"] = history["season"].astype(str).str[:4].astype(int)
    history["context_home_intercept"] = pd.to_numeric(
        history["context_home_intercept"], errors="coerce"
    )
    history = (
        history.loc[
            history["season_start"].lt(target_start) & history["context_home_intercept"].notna()
        ]
        .sort_values("season_start", kind="stable")
        .reset_index(drop=True)
    )
    if len(history) <= minimum_training_seasons:
        raise ValueError("Not enough completed HCA seasons for forward validation")
    if history["season"].duplicated().any():
        raise ValueError("HCA history must contain one row per season")

    values = history["context_home_intercept"].to_numpy(dtype=float)
    validation: dict[str, list[float]] = {method: [] for method in TRANSITION_METHODS}
    actuals: list[float] = []
    for index in range(minimum_training_seasons, len(values)):
        training = values[:index]
        actuals.append(float(values[index]))
        forecasts = _transition_forecasts(training)
        for method in TRANSITION_METHODS:
            validation[method].append(float(forecasts[method]))
    actual = np.asarray(actuals, dtype=float)
    metrics: dict[str, dict[str, float]] = {}
    for method in TRANSITION_METHODS:
        error = np.asarray(validation[method], dtype=float) - actual
        metrics[method] = {
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    selected = min(
        TRANSITION_METHODS,
        key=lambda method: (metrics[method]["rmse"], metrics[method]["mae"], method),
    )
    selected_error = np.asarray(validation[selected], dtype=float) - actual
    prior_variance = float(np.mean(np.square(selected_error)))
    if not np.isfinite(prior_variance) or prior_variance <= 0:
        raise ValueError("HCA transition variance must be finite and positive")

    residual = (
        pd.to_numeric(possession_predictions["residual_offense_margin"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if not len(residual):
        raise ValueError("Completed possession residuals are empty")
    observation_variance = float(10000.0 * np.mean(np.square(residual)))
    prior_precision = observation_variance / prior_variance
    if not np.isfinite(prior_precision) or prior_precision <= 0:
        raise ValueError("HCA prior precision must be finite and positive")
    prior_mean = float(_transition_forecasts(values)[selected])
    return HomeCourtPriorSelection(
        candidate_id=CANDIDATE_ID,
        target_season=str(target_season),
        source_through_season=str(history.iloc[-1]["season"]),
        transition_method=selected,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        observation_variance_per_100=observation_variance,
        prior_equivalent_possessions=prior_precision,
        forward_validation_rows=len(actual),
        forward_rmse=metrics[selected]["rmse"],
        forward_mae=metrics[selected]["mae"],
        method_metrics_json=json.dumps(metrics, sort_keys=True),
    )


def augment_game_contribution_with_home_court(
    base: GameContribution,
    stints: pd.DataFrame,
    *,
    preseason_nail: Mapping[int, float] | Sequence[float] | np.ndarray,
    fixed_neutral_context_offset: float | Sequence[float] | np.ndarray,
) -> JointHomeCourtContribution:
    """Append a home-court column to a neutral-context game contribution."""

    vocabulary = base.player_ids
    player_to_column = vocabulary_mapping(vocabulary)
    design = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_to_column,
        multiple=True,
    )
    weights = stints["possessions"].to_numpy(dtype=float)
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    prior = _prior_vector(preseason_nail, vocabulary)
    context = _offset_vector(fixed_neutral_context_offset, len(stints))
    residual = target - context - np.asarray(design @ prior).reshape(-1)
    cross = np.asarray(design.T @ weights, dtype=float).reshape(-1)
    hca_information = float(weights.sum())
    hca_score = float(np.dot(weights, residual))
    augmented_gram = sparse.bmat(
        [
            [base.xtwx, sparse.csr_matrix(cross[:, None])],
            [sparse.csr_matrix(cross[None, :]), sparse.csr_matrix([[hca_information]])],
        ],
        format="csr",
    )
    return JointHomeCourtContribution(
        base=base,
        xtwx=augmented_gram,
        xtwy=np.concatenate([base.xtwy, np.asarray([hca_score])]),
        exposure=np.concatenate([base.on_court_possessions, np.asarray([hca_information])]),
    )


def solve_joint_home_court_state(
    gram: sparse.csr_matrix,
    score: np.ndarray,
    *,
    player_count: int,
    player_prior_equivalent_possessions: float,
    home_court_prior_mean: float,
    home_court_prior_equivalent_possessions: float,
) -> JointHomeCourtSolve:
    """Solve the joint player-residual and absolute-HCA posterior."""

    expected = player_count + 1
    matrix = sparse.csr_matrix(gram, dtype=float)
    vector = np.asarray(score, dtype=float)
    if matrix.shape != (expected, expected) or vector.shape != (expected,):
        raise ValueError("Joint HCA state dimensions do not match the player vocabulary")
    prior_precision = np.concatenate(
        [
            np.full(player_count, float(player_prior_equivalent_possessions)),
            np.asarray([float(home_court_prior_equivalent_possessions)]),
        ]
    )
    if not np.isfinite(prior_precision).all() or (prior_precision <= 0).any():
        raise ValueError("Joint HCA prior precision must be finite and positive")
    rhs = vector.copy()
    rhs[-1] += prior_precision[-1] * float(home_court_prior_mean)
    precision = matrix + sparse.diags(prior_precision, format="csr")
    posterior = np.asarray(spsolve(precision, rhs), dtype=float)
    if not np.isfinite(posterior).all():
        raise ValueError("Joint HCA solve produced non-finite values")
    variance = np.diag(np.linalg.inv(precision.toarray()))
    hca_information = float(matrix[-1, -1])
    hca_evidence_weight = hca_information / (
        hca_information + float(home_court_prior_equivalent_possessions)
    )
    return JointHomeCourtSolve(
        player_adjustments=posterior[:player_count],
        player_standard_errors=np.sqrt(np.maximum(variance[:player_count], 0.0)),
        home_court=float(posterior[-1]),
        home_court_standard_error=float(np.sqrt(max(variance[-1], 0.0))),
        home_court_information=hca_information,
        home_court_prior_weight=1.0 - hca_evidence_weight,
        home_court_evidence_weight=hca_evidence_weight,
    )


def _transition_forecasts(values: np.ndarray) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("HCA transition forecast requires at least two seasons")
    x = values[:-1]
    y = values[1:]
    variance = float(np.var(x))
    phi = 0.0 if variance <= 0 else float(np.cov(x, y, bias=True)[0, 1] / variance)
    intercept = float(np.mean(y) - phi * np.mean(x))
    return {
        "last_season": float(values[-1]),
        "rolling_3": float(np.mean(values[-3:])),
        "expanding_mean": float(np.mean(values)),
        "ar1": float(intercept + phi * values[-1]),
    }


def _prior_vector(
    values: Mapping[int, float] | Sequence[float] | np.ndarray,
    player_ids: tuple[int, ...],
) -> np.ndarray:
    if isinstance(values, Mapping):
        missing = set(player_ids) - {int(player_id) for player_id in values}
        if missing:
            raise ValueError(f"Preseason NAIL is missing player IDs: {sorted(missing)}")
        vector = np.asarray([values[player_id] for player_id in player_ids], dtype=float)
    else:
        vector = np.asarray(values, dtype=float)
    if vector.shape != (len(player_ids),) or not np.isfinite(vector).all():
        raise ValueError("Preseason NAIL vector must be finite and complete")
    return vector


def _offset_vector(
    values: float | Sequence[float] | np.ndarray,
    length: int,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 0:
        vector = np.full(length, float(vector), dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError("Neutral context offset must be finite and match the stints")
    return vector
