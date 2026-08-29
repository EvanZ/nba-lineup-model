"""Foundation for a forward offense/defense Split NAIL model.

Split NAIL uses one row for each team's offensive possession segment. Every
player profile and retained non-additive unit feature has an offense-side and a
defense-side coefficient. Basketball labels do not impose zero constraints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES,
    lineup_side_context_features,
)
from nba_lineup_model.models.baselines import PriorPrecisionRidgeLineupModel

SPLIT_NAIL_ADDITIVE_FEATURES = LINEAR_NAIL_V1211_BASKETBALL_ADDITIVE_FEATURES
SPLIT_NAIL_NONADDITIVE_FEATURES = ("top_two_assists", "usage_concentration")
DEFAULT_SPECIALIZATION_RELATIVE_PRECISION = 4.0


@dataclass(frozen=True)
class SplitNailDesign:
    """Two scoring-side observations per stint under the Split NAIL contract."""

    player_ids: tuple[int, ...]
    additive_features: tuple[str, ...]
    nonadditive_features: tuple[str, ...]
    feature_scales: tuple[float, ...]
    includes_back_to_back: bool
    back_to_back_scale: float | None
    features: sparse.csr_matrix
    target: np.ndarray
    weights: np.ndarray
    game_ids: np.ndarray
    home_offense: np.ndarray

    @property
    def player_count(self) -> int:
        return len(self.player_ids)

    @property
    def coefficient_count(self) -> int:
        return self.features.shape[1]

    def player_column(self, player_id: int, *, side: str) -> int:
        """Return the stable player coefficient column for offense or defense."""

        index = self.player_ids.index(int(player_id))
        if side == "offense":
            return index
        if side == "defense":
            return self.player_count + index
        raise ValueError(f"Unknown Split NAIL side: {side}")

    def feature_column(self, feature: str, *, side: str) -> int:
        """Return the stable profile/context coefficient column for one side."""

        all_features = (*self.additive_features, *self.nonadditive_features)
        try:
            index = all_features.index(feature)
        except ValueError as error:
            raise ValueError(f"Unknown Split NAIL feature: {feature}") from error
        offset = 2 * self.player_count
        if side == "offense":
            return offset + index
        if side == "defense":
            return offset + len(all_features) + index
        raise ValueError(f"Unknown Split NAIL side: {side}")

    def feature_scale(self, feature: str) -> float:
        """Return the possession-weighted scale applied to a raw feature."""

        all_features = (*self.additive_features, *self.nonadditive_features)
        try:
            index = all_features.index(feature)
        except ValueError as error:
            raise ValueError(f"Unknown Split NAIL feature: {feature}") from error
        return self.feature_scales[index]

    def back_to_back_column(self, *, side: str) -> int:
        """Return the optional schedule-control column for one scoring side."""

        if not self.includes_back_to_back:
            raise ValueError("Split NAIL design does not include back-to-back controls")
        offset = 2 * self.player_count + 2 * len(
            (*self.additive_features, *self.nonadditive_features)
        )
        if side == "offense":
            return offset
        if side == "defense":
            return offset + 1
        raise ValueError(f"Unknown Split NAIL side: {side}")

    def home_court_column(self, *, side: str) -> int:
        """Return the home-court column for the offensive or defensive pathway.

        The home offensive effect raises the home scoring row.  The home
        defensive effect lowers the away scoring row.  Keeping the two paths
        separate lets the data decide whether home advantage is principally
        offensive, defensive, or both.
        """

        offset = 2 * self.player_count + 2 * len(
            (*self.additive_features, *self.nonadditive_features)
        ) + 2 * int(self.includes_back_to_back)
        if side == "offense":
            return offset
        if side == "defense":
            return offset + 1
        raise ValueError(f"Unknown Split NAIL home-court side: {side}")


@dataclass(frozen=True)
class SplitNailSeasonFit:
    """One constrained regularized Split NAIL fit around its forward scalar state."""

    design: SplitNailDesign
    model: PriorPrecisionRidgeLineupModel
    player_coefficients: pd.DataFrame
    feature_coefficients: pd.DataFrame
    schedule_coefficients: pd.DataFrame


@dataclass(frozen=True)
class ConstrainedSplitNailModel:
    """Ridge fit in paired mean/specialization coordinates.

    ``coef_`` is exposed in the original offense/defense column basis so the
    scoring code remains explicit, but the penalty is imposed on each pair's
    combined effect and its O/D specialization instead of on independent O/D
    coefficients.
    """

    model: PriorPrecisionRidgeLineupModel
    transform: sparse.csr_matrix

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self.transform @ self.model.coef_, dtype=float).reshape(-1)

    @property
    def intercept_(self) -> float:
        return self.model.intercept_


@dataclass(frozen=True)
class FixedTotalSplitNailModel:
    """O/D allocation around coefficients whose total coordinate is locked.

    The scalar production model supplies the paired ``mean`` coordinate
    (offense plus defense) for every player, feature, and schedule control.
    This model fits only the paired specialization coordinate (offense minus
    defense).  It is therefore an attribution model: reconstructing a net
    margin from both scoring sides returns the locked scalar prediction.
    """

    model: PriorPrecisionRidgeLineupModel
    transform: sparse.csr_matrix
    fixed_mean: np.ndarray

    @property
    def coef_(self) -> np.ndarray:
        pair_count = len(self.fixed_mean)
        mean_design = self.transform[:, :pair_count]
        specialization_design = self.transform[:, pair_count:]
        coefficients = mean_design @ self.fixed_mean
        coefficients = coefficients + specialization_design @ self.model.coef_
        return np.asarray(coefficients, dtype=float).reshape(-1)

    @property
    def intercept_(self) -> float:
        return self.model.intercept_


def build_split_nail_design(
    stints: pd.DataFrame,
    profiles: pd.DataFrame,
) -> SplitNailDesign:
    """Build the O/D design using strictly lagged NAIL player profiles."""

    home_features = lineup_side_context_features(
        stints["home_player_ids"].tolist(),
        profiles,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    )
    away_features = lineup_side_context_features(
        stints["away_player_ids"].tolist(),
        profiles,
        feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    )
    return build_split_nail_design_from_side_features(stints, home_features, away_features)


def build_split_nail_design_from_side_features(
    stints: pd.DataFrame,
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
) -> SplitNailDesign:
    """Build the sparse O/D scoring design from validated lineup feature tables."""

    required_stint_columns = {
        "game_id",
        "home_player_ids",
        "away_player_ids",
        "points_home",
        "points_away",
        "home_offensive_possessions",
        "away_offensive_possessions",
    }
    missing_stints = required_stint_columns - set(stints)
    if missing_stints:
        raise ValueError(f"Split NAIL stints missing: {sorted(missing_stints)}")
    feature_columns = (*SPLIT_NAIL_ADDITIVE_FEATURES, *SPLIT_NAIL_NONADDITIVE_FEATURES)
    for name, frame in (("home", home_features), ("away", away_features)):
        missing_features = set(feature_columns) - set(frame)
        if missing_features:
            raise ValueError(
                f"Split NAIL {name} features missing: {sorted(missing_features)}"
            )
        if len(frame) != len(stints):
            raise ValueError(f"Split NAIL {name} features do not align with stints")

    player_ids = tuple(
        sorted(
            {
                int(player_id)
                for lineup_column in ("home_player_ids", "away_player_ids")
                for lineup in stints[lineup_column]
                for player_id in lineup
            }
        )
    )
    if not player_ids:
        raise ValueError("Split NAIL requires at least one player")
    player_columns = {player_id: index for index, player_id in enumerate(player_ids)}
    player_count = len(player_ids)
    feature_count = len(feature_columns)
    includes_back_to_back = {"home_back_to_back", "away_back_to_back"}.issubset(stints)
    if includes_back_to_back:
        schedule_values = stints.loc[:, ["home_back_to_back", "away_back_to_back"]].to_numpy(
            dtype=float
        )
        if not np.isfinite(schedule_values).all():
            raise ValueError("Split NAIL back-to-back controls must be finite")
    feature_scales = _feature_scales(
        stints,
        home_features,
        away_features,
        feature_columns,
    )
    back_to_back_scale = _back_to_back_scale(stints) if includes_back_to_back else None
    row_count = 0
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    target: list[float] = []
    weights: list[float] = []
    game_ids: list[str] = []
    home_offense: list[bool] = []

    for stint_index, stint in stints.reset_index(drop=True).iterrows():
        scoring_sides = (
            (
                True,
                stint["home_player_ids"],
                stint["away_player_ids"],
                float(stint["points_home"]),
                float(stint["home_offensive_possessions"]),
                home_features.iloc[stint_index],
                away_features.iloc[stint_index],
            ),
            (
                False,
                stint["away_player_ids"],
                stint["home_player_ids"],
                float(stint["points_away"]),
                float(stint["away_offensive_possessions"]),
                away_features.iloc[stint_index],
                home_features.iloc[stint_index],
            ),
        )
        for (
            is_home,
            offense_players,
            defense_players,
            points,
            possessions,
            own_features,
            opponent_features,
        ) in scoring_sides:
            if possessions <= 0.0:
                continue
            for player_id in offense_players:
                rows.append(row_count)
                columns.append(player_columns[int(player_id)])
                values.append(1.0)
            for player_id in defense_players:
                rows.append(row_count)
                columns.append(player_count + player_columns[int(player_id)])
                values.append(-1.0)
            for feature_index, feature in enumerate(feature_columns):
                scale = feature_scales[feature_index]
                own_value = float(own_features[feature]) / scale
                opponent_value = float(opponent_features[feature]) / scale
                if own_value:
                    rows.append(row_count)
                    columns.append(2 * player_count + feature_index)
                    values.append(own_value)
                if opponent_value:
                    rows.append(row_count)
                    columns.append(2 * player_count + feature_count + feature_index)
                    values.append(-opponent_value)
            if includes_back_to_back:
                own_back_to_back = float(
                    stint["home_back_to_back"] if is_home else stint["away_back_to_back"]
                ) / float(back_to_back_scale)
                opponent_back_to_back = float(
                    stint["away_back_to_back"] if is_home else stint["home_back_to_back"]
                ) / float(back_to_back_scale)
                if own_back_to_back:
                    rows.append(row_count)
                    columns.append(2 * player_count + 2 * feature_count)
                    values.append(own_back_to_back)
                if opponent_back_to_back:
                    rows.append(row_count)
                    columns.append(2 * player_count + 2 * feature_count + 1)
                    values.append(-opponent_back_to_back)
            if is_home:
                rows.append(row_count)
                columns.append(
                    2 * player_count + 2 * feature_count + 2 * int(includes_back_to_back)
                )
                values.append(1.0)
            else:
                rows.append(row_count)
                columns.append(
                    2 * player_count
                    + 2 * feature_count
                    + 2 * int(includes_back_to_back)
                    + 1
                )
                values.append(-1.0)
            target.append(100.0 * points / possessions)
            weights.append(possessions)
            game_ids.append(str(stint["game_id"]))
            home_offense.append(is_home)
            row_count += 1

    if not target:
        raise ValueError("Split NAIL has no positive-exposure offense rows")
    features = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(
            row_count,
            2 * player_count + 2 * feature_count + 2 * int(includes_back_to_back) + 2,
        ),
        dtype=np.float64,
    ).tocsr()
    return SplitNailDesign(
        player_ids=player_ids,
        additive_features=SPLIT_NAIL_ADDITIVE_FEATURES,
        nonadditive_features=SPLIT_NAIL_NONADDITIVE_FEATURES,
        feature_scales=tuple(feature_scales),
        includes_back_to_back=includes_back_to_back,
        back_to_back_scale=back_to_back_scale,
        features=features,
        target=np.asarray(target, dtype=float),
        weights=np.asarray(weights, dtype=float),
        game_ids=np.asarray(game_ids, dtype=str),
        home_offense=np.asarray(home_offense, dtype=bool),
    )


def split_nail_prior_vector(
    design: SplitNailDesign,
    scalar_priors: Mapping[int, float],
    carried_side_differences: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Split a forward scalar prior while carrying only O-minus-D specialization.

    The offense and defense prior always sum to the existing scalar NAIL prior.
    A player's prior O/D difference is carried from the preceding Split NAIL
    state; players without prior side state receive the neutral equal split.
    """

    prior = np.zeros(design.coefficient_count, dtype=float)
    differences = carried_side_differences or {}
    for index, player_id in enumerate(design.player_ids):
        total = float(scalar_priors.get(player_id, 0.0))
        difference = float(differences.get(player_id, 0.0))
        prior[index] = 0.5 * (total + difference)
        prior[design.player_count + index] = 0.5 * (total - difference)
    return prior


def fit_split_nail_season(
    design: SplitNailDesign,
    scalar_priors: Mapping[int, float],
    *,
    carried_side_differences: Mapping[int, float] | None = None,
    regularization: float,
    feature_relative_precision: float = 1.0,
    schedule_relative_precision: float | None = None,
    player_specialization_relative_precision: float = DEFAULT_SPECIALIZATION_RELATIVE_PRECISION,
) -> SplitNailSeasonFit:
    """Fit one constrained O/D Split NAIL season around a forward scalar state."""

    if feature_relative_precision <= 0:
        raise ValueError("Split NAIL feature relative precision must be positive")
    resolved_schedule_precision = (
        feature_relative_precision
        if schedule_relative_precision is None
        else schedule_relative_precision
    )
    if resolved_schedule_precision <= 0:
        raise ValueError("Split NAIL schedule relative precision must be positive")
    if player_specialization_relative_precision <= 0:
        raise ValueError("Split NAIL player specialization relative precision must be positive")
    raw_prior = split_nail_prior_vector(design, scalar_priors, carried_side_differences)
    pairs = _raw_coefficient_pairs(design)
    transform = _paired_mean_specialization_transform(design.coefficient_count, pairs)
    parameter_prior = _paired_raw_to_parameters(raw_prior, pairs)
    precision = _constrained_precision(
        design,
        feature_relative_precision=feature_relative_precision,
        schedule_relative_precision=resolved_schedule_precision,
        player_specialization_relative_precision=player_specialization_relative_precision,
    )
    fitted = PriorPrecisionRidgeLineupModel(regularization).fit(
        design.features @ transform,
        design.target,
        design.weights,
        parameter_prior,
        precision,
    )
    model = ConstrainedSplitNailModel(fitted, transform)
    coefficients = model.coef_
    player_rows = []
    for index, player_id in enumerate(design.player_ids):
        offense = float(coefficients[index])
        defense = float(coefficients[design.player_count + index])
        player_rows.append(
            {
                "player_id": player_id,
                "offense_base_rating": offense,
                "defense_base_rating": defense,
                "net_base_rating": offense + defense,
                "scalar_prior": float(scalar_priors.get(player_id, 0.0)),
                "carried_side_difference": float(
                    (carried_side_differences or {}).get(player_id, 0.0)
                ),
            }
        )
    feature_rows = []
    for feature in (*design.additive_features, *design.nonadditive_features):
        feature_rows.append(
            {
                "feature": feature,
                "feature_layer": (
                    "additive_profile"
                    if feature in design.additive_features
                    else "nonadditive_lineup"
                ),
                "offense_coefficient": float(
                    coefficients[design.feature_column(feature, side="offense")]
                ),
                "defense_coefficient": float(
                    coefficients[design.feature_column(feature, side="defense")]
                ),
                "feature_standard_deviation": design.feature_scale(feature),
                "offense_raw_coefficient": float(
                    coefficients[design.feature_column(feature, side="offense")]
                    / design.feature_scale(feature)
                ),
                "defense_raw_coefficient": float(
                    coefficients[design.feature_column(feature, side="defense")]
                    / design.feature_scale(feature)
                ),
            }
        )
    schedule_rows = []
    if design.includes_back_to_back:
        if design.back_to_back_scale is None:
            raise ValueError("Split NAIL B2B scale is missing")
        offense = float(coefficients[design.back_to_back_column(side="offense")])
        defense = float(coefficients[design.back_to_back_column(side="defense")])
        schedule_rows.append(
            {
                "schedule_control": "back_to_back",
                "offense_coefficient": offense,
                "defense_coefficient": defense,
                "feature_standard_deviation": design.back_to_back_scale,
                "offense_raw_coefficient": offense / design.back_to_back_scale,
                "defense_raw_coefficient": defense / design.back_to_back_scale,
                "net_raw_coefficient": (offense + defense) / design.back_to_back_scale,
            }
        )
    return SplitNailSeasonFit(
        design=design,
        model=model,
        player_coefficients=pd.DataFrame(player_rows),
        feature_coefficients=pd.DataFrame(feature_rows),
        schedule_coefficients=pd.DataFrame(schedule_rows),
    )


def fit_fixed_total_split_nail(
    design: SplitNailDesign,
    fixed_mean: np.ndarray,
    *,
    specialization_prior: np.ndarray | None = None,
    regularization: float,
    player_specialization_relative_precision: float = 1.0,
    context_specialization_relative_precision: float = 1.0,
) -> FixedTotalSplitNailModel:
    """Fit only O/D allocation around a locked scalar NAIL state.

    ``fixed_mean`` follows the paired-coordinate order returned by
    :func:`_raw_coefficient_pairs`: players, every profile/context feature,
    back-to-back when present, then home court.  Its values are production
    totals, not half-ratings.  For example, a locked player value ``R`` is
    represented by raw coefficients ``R / 2`` on both offense and defense
    before a learned specialization is applied.
    """

    pair_count = design.coefficient_count // 2
    means = np.asarray(fixed_mean, dtype=float)
    if means.shape != (pair_count,) or not np.isfinite(means).all():
        raise ValueError("Fixed total coordinates must be finite and match O/D pairs")
    if player_specialization_relative_precision <= 0:
        raise ValueError("Player specialization precision must be positive")
    if context_specialization_relative_precision <= 0:
        raise ValueError("Context specialization precision must be positive")

    transform = _paired_mean_specialization_transform(
        design.coefficient_count, _raw_coefficient_pairs(design)
    )
    mean_design = design.features @ transform[:, :pair_count]
    specialization_design = design.features @ transform[:, pair_count:]
    prior = (
        np.zeros(pair_count, dtype=float)
        if specialization_prior is None
        else np.asarray(specialization_prior, dtype=float)
    )
    if prior.shape != (pair_count,) or not np.isfinite(prior).all():
        raise ValueError("Specialization prior must be finite and match O/D pairs")
    precision = np.full(
        pair_count, context_specialization_relative_precision, dtype=float
    )
    precision[: design.player_count] = player_specialization_relative_precision
    fitted = PriorPrecisionRidgeLineupModel(regularization).fit(
        specialization_design,
        design.target - np.asarray(mean_design @ means, dtype=float).reshape(-1),
        design.weights,
        prior,
        precision,
    )
    return FixedTotalSplitNailModel(
        model=fitted,
        transform=transform,
        fixed_mean=means,
    )


def _raw_coefficient_pairs(design: SplitNailDesign) -> list[tuple[int, int]]:
    """Return raw offense/defense column pairs in their physical order."""

    pairs = [
        (index, design.player_count + index) for index in range(design.player_count)
    ]
    feature_start = 2 * design.player_count
    feature_count = len((*design.additive_features, *design.nonadditive_features))
    pairs.extend(
        (feature_start + index, feature_start + feature_count + index)
        for index in range(feature_count)
    )
    cursor = feature_start + 2 * feature_count
    if design.includes_back_to_back:
        pairs.append((cursor, cursor + 1))
        cursor += 2
    pairs.append((cursor, cursor + 1))
    if 2 * len(pairs) != design.coefficient_count:
        raise AssertionError("Split NAIL coefficient pairs do not cover the raw design")
    return pairs


def _paired_mean_specialization_transform(
    column_count: int,
    pairs: list[tuple[int, int]],
) -> sparse.csr_matrix:
    """Map paired mean/specialization parameters to raw O/D coefficients."""

    row_values: list[int] = []
    col_values: list[int] = []
    data: list[float] = []
    pair_count = len(pairs)
    for pair, (raw_offense, raw_defense) in enumerate(pairs):
        mean, specialization = pair, pair + pair_count
        row_values.extend([raw_offense, raw_offense, raw_defense, raw_defense])
        col_values.extend([mean, specialization, mean, specialization])
        data.extend([0.5, 0.5, 0.5, -0.5])
    return sparse.coo_matrix(
        (data, (row_values, col_values)), shape=(column_count, column_count)
    ).tocsr()


def _paired_raw_to_parameters(raw: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    offense = np.asarray([raw[offense] for offense, _ in pairs], dtype=float)
    defense = np.asarray([raw[defense] for _, defense in pairs], dtype=float)
    return np.concatenate([offense + defense, offense - defense])


def _constrained_precision(
    design: SplitNailDesign,
    *,
    feature_relative_precision: float,
    schedule_relative_precision: float,
    player_specialization_relative_precision: float = DEFAULT_SPECIALIZATION_RELATIVE_PRECISION,
) -> np.ndarray:
    """Set stronger shrinkage on every O/D specialization coordinate."""

    pair_count = design.coefficient_count // 2
    mean = np.ones(pair_count, dtype=float)
    specialization = np.full(pair_count, DEFAULT_SPECIALIZATION_RELATIVE_PRECISION, dtype=float)
    specialization[: design.player_count] = player_specialization_relative_precision
    feature_count = len((*design.additive_features, *design.nonadditive_features))
    feature_start = design.player_count
    mean[feature_start : feature_start + feature_count] = feature_relative_precision
    specialization[feature_start : feature_start + feature_count] = (
        DEFAULT_SPECIALIZATION_RELATIVE_PRECISION * feature_relative_precision
    )
    cursor = feature_start + feature_count
    if design.includes_back_to_back:
        mean[cursor] = schedule_relative_precision
        specialization[cursor] = (
            DEFAULT_SPECIALIZATION_RELATIVE_PRECISION * schedule_relative_precision
        )
        cursor += 1
    # The final pair is home-offense/home-defense. Its difference is a
    # low-information scoring-environment specialization, so shrink it harder.
    mean[cursor] = 1.0
    specialization[cursor] = DEFAULT_SPECIALIZATION_RELATIVE_PRECISION
    return np.concatenate([mean, specialization])


def _feature_scales(
    stints: pd.DataFrame,
    home_features: pd.DataFrame,
    away_features: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    """Return possession-weighted scales without densifying the sparse design."""

    home_weights = stints["home_offensive_possessions"].to_numpy(dtype=float)
    away_weights = stints["away_offensive_possessions"].to_numpy(dtype=float)
    values = pd.concat(
        [
            home_features.loc[:, feature_columns],
            away_features.loc[:, feature_columns],
        ],
        ignore_index=True,
    ).to_numpy(dtype=float)
    weights = np.concatenate([home_weights, away_weights])
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average(np.square(values - mean), axis=0, weights=weights)
    scales = np.sqrt(variance)
    return np.where(scales > np.finfo(float).eps, scales, 1.0)


def _back_to_back_scale(stints: pd.DataFrame) -> float:
    """Return one possession-weighted B2B scale shared across scoring sides."""

    values = np.concatenate(
        [
            stints["home_back_to_back"].to_numpy(dtype=float),
            stints["away_back_to_back"].to_numpy(dtype=float),
        ]
    )
    weights = np.concatenate(
        [
            stints["home_offensive_possessions"].to_numpy(dtype=float),
            stints["away_offensive_possessions"].to_numpy(dtype=float),
        ]
    )
    mean = float(np.average(values, weights=weights))
    variance = float(np.average(np.square(values - mean), weights=weights))
    return max(float(np.sqrt(variance)), 1.0)
