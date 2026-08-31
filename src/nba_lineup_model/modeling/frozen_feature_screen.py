"""Fast frozen residual screens for proposed lineup-context features.

This is a diagnostic gate, not a model-selection substitute. It scores a new
feature against strictly frozen residuals from the promoted NAIL release before
the project spends time on a recursive refit. Continuous candidates are shown
in deciles; discrete candidates are shown by their observed values.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_lineup_model.modeling.contextual_features import (
    CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
    lineup_side_context_features,
)
from nba_lineup_model.modeling.contextual_prior import _lineup_effects
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_contextual_rapm import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CURATED_DIR,
    DEFAULT_PANEL_PATH,
    _previous_season,
)
from nba_lineup_model.modeling.forward_exposure_gated_rapm import _latest_run
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.modeling.teammate_continuity import (
    build_teammate_pair_exposure,
    teammate_continuity_side_feature,
)

DEFAULT_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_MODEL_ARTIFACT_SEASON = "2025-26"
DEFAULT_OUTPUT_ROOT = Path("artifacts/models/analysis/frozen_feature_screen")
DEFAULT_CHART_ROOT = Path("docs/assets/images/frozen-feature-screens")

SideFeature = Callable[[Sequence[Sequence[int]], pd.DataFrame], np.ndarray]

ROLE_REDUNDANCY_PROFILE_COLUMNS = (
    "usage_per_100",
    "assists_per_100",
    "three_pa_per_100",
    "unassisted_rim_makes_per_100",
    "offensive_rebounds_per_100",
    "free_throw_attempts_per_100",
)


@dataclass(frozen=True)
class FeatureCandidate:
    """One candidate whose home-minus-away contrast is screened."""

    name: str
    label: str
    description: str
    side_feature: SideFeature | None = None
    production_context_column: str | None = None
    uses_prior_pair_exposure: bool = False
    source_profile_scale_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrozenFeatureScreenRun:
    run_dir: Path
    chart_path: Path
    feature: FeatureCandidate


def _median_three_pm_per_100(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    values = profiles.set_index("player_id")["three_pm_per_100"]
    return np.asarray([float(values.loc[list(lineup)].median()) for lineup in lineups], dtype=float)


def _max_blocks_per_100(lineups: Sequence[Sequence[int]], profiles: pd.DataFrame) -> np.ndarray:
    """Return each unit's strongest strictly lagged rim-protection profile."""

    values = profiles.set_index("player_id")["blocks_per_100"]
    return np.asarray([float(values.loc[list(lineup)].max()) for lineup in lineups], dtype=float)


def _creation_spacing_alignment(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    """Weight each player's shooting by their share of unit usage events.

    The coordinate is non-additive because every player's weight depends on
    the other four players' usage rates. Both inputs are strictly prior-season,
    shrinkage-adjusted per-100 profile values.
    """

    values = profiles.set_index("player_id")
    output: list[float] = []
    for lineup in lineups:
        unit = values.loc[list(lineup)]
        usage = unit["usage_per_100"].to_numpy(dtype=float)
        shooting = unit["three_pm_per_100"].to_numpy(dtype=float)
        total_usage = float(usage.sum())
        output.append(float(np.dot(usage, shooting) / total_usage) if total_usage > 0 else 0.0)
    return np.asarray(output, dtype=float)


def _secondary_creator_floor(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    """Return each unit's second-highest shrunken assist profile."""

    values = profiles.set_index("player_id")["assists_per_100"]
    return np.asarray(
        [
            float(np.partition(values.loc[list(lineup)].to_numpy(dtype=float), -2)[-2])
            for lineup in lineups
        ],
        dtype=float,
    )


def _rim_pressure_by_spacing_floor(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    """Combine unit rim pressure with the average of its two weakest spacers."""

    values = profiles.set_index("player_id")
    output: list[float] = []
    for lineup in lineups:
        unit = values.loc[list(lineup)]
        rim_pressure = float(unit["unassisted_rim_makes_per_100"].sum())
        shooting = unit["three_pm_per_100"].to_numpy(dtype=float)
        spacing_floor = float(np.partition(shooting, 1)[:2].mean())
        output.append(rim_pressure * spacing_floor)
    return np.asarray(output, dtype=float)


def _defensive_anchor_by_perimeter_pressure(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    """Combine each unit's best rim protector with its other perimeter disruptors."""

    values = profiles.set_index("player_id")
    output: list[float] = []
    for lineup in lineups:
        unit = values.loc[list(lineup)]
        blocks = unit["blocks_per_100"].to_numpy(dtype=float)
        steals = unit["steals_per_100"].to_numpy(dtype=float)
        anchor_index = int(np.argmax(blocks))
        output.append(float(blocks[anchor_index] * (steals.sum() - steals[anchor_index])))
    return np.asarray(output, dtype=float)


def _offensive_role_redundancy(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    """Return mean pairwise cosine similarity among normalized role vectors."""

    values = profiles.set_index("player_id")
    output: list[float] = []
    for lineup in lineups:
        vectors = values.loc[list(lineup), list(ROLE_REDUNDANCY_PROFILE_COLUMNS)].to_numpy(
            dtype=float
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0.0)
        similarity = normalized @ normalized.T
        output.append(float(similarity[np.triu_indices(len(lineup), k=1)].mean()))
    return np.asarray(output, dtype=float)


def _lead_secondary_usage_gap(
    lineups: Sequence[Sequence[int]], profiles: pd.DataFrame
) -> np.ndarray:
    """Return the difference between a unit's highest and second-highest USG%."""

    values = profiles.set_index("player_id")["usage_pct"]
    return np.asarray(
        [
            float(
                np.partition(values.loc[list(lineup)].to_numpy(dtype=float), -2)[-1]
                - np.partition(values.loc[list(lineup)].to_numpy(dtype=float), -2)[-2]
            )
            for lineup in lineups
        ],
        dtype=float,
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Return a finite weighted quantile for positive weights."""

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    return float(
        np.interp(quantile, np.cumsum(ordered_weights) / ordered_weights.sum(), ordered_values)
    )


def _source_profile_scales(
    source_profiles: pd.DataFrame,
    source_panel: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, float]:
    """Compute source-season possession-weighted 90th-percentile profile scales."""

    weights = source_panel.loc[:, ["player_id", "rapm_possessions"]].copy()
    merged = source_profiles.merge(weights, on="player_id", how="inner", validate="one_to_one")
    if merged.empty or not merged["rapm_possessions"].gt(0).all():
        raise ValueError("Role-profile scaling requires positive source-season exposure")
    exposure = merged["rapm_possessions"].to_numpy(dtype=float)
    return {
        column: _weighted_quantile(merged[column].to_numpy(dtype=float), exposure, 0.90)
        for column in columns
    }


def _apply_profile_scales(profiles: pd.DataFrame, scales: Mapping[str, float]) -> pd.DataFrame:
    """Return a copy of profiles with selected coordinates source-season scaled."""

    output = profiles.copy()
    for column, scale in scales.items():
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Role-profile scale for {column} must be positive and finite")
        output[column] = output[column].to_numpy(dtype=float) / scale
    return output


def _production_nonadditive_side_feature(
    column: str,
) -> SideFeature:
    """Return one retained production unit feature without redefining it."""

    def evaluate(lineups: Sequence[Sequence[int]], profiles: pd.DataFrame) -> np.ndarray:
        values = lineup_side_context_features(
            lineups,
            profiles,
            feature_set=CONTEXT_FEATURE_SET_NAIL_V1211_STANDARD_USAGE,
        )
        return values[column].to_numpy(dtype=float)

    return evaluate


FEATURE_CANDIDATES: dict[str, FeatureCandidate] = {
    "median_three_pm_per_100": FeatureCandidate(
        name="median_three_pm_per_100",
        label="Median lineup 3PM per 100",
        description=(
            "The third-ranked player's prior-season, shrinkage-adjusted made threes "
            "per 100 possessions within a five-man unit."
        ),
        side_feature=_median_three_pm_per_100,
    ),
    "rim_protection_ceiling": FeatureCandidate(
        name="rim_protection_ceiling",
        label="Rim-protection ceiling (max BLK / 100)",
        description=(
            "The maximum prior-season, shrinkage-adjusted block rate per 100 possessions "
            "among the five players in a unit. Low values represent lineups lacking a "
            "credible rim-protection presence."
        ),
        side_feature=_max_blocks_per_100,
    ),
    "creation_spacing_alignment": FeatureCandidate(
        name="creation_spacing_alignment",
        label="Creation-spacing alignment",
        description=(
            "Usage-event-share-weighted prior-season, shrinkage-adjusted three-point "
            "makes per 100. High values mean the unit's likely creators are also its "
            "more credible three-point shooters."
        ),
        side_feature=_creation_spacing_alignment,
    ),
    "secondary_creator_floor": FeatureCandidate(
        name="secondary_creator_floor",
        label="Secondary-creator floor",
        description=(
            "The second-highest prior-season, shrinkage-adjusted assist rate per 100 "
            "within a five-man unit. High values indicate a credible second creator."
        ),
        side_feature=_secondary_creator_floor,
    ),
    "rim_pressure_by_spacing_floor": FeatureCandidate(
        name="rim_pressure_by_spacing_floor",
        label="Rim pressure by spacing floor",
        description=(
            "The unit's summed prior-season, shrinkage-adjusted unassisted rim makes "
            "per 100 multiplied by the mean 3PM per 100 of its two weakest spacers."
        ),
        side_feature=_rim_pressure_by_spacing_floor,
    ),
    "defensive_anchor_by_perimeter_pressure": FeatureCandidate(
        name="defensive_anchor_by_perimeter_pressure",
        label="Defensive anchor by perimeter pressure",
        description=(
            "The highest prior-season, shrinkage-adjusted block rate per 100 in a unit "
            "multiplied by the summed steal rates per 100 of the other four players."
        ),
        side_feature=_defensive_anchor_by_perimeter_pressure,
    ),
    "offensive_role_redundancy": FeatureCandidate(
        name="offensive_role_redundancy",
        label="Offensive role redundancy",
        description=(
            "Mean pairwise cosine similarity of five source-season-scaled, normalized "
            "offensive role vectors: usage, assists, 3PA, unassisted rim makes, offensive "
            "rebounds, and free-throw attempts per 100."
        ),
        side_feature=_offensive_role_redundancy,
        source_profile_scale_columns=ROLE_REDUNDANCY_PROFILE_COLUMNS,
    ),
    "lead_secondary_usage_gap": FeatureCandidate(
        name="lead_secondary_usage_gap",
        label="Lead-secondary usage gap",
        description=(
            "The difference between the highest and second-highest prior-season, "
            "shrinkage-adjusted conventional USG% profiles in a five-man unit."
        ),
        side_feature=_lead_secondary_usage_gap,
    ),
    "usage_concentration": FeatureCandidate(
        name="usage_concentration",
        label="Usage concentration",
        description=(
            "The retained production term: the combined share of a five-man unit's "
            "two highest conventional USG% values."
        ),
        side_feature=_production_nonadditive_side_feature("usage_concentration"),
        production_context_column="usage_concentration",
    ),
    "top_two_assists": FeatureCandidate(
        name="top_two_assists",
        label="Top-two assists per 100",
        description=(
            "The retained production term: the sum of a five-man unit's two highest "
            "prior-season, shrinkage-adjusted assist rates per 100 possessions."
        ),
        side_feature=_production_nonadditive_side_feature("top_two_assists"),
        production_context_column="top_two_assists",
    ),
    "prior_teammate_continuity": FeatureCandidate(
        name="prior_teammate_continuity",
        label="Prior-season teammate continuity",
        description=(
            "The mean log(1 + shared possessions) over a unit's ten teammate pairs, "
            "using only same-unit exposure from the immediately preceding regular season. "
            "Pairs without prior shared possessions contribute zero."
        ),
        uses_prior_pair_exposure=True,
    ),
}


def available_feature_candidates() -> tuple[str, ...]:
    """Return the stable CLI names of registered candidate features."""

    return tuple(FEATURE_CANDIDATES)


def weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Return a finite possession-weighted Pearson correlation."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    weights = np.asarray(weight, dtype=float)
    if len(x_values) != len(y_values) or len(x_values) != len(weights):
        raise ValueError("Weighted correlation inputs must align")
    if len(x_values) == 0 or (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError("Weighted correlation requires positive, finite weights")
    x_centered = x_values - np.average(x_values, weights=weights)
    y_centered = y_values - np.average(y_values, weights=weights)
    denominator = np.sqrt(
        np.average(np.square(x_centered), weights=weights)
        * np.average(np.square(y_centered), weights=weights)
    )
    if denominator == 0.0:
        return 0.0
    return float(np.average(x_centered * y_centered, weights=weights) / denominator)


def summarize_feature_bins(
    frame: pd.DataFrame,
    *,
    feature_column: str = "feature_edge",
    residual_column: str = "frozen_residual_net_rating",
    weight_column: str = "possessions",
    max_bins: int = 10,
) -> pd.DataFrame:
    """Summarize residuals in deciles or natural discrete feature groups.

    Decile membership is based on deterministic rank so equal values never make
    ``qcut`` fail. Candidates with fewer than ``max_bins`` unique values retain
    their natural groups instead, which makes binary lineup indicators legible.
    The confidence interval is descriptive: a normal approximation from the
    effective possession-weighted sample size, not a game-clustered test.
    """

    required = {feature_column, residual_column, weight_column}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Feature screen frame lacks: {sorted(missing)}")
    values = frame.loc[:, [feature_column, residual_column, weight_column]].copy()
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("Feature screen has no finite values")
    values[weight_column] = values[weight_column].astype(float)
    if not values[weight_column].gt(0).all():
        raise ValueError("Feature screen weights must be positive")
    unique_count = int(values[feature_column].nunique())
    if unique_count >= max_bins:
        ranks = values[feature_column].rank(method="first")
        values["bin"] = pd.qcut(ranks, q=max_bins, labels=False).astype(int) + 1
        bin_kind = "decile"
    else:
        values["bin"] = values[feature_column]
        bin_kind = "discrete_value"

    rows: list[dict[str, float | int | str]] = []
    for bin_value, group in values.groupby("bin", sort=True):
        weights = group[weight_column].to_numpy(dtype=float)
        residual = group[residual_column].to_numpy(dtype=float)
        mean = float(np.average(residual, weights=weights))
        variance = float(np.average(np.square(residual - mean), weights=weights))
        effective_n = float(np.square(weights.sum()) / np.square(weights).sum())
        standard_error = float(np.sqrt(variance / effective_n)) if effective_n > 1 else 0.0
        rows.append(
            {
                "bin_kind": bin_kind,
                "bin": int(bin_value) if bin_kind == "decile" else float(bin_value),
                "stint_count": int(len(group)),
                "possession_count": float(weights.sum()),
                "feature_mean": float(
                    np.average(group[feature_column].to_numpy(dtype=float), weights=weights)
                ),
                "residual_mean": mean,
                "residual_ci_low": mean - 1.96 * standard_error,
                "residual_ci_high": mean + 1.96 * standard_error,
                "effective_stint_count": effective_n,
            }
        )
    return pd.DataFrame(rows)


def build_frozen_feature_screen(
    feature_name: str,
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    game_catalog_path: Path | str = Path("data/catalog/games.parquet"),
    model_artifact_season: str = DEFAULT_MODEL_ARTIFACT_SEASON,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    chart_path: Path | str | None = None,
) -> FrozenFeatureScreenRun:
    """Score one registered candidate against strictly frozen NAIL residuals."""

    if feature_name not in FEATURE_CANDIDATES:
        raise ValueError(
            f"Unknown candidate {feature_name!r}; choices: {sorted(FEATURE_CANDIDATES)}"
        )
    candidate = FEATURE_CANDIDATES[feature_name]
    target_seasons = tuple(sorted({str(season) for season in seasons}))
    if not target_seasons:
        raise ValueError("At least one target season is required")

    artifacts = Path(artifacts_dir)
    analytical = Path(analytical_dir)
    panel = pd.read_parquet(player_season_panel_path)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(target_seasons[-1])],
        through_season=target_seasons[-1],
        analytical_dir=analytical,
    )
    model_root = _latest_run(artifacts / MODEL_NAME / model_artifact_season)
    metadata = json.loads((model_root / "metadata.json").read_text())
    if metadata.get("model") != MODEL_NAME:
        raise ValueError("Feature screen requires the promoted NAIL-RAPM v1.2.1.3 artifact")
    priors = pd.read_parquet(model_root / "season_player_priors.parquet")
    coefficients = pd.read_parquet(model_root / "historical_player_coefficients.parquet")
    context_models = joblib.load(model_root / "season_context_models.joblib")
    schedule_models = joblib.load(model_root / "season_schedule_models.joblib")
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    profile_builder = partial(
        build_contextual_player_profiles,
        padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
        use_last_observed_profile=True,
    )

    screens: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    pair_exposure_frames: list[pd.DataFrame] = []
    for target in target_seasons:
        print(f"Screening {candidate.name}: {target}", flush=True)
        source = _previous_season(target)
        stints = read_rapm_stints(target, analytical_dir=analytical)
        participants = set().union(*stints["home_player_ids"], *stints["away_player_ids"])
        profiles = profile_builder(
            panel,
            target_season=target,
            target_player_ids=participants,
            analytical_dir=str(analytical),
            curated_dir=str(curated_dir),
            exposure_cohort=exposure_cohort,
        )
        candidate_profiles = profiles
        if candidate.source_profile_scale_columns:
            source_panel = panel.loc[
                panel["season"].astype(str).eq(source), ["player_id", "rapm_possessions"]
            ].copy()
            source_profiles = profile_builder(
                panel,
                target_season=target,
                target_player_ids=source_panel["player_id"].astype(int).tolist(),
                analytical_dir=str(analytical),
                curated_dir=str(curated_dir),
                exposure_cohort=exposure_cohort,
            )
            candidate_profiles = _apply_profile_scales(
                profiles,
                _source_profile_scales(
                    source_profiles,
                    source_panel,
                    candidate.source_profile_scale_columns,
                ),
            )
        target_priors = priors.loc[priors["season"].eq(target), ["player_id", "prior_rapm"]].rename(
            columns={"prior_rapm": "prior_rapm_mean"}
        )
        source_coefficients = coefficients.loc[
            coefficients["season"].eq(source), ["player_id", "rapm"]
        ]
        if target_priors.empty or source_coefficients.empty:
            raise ValueError(f"Promoted model lacks frozen state for {target}")
        context_model = context_models.get(source)
        schedule_model = schedule_models.get(source)
        if context_model is None or schedule_model is None:
            raise ValueError(f"Promoted model lacks context or schedule state for {source}")
        source_stints = read_rapm_stints(source, analytical_dir=analytical)
        source_intercept = _recover_home_intercept(source_stints, source_coefficients)
        prior_map = dict(
            zip(
                target_priors["player_id"].astype(int),
                target_priors["prior_rapm_mean"],
                strict=True,
            )
        )
        player_edge, unknown = _lineup_effects(stints, prior_map)
        home_context = lineup_side_context_features(
            stints["home_player_ids"].tolist(),
            profiles,
            feature_set=context_model.feature_set,
        )
        away_context = lineup_side_context_features(
            stints["away_player_ids"].tolist(),
            profiles,
            feature_set=context_model.feature_set,
        )
        context_edge = context_model.predict_side_pairs(home_context, away_context)
        baseline_contract = "full_production_prediction"
        if candidate.production_context_column is not None:
            ablated_home = home_context.copy()
            ablated_away = away_context.copy()
            ablated_home[candidate.production_context_column] = 0.0
            ablated_away[candidate.production_context_column] = 0.0
            context_edge = context_model.predict_side_pairs(ablated_home, ablated_away)
            baseline_contract = "leave_one_production_context_term_out"
        schedule_edge = schedule_model.predict_games(stints, schedule_features)
        if candidate.uses_prior_pair_exposure:
            pair_exposure = build_teammate_pair_exposure(source_stints)
            pair_exposure.insert(0, "target_season", target)
            pair_exposure.insert(0, "source_season", source)
            pair_exposure_frames.append(pair_exposure)
            home_feature = teammate_continuity_side_feature(
                stints["home_player_ids"].tolist(), pair_exposure
            )
            away_feature = teammate_continuity_side_feature(
                stints["away_player_ids"].tolist(), pair_exposure
            )
        elif candidate.production_context_column is None:
            if candidate.side_feature is None:
                raise ValueError(f"Candidate {candidate.name} lacks a side feature")
            home_feature = candidate.side_feature(
                stints["home_player_ids"].tolist(), candidate_profiles
            )
            away_feature = candidate.side_feature(
                stints["away_player_ids"].tolist(), candidate_profiles
            )
        else:
            home_feature = home_context[candidate.production_context_column].to_numpy(dtype=float)
            away_feature = away_context[candidate.production_context_column].to_numpy(dtype=float)
        predicted = player_edge + source_intercept + context_edge + schedule_edge
        screen = stints.loc[
            :,
            ["season", "game_id", "stint_index", "possessions", "target_home_net_rating"],
        ].copy()
        screen["source_season"] = source
        screen["baseline_contract"] = baseline_contract
        screen["frozen_prediction_net_rating"] = predicted
        screen["frozen_residual_net_rating"] = (
            screen["target_home_net_rating"].to_numpy(dtype=float) - predicted
        )
        screen["home_feature"] = home_feature
        screen["away_feature"] = away_feature
        screen["feature_edge"] = home_feature - away_feature
        screen["unknown_player_exposures"] = unknown
        bins = summarize_feature_bins(screen)
        bins.insert(0, "season", target)
        bins.insert(1, "source_season", source)
        bins.insert(2, "feature", candidate.name)
        summaries.append(bins)
        weights = screen["possessions"].to_numpy(dtype=float)
        feature_edge = screen["feature_edge"].to_numpy(dtype=float)
        residual = screen["frozen_residual_net_rating"].to_numpy(dtype=float)
        summaries.append(
            pd.DataFrame(
                [
                    {
                        "season": target,
                        "source_season": source,
                        "feature": candidate.name,
                        "bin_kind": "season_summary",
                        "bin": np.nan,
                        "stint_count": len(screen),
                        "possession_count": float(weights.sum()),
                        "feature_mean": float(np.average(feature_edge, weights=weights)),
                        "residual_mean": float(np.average(residual, weights=weights)),
                        "residual_ci_low": np.nan,
                        "residual_ci_high": np.nan,
                        "effective_stint_count": np.nan,
                        "weighted_correlation": weighted_correlation(
                            feature_edge, residual, weights
                        ),
                    }
                ]
            )
        )
        screens.append(screen)
        print(
            f"Completed {target}: {len(screen):,} stints, {int(weights.sum()):,} possessions",
            flush=True,
        )

    screen_frame = pd.concat(screens, ignore_index=True)
    bin_frame = pd.concat(summaries, ignore_index=True, sort=False)
    pooled_bins = summarize_feature_bins(screen_frame)
    pooled_bins.insert(0, "season", "pooled")
    pooled_bins.insert(1, "source_season", "multiple")
    pooled_bins.insert(2, "feature", candidate.name)
    pooled_weights = screen_frame["possessions"].to_numpy(dtype=float)
    pooled_bins = pd.concat(
        [
            pooled_bins,
            pd.DataFrame(
                [
                    {
                        "season": "pooled",
                        "source_season": "multiple",
                        "feature": candidate.name,
                        "bin_kind": "season_summary",
                        "bin": np.nan,
                        "stint_count": len(screen_frame),
                        "possession_count": float(pooled_weights.sum()),
                        "feature_mean": float(
                            np.average(screen_frame["feature_edge"], weights=pooled_weights)
                        ),
                        "residual_mean": float(
                            np.average(
                                screen_frame["frozen_residual_net_rating"], weights=pooled_weights
                            )
                        ),
                        "residual_ci_low": np.nan,
                        "residual_ci_high": np.nan,
                        "effective_stint_count": np.nan,
                        "weighted_correlation": weighted_correlation(
                            screen_frame["feature_edge"],
                            screen_frame["frozen_residual_net_rating"],
                            pooled_weights,
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    bin_frame = pd.concat([bin_frame, pooled_bins], ignore_index=True, sort=False)

    resolved_chart = (
        Path(chart_path)
        if chart_path is not None
        else DEFAULT_CHART_ROOT / f"{candidate.name}-residual-screen.svg"
    )
    resolved_chart.parent.mkdir(parents=True, exist_ok=True)
    _render_residual_screen(bin_frame, candidate, target_seasons, resolved_chart)
    root = Path(output_root)
    run_id = (
        f"frozen-feature-screen-{candidate.name}-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    run_dir = root / candidate.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    screen_frame.to_parquet(run_dir / "stint_residuals.parquet", index=False)
    bin_frame.to_parquet(run_dir / "residual_bins.parquet", index=False)
    if pair_exposure_frames:
        pd.concat(pair_exposure_frames, ignore_index=True).to_parquet(
            run_dir / "prior_pair_exposures.parquet", index=False
        )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "feature": candidate.name,
                "label": candidate.label,
                "description": candidate.description,
                "model": MODEL_NAME,
                "model_run_id": metadata["run_id"],
                "model_artifact_season": model_artifact_season,
                "target_seasons": list(target_seasons),
                "information_boundary": (
                    "prior-season padded player profiles, prior-season teammate pair exposure, "
                    "and the immediately prior completed NAIL state only; target-season "
                    "outcomes are evaluation-only"
                ),
                "baseline_contract": (
                    "The full promoted prediction for novel candidates; a leave-one-term-out "
                    "prediction for candidates already represented in the production context model."
                ),
                "unknown_player_prior_contract": (
                    "Exact promoted-model fallback: player IDs absent from the frozen prior "
                    "map receive a zero player edge; their exposure is retained in "
                    "stint_residuals.parquet for audit."
                ),
                "chart_path": str(resolved_chart),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    latest = root / candidate.name / "latest.json"
    latest.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    return FrozenFeatureScreenRun(run_dir=run_dir, chart_path=resolved_chart, feature=candidate)


def _render_residual_screen(
    bins: pd.DataFrame,
    candidate: FeatureCandidate,
    seasons: Sequence[str],
    chart_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(seasons), figsize=(5.2 * len(seasons), 4.4), sharey=True)
    if len(seasons) == 1:
        axes = [axes]
    for axis, season in zip(axes, seasons, strict=True):
        frame = bins.loc[
            (bins["season"].eq(season)) & (bins["bin_kind"].isin(["decile", "discrete_value"]))
        ].sort_values("bin", kind="stable")
        axis.axhline(0.0, color="#69736c", linewidth=1.0, linestyle=(0, (4, 4)))
        axis.errorbar(
            frame["feature_mean"],
            frame["residual_mean"],
            yerr=np.vstack(
                [
                    frame["residual_mean"] - frame["residual_ci_low"],
                    frame["residual_ci_high"] - frame["residual_mean"],
                ]
            ),
            color="#25644f",
            marker="o",
            linewidth=2.0,
            capsize=3,
        )
        summary = bins.loc[
            (bins["season"].eq(season)) & (bins["bin_kind"].eq("season_summary"))
        ].iloc[0]
        axis.set_title(f"{season}\nr = {summary['weighted_correlation']:+.3f}", weight="bold")
        axis.set_xlabel("Home-minus-away feature edge")
        axis.grid(axis="y", color="#d9d4ca", linewidth=0.8)
    axes[0].set_ylabel("Mean frozen residual net rating\n(observed minus promoted NAIL prediction)")
    figure.suptitle(
        f"Frozen residual screen: {candidate.label}",
        fontsize=15,
        fontweight="bold",
        x=0.5,
        y=1.03,
    )
    figure.text(
        0.5,
        -0.02,
        (
            "Points are possession-weighted deciles; error bars are descriptive "
            "95% normal-approximation intervals."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(chart_path, bbox_inches="tight", format="svg")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen a candidate lineup feature against frozen NAIL residuals"
    )
    parser.add_argument("feature", choices=available_feature_candidates())
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--model-artifact-season", default=DEFAULT_MODEL_ARTIFACT_SEASON)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--chart-path")
    args = parser.parse_args()
    run = build_frozen_feature_screen(
        args.feature,
        seasons=tuple(args.seasons),
        model_artifact_season=args.model_artifact_season,
        output_root=args.output_root,
        chart_path=args.chart_path,
    )
    print(f"Frozen feature screen: run={run.run_dir}; chart={run.chart_path}")


if __name__ == "__main__":
    main()
