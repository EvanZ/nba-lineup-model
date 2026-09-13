"""Forward-safe player allocation residuals on top of the production W0 squash.

The candidate is deliberately downstream of Forward Availability and Forward
Conditional Minutes (FCM).  It first recreates the exact production opening
roster allocation, measures each eligible player's signed allocation residual,
and carries only the immediately prior same-team residual into the next season.
No target-season minutes, availability, or roster outcome enters a forecast.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    DEFAULT_PLAYER_PANEL_PATH,
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    attach_cold_start_biographies,
    normalize_opening_roster_minutes,
)
from nba_lineup_model.rotation.forward_conditional_team_strength_roster_squash import (
    paired_team_bootstrap,
)
from nba_lineup_model.rotation.forward_nail_roster_squash import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INITIAL_ROTATION_SIZE,
    _season_allocation_targets,
    assert_full_regular_season_minute_coverage,
    build_forward_conditional_raw_weights,
    build_static_opening_roster_bios,
)
from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    _assert_distribution,
    _validate_game_minutes,
)

MODEL_NAME = "forward_allocation_residual_roster_squash"
MODEL_VERSION = "v0.1"
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_RESIDUAL_BETA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_PRIOR_MINUTES_GRID = (0.0, 240.0, 480.0, 960.0, 1_920.0)
DEFAULT_MIN_RESIDUAL_MINUTES = 240.0
_PAIRED_METRICS = (
    "allocation_total_variation",
    "brier_score",
    "player_share_mae",
    "player_share_mse",
    "actual_top_rotation_overlap",
)


@dataclass(frozen=True)
class AllocationResidualConfig:
    """Strength and exposure shrinkage for a prior allocation residual."""

    residual_beta: float
    prior_minutes: float


@dataclass(frozen=True)
class AllocationResidualInputs:
    """One frozen opening roster and its held-out, full-season label."""

    season: str
    opening_roster: pd.DataFrame
    static_roster_bios: pd.DataFrame
    target_game_minutes: pd.DataFrame


@dataclass(frozen=True)
class AllocationResidualRun:
    """Immutable output location for one residual-squash replay."""

    run_dir: Path
    run_id: str


def build_allocation_residual_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> AllocationResidualInputs:
    """Build frozen target inputs without reading target outcomes into features."""

    opening_roster = read_preseason_roster_candidates(season, curated_dir=curated_dir)
    return AllocationResidualInputs(
        season=season,
        opening_roster=opening_roster,
        static_roster_bios=build_static_opening_roster_bios(
            season, opening_roster, panel=panel, curated_dir=curated_dir
        ),
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
    )


def build_forward_allocation_residual_features(
    inputs: AllocationResidualInputs,
    *,
    residual_observations: pd.DataFrame,
    config: AllocationResidualConfig,
) -> pd.DataFrame:
    """Return the prior same-team allocation residual available before ``inputs.season``.

    An eligible residual must come from the immediately preceding completed
    season.  A team change intentionally receives a zero state in v0.1; the
    diagnostics persist both same-team and team-change correlations so that
    choice can be revisited from evidence rather than assumed portability.
    """

    if config.prior_minutes < 0.0:
        raise ValueError("Prior residual minutes must be non-negative")
    required = {
        "season_start_year",
        "team_id",
        "player_id",
        "allocation_residual",
        "actual_minutes",
    }
    missing = sorted(required - set(residual_observations))
    if missing:
        raise ValueError("Residual observations lack columns: " + ", ".join(missing))
    target_year = _season_year(inputs.season)
    roster = inputs.opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]].copy()
    roster["team_id"] = pd.to_numeric(roster["team_id"], errors="raise").astype(int)
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="raise").astype(int)
    prior = residual_observations.loc[
        pd.to_numeric(residual_observations["season_start_year"], errors="raise").eq(
            target_year - 1
        ),
        ["team_id", "player_id", "allocation_residual", "actual_minutes"],
    ].copy()
    prior["team_id"] = pd.to_numeric(prior["team_id"], errors="raise").astype(int)
    prior["player_id"] = pd.to_numeric(prior["player_id"], errors="raise").astype(int)
    if prior.duplicated("player_id").any():
        raise ValueError(f"Prior residual observations duplicate players for {inputs.season}")
    prior = prior.rename(
        columns={
            "team_id": "prior_team_id",
            "allocation_residual": "prior_allocation_residual",
            "actual_minutes": "prior_actual_minutes",
        }
    )
    output = roster.merge(prior, on="player_id", how="left", validate="one_to_one")
    output["same_team_return"] = output["prior_team_id"].eq(output["team_id"])
    output["prior_actual_minutes"] = pd.to_numeric(
        output["prior_actual_minutes"], errors="coerce"
    ).fillna(0.0)
    output["prior_allocation_residual"] = pd.to_numeric(
        output["prior_allocation_residual"], errors="coerce"
    ).fillna(0.0)
    output.loc[
        ~output["same_team_return"], ["prior_actual_minutes", "prior_allocation_residual"]
    ] = 0.0
    denominator = output["prior_actual_minutes"] + float(config.prior_minutes)
    output["residual_reliability"] = np.where(
        denominator > 0.0,
        output["prior_actual_minutes"] / denominator,
        0.0,
    )
    output["carried_allocation_residual"] = (
        output["prior_allocation_residual"] * output["residual_reliability"]
    )
    return output.loc[
        :,
        [
            "team_id",
            "team",
            "player_id",
            "player_name",
            "same_team_return",
            "prior_actual_minutes",
            "prior_allocation_residual",
            "residual_reliability",
            "carried_allocation_residual",
        ],
    ]


def apply_forward_allocation_residual_tilt(
    raw_predictions: pd.DataFrame,
    *,
    residual_features: pd.DataFrame,
    config: AllocationResidualConfig,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> pd.DataFrame:
    """Apply a frozen allocation-residual tilt before the production W0 gate."""

    required = {"team_id", "team", "player_id", "player_name", "raw_expected_total_minutes"}
    missing = sorted(required - set(raw_predictions))
    if missing:
        raise ValueError("Raw predictions lack required columns: " + ", ".join(missing))
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    if not np.isfinite(config.residual_beta):
        raise ValueError("Residual beta must be finite")
    feature_columns = {
        "team_id",
        "player_id",
        "same_team_return",
        "prior_actual_minutes",
        "prior_allocation_residual",
        "residual_reliability",
        "carried_allocation_residual",
    }
    missing_features = sorted(feature_columns - set(residual_features))
    if missing_features:
        raise ValueError("Residual features lack columns: " + ", ".join(missing_features))
    output = raw_predictions.copy()
    output["team_id"] = pd.to_numeric(output["team_id"], errors="raise").astype(int)
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    features = residual_features.loc[:, list(feature_columns)].copy()
    if features.duplicated(["team_id", "player_id"]).any():
        raise ValueError("Residual features duplicate roster players")
    output = output.merge(features, on=["team_id", "player_id"], how="left", validate="one_to_one")
    output["same_team_return"] = output["same_team_return"].fillna(False).astype(bool)
    for column in (
        "prior_actual_minutes",
        "prior_allocation_residual",
        "residual_reliability",
        "carried_allocation_residual",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["base_raw_expected_total_minutes"] = pd.to_numeric(
        output["raw_expected_total_minutes"], errors="raise"
    )
    output["allocation_residual_tilt"] = np.exp(
        np.clip(
            float(config.residual_beta)
            * output["carried_allocation_residual"].to_numpy(dtype=float),
            -30.0,
            30.0,
        )
    )
    output["tilted_raw_expected_total_minutes"] = (
        output["base_raw_expected_total_minutes"] * output["allocation_residual_tilt"]
    )
    selected_ids: set[int] = set()
    for _, team in output.groupby("team_id", sort=False):
        selected_ids.update(
            team.sort_values(
                ["tilted_raw_expected_total_minutes", "player_name", "player_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(initial_rotation_size)["player_id"]
            .astype(int)
            .tolist()
        )
    output["is_rotation_candidate"] = output["player_id"].isin(selected_ids)
    output["raw_expected_total_minutes"] = output["tilted_raw_expected_total_minutes"].where(
        output["is_rotation_candidate"], 0.0
    )
    return output


def evaluate_forward_allocation_residual_roster_squash(
    inputs: AllocationResidualInputs,
    *,
    raw_predictions: pd.DataFrame,
    residual_features: pd.DataFrame,
    config: AllocationResidualConfig,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one preseason allocation against the complete target season."""

    _validate_game_minutes(inputs.target_game_minutes, label="target")
    adjusted = apply_forward_allocation_residual_tilt(
        raw_predictions,
        residual_features=residual_features,
        config=config,
        initial_rotation_size=initial_rotation_size,
    )
    opening = inputs.opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]]
    normalized = normalize_opening_roster_minutes(adjusted, opening_roster=opening)
    predicted = opening.merge(
        adjusted,
        on=["team_id", "team", "player_id", "player_name"],
        how="left",
        validate="one_to_one",
    ).merge(
        normalized.loc[:, ["player_id", "projected_minute_share", "projected_total_minutes"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    targets = {
        target.team_id: target
        for target in _season_allocation_targets(inputs.target_game_minutes, team_games=None)
    }
    rostered_games = inputs.target_game_minutes.groupby(["team_id", "player_id"])[
        "game_id"
    ].nunique()
    team_games = inputs.target_game_minutes.groupby("team_id")["game_id"].nunique()
    player_team_counts = inputs.target_game_minutes.groupby("player_id")["team_id"].nunique()
    team_minutes = inputs.target_game_minutes.groupby("team_id")["minutes"].sum()
    records: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for team_id, team_predictions in predicted.groupby("team_id", sort=True):
        target = targets.get(int(team_id))
        if target is None:
            raise ValueError(f"Target allocation missing team {team_id}")
        predicted_map = team_predictions.set_index("player_id")["projected_minute_share"].to_dict()
        actual_map = target.player_shares
        player_ids = sorted(set(predicted_map) | set(actual_map))
        actual_values = np.asarray(
            [float(actual_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        predicted_values = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        _assert_distribution(actual_values, label=f"actual for {target.team}")
        _assert_distribution(predicted_values, label=f"prediction for {target.team}")
        absolute_error = np.abs(actual_values - predicted_values)
        squared_error = np.square(actual_values - predicted_values)
        actual_top_ids = {
            player_id
            for player_id, _share in sorted(
                actual_map.items(), key=lambda item: (-item[1], item[0])
            )[:initial_rotation_size]
        }
        selected_ids = set(
            team_predictions.loc[team_predictions["is_rotation_candidate"], "player_id"]
            .astype(int)
            .tolist()
        )
        zero_probability_active = (actual_values > 0.0) & (predicted_values <= 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            positive_actual = actual_values > 0.0
            cross_entropy = float(
                -(actual_values[positive_actual] * np.log(predicted_values[positive_actual])).sum()
            )
        metric_rows.append(
            {
                "season": inputs.season,
                "team_id": int(team_id),
                "team": target.team,
                "target_game_count": target.target_game_count,
                **asdict(config),
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "actual_top_rotation_overlap": float(
                    len(actual_top_ids & selected_ids) / max(len(actual_top_ids), 1)
                ),
                "selected_rotation_size": int(len(selected_ids)),
            }
        )
        player_rows = team_predictions.set_index("player_id")
        target_total_minutes = float(team_minutes.loc[int(team_id)])
        target_game_count = int(team_games.loc[int(team_id)])
        for player_id, actual_share, predicted_share in zip(
            player_ids, actual_values, predicted_values, strict=True
        ):
            source = player_rows.loc[player_id] if player_id in player_rows.index else None
            rostered_count = int(rostered_games.get((int(team_id), int(player_id)), 0))
            records.append(
                {
                    "season": inputs.season,
                    "season_start_year": _season_year(inputs.season),
                    "team_id": int(team_id),
                    "team": target.team,
                    "player_id": int(player_id),
                    "player_name": str(source["player_name"])
                    if source is not None
                    else str(player_id),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "actual_minutes": float(actual_share * target_total_minutes),
                    "target_team_minutes": target_total_minutes,
                    "target_rostered_games": rostered_count,
                    "target_team_games": target_game_count,
                    "is_trade_contaminated": bool(player_team_counts.get(int(player_id), 0) > 1),
                    "has_full_team_roster_coverage": bool(rostered_count == target_game_count),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "base_raw_expected_total_minutes": (
                        float(source["base_raw_expected_total_minutes"])
                        if source is not None
                        else 0.0
                    ),
                    "allocation_residual_tilt": (
                        float(source["allocation_residual_tilt"]) if source is not None else 1.0
                    ),
                    "carried_allocation_residual": (
                        float(source["carried_allocation_residual"]) if source is not None else 0.0
                    ),
                    "was_selected_rotation": (
                        bool(source["is_rotation_candidate"]) if source is not None else False
                    ),
                    "is_opening_roster_player": source is not None,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metric_rows)


def derive_allocation_residual_observations(
    control_predictions: pd.DataFrame,
    *,
    minimum_minutes: float = DEFAULT_MIN_RESIDUAL_MINUTES,
) -> pd.DataFrame:
    """Keep only stable, observed same-team-season residual labels.

    The residual is a log ratio of actual and production-predicted *team-minute
    shares*.  Team totals cancel, so overtime cannot create a spurious player
    residual.  Opening players who later move, waive, or fail to appear on the
    team roster for every game are excluded from the state label.
    """

    required = {
        "season",
        "season_start_year",
        "team_id",
        "player_id",
        "actual_minute_share",
        "predicted_minute_share",
        "actual_minutes",
        "is_trade_contaminated",
        "has_full_team_roster_coverage",
        "was_selected_rotation",
        "is_opening_roster_player",
    }
    missing = sorted(required - set(control_predictions))
    if missing:
        raise ValueError("Control predictions lack columns: " + ", ".join(missing))
    if minimum_minutes <= 0.0:
        raise ValueError("Minimum residual minutes must be positive")
    output = control_predictions.copy()
    output["actual_minutes"] = pd.to_numeric(output["actual_minutes"], errors="coerce")
    output["actual_minute_share"] = pd.to_numeric(output["actual_minute_share"], errors="coerce")
    output["predicted_minute_share"] = pd.to_numeric(
        output["predicted_minute_share"], errors="coerce"
    )
    eligible = (
        output["is_opening_roster_player"].astype(bool)
        & output["was_selected_rotation"].astype(bool)
        & ~output["is_trade_contaminated"].astype(bool)
        & output["has_full_team_roster_coverage"].astype(bool)
        & output["actual_minutes"].ge(float(minimum_minutes))
        & output["actual_minute_share"].gt(0.0)
        & output["predicted_minute_share"].gt(0.0)
    )
    output = output.loc[eligible].copy()
    output["allocation_residual"] = np.log(
        output["actual_minute_share"] / output["predicted_minute_share"]
    )
    result_columns = [
        "season",
        "season_start_year",
        "team_id",
        "team",
        "player_id",
        "player_name",
        "actual_minutes",
        "actual_minute_share",
        "predicted_minute_share",
        "allocation_residual",
    ]
    result = (
        output.loc[:, result_columns]
        .sort_values(["season_start_year", "team_id", "player_id"], kind="stable")
        .reset_index(drop=True)
    )
    if result.duplicated(["season_start_year", "player_id"]).any():
        raise ValueError("Stable allocation residual labels duplicate a player-season")
    return result


def summarize_allocation_residual_persistence(
    residual_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Describe one-year residual persistence for same-team and moved players."""

    current = residual_observations.loc[
        :, ["season", "season_start_year", "team_id", "player_id", "allocation_residual"]
    ].copy()
    prior = current.rename(
        columns={
            "season": "prior_season",
            "season_start_year": "prior_season_start_year",
            "team_id": "prior_team_id",
            "allocation_residual": "prior_allocation_residual",
        }
    )
    paired = current.merge(prior, on="player_id", how="inner", validate="many_to_many")
    paired = paired.loc[
        paired["season_start_year"].eq(paired["prior_season_start_year"] + 1)
    ].copy()
    paired["continuity"] = np.where(
        paired["team_id"].eq(paired["prior_team_id"]), "same_team", "team_change"
    )
    rows: list[dict[str, object]] = []
    for continuity, group in paired.groupby("continuity", sort=True):
        previous = group["prior_allocation_residual"].to_numpy(dtype=float)
        current_values = group["allocation_residual"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(previous, current_values)[0, 1]) if len(group) > 1 else np.nan
        )
        rows.append(
            {
                "continuity": continuity,
                "pair_count": int(len(group)),
                "pearson_correlation": correlation,
                "mean_prior_residual": float(previous.mean()),
                "mean_current_residual": float(current_values.mean()),
                "mean_absolute_prior_residual": float(np.abs(previous).mean()),
                "mean_absolute_current_residual": float(np.abs(current_values).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_allocation_residual_exposure(
    residual_observations: pd.DataFrame,
    *,
    bins: int = 4,
) -> pd.DataFrame:
    """Summarize residual dispersion across observed-minute exposure groups."""

    if bins < 2:
        raise ValueError("Residual exposure diagnostics require at least two bins")
    source = residual_observations.loc[:, ["actual_minutes", "allocation_residual"]].copy()
    source["actual_minutes"] = pd.to_numeric(source["actual_minutes"], errors="coerce")
    source["allocation_residual"] = pd.to_numeric(source["allocation_residual"], errors="coerce")
    source = source.dropna()
    if len(source) < bins:
        raise ValueError("Not enough residual observations for exposure diagnostics")
    source["exposure_bin"] = pd.qcut(source["actual_minutes"], q=bins, duplicates="drop")
    rows: list[dict[str, object]] = []
    for exposure_bin, group in source.groupby("exposure_bin", observed=True, sort=True):
        values = group["allocation_residual"].to_numpy(dtype=float)
        rows.append(
            {
                "exposure_bin": str(exposure_bin),
                "observation_count": int(len(group)),
                "minimum_actual_minutes": float(group["actual_minutes"].min()),
                "maximum_actual_minutes": float(group["actual_minutes"].max()),
                "mean_actual_minutes": float(group["actual_minutes"].mean()),
                "mean_allocation_residual": float(values.mean()),
                "mean_absolute_allocation_residual": float(np.abs(values).mean()),
                "allocation_residual_standard_deviation": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def select_allocation_residual_config(
    source_inputs: list[AllocationResidualInputs],
    *,
    raw_predictions_by_season: dict[str, pd.DataFrame],
    residual_observations: pd.DataFrame,
    residual_beta_grid: tuple[float, ...] = DEFAULT_RESIDUAL_BETA_GRID,
    prior_minutes_grid: tuple[float, ...] = DEFAULT_PRIOR_MINUTES_GRID,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> tuple[AllocationResidualConfig, pd.DataFrame]:
    """Tune only the residual carry against completed pre-frozen seasons."""

    rows: list[dict[str, object]] = []
    for beta, prior_minutes in product(
        sorted(set(residual_beta_grid)), sorted(set(prior_minutes_grid))
    ):
        config = AllocationResidualConfig(
            residual_beta=float(beta), prior_minutes=float(prior_minutes)
        )
        metrics = []
        for inputs in source_inputs:
            features = build_forward_allocation_residual_features(
                inputs, residual_observations=residual_observations, config=config
            )
            _predictions, season_metrics = evaluate_forward_allocation_residual_roster_squash(
                inputs,
                raw_predictions=raw_predictions_by_season[inputs.season],
                residual_features=features,
                config=config,
                initial_rotation_size=initial_rotation_size,
            )
            metrics.append(season_metrics)
        combined = pd.concat(metrics, ignore_index=True)
        rows.append(
            {
                **asdict(config),
                "source_season_count": len(source_inputs),
                "mean_allocation_total_variation": float(
                    combined["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(combined["brier_score"].mean()),
                "mean_player_share_mae": float(combined["player_share_mae"].mean()),
                "mean_cross_entropy": float(combined["cross_entropy"].mean()),
                "mean_actual_top_rotation_overlap": float(
                    combined["actual_top_rotation_overlap"].mean()
                ),
            }
        )
    grid = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "mean_allocation_total_variation",
                "mean_brier_score",
                "mean_player_share_mae",
                "residual_beta",
                "prior_minutes",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    winner = grid.loc[0]
    return (
        AllocationResidualConfig(
            residual_beta=float(winner["residual_beta"]),
            prior_minutes=float(winner["prior_minutes"]),
        ),
        grid,
    )


def run_forward_allocation_residual_roster_squash(
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> AllocationResidualRun:
    """Fit and freeze the same-team allocation-residual roster-squash candidate."""

    if not tuning_seasons or not frozen_seasons:
        raise ValueError("Both tuning and frozen seasons are required")
    if set(tuning_seasons) & set(frozen_seasons):
        raise ValueError("Tuning and frozen seasons must not overlap")
    all_scored_seasons = tuple(dict.fromkeys((*tuning_seasons, *frozen_seasons)))
    final_year = max(_season_year(season) for season in all_scored_seasons)
    source_seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2016, final_year + 1))
    history_seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, final_year))
    print(
        f"[Allocation residual] loading production state through {history_seasons[-1]}",
        flush=True,
    )
    availability_summary = attach_cold_start_biographies(
        build_availability_season_summary(history_seasons, curated_dir=curated_dir)
    )
    panel = pd.read_parquet(player_panel_path)
    inputs_by_season: dict[str, AllocationResidualInputs] = {}
    raw_predictions_by_season: dict[str, pd.DataFrame] = {}
    control_predictions: list[pd.DataFrame] = []
    control_metrics: list[pd.DataFrame] = []
    neutral_config = AllocationResidualConfig(residual_beta=0.0, prior_minutes=0.0)
    for season in source_seasons:
        print(f"[Allocation residual] building production allocation for {season}", flush=True)
        inputs = build_allocation_residual_inputs(season, panel=panel, curated_dir=curated_dir)
        assert_full_regular_season_minute_coverage(
            season, inputs.target_game_minutes, catalog_path=catalog_path
        )
        raw = build_forward_conditional_raw_weights(
            inputs, availability_summary=availability_summary
        )
        empty_features = inputs.opening_roster.loc[
            :, ["team_id", "team", "player_id", "player_name"]
        ].copy()
        empty_features["same_team_return"] = False
        empty_features["prior_actual_minutes"] = 0.0
        empty_features["prior_allocation_residual"] = 0.0
        empty_features["residual_reliability"] = 0.0
        empty_features["carried_allocation_residual"] = 0.0
        predictions, metrics = evaluate_forward_allocation_residual_roster_squash(
            inputs,
            raw_predictions=raw,
            residual_features=empty_features,
            config=neutral_config,
            initial_rotation_size=initial_rotation_size,
        )
        inputs_by_season[season] = inputs
        raw_predictions_by_season[season] = raw
        control_predictions.append(predictions)
        control_metrics.append(metrics)
    all_control_predictions = pd.concat(control_predictions, ignore_index=True)
    residual_observations = derive_allocation_residual_observations(all_control_predictions)
    persistence = summarize_allocation_residual_persistence(residual_observations)
    exposure_diagnostics = summarize_allocation_residual_exposure(residual_observations)
    print("[Allocation residual] tuning prior same-team residual carry", flush=True)
    selected_config, tuning_grid = select_allocation_residual_config(
        [inputs_by_season[season] for season in tuning_seasons],
        raw_predictions_by_season=raw_predictions_by_season,
        residual_observations=residual_observations,
        initial_rotation_size=initial_rotation_size,
    )
    print(f"[Allocation residual] selected {asdict(selected_config)}", flush=True)
    candidate_results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    production_results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    control_config = AllocationResidualConfig(
        residual_beta=0.0, prior_minutes=selected_config.prior_minutes
    )
    for season in frozen_seasons:
        inputs = inputs_by_season[season]
        features = build_forward_allocation_residual_features(
            inputs, residual_observations=residual_observations, config=selected_config
        )
        candidate_results.append(
            evaluate_forward_allocation_residual_roster_squash(
                inputs,
                raw_predictions=raw_predictions_by_season[season],
                residual_features=features,
                config=selected_config,
                initial_rotation_size=initial_rotation_size,
            )
        )
        control_features = build_forward_allocation_residual_features(
            inputs, residual_observations=residual_observations, config=control_config
        )
        production_results.append(
            evaluate_forward_allocation_residual_roster_squash(
                inputs,
                raw_predictions=raw_predictions_by_season[season],
                residual_features=control_features,
                config=control_config,
                initial_rotation_size=initial_rotation_size,
            )
        )
    candidate_predictions = pd.concat(
        [result[0] for result in candidate_results], ignore_index=True
    )
    candidate_metrics = pd.concat([result[1] for result in candidate_results], ignore_index=True)
    production_predictions = pd.concat(
        [result[0] for result in production_results], ignore_index=True
    )
    production_metrics = pd.concat([result[1] for result in production_results], ignore_index=True)
    comparison = candidate_metrics.loc[:, ["season", "team_id", *_PAIRED_METRICS]].merge(
        production_metrics.loc[:, ["season", "team_id", *_PAIRED_METRICS]],
        on=["season", "team_id"],
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    for metric in _PAIRED_METRICS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_candidate"] - comparison[f"{metric}_control"]
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"forward-allocation-residual-roster-squash-{timestamp}-{uuid4().hex[:7]}"
    output_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    all_control_predictions.to_parquet(
        output_dir / "production_history_predictions.parquet", index=False
    )
    residual_observations.to_parquet(
        output_dir / "allocation_residual_observations.parquet", index=False
    )
    persistence.to_parquet(output_dir / "residual_persistence.parquet", index=False)
    exposure_diagnostics.to_parquet(
        output_dir / "residual_exposure_diagnostics.parquet", index=False
    )
    tuning_grid.to_parquet(output_dir / "tuning_grid.parquet", index=False)
    candidate_predictions.to_parquet(output_dir / "frozen_predictions.parquet", index=False)
    candidate_metrics.to_parquet(output_dir / "frozen_team_metrics.parquet", index=False)
    production_predictions.to_parquet(
        output_dir / "frozen_control_predictions.parquet", index=False
    )
    production_metrics.to_parquet(output_dir / "frozen_control_team_metrics.parquet", index=False)
    comparison.to_parquet(output_dir / "frozen_comparison.parquet", index=False)
    paired_team_bootstrap(comparison).to_parquet(
        output_dir / "paired_team_bootstrap.parquet", index=False
    )
    _summarize_metrics(candidate_metrics, frozen_seasons=frozen_seasons).to_parquet(
        output_dir / "frozen_summary.parquet", index=False
    )
    _summarize_metrics(production_metrics, frozen_seasons=frozen_seasons).to_parquet(
        output_dir / "frozen_control_summary.parquet", index=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "evaluation_target": "all regular-season games for each team",
                "initial_rotation_size": initial_rotation_size,
                "selected_config": asdict(selected_config),
                "control_config": asdict(control_config),
                "availability_config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                "conditional_minutes_config": asdict(PROMOTED_CONDITIONAL_MINUTES_CONFIG),
                "cold_start_conditional_minutes_config": asdict(
                    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG
                ),
                "residual_label": (
                    "log(actual full-season team-minute share / production W0 predicted "
                    "team-minute share), restricted to selected opening-roster players "
                    "with full team roster coverage, no in-season team move, and at least "
                    f"{DEFAULT_MIN_RESIDUAL_MINUTES:.0f} actual minutes"
                ),
                "contract": (
                    "The candidate uses the exact promoted Forward Availability and FCM raw "
                    "weights. Only a player's immediately prior, same-team, exposure-shrunk "
                    "post-W0 allocation residual multiplies that raw weight before the "
                    "unchanged active-15 gate and 240-minute normalization. Team changes "
                    "receive zero residual state."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return AllocationResidualRun(run_dir=output_dir, run_id=run_id)


def _season_year(season: str) -> int:
    return int(str(season)[:4])


def _summarize_metrics(metrics: pd.DataFrame, *, frozen_seasons: tuple[str, ...]) -> pd.DataFrame:
    """Summarize all required allocation metrics by frozen season and pooled."""

    rows: list[dict[str, object]] = []
    for season in (*frozen_seasons, "pooled_frozen"):
        source = metrics if season == "pooled_frozen" else metrics.loc[metrics["season"].eq(season)]
        finite_cross_entropy = source.loc[np.isfinite(source["cross_entropy"]), "cross_entropy"]
        rows.append(
            {
                "season": season,
                "evaluated_teams": int(len(source)),
                "mean_allocation_total_variation": float(
                    source["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(source["brier_score"].mean()),
                "player_share_mae": float(source["player_share_mae"].mean()),
                "player_share_rmse": float(np.sqrt(source["player_share_mse"].mean())),
                "strict_cross_entropy": float(source["cross_entropy"].mean()),
                "mean_cross_entropy_when_finite": float(finite_cross_entropy.mean()),
                "share_teams_infinite_cross_entropy": float(
                    source["has_zero_probability_active_player"].mean()
                ),
                "mean_actual_top_rotation_overlap": float(
                    source["actual_top_rotation_overlap"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the forward same-team allocation-residual roster-squash experiment."""

    parser = argparse.ArgumentParser(
        description="Evaluate a forward player allocation residual on the production W0 squash"
    )
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument("--player-panel-path", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()
    run = run_forward_allocation_residual_roster_squash(
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        player_panel_path=args.player_panel_path,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        catalog_path=args.catalog_path,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
