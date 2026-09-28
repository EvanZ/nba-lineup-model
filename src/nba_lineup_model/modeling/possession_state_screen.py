"""Conditional live/dead possession-state screen on top of frozen Split NAIL.

This module deliberately answers an at-possession question, not a pregame one.
The start state is known when a possession begins, so a conditional model may
use it.  It must not be folded into the frozen NAIL leaderboard unless a later
model forecasts the start-state mix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.modeling.constrained_split_nail import (
    DISPLAY_SEASON,
    PRODUCTION_RUN_ID,
    PRODUCTION_MODEL_ARTIFACT,
    ConstrainedSplitState,
    ProductionState,
    _forecast_mean_coordinates,
    _load_production_state,
    _raw_coefficients,
    _state_specialization_for_design,
)
from nba_lineup_model.modeling.forward_contextual_rapm import _previous_season
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import _Tee
from nba_lineup_model.modeling.single_lineup_possessions import (
    single_lineup_possessions_frame,
)
from nba_lineup_model.modeling.split_nail import SplitNailDesign
from nba_lineup_model.modeling.stints import modeling_code_fingerprint, read_rapm_stints
from nba_lineup_model.models.baselines import (
    PriorPrecisionRidgeLineupModel,
    RidgeLineupModel,
    vocabulary_mapping,
)


MODEL_NAME = "possession_state_screen"
RUN_PREFIX = "possession-state-screen"
FIRST_SCREEN_SEASON = "2015-16"
DEVELOPMENT_TARGET_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
FROZEN_TARGET_SEASONS = ("2023-24", "2024-25", "2025-26")
SOURCE_WINDOW_SEASONS = 4
PLAYER_DEVIATION_REGULARIZATION_GRID = (
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
)
PLAYER_DEVIATION_RELATIVE_PRECISION = 32.0
CONTROL_RELATIVE_PRECISION = 0.01
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_CONSTRAINED_RUN_DIR = Path(
    "artifacts/models/constrained_split_nail/2025-26/"
    "constrained-split-nail-2025-26-20260829T130012Z-81a5fad3"
)
DEFAULT_PRODUCTION_RUN_DIR = (
    DEFAULT_ARTIFACTS_DIR
    / PRODUCTION_MODEL_ARTIFACT
    / DISPLAY_SEASON
    / PRODUCTION_RUN_ID
)
DEFAULT_DESIGN_CACHE_PATH = Path(
    "artifacts/cache/constrained_split_nail/2025-26/season_designs.joblib"
)
DEFAULT_OBSERVATION_CACHE_DIR = Path("artifacts/cache/possession_state_screen")


@dataclass(frozen=True)
class ConditionalStateFit:
    """Control-only and player-deviation fits for one source window."""

    player_ids: tuple[int, ...]
    control_model: RidgeLineupModel
    full_model: PriorPrecisionRidgeLineupModel


@dataclass(frozen=True)
class PossessionStateScreenRun:
    """Identity of a completed conditional possession-state screen."""

    run_dir: Path
    run_id: str
    selected_regularization: float


def build_possession_state_observations(
    season: str,
    *,
    design: SplitNailDesign,
    baseline_prediction: np.ndarray,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    include_start_state: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join known start states to single-lineup possessions and frozen baseline.

    The frozen Split NAIL baseline is scored from the cached two-side stint
    design.  A possession inherits that score only when its lineup is an exact
    match to one of the design's valid scoring rows.  This avoids rebuilding a
    second profile/context implementation for the screen.
    """

    stints = read_rapm_stints(season, analytical_dir=analytical_dir)
    lookup = _baseline_lookup(stints, design, baseline_prediction)
    partition = Path(curated_dir) / "possession_segments" / season / "regular"
    segments = pd.read_parquet(partition)
    possessions = single_lineup_possessions_frame(segments)
    states = pd.read_parquet(
        Path(analytical_dir) / "possession_start_states" / season / "regular" / "part-00000.parquet",
        columns=[
            "game_id",
            "possession_id",
            "start_state",
            "coarse_start_state",
            "is_audited_fallback",
        ],
    )
    states["game_id"] = states["game_id"].astype(str)
    states["possession_id"] = states["possession_id"].astype(str)
    merged = possessions.merge(
        states,
        on=["game_id", "possession_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["coarse_start_state"].isna().any():
        raise ValueError(f"{season} has single-lineup possessions without start-state rows")

    baseline = np.empty(len(merged), dtype=float)
    missing = []
    for index, row in enumerate(merged.itertuples(index=False)):
        if row.home_offense:
            home, away = row.offense_player_ids, row.defense_player_ids
        else:
            home, away = row.defense_player_ids, row.offense_player_ids
        key = _side_key(row.game_id, bool(row.home_offense), home, away)
        value = lookup.get(key)
        if value is None:
            missing.append(key)
        else:
            baseline[index] = value
    if missing:
        preview = ", ".join(str(key[:2]) for key in missing[:5])
        raise ValueError(
            f"{season} possession baseline has {len(missing)} unmatched exact lineup keys: {preview}"
        )

    known = merged.loc[~merged["is_audited_fallback"].astype(bool)].copy()
    baseline = baseline[~merged["is_audited_fallback"].to_numpy(dtype=bool)]
    if known.empty:
        raise ValueError(f"{season} has no non-fallback single-lineup possessions")
    output_columns = [
        "game_id",
        "possession_id",
        "home_offense",
        "offense_player_ids",
        "defense_player_ids",
        "offense_points",
        "coarse_start_state",
    ]
    if include_start_state:
        output_columns.insert(-1, "start_state")
    output = known.loc[:, output_columns].copy()
    output.insert(0, "season", str(season))
    output["game_id"] = output["game_id"].astype(str)
    output["possession_id"] = output["possession_id"].astype(str)
    output["target"] = 100.0 * output["offense_points"].to_numpy(dtype=float)
    output["baseline_prediction"] = baseline
    output["is_live_ball"] = output["coarse_start_state"].eq("live_ball")
    output["weight"] = 1.0
    _validate_observations(output)
    coverage = {
        "season": str(season),
        "segment_rows": int(len(segments)),
        "single_lineup_possessions": int(len(possessions)),
        "known_state_single_lineup_possessions": int(len(output)),
        "excluded_fallback_single_lineup_possessions": int(len(merged) - len(output)),
        "live_ball_possessions": int(output["is_live_ball"].sum()),
        "dead_ball_possessions": int((~output["is_live_ball"]).sum()),
        "frozen_baseline_rmse": _rmse(output["target"], output["baseline_prediction"]),
    }
    return output, coverage


def build_live_ball_interaction_design(
    rows: pd.DataFrame,
    player_columns: dict[int, int],
) -> sparse.csr_matrix:
    """Encode zero-at-dead-ball O/D player deviations for live possessions."""

    _validate_observations(rows)
    player_count = len(player_columns)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows.itertuples(index=False)):
        if not row.is_live_ball:
            continue
        for player_id in row.offense_player_ids:
            column = player_columns.get(int(player_id))
            if column is not None:
                row_indices.append(row_index)
                column_indices.append(column)
                values.append(1.0)
        for player_id in row.defense_player_ids:
            column = player_columns.get(int(player_id))
            if column is not None:
                row_indices.append(row_index)
                column_indices.append(player_count + column)
                values.append(-1.0)
    return sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), 2 * player_count),
        dtype=float,
    ).tocsr()


def fit_conditional_state_model(
    source: pd.DataFrame,
    *,
    regularization: float,
) -> ConditionalStateFit:
    """Fit live/dead control and strongly pooled player live-ball deviations."""

    _validate_observations(source)
    player_ids = tuple(
        sorted(
            {
                int(player_id)
                for column in ("offense_player_ids", "defense_player_ids")
                for lineup in source[column]
                for player_id in lineup
            }
        )
    )
    player_columns = vocabulary_mapping(player_ids)
    live = source["is_live_ball"].to_numpy(dtype=float)
    controls = sparse.csr_matrix(live[:, None])
    residual = (
        source["target"].to_numpy(dtype=float)
        - source["baseline_prediction"].to_numpy(dtype=float)
    )
    weights = source["weight"].to_numpy(dtype=float)
    control_model = RidgeLineupModel(regularization=0.0).fit(controls, residual, weights)
    interactions = build_live_ball_interaction_design(source, player_columns)
    features = sparse.hstack((controls, interactions), format="csr")
    prior = np.zeros(features.shape[1], dtype=float)
    precision = np.concatenate(
        (
            np.array([CONTROL_RELATIVE_PRECISION], dtype=float),
            np.full(interactions.shape[1], PLAYER_DEVIATION_RELATIVE_PRECISION, dtype=float),
        )
    )
    full_model = PriorPrecisionRidgeLineupModel(regularization).fit(
        features,
        residual,
        weights,
        prior,
        precision,
    )
    return ConditionalStateFit(
        player_ids=player_ids,
        control_model=control_model,
        full_model=full_model,
    )


def predict_conditional_state_models(
    rows: pd.DataFrame,
    fit: ConditionalStateFit,
) -> pd.DataFrame:
    """Score fixed baseline, state-only, and state-plus-player models."""

    _validate_observations(rows)
    player_columns = vocabulary_mapping(fit.player_ids)
    controls = sparse.csr_matrix(rows["is_live_ball"].to_numpy(dtype=float)[:, None])
    interactions = build_live_ball_interaction_design(rows, player_columns)
    features = sparse.hstack((controls, interactions), format="csr")
    output = rows.loc[
        :,
        ["season", "game_id", "possession_id", "home_offense", "target", "weight", "coarse_start_state"],
    ].copy()
    output["baseline_prediction"] = rows["baseline_prediction"].to_numpy(dtype=float)
    output["state_control_prediction"] = (
        output["baseline_prediction"].to_numpy(dtype=float) + fit.control_model.predict(controls)
    )
    output["state_player_prediction"] = (
        output["baseline_prediction"].to_numpy(dtype=float) + fit.full_model.predict(features)
    )
    return output


def conditional_state_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return conditional side and full-game margin RMSE for all three models."""

    required = {
        "game_id",
        "home_offense",
        "target",
        "weight",
        "baseline_prediction",
        "state_control_prediction",
        "state_player_prediction",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Conditional predictions missing: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for label, column in (
        ("fixed_split_nail", "baseline_prediction"),
        ("state_control", "state_control_prediction"),
        ("state_plus_player", "state_player_prediction"),
    ):
        rows.append(
            {
                "model": label,
                "metric": "conditional_possession_rmse",
                "value": _rmse(predictions["target"], predictions[column]),
                "observation_count": len(predictions),
                "exposure": float(predictions["weight"].sum()),
            }
        )
        margins = _game_margin_frame(predictions, column)
        rows.append(
            {
                "model": label,
                "metric": "conditional_game_margin_rmse",
                "value": float(np.sqrt(np.mean(np.square(margins["actual_margin"] - margins["prediction_margin"])))),
                "observation_count": len(margins),
                "exposure": float(len(margins)),
            }
        )
    output = pd.DataFrame(rows)
    reference = output.loc[output["model"].eq("fixed_split_nail"), ["metric", "value"]].rename(
        columns={"value": "reference_value"}
    )
    output = output.merge(reference, on="metric", how="left", validate="many_to_one")
    output["minus_fixed_split_nail"] = output["value"] - output["reference_value"]
    return output


def paired_conditional_state_bootstrap(
    predictions: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 20_260_928,
) -> pd.DataFrame:
    """Game-cluster bootstrap for state-plus-player against fixed Split NAIL."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    rows = predictions.copy()
    game_ids = np.array(sorted(rows["game_id"].astype(str).unique()))
    groups = {game_id: frame for game_id, frame in rows.groupby("game_id", sort=True)}
    side_reference = np.array(
        [
            np.square(
                groups[game_id]["target"].to_numpy(dtype=float)
                - groups[game_id]["baseline_prediction"].to_numpy(dtype=float)
            ).sum()
            for game_id in game_ids
        ]
    )
    side_candidate = np.array(
        [
            np.square(
                groups[game_id]["target"].to_numpy(dtype=float)
                - groups[game_id]["state_player_prediction"].to_numpy(dtype=float)
            ).sum()
            for game_id in game_ids
        ]
    )
    side_count = np.array([len(groups[game_id]) for game_id in game_ids], dtype=float)
    baseline_game = _game_margin_frame(rows, "baseline_prediction").set_index("game_id").loc[game_ids]
    candidate_game = _game_margin_frame(rows, "state_player_prediction").set_index("game_id").loc[game_ids]
    game_reference = np.square(
        baseline_game["actual_margin"].to_numpy(dtype=float)
        - baseline_game["prediction_margin"].to_numpy(dtype=float)
    )
    game_candidate = np.square(
        candidate_game["actual_margin"].to_numpy(dtype=float)
        - candidate_game["prediction_margin"].to_numpy(dtype=float)
    )
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(game_ids), size=(draws, len(game_ids)))
    side_delta = np.sqrt(side_candidate[samples].sum(axis=1) / side_count[samples].sum(axis=1)) - np.sqrt(
        side_reference[samples].sum(axis=1) / side_count[samples].sum(axis=1)
    )
    game_delta = np.sqrt(game_candidate[samples].mean(axis=1)) - np.sqrt(
        game_reference[samples].mean(axis=1)
    )
    point = conditional_state_metrics(rows)
    point = point.loc[point["model"].eq("state_plus_player")].set_index("metric")
    return pd.DataFrame(
        [
            _bootstrap_row(
                "conditional_possession_rmse",
                float(point.loc["conditional_possession_rmse", "minus_fixed_split_nail"]),
                side_delta,
                draws,
                seed,
            ),
            _bootstrap_row(
                "conditional_game_margin_rmse",
                float(point.loc["conditional_game_margin_rmse", "minus_fixed_split_nail"]),
                game_delta,
                draws,
                seed,
            ),
        ]
    )


def run_possession_state_screen(
    *,
    constrained_run_dir: Path | str = DEFAULT_CONSTRAINED_RUN_DIR,
    production_run_dir: Path | str = DEFAULT_PRODUCTION_RUN_DIR,
    design_cache_path: Path | str = DEFAULT_DESIGN_CACHE_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    observation_cache_dir: Path | str = DEFAULT_OBSERVATION_CACHE_DIR,
    bootstrap_draws: int = 2_000,
) -> PossessionStateScreenRun:
    """Tune live-ball player deviations, then run the frozen conditional replay."""

    print("[possession-state] loading frozen Split NAIL state", flush=True)
    production = _load_production_state(Path(production_run_dir))
    states: dict[str, ConstrainedSplitState] = joblib.load(
        Path(constrained_run_dir) / "season_states.joblib"
    )
    designs: dict[str, SplitNailDesign] = joblib.load(Path(design_cache_path))
    all_targets = (*DEVELOPMENT_TARGET_SEASONS, *FROZEN_TARGET_SEASONS)
    required_seasons = _screen_seasons(all_targets)
    if not set(required_seasons).issubset(designs) or not set(required_seasons).issubset(states):
        raise ValueError("Constrained Split artifacts do not cover the conditional screen seasons")

    cache_dir = Path(observation_cache_dir) / Path(constrained_run_dir).name
    coverage_rows: list[dict[str, object]] = []
    for index, season in enumerate(required_seasons, start=1):
        path = cache_dir / f"{season}.parquet"
        if path.is_file():
            print(f"[possession-state] cache {season} ({index}/{len(required_seasons)})", flush=True)
            coverage = _coverage_from_rows(pd.read_parquet(path))
        else:
            print(f"[possession-state] baseline and states {season} ({index}/{len(required_seasons)})", flush=True)
            baseline = _frozen_split_baseline(season, designs[season], states, production)
            rows, coverage = build_possession_state_observations(
                season,
                design=designs[season],
                baseline_prediction=baseline,
                curated_dir=curated_dir,
                analytical_dir=analytical_dir,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(path, index=False)
        coverage_rows.append(coverage)

    selection_rows: list[dict[str, object]] = []
    for regularization in PLAYER_DEVIATION_REGULARIZATION_GRID:
        for season in DEVELOPMENT_TARGET_SEASONS:
            print(
                f"[possession-state] select lambda={regularization:g}; target={season}",
                flush=True,
            )
            source = _read_source_window(cache_dir, season)
            target = pd.read_parquet(cache_dir / f"{season}.parquet")
            fit = fit_conditional_state_model(source, regularization=regularization)
            predictions = predict_conditional_state_models(target, fit)
            metrics = conditional_state_metrics(predictions)
            for metric in metrics.itertuples(index=False):
                selection_rows.append(
                    {
                        "regularization": regularization,
                        "target_season": season,
                        "model": metric.model,
                        "metric": metric.metric,
                        "value": metric.value,
                        "minus_fixed_split_nail": metric.minus_fixed_split_nail,
                        "source_possessions": len(source),
                        "target_possessions": len(target),
                    }
                )
    selection = pd.DataFrame(selection_rows)
    summary = (
        selection.groupby(["regularization", "model", "metric"], as_index=False)
        .agg(
            value=("value", "mean"),
            minus_fixed_split_nail=("minus_fixed_split_nail", "mean"),
            median_minus_fixed_split_nail=("minus_fixed_split_nail", "median"),
        )
        .sort_values(["metric", "model", "value", "regularization"], kind="stable")
        .reset_index(drop=True)
    )
    candidates = summary.loc[
        summary["model"].eq("state_plus_player")
        & summary["metric"].eq("conditional_possession_rmse")
    ].sort_values(["value", "regularization"], kind="stable")
    selected_regularization = float(candidates.iloc[0]["regularization"])
    boundary = {
        PLAYER_DEVIATION_REGULARIZATION_GRID[0],
        PLAYER_DEVIATION_REGULARIZATION_GRID[-1],
    }
    if selected_regularization in boundary:
        raise ValueError(
            "Conditional player-deviation regularization selected a grid boundary: "
            f"{selected_regularization:g}. Expand the grid before interpreting the screen."
        )
    print(
        f"[possession-state] selected lambda={selected_regularization:g}; interior=True",
        flush=True,
    )

    frozen_predictions: list[pd.DataFrame] = []
    frozen_metrics: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    for season in FROZEN_TARGET_SEASONS:
        print(f"[possession-state] frozen replay {season}", flush=True)
        source = _read_source_window(cache_dir, season)
        target = pd.read_parquet(cache_dir / f"{season}.parquet")
        fit = fit_conditional_state_model(source, regularization=selected_regularization)
        prediction = predict_conditional_state_models(target, fit)
        prediction.insert(1, "target_season", season)
        frozen_predictions.append(prediction)
        frozen_metrics.append(conditional_state_metrics(prediction).assign(target_season=season))
        bootstrap_rows.append(
            paired_conditional_state_bootstrap(prediction, draws=bootstrap_draws).assign(
                target_season=season
            )
        )
    frozen = pd.concat(frozen_predictions, ignore_index=True)
    frozen_metrics.append(conditional_state_metrics(frozen).assign(target_season="pooled"))
    bootstrap_rows.append(
        paired_conditional_state_bootstrap(frozen, draws=bootstrap_draws).assign(target_season="pooled")
    )
    # The diagnostic state is the same four-completed-season window that would
    # feed the next season, not an ad hoc pool of the frozen target rows.
    final_source = _read_source_window(cache_dir, "2026-27")
    final_fit = fit_conditional_state_model(
        final_source, regularization=selected_regularization
    )
    player_diagnostics = _player_diagnostics(final_source, final_fit)
    control_diagnostics = _control_diagnostics(final_fit)
    run = _write_run(
        coverage=pd.DataFrame(coverage_rows),
        selection=selection,
        selection_summary=summary,
        frozen_predictions=frozen,
        frozen_metrics=pd.concat(frozen_metrics, ignore_index=True),
        bootstrap=pd.concat(bootstrap_rows, ignore_index=True),
        player_diagnostics=player_diagnostics,
        control_diagnostics=control_diagnostics,
        selected_regularization=selected_regularization,
        constrained_run_dir=Path(constrained_run_dir),
        production_run_dir=Path(production_run_dir),
        artifacts_dir=Path(artifacts_dir),
        bootstrap_draws=bootstrap_draws,
    )
    print(f"[possession-state] complete: {run.run_dir}", flush=True)
    return run


def _frozen_split_baseline(
    season: str,
    design: SplitNailDesign,
    states: dict[str, ConstrainedSplitState],
    production: ProductionState,
) -> np.ndarray:
    source = _previous_season(season)
    if source not in states:
        raise ValueError(f"Constrained O/D source state is missing for {season}")
    mean = _forecast_mean_coordinates(design, season, production)
    specialization = _state_specialization_for_design(design, states[source])
    raw = _raw_coefficients(design, mean, specialization)
    return np.asarray(design.features @ raw, dtype=float).reshape(-1) + states[source].intercept


def _baseline_lookup(
    stints: pd.DataFrame,
    design: SplitNailDesign,
    prediction: np.ndarray,
) -> dict[tuple[str, bool, tuple[int, ...], tuple[int, ...]], float]:
    if len(prediction) != len(design.target):
        raise ValueError("Frozen Split baseline must align to every design scoring row")
    rows = stints.reset_index(drop=True)
    expected = int(rows["home_offensive_possessions"].gt(0).sum() + rows["away_offensive_possessions"].gt(0).sum())
    if expected != len(prediction):
        raise ValueError("Source stints do not match the cached Split NAIL design")
    lookup: dict[tuple[str, bool, tuple[int, ...], tuple[int, ...]], float] = {}
    cursor = 0
    for row in rows.itertuples(index=False):
        for home_offense, possessions in (
            (True, row.home_offensive_possessions),
            (False, row.away_offensive_possessions),
        ):
            if float(possessions) <= 0.0:
                continue
            key = _side_key(row.game_id, home_offense, row.home_player_ids, row.away_player_ids)
            value = float(prediction[cursor])
            cursor += 1
            previous = lookup.setdefault(key, value)
            if not np.isclose(previous, value, atol=1e-10):
                raise ValueError("One Split NAIL lineup key has inconsistent frozen predictions")
    return lookup


def _side_key(
    game_id: object,
    home_offense: bool,
    home_players: object,
    away_players: object,
) -> tuple[str, bool, tuple[int, ...], tuple[int, ...]]:
    return (
        str(game_id),
        bool(home_offense),
        tuple(int(player) for player in home_players),
        tuple(int(player) for player in away_players),
    )


def _screen_seasons(targets: tuple[str, ...]) -> tuple[str, ...]:
    earliest = int(FIRST_SCREEN_SEASON[:4])
    latest = int(max(targets)[:4])
    return tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(earliest, latest + 1))


def _read_source_window(cache_dir: Path, target_season: str) -> pd.DataFrame:
    target_year = int(target_season[:4])
    start = target_year - SOURCE_WINDOW_SEASONS
    seasons = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(start, target_year))
    missing = [season for season in seasons if not (cache_dir / f"{season}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(f"Conditional source window is missing: {missing}")
    return pd.concat(
        [pd.read_parquet(cache_dir / f"{season}.parquet") for season in seasons],
        ignore_index=True,
    )


def _coverage_from_rows(rows: pd.DataFrame) -> dict[str, object]:
    _validate_observations(rows)
    return {
        "season": str(rows["season"].iloc[0]),
        "segment_rows": np.nan,
        "single_lineup_possessions": np.nan,
        "known_state_single_lineup_possessions": int(len(rows)),
        "excluded_fallback_single_lineup_possessions": np.nan,
        "live_ball_possessions": int(rows["is_live_ball"].sum()),
        "dead_ball_possessions": int((~rows["is_live_ball"]).sum()),
        "frozen_baseline_rmse": _rmse(rows["target"], rows["baseline_prediction"]),
    }


def _game_margin_frame(rows: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    signs = np.where(rows["home_offense"].to_numpy(dtype=bool), 1.0, -1.0)
    frame = pd.DataFrame(
        {
            "game_id": rows["game_id"].astype(str),
            "actual_signed": signs * rows["target"].to_numpy(dtype=float) / 100.0,
            "prediction_signed": signs * rows[prediction_column].to_numpy(dtype=float) / 100.0,
        }
    )
    return frame.groupby("game_id", as_index=False, sort=True).sum().rename(
        columns={"actual_signed": "actual_margin", "prediction_signed": "prediction_margin"}
    )


def _player_diagnostics(source: pd.DataFrame, fit: ConditionalStateFit) -> pd.DataFrame:
    columns = vocabulary_mapping(fit.player_ids)
    design = build_live_ball_interaction_design(source, columns)
    player_count = len(fit.player_ids)
    coefficients = fit.full_model.coef_[1:]
    variance = fit.full_model.posterior_variance_[1:]
    rows: list[dict[str, object]] = []
    for index, player_id in enumerate(fit.player_ids):
        offense_exposure = float(design[:, index].count_nonzero())
        defense_exposure = float(design[:, player_count + index].count_nonzero())
        rows.append(
            {
                "player_id": player_id,
                "offense_live_ball_deviation": float(coefficients[index]),
                "defense_live_ball_deviation": float(coefficients[player_count + index]),
                "offense_live_ball_standard_error": float(np.sqrt(variance[index])),
                "defense_live_ball_standard_error": float(np.sqrt(variance[player_count + index])),
                "offense_live_ball_possessions": offense_exposure,
                "defense_live_ball_possessions": defense_exposure,
                "source_seasons": ",".join(sorted(source["season"].astype(str).unique())),
            }
        )
    output = pd.DataFrame(rows)
    output["total_live_ball_deviation"] = (
        output["offense_live_ball_deviation"] + output["defense_live_ball_deviation"]
    )
    catalog_path = Path("data/catalog/players.parquet")
    if catalog_path.is_file():
        names = pd.read_parquet(catalog_path, columns=["player_id", "display_name"])
        names["player_id"] = pd.to_numeric(names["player_id"], errors="raise").astype(int)
        names = names.drop_duplicates("player_id")
        output = output.merge(names, on="player_id", how="left", validate="one_to_one")
        output = output.loc[
            :,
            ["player_id", "display_name", *[column for column in output if column not in {"player_id", "display_name"}]],
        ]
    return output.sort_values("total_live_ball_deviation", key=np.abs, ascending=False).reset_index(drop=True)


def _control_diagnostics(fit: ConditionalStateFit) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "state_control",
                "baseline_recalibration": float(fit.control_model.intercept_),
                "live_ball_adjustment": float(fit.control_model.coef_[0]),
            },
            {
                "model": "state_plus_player",
                "baseline_recalibration": float(fit.full_model.intercept_),
                "live_ball_adjustment": float(fit.full_model.coef_[0]),
            },
        ]
    )


def _rmse(target: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    values = np.asarray(target, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(np.square(values - forecast))))


def _bootstrap_row(
    metric: str,
    point: float,
    differences: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, object]:
    return {
        "metric": metric,
        "candidate_minus_fixed_split_nail": point,
        "ci_lower": float(np.quantile(differences, 0.025)),
        "ci_upper": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float(np.mean(differences < 0.0)),
        "bootstrap_unit": "complete_game",
        "bootstrap_draws": draws,
        "random_seed": seed,
    }


def _validate_observations(rows: pd.DataFrame) -> None:
    required = {
        "season",
        "game_id",
        "possession_id",
        "home_offense",
        "offense_player_ids",
        "defense_player_ids",
        "target",
        "baseline_prediction",
        "is_live_ball",
        "weight",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Conditional possession observations missing: {sorted(missing)}")
    if rows.empty or rows.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Conditional possession observations must be non-empty and unique")
    if not rows["coarse_start_state"].isin(("live_ball", "dead_ball")).all():
        raise ValueError("Conditional observations may contain only known live/dead states")
    if not rows["is_live_ball"].eq(rows["coarse_start_state"].eq("live_ball")).all():
        raise ValueError("Live-ball state indicator is inconsistent")
    if not np.isfinite(rows[["target", "baseline_prediction", "weight"]].to_numpy(dtype=float)).all():
        raise ValueError("Conditional observations must be finite")
    if (rows["weight"].to_numpy(dtype=float) <= 0.0).any():
        raise ValueError("Conditional observation weights must be positive")
    for column in ("offense_player_ids", "defense_player_ids"):
        if not rows[column].map(lambda lineup: len(lineup) == 5).all():
            raise ValueError("Conditional possession lineups must have five players")


def _write_run(
    *,
    coverage: pd.DataFrame,
    selection: pd.DataFrame,
    selection_summary: pd.DataFrame,
    frozen_predictions: pd.DataFrame,
    frozen_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    player_diagnostics: pd.DataFrame,
    control_diagnostics: pd.DataFrame,
    selected_regularization: float,
    constrained_run_dir: Path,
    production_run_dir: Path,
    artifacts_dir: Path,
    bootstrap_draws: int,
) -> PossessionStateScreenRun:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{RUN_PREFIX}-{timestamp}-{uuid4().hex[:8]}"
    root = artifacts_dir / MODEL_NAME
    staging = root / f".{run_id}.tmp"
    run_dir = root / run_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        coverage.to_parquet(staging / "coverage.parquet", index=False)
        selection.to_parquet(staging / "selection_by_season.parquet", index=False)
        selection_summary.to_parquet(staging / "selection_summary.parquet", index=False)
        frozen_predictions.to_parquet(staging / "frozen_predictions.parquet", index=False)
        frozen_metrics.to_parquet(staging / "frozen_metrics.parquet", index=False)
        bootstrap.to_parquet(staging / "paired_bootstrap_metrics.parquet", index=False)
        player_diagnostics.to_parquet(staging / "player_live_ball_diagnostics.parquet", index=False)
        control_diagnostics.to_parquet(staging / "state_control_diagnostics.parquet", index=False)
        metadata = {
            "model": MODEL_NAME,
            "contract": "conditional_realized_live_dead_state_on_frozen_split_nail",
            "not_a_pregame_candidate": True,
            "coarse_state_taxonomy": ["live_ball", "dead_ball"],
            "excluded_start_state": "audited_fallback",
            "source_window_seasons": SOURCE_WINDOW_SEASONS,
            "development_target_seasons": list(DEVELOPMENT_TARGET_SEASONS),
            "frozen_target_seasons": list(FROZEN_TARGET_SEASONS),
            "player_deviation_regularization_grid": list(PLAYER_DEVIATION_REGULARIZATION_GRID),
            "selected_player_deviation_regularization": selected_regularization,
            "player_deviation_relative_precision": PLAYER_DEVIATION_RELATIVE_PRECISION,
            "control_relative_precision": CONTROL_RELATIVE_PRECISION,
            "constrained_split_run_dir": str(constrained_run_dir),
            "production_run_dir": str(production_run_dir),
            "bootstrap_draws": bootstrap_draws,
            "created_at": datetime.now(UTC).isoformat(),
            "code_fingerprint": modeling_code_fingerprint((Path(__file__),)),
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest = {"run_id": run_id, "files": sorted(path.name for path in staging.iterdir())}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.replace(run_dir)
        (root / "latest.json").write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return PossessionStateScreenRun(
        run_dir=run_dir,
        run_id=run_id,
        selected_regularization=selected_regularization,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live/dead conditional Split NAIL screen")
    parser.add_argument("--log-path")
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()
    if args.log_path:
        path = Path(args.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            stdout = sys.stdout
            sys.stdout = _Tee(stdout, handle)  # type: ignore[assignment]
            try:
                run = run_possession_state_screen(bootstrap_draws=args.bootstrap_draws)
                print(f"Possession-state screen: run={run.run_dir}", flush=True)
            finally:
                sys.stdout = stdout
        return
    run = run_possession_state_screen(bootstrap_draws=args.bootstrap_draws)
    print(f"Possession-state screen: run={run.run_dir}")


if __name__ == "__main__":
    main()
