"""Hierarchically pooled granular possession-state screen on frozen Split NAIL.

The realized possession start state is available only after a possession begins.
This is therefore a conditional, in-game screen. It explicitly preserves total
player coordinates: source-frequency-weighted coarse and fine state contrasts
sum to zero for every player-side whose contrast is represented.
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
    PRODUCTION_MODEL_ARTIFACT,
    PRODUCTION_RUN_ID,
    ConstrainedSplitState,
    ProductionState,
    _load_production_state,
)
from nba_lineup_model.modeling.forward_nail_v1212_back_to_back import _Tee
from nba_lineup_model.modeling.possession_start_state_mart import (
    DEAD_BALL_STATES,
    LIVE_BALL_STATES,
)
from nba_lineup_model.modeling.possession_state_screen import (
    CONTROL_RELATIVE_PRECISION,
    DEVELOPMENT_TARGET_SEASONS,
    FIRST_SCREEN_SEASON,
    FROZEN_TARGET_SEASONS,
    PLAYER_DEVIATION_RELATIVE_PRECISION,
    SOURCE_WINDOW_SEASONS,
    _bootstrap_row,
    _frozen_split_baseline,
    _game_margin_frame,
    _read_source_window,
    _rmse,
    _screen_seasons,
    build_possession_state_observations,
)
from nba_lineup_model.modeling.split_nail import SplitNailDesign
from nba_lineup_model.modeling.stints import modeling_code_fingerprint
from nba_lineup_model.models.baselines import (
    PriorPrecisionRidgeLineupModel,
    RidgeLineupModel,
    vocabulary_mapping,
)


MODEL_NAME = "hierarchical_possession_state_screen"
RUN_PREFIX = "hierarchical-possession-state-screen"
MINIMUM_STATE_SOURCE_POSSESSIONS = 10_000
CHILD_RELATIVE_PRECISION_GRID = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
REFERENCE_CONTROL_STATE = "after_made_2pt"
DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/models")
DEFAULT_CONSTRAINED_RUN_DIR = Path(
    "artifacts/models/constrained_split_nail/2025-26/"
    "constrained-split-nail-2025-26-20260829T130012Z-81a5fad3"
)
DEFAULT_PRODUCTION_RUN_DIR = (
    DEFAULT_ARTIFACTS_DIR / PRODUCTION_MODEL_ARTIFACT / DISPLAY_SEASON / PRODUCTION_RUN_ID
)
DEFAULT_DESIGN_CACHE_PATH = Path(
    "artifacts/cache/constrained_split_nail/2025-26/season_designs.joblib"
)
DEFAULT_OBSERVATION_CACHE_DIR = Path("artifacts/cache/hierarchical_possession_state_screen")


@dataclass(frozen=True)
class FineStateSpecification:
    """Source-fixed dictionary and centering weights for one target-season fit."""

    player_ids: tuple[int, ...]
    control_states: tuple[str, ...]
    eligible_states: tuple[str, ...]
    child_states: tuple[str, ...]
    anchors: dict[str, str]
    offense_parent_ratio: np.ndarray
    defense_parent_ratio: np.ndarray
    offense_child_ratio: np.ndarray
    defense_child_ratio: np.ndarray
    source_state_exposure: dict[str, int]


@dataclass(frozen=True)
class PreparedFineStateFeatures:
    """Sparse state controls and player contrasts for a fixed source dictionary."""

    rows: pd.DataFrame
    specification: FineStateSpecification
    controls: sparse.csr_matrix
    parent: sparse.csr_matrix
    children: sparse.csr_matrix


@dataclass(frozen=True)
class HierarchicalStateFit:
    """Nested conditional fits evaluated on the same realized state rows."""

    specification: FineStateSpecification
    control_model: RidgeLineupModel
    coarse_model: PriorPrecisionRidgeLineupModel
    full_model: PriorPrecisionRidgeLineupModel
    child_relative_precision: float


@dataclass(frozen=True)
class HierarchicalStateScreenRun:
    run_dir: Path
    run_id: str
    selected_child_relative_precision: float


def build_fine_state_specification(source: pd.DataFrame) -> FineStateSpecification:
    """Select supported states and derive source-only player contrast weights."""

    _validate_fine_state_rows(source)
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
    state_exposure = {
        str(state): int(count)
        for state, count in source["start_state"].value_counts().items()
    }
    eligible_states = tuple(
        state
        for state, count in sorted(state_exposure.items())
        if count >= MINIMUM_STATE_SOURCE_POSSESSIONS
    )
    state_to_coarse = _state_to_coarse(source)
    anchors: dict[str, str] = {}
    child_states: list[str] = []
    for coarse, available in (("live_ball", LIVE_BALL_STATES), ("dead_ball", DEAD_BALL_STATES)):
        represented = [state for state in eligible_states if state in available]
        if not represented:
            continue
        anchor = max(represented, key=lambda state: (state_exposure[state], state))
        anchors[coarse] = anchor
        child_states.extend(state for state in represented if state != anchor)
    if not child_states:
        raise ValueError("No granular child states meet the predeclared source-exposure threshold")
    control_states = tuple(
        state
        for state in sorted(state_exposure)
        if state != REFERENCE_CONTROL_STATE
    )
    if REFERENCE_CONTROL_STATE not in state_exposure:
        raise ValueError(f"Fine state control reference is missing: {REFERENCE_CONTROL_STATE}")

    columns = vocabulary_mapping(player_ids)
    player_count = len(player_ids)
    child_count = len(child_states)
    offense_counts = np.zeros((player_count, len(state_exposure)), dtype=float)
    defense_counts = np.zeros((player_count, len(state_exposure)), dtype=float)
    states = tuple(sorted(state_exposure))
    state_columns = {state: index for index, state in enumerate(states)}
    for row in source.itertuples(index=False):
        state_column = state_columns[str(row.start_state)]
        for player_id in row.offense_player_ids:
            offense_counts[columns[int(player_id)], state_column] += 1.0
        for player_id in row.defense_player_ids:
            defense_counts[columns[int(player_id)], state_column] += 1.0
    live_columns = [state_columns[state] for state in states if state_to_coarse[state] == "live_ball"]
    dead_columns = [state_columns[state] for state in states if state_to_coarse[state] == "dead_ball"]
    offense_live = offense_counts[:, live_columns].sum(axis=1)
    defense_live = defense_counts[:, live_columns].sum(axis=1)
    offense_dead = offense_counts[:, dead_columns].sum(axis=1)
    defense_dead = defense_counts[:, dead_columns].sum(axis=1)
    offense_parent_ratio = _ratio(offense_live, offense_dead)
    defense_parent_ratio = _ratio(defense_live, defense_dead)

    offense_child_ratio = np.zeros((player_count, child_count), dtype=float)
    defense_child_ratio = np.zeros((player_count, child_count), dtype=float)
    for child_column, state in enumerate(child_states):
        coarse = state_to_coarse[state]
        anchor = anchors[coarse]
        state_index = state_columns[state]
        anchor_index = state_columns[anchor]
        offense_child_ratio[:, child_column] = _ratio(
            offense_counts[:, state_index], offense_counts[:, anchor_index]
        )
        defense_child_ratio[:, child_column] = _ratio(
            defense_counts[:, state_index], defense_counts[:, anchor_index]
        )
    return FineStateSpecification(
        player_ids=player_ids,
        control_states=control_states,
        eligible_states=eligible_states,
        child_states=tuple(child_states),
        anchors=anchors,
        offense_parent_ratio=offense_parent_ratio,
        defense_parent_ratio=defense_parent_ratio,
        offense_child_ratio=offense_child_ratio,
        defense_child_ratio=defense_child_ratio,
        source_state_exposure=state_exposure,
    )


def prepare_fine_state_features(
    rows: pd.DataFrame,
    specification: FineStateSpecification,
) -> PreparedFineStateFeatures:
    """Build controls and source-centered O/D state contrasts for possession rows."""

    _validate_fine_state_rows(rows)
    player_columns = vocabulary_mapping(specification.player_ids)
    controls = _build_control_design(rows, specification.control_states)
    parent = _build_parent_design(rows, specification, player_columns)
    children = _build_child_design(rows, specification, player_columns)
    return PreparedFineStateFeatures(
        rows=rows,
        specification=specification,
        controls=controls,
        parent=parent,
        children=children,
    )


def fit_hierarchical_state_model(
    source: PreparedFineStateFeatures,
    *,
    child_relative_precision: float,
) -> HierarchicalStateFit:
    """Fit nested fine-state controls, coarse effects, and child contrasts."""

    if child_relative_precision <= 0.0:
        raise ValueError("Child relative precision must be positive")
    rows = source.rows
    residual = rows["target"].to_numpy(dtype=float) - rows["baseline_prediction"].to_numpy(dtype=float)
    weights = rows["weight"].to_numpy(dtype=float)
    control_model = RidgeLineupModel(regularization=0.0).fit(source.controls, residual, weights)
    coarse_features = sparse.hstack((source.controls, source.parent), format="csr")
    coarse_precision = np.concatenate(
        (
            np.full(source.controls.shape[1], CONTROL_RELATIVE_PRECISION, dtype=float),
            np.full(source.parent.shape[1], PLAYER_DEVIATION_RELATIVE_PRECISION, dtype=float),
        )
    )
    coarse_model = PriorPrecisionRidgeLineupModel(0.0001).fit(
        coarse_features,
        residual,
        weights,
        np.zeros(coarse_features.shape[1], dtype=float),
        coarse_precision,
    )
    full_features = sparse.hstack((source.controls, source.parent, source.children), format="csr")
    child_precision = _child_precision_vector(source.specification, child_relative_precision)
    full_precision = np.concatenate((coarse_precision, child_precision))
    full_model = PriorPrecisionRidgeLineupModel(0.0001).fit(
        full_features,
        residual,
        weights,
        np.zeros(full_features.shape[1], dtype=float),
        full_precision,
    )
    return HierarchicalStateFit(
        specification=source.specification,
        control_model=control_model,
        coarse_model=coarse_model,
        full_model=full_model,
        child_relative_precision=float(child_relative_precision),
    )


def predict_hierarchical_state_models(
    prepared: PreparedFineStateFeatures,
    fit: HierarchicalStateFit,
) -> pd.DataFrame:
    """Score fixed, fine-control, coarse-player, and full hierarchical models."""

    if prepared.specification is not fit.specification:
        raise ValueError("Target state design must use the source fit specification")
    coarse_features = sparse.hstack((prepared.controls, prepared.parent), format="csr")
    full_features = sparse.hstack((prepared.controls, prepared.parent, prepared.children), format="csr")
    output = prepared.rows.loc[
        :,
        [
            "season",
            "game_id",
            "possession_id",
            "home_offense",
            "target",
            "weight",
            "start_state",
            "coarse_start_state",
        ],
    ].copy()
    output["baseline_prediction"] = prepared.rows["baseline_prediction"].to_numpy(dtype=float)
    output["fine_control_prediction"] = (
        output["baseline_prediction"].to_numpy(dtype=float) + fit.control_model.predict(prepared.controls)
    )
    output["coarse_player_prediction"] = (
        output["baseline_prediction"].to_numpy(dtype=float) + fit.coarse_model.predict(coarse_features)
    )
    output["hierarchical_player_prediction"] = (
        output["baseline_prediction"].to_numpy(dtype=float) + fit.full_model.predict(full_features)
    )
    return output


def hierarchical_state_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate nested conditional predictions at possession and game level."""

    rows: list[dict[str, object]] = []
    for label, column in (
        ("fixed_split_nail", "baseline_prediction"),
        ("fine_state_control", "fine_control_prediction"),
        ("coarse_player_state", "coarse_player_prediction"),
        ("hierarchical_fine_state", "hierarchical_player_prediction"),
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
                "value": float(
                    np.sqrt(np.mean(np.square(margins["actual_margin"] - margins["prediction_margin"])))
                ),
                "observation_count": len(margins),
                "exposure": float(len(margins)),
            }
        )
    output = pd.DataFrame(rows)
    baseline = output.loc[output["model"].eq("fixed_split_nail"), ["metric", "value"]].rename(
        columns={"value": "fixed_split_nail_value"}
    )
    coarse = output.loc[output["model"].eq("coarse_player_state"), ["metric", "value"]].rename(
        columns={"value": "coarse_player_value"}
    )
    output = output.merge(baseline, on="metric", how="left", validate="many_to_one")
    output = output.merge(coarse, on="metric", how="left", validate="many_to_one")
    output["minus_fixed_split_nail"] = output["value"] - output["fixed_split_nail_value"]
    output["minus_coarse_player_state"] = output["value"] - output["coarse_player_value"]
    return output


def paired_hierarchical_state_bootstrap(
    predictions: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 20_260_928,
) -> pd.DataFrame:
    """Game-cluster intervals for fine hierarchy against fixed and coarse models."""

    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    game_ids = np.array(sorted(predictions["game_id"].astype(str).unique()))
    groups = {game_id: frame for game_id, frame in predictions.groupby("game_id", sort=True)}
    counts = np.array([len(groups[game_id]) for game_id in game_ids], dtype=float)
    errors = {
        label: np.array(
            [
                np.square(
                    groups[game_id]["target"].to_numpy(dtype=float)
                    - groups[game_id][column].to_numpy(dtype=float)
                ).sum()
                for game_id in game_ids
            ]
        )
        for label, column in (
            ("fixed_split_nail", "baseline_prediction"),
            ("coarse_player_state", "coarse_player_prediction"),
            ("hierarchical_fine_state", "hierarchical_player_prediction"),
        )
    }
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(game_ids), size=(draws, len(game_ids)))
    reference = np.sqrt(errors["fixed_split_nail"][samples].sum(axis=1) / counts[samples].sum(axis=1))
    coarse = np.sqrt(errors["coarse_player_state"][samples].sum(axis=1) / counts[samples].sum(axis=1))
    candidate = np.sqrt(
        errors["hierarchical_fine_state"][samples].sum(axis=1) / counts[samples].sum(axis=1)
    )
    metrics = hierarchical_state_metrics(predictions)
    candidate_metric = metrics.loc[
        (metrics["model"] == "hierarchical_fine_state")
        & (metrics["metric"] == "conditional_possession_rmse")
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                **_bootstrap_row(
                    "conditional_possession_rmse",
                    float(candidate_metric["minus_fixed_split_nail"]),
                    candidate - reference,
                    draws,
                    seed,
                ),
                "comparison": "hierarchical_fine_state_minus_fixed_split_nail",
            },
            {
                **_bootstrap_row(
                    "conditional_possession_rmse",
                    float(candidate_metric["minus_coarse_player_state"]),
                    candidate - coarse,
                    draws,
                    seed,
                ),
                "comparison": "hierarchical_fine_state_minus_coarse_player_state",
            },
        ]
    )


def run_hierarchical_possession_state_screen(
    *,
    constrained_run_dir: Path | str = DEFAULT_CONSTRAINED_RUN_DIR,
    production_run_dir: Path | str = DEFAULT_PRODUCTION_RUN_DIR,
    design_cache_path: Path | str = DEFAULT_DESIGN_CACHE_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    observation_cache_dir: Path | str = DEFAULT_OBSERVATION_CACHE_DIR,
    bootstrap_draws: int = 2_000,
) -> HierarchicalStateScreenRun:
    """Tune fine-state pooling on development seasons and run frozen replay."""

    print("[hierarchical-state] loading frozen Split NAIL state", flush=True)
    production = _load_production_state(Path(production_run_dir))
    states: dict[str, ConstrainedSplitState] = joblib.load(
        Path(constrained_run_dir) / "season_states.joblib"
    )
    designs: dict[str, SplitNailDesign] = joblib.load(Path(design_cache_path))
    targets = (*DEVELOPMENT_TARGET_SEASONS, *FROZEN_TARGET_SEASONS)
    required_seasons = _screen_seasons(targets)
    if not set(required_seasons).issubset(designs) or not set(required_seasons).issubset(states):
        raise ValueError("Constrained Split artifacts do not cover the hierarchical screen seasons")

    cache_dir = Path(observation_cache_dir) / Path(constrained_run_dir).name
    coverage_rows: list[dict[str, object]] = []
    for index, season in enumerate(required_seasons, start=1):
        path = cache_dir / f"{season}.parquet"
        if path.is_file():
            rows = pd.read_parquet(path)
            _validate_fine_state_rows(rows)
            print(f"[hierarchical-state] cache {season} ({index}/{len(required_seasons)})", flush=True)
        else:
            print(
                f"[hierarchical-state] baseline and states {season} ({index}/{len(required_seasons)})",
                flush=True,
            )
            baseline = _frozen_split_baseline(season, designs[season], states, production)
            rows, _ = build_possession_state_observations(
                season,
                design=designs[season],
                baseline_prediction=baseline,
                curated_dir=curated_dir,
                analytical_dir=analytical_dir,
                include_start_state=True,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(path, index=False)
        coverage_rows.append(_coverage_from_fine_rows(rows))

    prepared_by_target, eligibility_rows = _prepare_target_windows(cache_dir, targets)
    selection_rows: list[dict[str, object]] = []
    for child_precision in CHILD_RELATIVE_PRECISION_GRID:
        for season in DEVELOPMENT_TARGET_SEASONS:
            print(
                f"[hierarchical-state] select child_precision={child_precision:g}; target={season}",
                flush=True,
            )
            source, target = prepared_by_target[season]
            fit = fit_hierarchical_state_model(source, child_relative_precision=child_precision)
            metrics = hierarchical_state_metrics(predict_hierarchical_state_models(target, fit))
            selection_rows.extend(
                _metric_rows(metrics, child_precision, season, len(source.rows), len(target.rows))
            )
    selection = pd.DataFrame(selection_rows)
    summary = (
        selection.groupby(["child_relative_precision", "model", "metric"], as_index=False)
        .agg(
            value=("value", "mean"),
            minus_fixed_split_nail=("minus_fixed_split_nail", "mean"),
            minus_coarse_player_state=("minus_coarse_player_state", "mean"),
        )
        .sort_values(["metric", "model", "value", "child_relative_precision"], kind="stable")
        .reset_index(drop=True)
    )
    candidates = summary.loc[
        (summary["model"] == "hierarchical_fine_state")
        & (summary["metric"] == "conditional_possession_rmse")
    ].sort_values(["value", "child_relative_precision"], kind="stable")
    selected_child_precision = float(candidates.iloc[0]["child_relative_precision"])
    if selected_child_precision in {CHILD_RELATIVE_PRECISION_GRID[0], CHILD_RELATIVE_PRECISION_GRID[-1]}:
        raise ValueError(
            "Fine-state child precision selected a grid boundary: "
            f"{selected_child_precision:g}. Expand the grid before interpreting the screen."
        )
    print(
        f"[hierarchical-state] selected child_precision={selected_child_precision:g}; interior=True",
        flush=True,
    )

    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    for season in FROZEN_TARGET_SEASONS:
        print(f"[hierarchical-state] frozen replay {season}", flush=True)
        source, target = prepared_by_target[season]
        fit = fit_hierarchical_state_model(source, child_relative_precision=selected_child_precision)
        prediction = predict_hierarchical_state_models(target, fit)
        prediction.insert(1, "target_season", season)
        prediction_frames.append(prediction)
        metric_frames.append(hierarchical_state_metrics(prediction).assign(target_season=season))
        bootstrap_frames.append(
            paired_hierarchical_state_bootstrap(prediction, draws=bootstrap_draws).assign(target_season=season)
        )
        control_frames.append(_control_time_series_row(fit, season))
    frozen_predictions = pd.concat(prediction_frames, ignore_index=True)
    metric_frames.append(hierarchical_state_metrics(frozen_predictions).assign(target_season="pooled"))
    bootstrap_frames.append(
        paired_hierarchical_state_bootstrap(frozen_predictions, draws=bootstrap_draws).assign(
            target_season="pooled"
        )
    )
    final_source_rows = _read_source_window(cache_dir, "2026-27")
    final_specification = build_fine_state_specification(final_source_rows)
    final_source = prepare_fine_state_features(final_source_rows, final_specification)
    final_fit = fit_hierarchical_state_model(
        final_source, child_relative_precision=selected_child_precision
    )
    control_frames.append(_control_time_series_row(final_fit, "2026-27"))
    state_eligibility = _state_eligibility(eligibility_rows, final_specification)
    run = _write_run(
        coverage=pd.DataFrame(coverage_rows),
        selection=selection,
        selection_summary=summary,
        frozen_predictions=frozen_predictions,
        frozen_metrics=pd.concat(metric_frames, ignore_index=True),
        bootstrap=pd.concat(bootstrap_frames, ignore_index=True),
        state_eligibility=state_eligibility,
        control_time_series=pd.concat(control_frames, ignore_index=True),
        player_diagnostics=_player_child_diagnostics(final_source, final_fit),
        selected_child_precision=selected_child_precision,
        constrained_run_dir=Path(constrained_run_dir),
        production_run_dir=Path(production_run_dir),
        artifacts_dir=Path(artifacts_dir),
        bootstrap_draws=bootstrap_draws,
    )
    print(f"[hierarchical-state] complete: {run.run_dir}", flush=True)
    return run


def _prepare_target_windows(
    cache_dir: Path,
    targets: tuple[str, ...],
) -> tuple[dict[str, tuple[PreparedFineStateFeatures, PreparedFineStateFeatures]], list[dict[str, object]]]:
    prepared: dict[str, tuple[PreparedFineStateFeatures, PreparedFineStateFeatures]] = {}
    eligibility_rows: list[dict[str, object]] = []
    for season in targets:
        print(f"[hierarchical-state] prepare source design for {season}", flush=True)
        source_rows = _read_source_window(cache_dir, season)
        specification = build_fine_state_specification(source_rows)
        source = prepare_fine_state_features(source_rows, specification)
        target = prepare_fine_state_features(pd.read_parquet(cache_dir / f"{season}.parquet"), specification)
        prepared[season] = (source, target)
        for state, exposure in specification.source_state_exposure.items():
            eligibility_rows.append(
                {
                    "target_season": season,
                    "start_state": state,
                    "coarse_start_state": _coarse_state(state),
                    "source_possessions": exposure,
                    "eligible_for_player_child": state in specification.eligible_states,
                    "is_anchor": state in specification.anchors.values(),
                    "is_child_contrast": state in specification.child_states,
                }
            )
    return prepared, eligibility_rows


def _build_control_design(rows: pd.DataFrame, control_states: tuple[str, ...]) -> sparse.csr_matrix:
    state_columns = {state: index for index, state in enumerate(control_states)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    for index, state in enumerate(rows["start_state"].astype(str)):
        column = state_columns.get(state)
        if column is not None:
            row_indices.append(index)
            column_indices.append(column)
    return sparse.coo_matrix(
        (np.ones(len(row_indices), dtype=float), (row_indices, column_indices)),
        shape=(len(rows), len(control_states)),
    ).tocsr()


def _build_parent_design(
    rows: pd.DataFrame,
    specification: FineStateSpecification,
    player_columns: dict[int, int],
) -> sparse.csr_matrix:
    player_count = len(player_columns)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows.itertuples(index=False)):
        live = row.coarse_start_state == "live_ball"
        for player_id in row.offense_player_ids:
            column = player_columns.get(int(player_id))
            if column is not None:
                row_indices.append(row_index)
                column_indices.append(column)
                values.append(1.0 if live else -float(specification.offense_parent_ratio[column]))
        for player_id in row.defense_player_ids:
            column = player_columns.get(int(player_id))
            if column is not None:
                row_indices.append(row_index)
                column_indices.append(player_count + column)
                values.append(-1.0 if live else float(specification.defense_parent_ratio[column]))
    return sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), 2 * player_count),
    ).tocsr()


def _build_child_design(
    rows: pd.DataFrame,
    specification: FineStateSpecification,
    player_columns: dict[int, int],
) -> sparse.csr_matrix:
    player_count = len(player_columns)
    child_columns = {state: index for index, state in enumerate(specification.child_states)}
    child_count = len(child_columns)
    anchor_to_children: dict[str, list[tuple[str, int]]] = {}
    for state, state_column in child_columns.items():
        anchor_to_children.setdefault(specification.anchors[_coarse_state(state)], []).append(
            (state, state_column)
        )
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows.itertuples(index=False)):
        state = str(row.start_state)
        state_column = child_columns.get(state)
        matching_anchor_children = anchor_to_children.get(state, ())
        for player_id in row.offense_player_ids:
            player_column = player_columns.get(int(player_id))
            if player_column is None:
                continue
            if state_column is not None:
                ratio = specification.offense_child_ratio[player_column, state_column]
                if ratio > 0.0:
                    row_indices.append(row_index)
                    column_indices.append(player_column * child_count + state_column)
                    values.append(1.0)
            for child_state, child_column in matching_anchor_children:
                ratio = specification.offense_child_ratio[player_column, child_column]
                if ratio > 0.0:
                    row_indices.append(row_index)
                    column_indices.append(player_column * child_count + child_column)
                    values.append(-float(ratio))
        defense_offset = player_count * child_count
        for player_id in row.defense_player_ids:
            player_column = player_columns.get(int(player_id))
            if player_column is None:
                continue
            if state_column is not None:
                ratio = specification.defense_child_ratio[player_column, state_column]
                if ratio > 0.0:
                    row_indices.append(row_index)
                    column_indices.append(defense_offset + player_column * child_count + state_column)
                    values.append(-1.0)
            for child_state, child_column in matching_anchor_children:
                ratio = specification.defense_child_ratio[player_column, child_column]
                if ratio > 0.0:
                    row_indices.append(row_index)
                    column_indices.append(defense_offset + player_column * child_count + child_column)
                    values.append(float(ratio))
    return sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), 2 * player_count * child_count),
    ).tocsr()


def _child_precision_vector(
    specification: FineStateSpecification,
    child_relative_precision: float,
) -> np.ndarray:
    player_count = len(specification.player_ids)
    state_share = {
        state: exposure / sum(specification.source_state_exposure.values())
        for state, exposure in specification.source_state_exposure.items()
    }
    precision_per_child: list[float] = []
    for state in specification.child_states:
        anchor = specification.anchors[_coarse_state(state)]
        ratio = max(state_share[anchor] / state_share[state], 1.0)
        precision_per_child.append(child_relative_precision * ratio)
    one_side = np.tile(np.asarray(precision_per_child, dtype=float), player_count)
    return np.concatenate((one_side, one_side))


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=float)
    eligible = (numerator > 0.0) & (denominator > 0.0)
    output[eligible] = numerator[eligible] / denominator[eligible]
    return output


def _metric_rows(
    metrics: pd.DataFrame,
    child_precision: float,
    target_season: str,
    source_size: int,
    target_size: int,
) -> list[dict[str, object]]:
    return [
        {
            "child_relative_precision": child_precision,
            "target_season": target_season,
            "model": metric.model,
            "metric": metric.metric,
            "value": metric.value,
            "minus_fixed_split_nail": metric.minus_fixed_split_nail,
            "minus_coarse_player_state": metric.minus_coarse_player_state,
            "source_possessions": source_size,
            "target_possessions": target_size,
        }
        for metric in metrics.itertuples(index=False)
    ]


def _control_time_series_row(fit: HierarchicalStateFit, target_season: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    control_count = len(fit.specification.control_states)
    for model_name, model in (
        ("fine_state_control", fit.control_model),
        ("coarse_player_state", fit.coarse_model),
        ("hierarchical_fine_state", fit.full_model),
    ):
        rows.append(
            {
                "target_season": target_season,
                "model": model_name,
                "start_state": REFERENCE_CONTROL_STATE,
                "state_adjustment": 0.0,
                "baseline_recalibration": float(model.intercept_),
                "is_reference": True,
            }
        )
        for index, state in enumerate(fit.specification.control_states):
            rows.append(
                {
                    "target_season": target_season,
                    "model": model_name,
                    "start_state": state,
                    "state_adjustment": float(model.coef_[index]),
                    "baseline_recalibration": float(model.intercept_),
                    "is_reference": False,
                }
            )
    return pd.DataFrame(rows)


def _state_eligibility(
    rows: list[dict[str, object]], final: FineStateSpecification
) -> pd.DataFrame:
    output = pd.DataFrame(rows)
    summary = (
        output.groupby(["start_state", "coarse_start_state"], as_index=False)
        .agg(
            minimum_source_possessions=("source_possessions", "min"),
            median_source_possessions=("source_possessions", "median"),
            maximum_source_possessions=("source_possessions", "max"),
            windows=("target_season", "nunique"),
            eligible_in_all_windows=("eligible_for_player_child", "all"),
        )
        .sort_values(["coarse_start_state", "minimum_source_possessions"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary["latest_source_eligible"] = summary["start_state"].isin(final.eligible_states)
    summary["latest_source_anchor"] = summary["start_state"].isin(final.anchors.values())
    summary["latest_source_child_contrast"] = summary["start_state"].isin(final.child_states)
    summary["decision"] = np.where(
        summary["eligible_in_all_windows"],
        "player_child_tested",
        "control_only_sparse_state",
    )
    return summary


def _player_child_diagnostics(
    source: PreparedFineStateFeatures,
    fit: HierarchicalStateFit,
) -> pd.DataFrame:
    spec = source.specification
    player_count = len(spec.player_ids)
    child_count = len(spec.child_states)
    control_count = len(spec.control_states)
    coefficients = fit.full_model.coef_[control_count + 2 * player_count :]
    variance = fit.full_model.posterior_variance_[control_count + 2 * player_count :]
    offense_exposure: dict[tuple[int, str], int] = {}
    defense_exposure: dict[tuple[int, str], int] = {}
    for row in source.rows.itertuples(index=False):
        state = str(row.start_state)
        for player_id in row.offense_player_ids:
            key = (int(player_id), state)
            offense_exposure[key] = offense_exposure.get(key, 0) + 1
        for player_id in row.defense_player_ids:
            key = (int(player_id), state)
            defense_exposure[key] = defense_exposure.get(key, 0) + 1
    rows: list[dict[str, object]] = []
    for player_index, player_id in enumerate(spec.player_ids):
        for child_index, state in enumerate(spec.child_states):
            anchor = spec.anchors[_coarse_state(state)]
            offense_column = player_index * child_count + child_index
            defense_column = player_count * child_count + offense_column
            rows.append(
                {
                    "player_id": player_id,
                    "start_state": state,
                    "anchor_state": anchor,
                    "offense_child_contrast": float(coefficients[offense_column]),
                    "defense_child_contrast": float(coefficients[defense_column]),
                    "offense_standard_error": float(np.sqrt(variance[offense_column])),
                    "defense_standard_error": float(np.sqrt(variance[defense_column])),
                    "offense_source_state_possessions": offense_exposure.get((player_id, state), 0),
                    "defense_source_state_possessions": defense_exposure.get((player_id, state), 0),
                    "source_seasons": ",".join(sorted(source.rows["season"].astype(str).unique())),
                }
            )
    output = pd.DataFrame(rows)
    catalog_path = Path("data/catalog/players.parquet")
    if catalog_path.is_file():
        names = pd.read_parquet(catalog_path, columns=["player_id", "display_name"])
        names["player_id"] = pd.to_numeric(names["player_id"], errors="raise").astype(int)
        output = output.merge(names.drop_duplicates("player_id"), on="player_id", how="left")
    output["absolute_offense_child_contrast"] = output["offense_child_contrast"].abs()
    return (
        output.sort_values(
            ["start_state", "absolute_offense_child_contrast"],
            ascending=[True, False],
            kind="stable",
        )
        .drop(columns="absolute_offense_child_contrast")
        .reset_index(drop=True)
    )


def _coverage_from_fine_rows(rows: pd.DataFrame) -> dict[str, object]:
    _validate_fine_state_rows(rows)
    counts = rows["start_state"].value_counts()
    return {
        "season": str(rows["season"].iloc[0]),
        "known_state_single_lineup_possessions": int(len(rows)),
        "frozen_baseline_rmse": _rmse(rows["target"], rows["baseline_prediction"]),
        **{f"state_{state}_possessions": int(count) for state, count in counts.items()},
    }


def _state_to_coarse(rows: pd.DataFrame) -> dict[str, str]:
    pairs = rows.loc[:, ["start_state", "coarse_start_state"]].drop_duplicates()
    if pairs.duplicated("start_state").any():
        raise ValueError("A granular start state must map to exactly one coarse state")
    return dict(zip(pairs["start_state"].astype(str), pairs["coarse_start_state"].astype(str), strict=True))


def _coarse_state(state: str) -> str:
    if state in LIVE_BALL_STATES:
        return "live_ball"
    if state in DEAD_BALL_STATES:
        return "dead_ball"
    raise ValueError(f"Unknown granular state: {state}")


def _validate_fine_state_rows(rows: pd.DataFrame) -> None:
    required = {
        "season",
        "game_id",
        "possession_id",
        "home_offense",
        "offense_player_ids",
        "defense_player_ids",
        "target",
        "baseline_prediction",
        "start_state",
        "coarse_start_state",
        "weight",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Fine possession-state rows missing: {sorted(missing)}")
    if rows.empty or rows.duplicated(["game_id", "possession_id"]).any():
        raise ValueError("Fine possession-state rows must be non-empty and unique")
    if not rows["start_state"].isin(set(LIVE_BALL_STATES) | set(DEAD_BALL_STATES)).all():
        raise ValueError("Fine possession-state rows include a fallback or unknown state")
    expected = rows["start_state"].map(_coarse_state)
    if not expected.eq(rows["coarse_start_state"].astype(str)).all():
        raise ValueError("Fine granular and coarse start states are inconsistent")


def _write_run(
    *,
    coverage: pd.DataFrame,
    selection: pd.DataFrame,
    selection_summary: pd.DataFrame,
    frozen_predictions: pd.DataFrame,
    frozen_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    state_eligibility: pd.DataFrame,
    control_time_series: pd.DataFrame,
    player_diagnostics: pd.DataFrame,
    selected_child_precision: float,
    constrained_run_dir: Path,
    production_run_dir: Path,
    artifacts_dir: Path,
    bootstrap_draws: int,
) -> HierarchicalStateScreenRun:
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
        state_eligibility.to_parquet(staging / "state_eligibility.parquet", index=False)
        control_time_series.to_parquet(staging / "state_control_time_series.parquet", index=False)
        player_diagnostics.to_parquet(staging / "player_child_diagnostics.parquet", index=False)
        metadata = {
            "model": MODEL_NAME,
            "contract": "conditional_realized_hierarchical_start_state_on_frozen_split_nail",
            "not_a_pregame_candidate": True,
            "minimum_state_source_possessions": MINIMUM_STATE_SOURCE_POSSESSIONS,
            "coarse_player_relative_precision": PLAYER_DEVIATION_RELATIVE_PRECISION,
            "child_relative_precision_grid": list(CHILD_RELATIVE_PRECISION_GRID),
            "selected_child_relative_precision": selected_child_precision,
            "child_precision_frequency_scaling": "max(anchor_share / child_state_share, 1)",
            "development_target_seasons": list(DEVELOPMENT_TARGET_SEASONS),
            "frozen_target_seasons": list(FROZEN_TARGET_SEASONS),
            "source_window_seasons": SOURCE_WINDOW_SEASONS,
            "excluded_start_state": "audited_fallback",
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
    return HierarchicalStateScreenRun(
        run_dir=run_dir,
        run_id=run_id,
        selected_child_relative_precision=selected_child_precision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the hierarchical granular possession-state conditional screen"
    )
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
                run = run_hierarchical_possession_state_screen(bootstrap_draws=args.bootstrap_draws)
                print(f"Hierarchical possession-state screen: run={run.run_dir}", flush=True)
            finally:
                sys.stdout = stdout
        return
    run = run_hierarchical_possession_state_screen(bootstrap_draws=args.bootstrap_draws)
    print(f"Hierarchical possession-state screen: run={run.run_dir}")


if __name__ == "__main__":
    main()
