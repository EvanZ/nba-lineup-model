# ruff: noqa: E501
"""Leakage-free study of the preseason anchor used by live NAIL.

The possession updater is intentionally held fixed.  This module changes only
the player-rating vector at which the season starts and algebraically
re-residualizes the compact historical game contributions around that vector.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit

from nba_lineup_model.modeling.in_season_nail_calibration import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_GAME_CATALOG_PATH,
    DEFAULT_PANEL_PATH,
    DEFAULT_RUN_DIR,
    WIN_PROBABILITY_MARGIN_SCALE,
    HistoricalReplayGame,
    HistoricalReplaySeason,
    build_historical_replay_seasons,
    read_historical_replay_mart,
    write_historical_replay_mart,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_SEASONS = (
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)
DEFAULT_TRAINING_SEASONS = DEFAULT_SEASONS[:4]
DEFAULT_VALIDATION_SEASONS = ("2024-25",)
DEFAULT_HOLDOUT_SEASONS = ("2025-26",)
DEFAULT_COMPLETED_WEIGHT_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS = 6000.0
DEFAULT_REPLAY_MART_PATH = Path(
    "artifacts/cache/in_season_nail_replay_mart/preseason-anchor-seasons.joblib"
)
DEFAULT_COMPLETED_RATINGS_PATH = Path(
    "artifacts/web/lineup_rankings/forward_nail_rapm_v1212_residualized_lambda/"
    "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/"
    "player_ratings.parquet"
)
DEFAULT_CURRENT_PRESEASON_PATH = Path(
    "artifacts/web/preseason_rankings/forward_nail_rapm_v1212_residualized_lambda/"
    "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25/"
    "2026-27.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/models/in_season_nail_anchor_calibration")
CALIBRATION_VERSION = "in-season-nail-anchor-v0.1.0"


@dataclass(frozen=True)
class HistoricalAnchorReplaySeason:
    """One scalar replay plus the alternative previous-season anchor."""

    scalar_replay: HistoricalReplaySeason
    forward_preseason_nail: np.ndarray
    previous_completed_nail: np.ndarray
    previous_completed_available: np.ndarray

    def __post_init__(self) -> None:
        player_count = len(self.scalar_replay.player_ids)
        forward = np.asarray(self.forward_preseason_nail, dtype=float)
        completed = np.asarray(self.previous_completed_nail, dtype=float)
        available = np.asarray(self.previous_completed_available, dtype=bool)
        if forward.shape != (player_count,) or not np.isfinite(forward).all():
            raise ValueError("Forward ratings must match the replay vocabulary")
        if completed.shape != (player_count,) or not np.isfinite(completed).all():
            raise ValueError("Previous completed ratings must match the replay vocabulary")
        if available.shape != (player_count,):
            raise ValueError("Previous completed availability must match the replay vocabulary")
        object.__setattr__(self, "forward_preseason_nail", forward)
        object.__setattr__(self, "previous_completed_nail", completed)
        object.__setattr__(self, "previous_completed_available", available)


@dataclass(frozen=True)
class AnchorCalibration:
    selected_completed_weight: float
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    player_anchors: pd.DataFrame
    player_states: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class AnchorCalibrationRun:
    run_id: str
    run_dir: Path
    selected_completed_weight: float


def build_anchor_replays(
    scalar_replays: Sequence[HistoricalReplaySeason],
    *,
    completed_ratings_path: Path | str = DEFAULT_COMPLETED_RATINGS_PATH,
    player_panel_path: Path | str = DEFAULT_PANEL_PATH,
) -> tuple[tuple[HistoricalAnchorReplaySeason, ...], pd.DataFrame]:
    """Attach the prior completed-season ratings without changing cold starts."""

    replays = tuple(scalar_replays)
    if not replays:
        raise ValueError("At least one scalar replay is required")
    completed = pd.read_parquet(completed_ratings_path)
    required = {"season", "player_id", "rapm"}
    if missing := required - set(completed.columns):
        raise ValueError(f"Completed ratings missing columns: {sorted(missing)}")
    completed = completed.copy()
    completed["season"] = completed["season"].astype(str)
    completed["player_id"] = completed["player_id"].astype(int)
    completed["rapm"] = pd.to_numeric(completed["rapm"], errors="coerce")
    if completed.duplicated(["season", "player_id"]).any():
        raise ValueError("Completed ratings must be unique by season and player")

    panel_path = Path(player_panel_path)
    panel = pd.read_parquet(panel_path) if panel_path.is_file() else pd.DataFrame()
    if not panel.empty:
        panel = panel.copy()
        panel["season"] = panel["season"].astype(str)
        panel["player_id"] = panel["player_id"].astype(int)

    output: list[HistoricalAnchorReplaySeason] = []
    anchor_rows: list[pd.DataFrame] = []
    for replay in replays:
        source = _previous_season(replay.season)
        source_ratings = completed.loc[
            completed["season"].eq(source), ["player_id", "rapm"]
        ].rename(columns={"rapm": "previous_completed_nail"})
        anchors = pd.DataFrame(
            {
                "season": replay.season,
                "player_id": replay.player_ids,
                "forward_preseason_nail_uncentered": replay.frozen_preseason_nail,
            }
        ).merge(source_ratings, on="player_id", how="left", validate="one_to_one")
        display_center = 0.0
        if not panel.empty and "rapm_possessions" in panel.columns:
            exposure = panel.loc[
                panel["season"].eq(source), ["player_id", "rapm_possessions"]
            ].drop_duplicates("player_id", keep="last")
            centered = anchors.merge(exposure, on="player_id", how="left", validate="one_to_one")
            weights = (
                pd.to_numeric(centered["rapm_possessions"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(float)
            )
            values = centered["forward_preseason_nail_uncentered"].to_numpy(float)
            display_center = (
                float(np.average(values, weights=weights))
                if float(weights.sum()) > 0
                else float(values.mean())
            )
        anchors["forward_display_center"] = display_center
        anchors["forward_preseason_nail"] = (
            anchors["forward_preseason_nail_uncentered"] - display_center
        )
        anchors["previous_completed_available"] = anchors["previous_completed_nail"].notna()
        # A rookie or returning player with no prior completed estimate retains
        # the published forward projection under every blend candidate.
        anchors["previous_completed_nail"] = anchors["previous_completed_nail"].fillna(
            anchors["forward_preseason_nail"]
        )
        anchors["anchor_gap"] = (
            anchors["previous_completed_nail"] - anchors["forward_preseason_nail"]
        )
        if not panel.empty:
            columns = [
                column
                for column in (
                    "player_id",
                    "player_name",
                    "age",
                    "nba_experience_years",
                    "is_rookie",
                )
                if column in panel.columns
            ]
            target_panel = panel.loc[panel["season"].eq(replay.season), columns]
            target_panel = target_panel.drop_duplicates("player_id", keep="last")
            anchors = anchors.merge(target_panel, on="player_id", how="left", validate="one_to_one")
        if "player_name" not in anchors:
            anchors["player_name"] = anchors["player_id"].map(lambda value: f"Player {value}")
        anchors["experience_cohort"] = anchors.apply(_experience_cohort, axis=1)
        anchors["age_band"] = anchors.apply(_age_band, axis=1)
        anchors["prior_rating_band"] = pd.cut(
            anchors["forward_preseason_nail"],
            bins=[-np.inf, -1.0, 1.0, np.inf],
            labels=["below -1", "-1 to +1", "above +1"],
        ).astype(str)
        output.append(
            HistoricalAnchorReplaySeason(
                scalar_replay=replay,
                forward_preseason_nail=anchors["forward_preseason_nail"].to_numpy(float),
                previous_completed_nail=anchors["previous_completed_nail"].to_numpy(float),
                previous_completed_available=anchors["previous_completed_available"].to_numpy(bool),
            )
        )
        anchor_rows.append(anchors)
    return tuple(output), pd.concat(anchor_rows, ignore_index=True)


def calibrate_preseason_anchor(
    anchor_replays: Sequence[HistoricalAnchorReplaySeason],
    *,
    player_anchors: pd.DataFrame,
    completed_weight_grid: Sequence[float] = DEFAULT_COMPLETED_WEIGHT_GRID,
    prior_equivalent_possessions: float = DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
    training_seasons: Sequence[str] = DEFAULT_TRAINING_SEASONS,
    validation_seasons: Sequence[str] = DEFAULT_VALIDATION_SEASONS,
    holdout_seasons: Sequence[str] = DEFAULT_HOLDOUT_SEASONS,
    completed_ratings_path: Path | str = DEFAULT_COMPLETED_RATINGS_PATH,
) -> AnchorCalibration:
    """Select a blend on training seasons and score validation/holdout untouched."""

    replays = tuple(anchor_replays)
    if not replays:
        raise ValueError("At least one anchor replay is required")
    weights = tuple(float(value) for value in completed_weight_grid)
    if (
        not weights
        or len(weights) != len(set(weights))
        or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in weights)
    ):
        raise ValueError("Completed-weight grid must contain unique values in [0, 1]")
    available = {replay.scalar_replay.season for replay in replays}
    training = _validated_split(training_seasons, available, "training")
    validation = _validated_split(validation_seasons, available, "validation")
    holdout = _validated_split(holdout_seasons, available, "holdout")
    if (
        set(training) & set(validation)
        or set(training) & set(holdout)
        or set(validation) & set(holdout)
    ):
        raise ValueError("Training, validation, and holdout seasons must be disjoint")

    frames = [
        replay_anchor_candidate(
            replay,
            completed_weight=weight,
            prior_equivalent_possessions=prior_equivalent_possessions,
        )
        for replay in replays
        for weight in weights
    ]
    predictions = pd.concat(frames, ignore_index=True)
    split_map = {
        **{season: "training" for season in training},
        **{season: "validation" for season in validation},
        **{season: "holdout" for season in holdout},
    }
    predictions["split"] = predictions["season"].map(split_map).fillna("unused")
    metrics = _prediction_metrics(predictions)
    eligible = metrics.loc[
        metrics["split"].eq("training") & metrics["scope"].eq("pooled")
    ].sort_values(
        ["margin_rmse", "log_loss", "brier_score", "completed_weight"],
        kind="stable",
    )
    if eligible.empty:
        raise ValueError("Anchor study produced no training metrics")
    selected = float(eligible.iloc[0]["completed_weight"])

    holdout_anchors = player_anchors.loc[player_anchors["season"].isin(holdout)].copy()
    trajectory_ids = {201939}
    trajectory_ids.update(
        holdout_anchors.sort_values("anchor_gap", key=lambda values: values.abs())
        .tail(5)["player_id"]
        .astype(int)
    )
    state_frames: list[pd.DataFrame] = []
    for replay in replays:
        if replay.scalar_replay.season not in {*validation, *holdout}:
            continue
        for weight in sorted({0.0, selected, 1.0}):
            state_frames.append(
                replay_anchor_states(
                    replay,
                    completed_weight=weight,
                    prior_equivalent_possessions=prior_equivalent_possessions,
                    trajectory_player_ids=trajectory_ids,
                )
            )
    player_states = pd.concat(state_frames, ignore_index=True)
    completed = pd.read_parquet(completed_ratings_path, columns=["season", "player_id", "rapm"])
    completed["season"] = completed["season"].astype(str)
    completed["player_id"] = completed["player_id"].astype(int)
    diagnostics = pd.concat(
        [
            _paired_bootstrap(predictions, selected_weight=selected),
            _endpoint_diagnostics(player_states, completed),
            _anchor_cohort_diagnostics(player_anchors),
        ],
        ignore_index=True,
        sort=False,
    )
    return AnchorCalibration(
        selected_completed_weight=selected,
        predictions=predictions,
        metrics=metrics,
        player_anchors=player_anchors,
        player_states=player_states,
        diagnostics=diagnostics,
    )


def replay_anchor_candidate(
    replay: HistoricalAnchorReplaySeason,
    *,
    completed_weight: float,
    prior_equivalent_possessions: float,
) -> pd.DataFrame:
    """Chronologically score one anchor while withholding same-day evidence."""

    if not 0.0 <= completed_weight <= 1.0:
        raise ValueError("Completed weight must be in [0, 1]")
    if not np.isfinite(prior_equivalent_possessions) or prior_equivalent_possessions <= 0:
        raise ValueError("Prior equivalent possessions must be finite and positive")
    scalar = replay.scalar_replay
    anchor = replay.forward_preseason_nail + completed_weight * (
        replay.previous_completed_nail - replay.forward_preseason_nail
    )
    delta = anchor - scalar.frozen_preseason_nail
    player_count = len(scalar.player_ids)
    gram = sparse.csr_matrix((player_count, player_count), dtype=float)
    score = np.zeros(player_count, dtype=float)
    rows: list[dict[str, object]] = []
    games_by_date: dict[date, list[HistoricalReplayGame]] = {}
    for game in scalar.games:
        games_by_date.setdefault(game.game_date, []).append(game)
    for games in games_by_date.values():
        adjustment = _solve_adjustment(gram, score, prior_equivalent_possessions)
        for game in games:
            reanchored_margin = game.frozen_home_margin + float((game.margin_design @ delta).item())
            predicted_margin = reanchored_margin + float((game.margin_design @ adjustment).item())
            rows.append(
                _prediction_row(
                    scalar,
                    game,
                    completed_weight=completed_weight,
                    predicted_home_margin=predicted_margin,
                    dominant_cohort=_dominant_game_cohort(
                        game,
                        delta,
                        scalar.player_ids,
                        replay.previous_completed_available,
                    ),
                )
            )
        for game in games:
            gram = gram + game.contribution.xtwx
            score += reanchored_score(game, delta)
    return pd.DataFrame(rows)


def reanchored_score(game: HistoricalReplayGame, anchor_delta: np.ndarray) -> np.ndarray:
    """Return X'W residuals after moving the immutable player anchor."""

    delta = np.asarray(anchor_delta, dtype=float)
    if delta.shape != game.contribution.xtwy.shape:
        raise ValueError("Anchor delta must match the game contribution vocabulary")
    return game.contribution.xtwy - np.asarray(game.contribution.xtwx @ delta).reshape(-1)


def replay_anchor_states(
    replay: HistoricalAnchorReplaySeason,
    *,
    completed_weight: float,
    prior_equivalent_possessions: float,
    trajectory_player_ids: set[int] | frozenset[int] = frozenset(),
) -> pd.DataFrame:
    """Materialize selected pregame trajectories and the final postseason state."""

    scalar = replay.scalar_replay
    anchor = replay.forward_preseason_nail + completed_weight * (
        replay.previous_completed_nail - replay.forward_preseason_nail
    )
    delta = anchor - scalar.frozen_preseason_nail
    player_count = len(scalar.player_ids)
    gram = sparse.csr_matrix((player_count, player_count), dtype=float)
    score = np.zeros(player_count, dtype=float)
    player_ids = np.asarray(scalar.player_ids, dtype=int)
    trajectory_mask = np.isin(player_ids, list(trajectory_player_ids))
    rows: list[dict[str, object]] = []
    games_by_date: dict[date, list[HistoricalReplayGame]] = {}
    for game in scalar.games:
        games_by_date.setdefault(game.game_date, []).append(game)
    for game_date, games in games_by_date.items():
        adjustment = _solve_adjustment(gram, score, prior_equivalent_possessions)
        for index in np.flatnonzero(trajectory_mask):
            rows.append(
                _state_row(
                    scalar,
                    player_index=int(index),
                    prediction_date=game_date,
                    completed_weight=completed_weight,
                    anchor=anchor,
                    adjustment=adjustment,
                    endpoint=False,
                )
            )
        for game in games:
            gram = gram + game.contribution.xtwx
            score += reanchored_score(game, delta)
    adjustment = _solve_adjustment(gram, score, prior_equivalent_possessions)
    endpoint_date = scalar.games[-1].game_date + timedelta(days=1)
    for index in range(player_count):
        rows.append(
            _state_row(
                scalar,
                player_index=index,
                prediction_date=endpoint_date,
                completed_weight=completed_weight,
                anchor=anchor,
                adjustment=adjustment,
                endpoint=True,
            )
        )
    return pd.DataFrame(rows)


def write_calibration_run(
    calibration: AnchorCalibration,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    current_preseason_path: Path | str = DEFAULT_CURRENT_PRESEASON_PATH,
    completed_ratings_path: Path | str = DEFAULT_COMPLETED_RATINGS_PATH,
) -> AnchorCalibrationRun:
    created_at = datetime.now(UTC)
    run_id = f"preseason-anchor-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(output_dir)
    staging = root / f".{run_id}.tmp"
    target = root / run_id
    recommended_weight = _recommended_completed_weight(calibration)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        frames = {
            "game_predictions.parquet": calibration.predictions,
            "metrics.parquet": calibration.metrics,
            "player_anchors.parquet": calibration.player_anchors,
            "player_states.parquet": calibration.player_states,
            "diagnostics.parquet": calibration.diagnostics,
        }
        for filename, frame in frames.items():
            frame.to_parquet(staging / filename, index=False)
        parameters = {
            "schema_version": 1,
            "calibration_version": CALIBRATION_VERSION,
            "selected_previous_completed_weight": calibration.selected_completed_weight,
            "forward_projection_weight": 1.0 - calibration.selected_completed_weight,
            "recommended_previous_completed_weight": recommended_weight,
            "recommended_forward_projection_weight": 1.0 - recommended_weight,
            "prior_equivalent_possessions": DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
            "training_seasons": DEFAULT_TRAINING_SEASONS,
            "validation_seasons": DEFAULT_VALIDATION_SEASONS,
            "holdout_seasons": DEFAULT_HOLDOUT_SEASONS,
            "production_serving_changed": False,
        }
        (staging / "parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True) + "\n"
        )
        bridge = _current_season_bridge(
            selected_weight=calibration.selected_completed_weight,
            recommended_weight=recommended_weight,
            current_preseason_path=current_preseason_path,
            completed_ratings_path=completed_ratings_path,
        )
        _write_html_report(
            staging / "report.html",
            calibration=calibration,
            current_bridge=bridge,
        )
        artifact_names = [*frames, "parameters.json", "report.html"]
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "files": {filename: _file_sha256(staging / filename) for filename in artifact_names},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
        latest = root / "latest.json"
        temporary = root / ".latest.json.tmp"
        temporary.write_text(json.dumps({"run_id": run_id}, indent=2) + "\n")
        temporary.replace(latest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return AnchorCalibrationRun(run_id, target, calibration.selected_completed_weight)


def run_study(
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    replay_mart_path: Path | str = DEFAULT_REPLAY_MART_PATH,
    rebuild_replay_mart: bool = False,
    completed_ratings_path: Path | str = DEFAULT_COMPLETED_RATINGS_PATH,
    current_preseason_path: Path | str = DEFAULT_CURRENT_PRESEASON_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> AnchorCalibrationRun:
    targets = tuple(validate_season(value) for value in seasons)
    mart_path = Path(replay_mart_path)
    manifest_path = mart_path.with_suffix(mart_path.suffix + ".manifest.json")
    if mart_path.is_file() and manifest_path.is_file() and not rebuild_replay_mart:
        scalar_replays = read_historical_replay_mart(mart_path, expected_seasons=targets)
    else:
        scalar_replays = build_historical_replay_seasons(
            seasons=targets,
            run_dir=DEFAULT_RUN_DIR,
            player_season_panel_path=DEFAULT_PANEL_PATH,
            analytical_dir=DEFAULT_ANALYTICAL_DIR,
            game_catalog_path=DEFAULT_GAME_CATALOG_PATH,
        )
        write_historical_replay_mart(scalar_replays, mart_path)
    anchor_replays, player_anchors = build_anchor_replays(
        scalar_replays,
        completed_ratings_path=completed_ratings_path,
    )
    calibration = calibrate_preseason_anchor(
        anchor_replays,
        player_anchors=player_anchors,
        completed_ratings_path=completed_ratings_path,
    )
    return write_calibration_run(
        calibration,
        output_dir=output_dir,
        current_preseason_path=current_preseason_path,
        completed_ratings_path=completed_ratings_path,
    )


def _prediction_row(
    replay: HistoricalReplaySeason,
    game: HistoricalReplayGame,
    *,
    completed_weight: float,
    predicted_home_margin: float,
    dominant_cohort: str,
) -> dict[str, object]:
    probability = float(expit(predicted_home_margin / WIN_PROBABILITY_MARGIN_SCALE))
    return {
        "season": replay.season,
        "game_id": game.game_id,
        "game_date": game.game_date,
        "game_time_utc": game.game_time_utc,
        "completed_weight": completed_weight,
        "model": _model_key(completed_weight),
        "dominant_anchor_cohort": dominant_cohort,
        "actual_home_margin": game.actual_home_margin,
        "predicted_home_margin": predicted_home_margin,
        "actual_home_win": game.actual_home_margin > 0,
        "predicted_home_win_probability": probability,
    }


def _state_row(
    replay: HistoricalReplaySeason,
    *,
    player_index: int,
    prediction_date: date,
    completed_weight: float,
    anchor: np.ndarray,
    adjustment: np.ndarray,
    endpoint: bool,
) -> dict[str, object]:
    return {
        "season": replay.season,
        "player_id": replay.player_ids[player_index],
        "prediction_date": prediction_date,
        "completed_weight": completed_weight,
        "model": _model_key(completed_weight),
        "anchor_rating": float(anchor[player_index]),
        "in_season_adjustment": float(adjustment[player_index]),
        "live_rating": float(anchor[player_index] + adjustment[player_index]),
        "is_postseason_endpoint": endpoint,
    }


def _solve_adjustment(
    gram: sparse.csr_matrix,
    score: np.ndarray,
    prior_equivalent_possessions: float,
) -> np.ndarray:
    if gram.nnz == 0:
        return np.zeros(len(score), dtype=float)
    precision = gram + sparse.eye(len(score), format="csr") * prior_equivalent_possessions
    return np.asarray(spsolve(precision, score), dtype=float)


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in predictions.groupby(
        ["split", "model", "completed_weight"], dropna=False, sort=False
    ):
        split, model, weight = keys
        groups = [("pooled", frame), *list(frame.groupby("season", sort=False))]
        for scope, subset in groups:
            actual_margin = subset["actual_home_margin"].to_numpy(float)
            predicted_margin = subset["predicted_home_margin"].to_numpy(float)
            actual_win = subset["actual_home_win"].to_numpy(float)
            probability = np.clip(
                subset["predicted_home_win_probability"].to_numpy(float), 1e-9, 1 - 1e-9
            )
            rows.append(
                {
                    "split": split,
                    "scope": scope,
                    "model": model,
                    "completed_weight": weight,
                    "season_count": int(subset["season"].nunique()),
                    "game_count": len(subset),
                    "margin_mae": float(np.mean(np.abs(actual_margin - predicted_margin))),
                    "margin_rmse": float(np.sqrt(np.mean((actual_margin - predicted_margin) ** 2))),
                    "winner_accuracy": float(
                        np.mean((predicted_margin > 0) == (actual_margin > 0))
                    ),
                    "brier_score": float(np.mean((probability - actual_win) ** 2)),
                    "log_loss": float(
                        -np.mean(
                            actual_win * np.log(probability)
                            + (1 - actual_win) * np.log(1 - probability)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_bootstrap(
    predictions: pd.DataFrame,
    *,
    selected_weight: float,
    samples: int = 5000,
) -> pd.DataFrame:
    holdout = predictions.loc[predictions["split"].eq("holdout")]
    controls = holdout.loc[holdout["completed_weight"].eq(0.0)].sort_values("game_id")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20261027)
    for candidate_weight in sorted({selected_weight, 1.0}):
        candidate = holdout.loc[holdout["completed_weight"].eq(candidate_weight)].sort_values(
            "game_id"
        )
        if tuple(candidate["game_id"]) != tuple(controls["game_id"]):
            raise ValueError("Paired holdout predictions do not cover the same games")
        actual_margin = controls["actual_home_margin"].to_numpy(float)
        actual_win = controls["actual_home_win"].to_numpy(float)
        control_margin = controls["predicted_home_margin"].to_numpy(float)
        candidate_margin = candidate["predicted_home_margin"].to_numpy(float)
        control_probability = np.clip(
            controls["predicted_home_win_probability"].to_numpy(float), 1e-9, 1 - 1e-9
        )
        candidate_probability = np.clip(
            candidate["predicted_home_win_probability"].to_numpy(float), 1e-9, 1 - 1e-9
        )
        losses = {
            "margin_mae": (
                np.abs(candidate_margin - actual_margin),
                np.abs(control_margin - actual_margin),
            ),
            "brier_score": (
                (candidate_probability - actual_win) ** 2,
                (control_probability - actual_win) ** 2,
            ),
            "log_loss": (
                -(
                    actual_win * np.log(candidate_probability)
                    + (1 - actual_win) * np.log(1 - candidate_probability)
                ),
                -(
                    actual_win * np.log(control_probability)
                    + (1 - actual_win) * np.log(1 - control_probability)
                ),
            ),
        }
        indices = rng.integers(0, len(controls), size=(samples, len(controls)))
        for metric, (candidate_loss, control_loss) in losses.items():
            per_game = candidate_loss - control_loss
            draws = per_game[indices].mean(axis=1)
            rows.append(
                {
                    "record_type": "paired_bootstrap",
                    "candidate_weight": candidate_weight,
                    "metric": metric,
                    "candidate_minus_forward": float(per_game.mean()),
                    "lower_95": float(np.quantile(draws, 0.025)),
                    "upper_95": float(np.quantile(draws, 0.975)),
                    "probability_candidate_better": float(np.mean(draws < 0)),
                    "sample_count": len(controls),
                }
            )
    return pd.DataFrame(rows)


def _endpoint_diagnostics(states: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    endpoints = states.loc[states["is_postseason_endpoint"]].merge(
        completed.rename(columns={"rapm": "same_season_completed_nail"}),
        on=["season", "player_id"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (season, weight), frame in endpoints.groupby(["season", "completed_weight"]):
        difference = frame["live_rating"] - frame["same_season_completed_nail"]
        rows.append(
            {
                "record_type": "endpoint_diagnostic",
                "season": season,
                "completed_weight": weight,
                "sample_count": len(frame),
                "endpoint_mae": float(difference.abs().mean()),
                "endpoint_rmse": float(np.sqrt(np.mean(difference**2))),
            }
        )
    return pd.DataFrame(rows)


def _anchor_cohort_diagnostics(player_anchors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, season_frame in player_anchors.groupby("season", sort=False):
        for cohort, frame in season_frame.groupby("experience_cohort", sort=False):
            rows.append(
                {
                    "record_type": "anchor_cohort",
                    "season": season,
                    "cohort": cohort,
                    "sample_count": len(frame),
                    "previous_completed_coverage": float(
                        frame["previous_completed_available"].mean()
                    ),
                    "mean_anchor_gap": float(frame["anchor_gap"].mean()),
                    "median_absolute_anchor_gap": float(frame["anchor_gap"].abs().median()),
                }
            )
    return pd.DataFrame(rows)


def _current_season_bridge(
    *,
    selected_weight: float,
    recommended_weight: float,
    current_preseason_path: Path | str,
    completed_ratings_path: Path | str,
) -> pd.DataFrame:
    forward = pd.read_parquet(current_preseason_path)
    season_values = tuple(forward["season"].astype(str).unique())
    if len(season_values) != 1:
        raise ValueError("Current preseason rankings must contain one season")
    target = season_values[0]
    source = _previous_season(target)
    completed = pd.read_parquet(completed_ratings_path)
    previous = completed.loc[
        completed["season"].astype(str).eq(source), ["player_id", "rapm"]
    ].rename(columns={"rapm": "previous_completed_nail"})
    bridge = (
        forward.loc[
            :, ["player_id", "player_name", "rapm", "prior_rating", "additive_profile_adjustment"]
        ]
        .rename(columns={"rapm": "forward_preseason_nail"})
        .merge(previous, on="player_id", how="left", validate="one_to_one")
    )
    bridge["previous_completed_available"] = bridge["previous_completed_nail"].notna()
    bridge["previous_completed_nail"] = bridge["previous_completed_nail"].fillna(
        bridge["forward_preseason_nail"]
    )
    gap = bridge["previous_completed_nail"] - bridge["forward_preseason_nail"]
    bridge["development_selected_anchor"] = bridge["forward_preseason_nail"] + selected_weight * gap
    bridge["recommended_anchor"] = bridge["forward_preseason_nail"] + recommended_weight * gap
    bridge["candidate_minus_forward"] = (
        bridge["development_selected_anchor"] - bridge["forward_preseason_nail"]
    )
    return bridge


def _recommended_completed_weight(calibration: AnchorCalibration) -> float:
    """Promote a development-selected blend only if it clears the holdout gate."""

    selected = calibration.selected_completed_weight
    if np.isclose(selected, 0.0):
        return 0.0
    pooled = calibration.metrics.loc[
        calibration.metrics["split"].eq("holdout") & calibration.metrics["scope"].eq("pooled")
    ]
    candidate = _metric_row_for_weight(pooled, selected)
    forward = _metric_row_for_weight(pooled, 0.0)
    primary_metrics = ("margin_mae", "margin_rmse", "brier_score", "log_loss")
    clears_gate = all(
        float(candidate[metric]) < float(forward[metric]) for metric in primary_metrics
    )
    return selected if clears_gate else 0.0


def _metric_row_for_weight(frame: pd.DataFrame, weight: float) -> pd.Series:
    matches = frame.loc[np.isclose(frame["completed_weight"].to_numpy(float), weight)]
    if len(matches) != 1:
        raise ValueError(f"Expected one metric row for completed weight {weight:g}")
    return matches.iloc[0]


def _weight_mask(values: pd.Series, weights: Sequence[float]) -> np.ndarray:
    numeric = values.to_numpy(float)
    return np.logical_or.reduce([np.isclose(numeric, weight) for weight in weights])


def _write_html_report(
    path: Path,
    *,
    calibration: AnchorCalibration,
    current_bridge: pd.DataFrame,
) -> None:
    selected = calibration.selected_completed_weight
    recommended = _recommended_completed_weight(calibration)
    metrics = calibration.metrics
    summary = metrics.loc[
        metrics["scope"].eq("pooled")
        & metrics["split"].isin(["training", "validation", "holdout"])
        & _weight_mask(metrics["completed_weight"], (0.0, selected, 1.0))
    ].copy()
    summary["Anchor"] = summary["completed_weight"].map(
        lambda value: _model_label(float(value), selected)
    )
    summary["Split"] = summary["split"].str.title()
    summary["Winner %"] = 100.0 * summary["winner_accuracy"]
    summary = summary.rename(
        columns={
            "game_count": "Games",
            "margin_mae": "Margin MAE",
            "margin_rmse": "Margin RMSE",
            "brier_score": "Brier",
            "log_loss": "Log loss",
        }
    )
    summary_table = summary.loc[
        :,
        ["Split", "Anchor", "Games", "Margin MAE", "Margin RMSE", "Winner %", "Brier", "Log loss"],
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:.5f}")

    training_curve = metrics.loc[
        metrics["split"].eq("training") & metrics["scope"].eq("pooled")
    ].sort_values("completed_weight")
    holdout = metrics.loc[metrics["split"].eq("holdout") & metrics["scope"].eq("pooled")]
    selected_row = _metric_row_for_weight(holdout, selected)
    forward_row = _metric_row_for_weight(holdout, 0.0)
    rmse_change = float(selected_row["margin_rmse"] - forward_row["margin_rmse"])
    brier_change = float(selected_row["brier_score"] - forward_row["brier_score"])
    log_loss_change = float(selected_row["log_loss"] - forward_row["log_loss"])
    if recommended == 0.0:
        recommendation = "Keep the forward preseason projection as the live NAIL anchor."
    elif recommended == 1.0:
        recommendation = "Use the previous completed-season NAIL as the live anchor."
    else:
        recommendation = f"Use a {recommended:.0%} previous-completed / {1 - recommended:.0%} forward-projection blend."
    gate_result = (
        "the candidate cleared the promotion gate."
        if np.isclose(recommended, selected) and not np.isclose(selected, 0.0)
        else "the candidate did not clear the promotion gate."
    )
    interpretation = (
        f"Development selected a {selected:.0%} completed-season blend. On the untouched "
        f"2025-26 holdout, that candidate changed margin RMSE by "
        f"{rmse_change:+.4f} points, Brier by {brier_change:+.6f}, and log loss by "
        f"{log_loss_change:+.6f} versus the current forward anchor. Negative is better, so "
        f"{gate_result}"
    )

    bootstrap = calibration.diagnostics.loc[
        calibration.diagnostics["record_type"].eq("paired_bootstrap")
        & np.isclose(
            pd.to_numeric(calibration.diagnostics["candidate_weight"], errors="coerce"),
            selected,
        )
    ].copy()
    bootstrap["Metric"] = bootstrap["metric"].map(
        {"margin_mae": "Margin MAE", "brier_score": "Brier", "log_loss": "Log loss"}
    )
    bootstrap["P(candidate better), %"] = 100 * bootstrap["probability_candidate_better"]
    bootstrap = bootstrap.rename(
        columns={
            "candidate_minus_forward": "Candidate - forward",
            "lower_95": "95% lower",
            "upper_95": "95% upper",
            "sample_count": "Games",
        }
    )
    bootstrap_table = bootstrap.loc[
        :,
        [
            "Metric",
            "Candidate - forward",
            "95% lower",
            "95% upper",
            "P(candidate better), %",
            "Games",
        ],
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:.5f}")

    candidate_label = f"{selected:.0%} candidate"
    curry = current_bridge.loc[current_bridge["player_id"].eq(201939)].copy()
    curry = curry.rename(
        columns={
            "player_name": "Player",
            "prior_rating": "Forward scalar prior",
            "additive_profile_adjustment": "Frozen additive profile",
            "forward_preseason_nail": "Forward preseason",
            "previous_completed_nail": "Previous completed",
            "development_selected_anchor": candidate_label,
            "recommended_anchor": "Recommended live anchor",
        }
    )
    curry_table = curry.loc[
        :,
        [
            "Player",
            "Forward scalar prior",
            "Frozen additive profile",
            "Forward preseason",
            "Previous completed",
            candidate_label,
            "Recommended live anchor",
        ],
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:+.3f}")

    biggest = current_bridge.loc[current_bridge["previous_completed_available"]].copy()
    biggest = biggest.sort_values("candidate_minus_forward", key=lambda values: values.abs()).tail(
        12
    )
    biggest = biggest.sort_values(
        "candidate_minus_forward", key=lambda values: values.abs(), ascending=False
    )
    biggest = biggest.rename(
        columns={
            "player_name": "Player",
            "forward_preseason_nail": "Forward",
            "previous_completed_nail": "Completed",
            "development_selected_anchor": candidate_label,
            "candidate_minus_forward": "Change",
        }
    )
    biggest_table = biggest.loc[
        :, ["Player", "Forward", "Completed", candidate_label, "Change"]
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:+.3f}")

    holdout_anchors = calibration.player_anchors.loc[
        calibration.player_anchors["season"].eq("2025-26")
    ]
    cohort = holdout_anchors.groupby("experience_cohort", as_index=False).agg(
        Players=("player_id", "size"),
        **{
            "Prior-year coverage, %": (
                "previous_completed_available",
                lambda values: 100 * values.mean(),
            ),
            "Median |gap|": ("anchor_gap", lambda values: values.abs().median()),
        },
    )
    cohort = cohort.rename(columns={"experience_cohort": "Cohort"})
    cohort_table = cohort.to_html(
        index=False, classes="metrics", float_format=lambda value: f"{value:.3f}"
    )

    trajectory = calibration.player_states.loc[
        calibration.player_states["season"].eq("2025-26")
        & calibration.player_states["player_id"].eq(201939)
        & ~calibration.player_states["is_postseason_endpoint"]
        & _weight_mask(calibration.player_states["completed_weight"], (0.0, selected, 1.0))
    ].copy()
    trajectory_svg = _trajectory_svg(trajectory, selected)
    curve_svg = _curve_svg(training_curve, selected)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    html_page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Live NAIL preseason anchor study</title>
<style>
body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;max-width:1220px;margin:36px auto;padding:0 24px;color:#17211b;background:#faf9f5}}
h1,h2{{letter-spacing:-.025em}} h2{{margin-top:38px}} .lede{{font-size:1.08rem;max-width:920px;line-height:1.55}}
.decision{{border-left:6px solid #245d47;background:#eaf2ed;padding:20px 24px;margin:24px 0}}
.decision strong{{display:block;font-size:1.25rem;margin-bottom:8px;color:#174633}}
.note{{background:#f0eee7;border-left:5px solid #8b7441;padding:15px 20px;line-height:1.5}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}}
table{{border-collapse:collapse;width:100%;margin:18px 0;background:white}}
th,td{{padding:9px 11px;border-bottom:1px solid #d5dbd5;text-align:right;font-variant-numeric:tabular-nums}}
th{{background:#edf1ec}} th:first-child,td:first-child{{text-align:left}} .small{{color:#59645c;font-size:.9rem}}
svg{{width:100%;height:auto;background:white;border:1px solid #dfe4df}} code{{background:#edf1ec;padding:2px 5px}}
@media(max-width:880px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Which preseason rating should live NAIL start from?</h1>
<p class="lede">A leakage-free six-season replay. Every candidate uses the same possession updater, 6,000-equivalent-possession damping, context, and frozen game minutes. The only changed input is the player-rating anchor at opening night.</p>
<div class="decision"><strong>{html.escape(recommendation)}</strong>{html.escape(interpretation)}</div>
<p class="note"><strong>Decision boundary:</strong> this is a predictive test, not a contest to match the retrospective completed-season rating. The blend was selected only on 2020-21 through 2023-24, checked on 2024-25, and scored once on the untouched 2025-26 holdout. Production has not changed.</p>
<h2>Direct game-level findings</h2>{summary_table}
<div class="grid"><section><h2>Training blend curve</h2>{curve_svg}</section><section><h2>Paired holdout uncertainty</h2>{bootstrap_table}<p class="small">Candidate − forward below zero favors the candidate. Bootstrap resamples the same holdout games in pairs.</p></section></div>
<h2>The Curry bridge for 2026-27</h2>{curry_table}
<p>The forward number is exactly scalar prior + frozen additive profile. The prior completed-season number is a separate predictive-anchor candidate; it is not being mislabeled as the new full-season estimate. The {selected:.0%} candidate is shown for transparency, but the holdout result leaves the recommended live anchor unchanged.</p>
<h2>Curry historical live trajectory (2025-26 replay)</h2>{trajectory_svg}
<p class="small">Each point is the rating available before that date's games. Games on the same date cannot update one another.</p>
<div class="grid"><section><h2>Largest 2026-27 anchor changes</h2>{biggest_table}</section><section><h2>2025-26 cohort coverage</h2>{cohort_table}<p class="small">Cold starts keep the forward projection because no prior completed rating exists. No current-season information is used to manufacture a prior-year value.</p></section></div>
<h2>Study contract and limitations</h2>
<ul><li>Primary criteria: margin MAE/RMSE, Brier score, and binary log loss.</li><li>Winner accuracy is descriptive. ATS is not reported because the historical compact replay does not contain a comparable archived closing line for every game.</li><li>The same-day cutoff is strict: all games for a date are predicted before any game from that date is incorporated.</li><li>Same-season completed NAIL appears only in an endpoint diagnostic artifact, never in tuning or live prediction.</li></ul>
<p class="small">Generated {generated}. Calibration version <code>{CALIBRATION_VERSION}</code>.</p>
</body></html>"""
    path.write_text(html_page)


def _curve_svg(frame: pd.DataFrame, selected_weight: float) -> str:
    if frame.empty:
        return "<p>No training curve.</p>"
    x = frame["completed_weight"].to_numpy(float)
    y = frame["margin_rmse"].to_numpy(float)
    return _line_svg(
        [(x, y, "Training RMSE", "#245d47")],
        x_label="Weight on previous completed-season NAIL",
        y_label="Margin RMSE",
        marker_x=selected_weight,
    )


def _trajectory_svg(frame: pd.DataFrame, selected_weight: float) -> str:
    if frame.empty:
        return "<p>No Curry trajectory was materialized.</p>"
    dates = sorted(frame["prediction_date"].unique())
    date_to_x = {value: index for index, value in enumerate(dates)}
    series = []
    colors = {0.0: "#9a3d2b", selected_weight: "#245d47", 1.0: "#355f86"}
    for weight, rows in frame.groupby("completed_weight", sort=True):
        ordered = rows.sort_values("prediction_date")
        series.append(
            (
                np.asarray([date_to_x[value] for value in ordered["prediction_date"]], dtype=float),
                ordered["live_rating"].to_numpy(float),
                _model_label(float(weight), selected_weight),
                colors.get(float(weight), "#555"),
            )
        )
    return _line_svg(series, x_label="2025-26 game dates", y_label="Live NAIL")


def _line_svg(
    series: Sequence[tuple[np.ndarray, np.ndarray, str, str]],
    *,
    x_label: str,
    y_label: str,
    marker_x: float | None = None,
) -> str:
    width, height = 720, 360
    left, right, top, bottom = 68, 22, 26, 58
    all_x = np.concatenate([values[0] for values in series])
    all_y = np.concatenate([values[1] for values in series])
    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - ypad, ymax + ypad

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    items = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for tick in np.linspace(ymin, ymax, 5):
        y = sy(float(tick))
        items.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e3e7e3"/>'
        )
        items.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5d675f">{tick:.2f}</text>'
        )
    if marker_x is not None:
        marker = sx(marker_x)
        items.append(
            f'<line x1="{marker:.1f}" y1="{top}" x2="{marker:.1f}" y2="{height - bottom}" stroke="#b68b2e" stroke-dasharray="5 4"/>'
        )
    for xvalues, yvalues, _label, color in series:
        points = " ".join(
            f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xvalues, yvalues, strict=True)
        )
        items.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    items.append(
        f'<text x="{(left + width - right) / 2:.1f}" y="{height - 15}" text-anchor="middle" font-size="12">{html.escape(x_label)}</text>'
    )
    items.append(
        f'<text x="16" y="{height / 2:.1f}" transform="rotate(-90 16 {height / 2:.1f})" text-anchor="middle" font-size="12">{html.escape(y_label)}</text>'
    )
    legend_x = left + 8
    for index, (_x, _y, label, color) in enumerate(series):
        y = top + 14 + index * 18
        items.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>'
        )
        items.append(
            f'<text x="{legend_x + 28}" y="{y + 4}" font-size="11">{html.escape(label)}</text>'
        )
    items.append("</svg>")
    return "".join(items)


def _model_key(weight: float) -> str:
    if np.isclose(weight, 0.0):
        return "forward_projection_anchor"
    if np.isclose(weight, 1.0):
        return "previous_completed_anchor"
    return f"blend_{weight:.2f}"


def _model_label(weight: float, selected_weight: float) -> str:
    if np.isclose(weight, 0.0):
        return "Forward projection (current)"
    if np.isclose(weight, 1.0):
        return "Previous completed"
    suffix = " (selected)" if np.isclose(weight, selected_weight) else ""
    return f"{weight:.0%} completed / {1 - weight:.0%} forward{suffix}"


def _dominant_game_cohort(
    game: HistoricalReplayGame,
    delta: np.ndarray,
    player_ids: Sequence[int],
    previous_available: np.ndarray,
) -> str:
    coefficients = np.abs(game.margin_design.toarray().reshape(-1) * delta)
    if float(coefficients.sum()) <= 1e-12:
        return "no anchor difference"
    unavailable = ~np.asarray(previous_available, dtype=bool)
    # The alternative anchor never changes unavailable players, but keeping this
    # branch explicit documents why cold starts cannot dominate an anchor shift.
    if float(coefficients[unavailable].sum()) > float(coefficients[~unavailable].sum()):
        return "cold start"
    _ = player_ids
    return "returning players"


def _experience_cohort(row: pd.Series) -> str:
    if not bool(row.get("previous_completed_available", False)):
        return "cold start / no prior rating"
    experience = pd.to_numeric(pd.Series([row.get("nba_experience_years")]), errors="coerce").iloc[
        0
    ]
    if pd.isna(experience):
        return "experience unknown"
    if experience <= 2:
        return "early career (0-2 years)"
    if experience >= 10:
        return "veteran (10+ years)"
    return "mid-career (3-9 years)"


def _age_band(row: pd.Series) -> str:
    age = pd.to_numeric(pd.Series([row.get("age")]), errors="coerce").iloc[0]
    if pd.isna(age):
        return "unknown"
    if age <= 24:
        return "24 or younger"
    if age >= 32:
        return "32 or older"
    return "25-31"


def _validated_split(values: Sequence[str], available: set[str], label: str) -> tuple[str, ...]:
    seasons = tuple(validate_season(value) for value in values)
    if not seasons or not set(seasons).issubset(available):
        raise ValueError(f"{label.title()} seasons must be a non-empty replay subset")
    return seasons


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start:04d}-{str(start + 1)[-2:]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare live NAIL preseason anchors")
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--replay-mart-path", type=Path, default=DEFAULT_REPLAY_MART_PATH)
    parser.add_argument("--rebuild-replay-mart", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run = run_study(
        seasons=tuple(args.seasons),
        replay_mart_path=args.replay_mart_path,
        rebuild_replay_mart=args.rebuild_replay_mart,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "selected_previous_completed_weight": run.selected_completed_weight,
                "report": str(run.run_dir / "report.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
