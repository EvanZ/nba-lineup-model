# ruff: noqa: E501
"""Compare the annual NAIL forecast with the prior live-posterior endpoint."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.in_season_nail_anchor_calibration import (
    DEFAULT_COMPLETED_RATINGS_PATH,
    DEFAULT_CURRENT_PRESEASON_PATH,
    DEFAULT_HOLDOUT_SEASONS,
    DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
    DEFAULT_SEASONS,
    DEFAULT_TRAINING_SEASONS,
    DEFAULT_VALIDATION_SEASONS,
    HistoricalAnchorReplaySeason,
    _line_svg,
    _paired_bootstrap,
    _prediction_metrics,
    _weight_mask,
    build_anchor_replays,
    replay_anchor_candidate,
    replay_anchor_states,
)
from nba_lineup_model.modeling.in_season_nail_calibration import (
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_GAME_CATALOG_PATH,
    DEFAULT_PANEL_PATH,
    DEFAULT_RUN_DIR,
    HistoricalReplaySeason,
    build_historical_replay_seasons,
    read_historical_replay_mart,
    write_historical_replay_mart,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_SOURCE_SEASONS = ("2019-20", *DEFAULT_SEASONS)
DEFAULT_LIVE_WEIGHT_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
DEFAULT_REPLAY_MART_PATH = Path(
    "artifacts/cache/in_season_nail_replay_mart/live-anchor-seasons.joblib"
)
DEFAULT_EXISTING_REPLAY_MART_PATH = Path(
    "artifacts/cache/in_season_nail_replay_mart/preseason-anchor-seasons.joblib"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/models/in_season_nail_live_anchor_calibration")
CALIBRATION_VERSION = "in-season-nail-live-anchor-v0.1.0"


@dataclass(frozen=True)
class HistoricalLiveAnchorReplaySeason:
    """Target-season replay with a forward-safe prior-season endpoint."""

    scalar_replay: HistoricalReplaySeason
    forward_preseason_nail: np.ndarray
    previous_live_endpoint_nail: np.ndarray
    previous_live_endpoint_available: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.scalar_replay.player_ids)
        forward = np.asarray(self.forward_preseason_nail, dtype=float)
        endpoint = np.asarray(self.previous_live_endpoint_nail, dtype=float)
        available = np.asarray(self.previous_live_endpoint_available, dtype=bool)
        if forward.shape != (count,) or not np.isfinite(forward).all():
            raise ValueError("Forward ratings must match the player vocabulary")
        if endpoint.shape != (count,) or not np.isfinite(endpoint).all():
            raise ValueError("Live endpoint ratings must match the player vocabulary")
        if available.shape != (count,):
            raise ValueError("Live endpoint availability must match the player vocabulary")
        object.__setattr__(self, "forward_preseason_nail", forward)
        object.__setattr__(self, "previous_live_endpoint_nail", endpoint)
        object.__setattr__(self, "previous_live_endpoint_available", available)


@dataclass(frozen=True)
class LiveAnchorCalibration:
    selected_live_weight: float
    recommended_live_weight: float
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    player_anchors: pd.DataFrame
    player_states: pd.DataFrame
    source_endpoints: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class LiveAnchorCalibrationRun:
    run_id: str
    run_dir: Path
    selected_live_weight: float
    recommended_live_weight: float


def build_live_anchor_replays(
    scalar_replays: Sequence[HistoricalReplaySeason],
    *,
    target_seasons: Sequence[str] = DEFAULT_SEASONS,
    completed_ratings_path: Path | str = DEFAULT_COMPLETED_RATINGS_PATH,
    player_panel_path: Path | str = DEFAULT_PANEL_PATH,
    prior_equivalent_possessions: float = DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
) -> tuple[
    tuple[HistoricalLiveAnchorReplaySeason, ...],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Replay every source season and carry only its forward-safe final state."""

    scalar = tuple(scalar_replays)
    if not scalar:
        raise ValueError("At least one scalar replay is required")
    base_replays, base_players = build_anchor_replays(
        scalar,
        completed_ratings_path=completed_ratings_path,
        player_panel_path=player_panel_path,
    )
    for optional_column in ("age", "nba_experience_years"):
        if optional_column not in base_players:
            base_players[optional_column] = np.nan
    base_by_season = {replay.scalar_replay.season: replay for replay in base_replays}
    endpoint_frames: list[pd.DataFrame] = []
    for season, replay in base_by_season.items():
        states = replay_anchor_states(
            replay,
            completed_weight=0.0,
            prior_equivalent_possessions=prior_equivalent_possessions,
        )
        endpoint = states.loc[states["is_postseason_endpoint"], ["player_id", "live_rating"]].copy()
        endpoint.insert(0, "season", season)
        endpoint = endpoint.rename(columns={"live_rating": "source_live_endpoint_nail"})
        endpoint_frames.append(endpoint)
    source_endpoints = pd.concat(endpoint_frames, ignore_index=True)

    targets = tuple(validate_season(value) for value in target_seasons)
    outputs: list[HistoricalLiveAnchorReplaySeason] = []
    player_rows: list[pd.DataFrame] = []
    for target in targets:
        if target not in base_by_season:
            raise ValueError(f"Missing target replay for {target}")
        source = _previous_season(target)
        source_frame = source_endpoints.loc[
            source_endpoints["season"].eq(source),
            ["player_id", "source_live_endpoint_nail"],
        ]
        if source_frame.empty:
            raise ValueError(f"Missing source live endpoint for {target}")
        base = base_by_season[target]
        players = base_players.loc[
            base_players["season"].eq(target),
            [
                "season",
                "player_id",
                "player_name",
                "forward_preseason_nail",
                "age",
                "nba_experience_years",
                "experience_cohort",
                "age_band",
                "prior_rating_band",
            ],
        ].merge(source_frame, on="player_id", how="left", validate="one_to_one")
        players["source_season"] = source
        players["previous_live_endpoint_available"] = players["source_live_endpoint_nail"].notna()
        players["previous_live_endpoint_nail"] = players["source_live_endpoint_nail"].fillna(
            players["forward_preseason_nail"]
        )
        players["live_endpoint_gap"] = (
            players["previous_live_endpoint_nail"] - players["forward_preseason_nail"]
        )
        outputs.append(
            HistoricalLiveAnchorReplaySeason(
                scalar_replay=base.scalar_replay,
                forward_preseason_nail=players["forward_preseason_nail"].to_numpy(float),
                previous_live_endpoint_nail=players["previous_live_endpoint_nail"].to_numpy(float),
                previous_live_endpoint_available=players[
                    "previous_live_endpoint_available"
                ].to_numpy(bool),
            )
        )
        player_rows.append(players)
    return tuple(outputs), pd.concat(player_rows, ignore_index=True), source_endpoints


def replay_live_anchor_candidate(
    replay: HistoricalLiveAnchorReplaySeason,
    *,
    live_endpoint_weight: float,
    prior_equivalent_possessions: float = DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
) -> pd.DataFrame:
    """Score a blend between annual forward and prior live endpoint."""

    proxy = _completed_proxy(replay)
    predictions = replay_anchor_candidate(
        proxy,
        completed_weight=live_endpoint_weight,
        prior_equivalent_possessions=prior_equivalent_possessions,
    ).rename(columns={"completed_weight": "live_endpoint_weight"})
    predictions["model"] = predictions["live_endpoint_weight"].map(_model_key)
    predictions["anchor_source"] = "previous_live_endpoint"
    return predictions


def calibrate_live_anchor(
    replays: Sequence[HistoricalLiveAnchorReplaySeason],
    *,
    player_anchors: pd.DataFrame,
    source_endpoints: pd.DataFrame,
    live_weight_grid: Sequence[float] = DEFAULT_LIVE_WEIGHT_GRID,
    prior_equivalent_possessions: float = DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
    training_seasons: Sequence[str] = DEFAULT_TRAINING_SEASONS,
    validation_seasons: Sequence[str] = DEFAULT_VALIDATION_SEASONS,
    holdout_seasons: Sequence[str] = DEFAULT_HOLDOUT_SEASONS,
) -> LiveAnchorCalibration:
    """Tune on development, verify on validation, and gate on untouched holdout."""

    payload = tuple(replays)
    weights = tuple(float(value) for value in live_weight_grid)
    if not payload:
        raise ValueError("At least one live-anchor replay is required")
    if (
        not weights
        or len(weights) != len(set(weights))
        or any(not np.isfinite(value) or value < 0 or value > 1 for value in weights)
    ):
        raise ValueError("Live-endpoint weight grid must be unique and within [0, 1]")
    available = {replay.scalar_replay.season for replay in payload}
    training = _validate_split(training_seasons, available, "training")
    validation = _validate_split(validation_seasons, available, "validation")
    holdout = _validate_split(holdout_seasons, available, "holdout")
    if (
        set(training) & set(validation)
        or set(training) & set(holdout)
        or set(validation) & set(holdout)
    ):
        raise ValueError("Calibration splits must be disjoint")

    predictions = pd.concat(
        [
            replay_live_anchor_candidate(
                replay,
                live_endpoint_weight=weight,
                prior_equivalent_possessions=prior_equivalent_possessions,
            )
            for replay in payload
            for weight in weights
        ],
        ignore_index=True,
    )
    split_map = {
        **{season: "training" for season in training},
        **{season: "validation" for season in validation},
        **{season: "holdout" for season in holdout},
    }
    predictions["split"] = predictions["season"].map(split_map).fillna("unused")
    metric_input = predictions.rename(columns={"live_endpoint_weight": "completed_weight"})
    metrics = _prediction_metrics(metric_input).rename(
        columns={"completed_weight": "live_endpoint_weight"}
    )
    training_metrics = metrics.loc[
        metrics["split"].eq("training") & metrics["scope"].eq("pooled")
    ].sort_values(
        ["margin_rmse", "log_loss", "brier_score", "live_endpoint_weight"],
        kind="stable",
    )
    if training_metrics.empty:
        raise ValueError("No training metrics were produced")
    selected = float(training_metrics.iloc[0]["live_endpoint_weight"])
    recommended = _recommended_weight(metrics, selected)

    states: list[pd.DataFrame] = []
    trajectory_ids = {201939}
    for replay in payload:
        if replay.scalar_replay.season not in {*validation, *holdout}:
            continue
        for weight in sorted({0.0, selected, 1.0}):
            frame = replay_anchor_states(
                _completed_proxy(replay),
                completed_weight=weight,
                prior_equivalent_possessions=prior_equivalent_possessions,
                trajectory_player_ids=trajectory_ids,
            ).rename(columns={"completed_weight": "live_endpoint_weight"})
            frame["model"] = frame["live_endpoint_weight"].map(_model_key)
            states.append(frame)
    player_states = pd.concat(states, ignore_index=True)
    bootstrap_input = predictions.rename(columns={"live_endpoint_weight": "completed_weight"})
    diagnostics = _paired_bootstrap(bootstrap_input, selected_weight=selected).rename(
        columns={"candidate_weight": "live_endpoint_weight"}
    )
    return LiveAnchorCalibration(
        selected_live_weight=selected,
        recommended_live_weight=recommended,
        predictions=predictions,
        metrics=metrics,
        player_anchors=player_anchors,
        player_states=player_states,
        source_endpoints=source_endpoints,
        diagnostics=diagnostics,
    )


def write_calibration_run(
    calibration: LiveAnchorCalibration,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    current_preseason_path: Path | str = DEFAULT_CURRENT_PRESEASON_PATH,
) -> LiveAnchorCalibrationRun:
    created_at = datetime.now(UTC)
    run_id = f"live-anchor-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(output_dir)
    staging = root / f".{run_id}.tmp"
    target = root / run_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        frames = {
            "game_predictions.parquet": calibration.predictions,
            "metrics.parquet": calibration.metrics,
            "player_anchors.parquet": calibration.player_anchors,
            "player_states.parquet": calibration.player_states,
            "source_endpoints.parquet": calibration.source_endpoints,
            "diagnostics.parquet": calibration.diagnostics,
        }
        for filename, frame in frames.items():
            frame.to_parquet(staging / filename, index=False)
        parameters = {
            "schema_version": 1,
            "calibration_version": CALIBRATION_VERSION,
            "selected_previous_live_weight": calibration.selected_live_weight,
            "recommended_previous_live_weight": calibration.recommended_live_weight,
            "prior_equivalent_possessions": DEFAULT_PRIOR_EQUIVALENT_POSSESSIONS,
            "training_seasons": DEFAULT_TRAINING_SEASONS,
            "validation_seasons": DEFAULT_VALIDATION_SEASONS,
            "holdout_seasons": DEFAULT_HOLDOUT_SEASONS,
            "production_serving_changed": False,
        }
        (staging / "parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True) + "\n"
        )
        bridge = _current_bridge(
            calibration,
            current_preseason_path=current_preseason_path,
        )
        _write_html_report(staging / "report.html", calibration, bridge)
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
    return LiveAnchorCalibrationRun(
        run_id,
        target,
        calibration.selected_live_weight,
        calibration.recommended_live_weight,
    )


def run_study(
    *,
    replay_mart_path: Path | str = DEFAULT_REPLAY_MART_PATH,
    rebuild_replay_mart: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> LiveAnchorCalibrationRun:
    targets = DEFAULT_SOURCE_SEASONS
    mart_path = Path(replay_mart_path)
    manifest_path = mart_path.with_suffix(mart_path.suffix + ".manifest.json")
    if mart_path.is_file() and manifest_path.is_file() and not rebuild_replay_mart:
        scalar_replays = read_historical_replay_mart(mart_path, expected_seasons=targets)
    else:
        scalar_replays = _load_or_extend_scalar_replays(targets)
        write_historical_replay_mart(scalar_replays, mart_path)
    live_replays, player_anchors, source_endpoints = build_live_anchor_replays(
        scalar_replays,
        target_seasons=DEFAULT_SEASONS,
    )
    calibration = calibrate_live_anchor(
        live_replays,
        player_anchors=player_anchors,
        source_endpoints=source_endpoints,
    )
    return write_calibration_run(calibration, output_dir=output_dir)


def _load_or_extend_scalar_replays(
    seasons: Sequence[str],
) -> tuple[HistoricalReplaySeason, ...]:
    requested = tuple(seasons)
    existing: dict[str, HistoricalReplaySeason] = {}
    existing_path = DEFAULT_EXISTING_REPLAY_MART_PATH
    existing_manifest = existing_path.with_suffix(existing_path.suffix + ".manifest.json")
    if existing_path.is_file() and existing_manifest.is_file():
        for replay in read_historical_replay_mart(existing_path):
            existing[replay.season] = replay
    missing = tuple(season for season in requested if season not in existing)
    if missing:
        built = build_historical_replay_seasons(
            seasons=missing,
            run_dir=DEFAULT_RUN_DIR,
            player_season_panel_path=DEFAULT_PANEL_PATH,
            analytical_dir=DEFAULT_ANALYTICAL_DIR,
            game_catalog_path=DEFAULT_GAME_CATALOG_PATH,
        )
        existing.update({replay.season: replay for replay in built})
    return tuple(existing[season] for season in requested)


def _completed_proxy(
    replay: HistoricalLiveAnchorReplaySeason,
) -> HistoricalAnchorReplaySeason:
    return HistoricalAnchorReplaySeason(
        scalar_replay=replay.scalar_replay,
        forward_preseason_nail=replay.forward_preseason_nail,
        previous_completed_nail=replay.previous_live_endpoint_nail,
        previous_completed_available=replay.previous_live_endpoint_available,
    )


def _recommended_weight(metrics: pd.DataFrame, selected: float) -> float:
    if np.isclose(selected, 0.0):
        return 0.0
    holdout = metrics.loc[metrics["split"].eq("holdout") & metrics["scope"].eq("pooled")]
    candidate = _metric_row(holdout, selected)
    forward = _metric_row(holdout, 0.0)
    primary = ("margin_mae", "margin_rmse", "brier_score", "log_loss")
    return selected if all(candidate[key] < forward[key] for key in primary) else 0.0


def _metric_row(frame: pd.DataFrame, weight: float) -> pd.Series:
    matches = frame.loc[np.isclose(frame["live_endpoint_weight"].to_numpy(float), weight)]
    if len(matches) != 1:
        raise ValueError(f"Expected one metric row for weight {weight:g}")
    return matches.iloc[0]


def _current_bridge(
    calibration: LiveAnchorCalibration,
    *,
    current_preseason_path: Path | str,
) -> pd.DataFrame:
    forward = pd.read_parquet(current_preseason_path)
    target_values = tuple(forward["season"].astype(str).unique())
    if len(target_values) != 1:
        raise ValueError("Current preseason rankings must contain one season")
    source = _previous_season(target_values[0])
    endpoints = calibration.source_endpoints.loc[
        calibration.source_endpoints["season"].eq(source),
        ["player_id", "source_live_endpoint_nail"],
    ]
    bridge = forward.loc[
        :, ["player_id", "player_name", "rapm", "prior_rating", "additive_profile_adjustment"]
    ].rename(columns={"rapm": "forward_preseason_nail"})
    bridge = bridge.merge(endpoints, on="player_id", how="left", validate="one_to_one")
    bridge["previous_live_available"] = bridge["source_live_endpoint_nail"].notna()
    bridge["previous_live_endpoint_nail"] = bridge["source_live_endpoint_nail"].fillna(
        bridge["forward_preseason_nail"]
    )
    gap = bridge["previous_live_endpoint_nail"] - bridge["forward_preseason_nail"]
    bridge["development_selected_anchor"] = (
        bridge["forward_preseason_nail"] + calibration.selected_live_weight * gap
    )
    bridge["recommended_anchor"] = (
        bridge["forward_preseason_nail"] + calibration.recommended_live_weight * gap
    )
    bridge["candidate_minus_forward"] = (
        bridge["development_selected_anchor"] - bridge["forward_preseason_nail"]
    )
    return bridge


def _write_html_report(
    path: Path,
    calibration: LiveAnchorCalibration,
    current_bridge: pd.DataFrame,
) -> None:
    selected = calibration.selected_live_weight
    recommended = calibration.recommended_live_weight
    metrics = calibration.metrics
    summary = metrics.loc[
        metrics["scope"].eq("pooled")
        & metrics["split"].isin(["training", "validation", "holdout"])
        & _weight_mask(metrics["live_endpoint_weight"], (0.0, selected, 1.0))
    ].copy()
    summary["Anchor"] = summary["live_endpoint_weight"].map(
        lambda value: _model_label(float(value), selected)
    )
    summary["Split"] = summary["split"].str.title()
    summary["Winner %"] = 100 * summary["winner_accuracy"]
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

    holdout = metrics.loc[metrics["split"].eq("holdout") & metrics["scope"].eq("pooled")]
    candidate = _metric_row(holdout, selected)
    forward = _metric_row(holdout, 0.0)
    deltas = {
        key: float(candidate[key] - forward[key])
        for key in ("margin_mae", "margin_rmse", "brier_score", "log_loss")
    }
    if np.isclose(recommended, 0.0):
        decision = "Keep the annual forward forecast; the live endpoint did not clear holdout."
    elif np.isclose(recommended, 1.0):
        decision = "Carry the prior live posterior forward as the opening anchor."
    else:
        decision = (
            f"Use {recommended:.0%} prior live posterior / {1 - recommended:.0%} annual forward."
        )
    interpretation = (
        f"Development selected {selected:.0%} prior-live weight. On the untouched 2025-26 "
        f"holdout, it changed margin RMSE by {deltas['margin_rmse']:+.4f}, Brier by "
        f"{deltas['brier_score']:+.6f}, and log loss by {deltas['log_loss']:+.6f}. "
        "Negative is better."
    )

    bootstrap = calibration.diagnostics.loc[
        np.isclose(
            pd.to_numeric(calibration.diagnostics["live_endpoint_weight"], errors="coerce"),
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

    candidate_label = f"{selected:.0%} live candidate"
    curry = current_bridge.loc[current_bridge["player_id"].eq(201939)].rename(
        columns={
            "player_name": "Player",
            "forward_preseason_nail": "Annual forward",
            "previous_live_endpoint_nail": "Prior live endpoint",
            "development_selected_anchor": candidate_label,
            "recommended_anchor": "Recommended anchor",
        }
    )
    curry_table = curry.loc[
        :,
        ["Player", "Annual forward", "Prior live endpoint", candidate_label, "Recommended anchor"],
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:+.3f}")

    biggest = current_bridge.loc[current_bridge["previous_live_available"]].copy()
    biggest = biggest.sort_values(
        "candidate_minus_forward", key=lambda values: values.abs(), ascending=False
    ).head(12)
    biggest = biggest.rename(
        columns={
            "player_name": "Player",
            "forward_preseason_nail": "Annual forward",
            "previous_live_endpoint_nail": "Prior live",
            "development_selected_anchor": candidate_label,
            "candidate_minus_forward": "Change",
        }
    )
    biggest_table = biggest.loc[
        :, ["Player", "Annual forward", "Prior live", candidate_label, "Change"]
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:+.3f}")

    training = metrics.loc[
        metrics["split"].eq("training") & metrics["scope"].eq("pooled")
    ].sort_values("live_endpoint_weight")
    curve_svg = _line_svg(
        [
            (
                training["live_endpoint_weight"].to_numpy(float),
                training["margin_rmse"].to_numpy(float),
                "Training RMSE",
                "#245d47",
            )
        ],
        x_label="Weight on prior live endpoint",
        y_label="Margin RMSE",
        marker_x=selected,
    )
    trajectory = calibration.player_states.loc[
        calibration.player_states["season"].eq("2025-26")
        & calibration.player_states["player_id"].eq(201939)
        & ~calibration.player_states["is_postseason_endpoint"]
        & _weight_mask(calibration.player_states["live_endpoint_weight"], (0.0, selected, 1.0))
    ].copy()
    dates = sorted(trajectory["prediction_date"].unique())
    date_index = {value: index for index, value in enumerate(dates)}
    trajectory_series = []
    colors = {0.0: "#9a3d2b", selected: "#245d47", 1.0: "#355f86"}
    for weight, frame in trajectory.groupby("live_endpoint_weight", sort=True):
        ordered = frame.sort_values("prediction_date")
        trajectory_series.append(
            (
                np.asarray([date_index[value] for value in ordered["prediction_date"]], float),
                ordered["live_rating"].to_numpy(float),
                _model_label(float(weight), selected),
                colors.get(float(weight), "#555"),
            )
        )
    trajectory_svg = _line_svg(
        trajectory_series,
        x_label="2025-26 game dates",
        y_label="Live NAIL",
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Live-posterior anchor study</title>
<style>body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;max-width:1220px;margin:36px auto;padding:0 24px;color:#17211b;background:#faf9f5}}h1,h2{{letter-spacing:-.025em}}h2{{margin-top:38px}}.lede{{font-size:1.08rem;max-width:930px;line-height:1.55}}.decision{{border-left:6px solid #245d47;background:#eaf2ed;padding:20px 24px;margin:24px 0}}.decision strong{{display:block;font-size:1.22rem;margin-bottom:8px}}.note{{background:#f0eee7;border-left:5px solid #8b7441;padding:15px 20px;line-height:1.5}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}}table{{border-collapse:collapse;width:100%;margin:18px 0;background:white}}th,td{{padding:9px 11px;border-bottom:1px solid #d5dbd5;text-align:right;font-variant-numeric:tabular-nums}}th{{background:#edf1ec}}th:first-child,td:first-child{{text-align:left}}.small{{color:#59645c;font-size:.9rem}}svg{{width:100%;height:auto;background:white;border:1px solid #dfe4df}}@media(max-width:880px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>Should live NAIL carry its prior-season posterior forward?</h1><p class="lede">This extension tests the missing forward-safe anchor. Each source season is replayed chronologically from its annual forecast; its final live posterior is then available as an opening-night input for the next season. The possession updater, damping, context, and frozen minutes are identical across candidates.</p>
<div class="decision"><strong>{html.escape(decision)}</strong>{html.escape(interpretation)}</div>
<p class="note"><strong>No hidden hindsight:</strong> the prior live endpoint uses only games completed before the target season. It is not the retrospective completed-season NAIL fit. Cold starts absent from the source season retain the annual forward forecast.</p>
<h2>Direct game-level findings</h2>{summary_table}
<div class="grid"><section><h2>Training blend curve</h2>{curve_svg}</section><section><h2>Paired holdout uncertainty</h2>{bootstrap_table}<p class="small">Candidate − annual forward below zero favors carrying the live posterior.</p></section></div>
<h2>Curry bridge into 2026-27</h2>{curry_table}<p>This directly tests the discontinuity that the earlier report left unexplained: annual +1.454 versus the forward-safe 2025-26 endpoint near +2.080.</p>
<h2>Curry 2025-26 replay under each opening anchor</h2>{trajectory_svg}<p class="small">Each point is available before that date's games; same-day games cannot update one another.</p>
<h2>Largest current-season candidate changes</h2>{biggest_table}
<h2>Study limits</h2><ul><li>The blend is selected only on 2020-21 through 2023-24, checked on 2024-25, and gated on untouched 2025-26.</li><li>ATS remains omitted because the compact historical replay lacks a comparable archived closing line for every game.</li><li>This tests direct carry-forward and global blends. It does not yet fit a separate offseason transition model.</li></ul>
<p class="small">Generated {generated}. Production serving remains unchanged.</p></body></html>"""
    path.write_text(page)


def _model_key(weight: float) -> str:
    if np.isclose(weight, 0.0):
        return "annual_forward_anchor"
    if np.isclose(weight, 1.0):
        return "prior_live_endpoint_anchor"
    return f"live_endpoint_blend_{weight:.2f}"


def _model_label(weight: float, selected: float) -> str:
    if np.isclose(weight, 0.0):
        return "Annual forward"
    if np.isclose(weight, 1.0):
        return "Prior live endpoint"
    suffix = " (selected)" if np.isclose(weight, selected) else ""
    return f"{weight:.0%} live / {1 - weight:.0%} annual{suffix}"


def _validate_split(values: Sequence[str], available: set[str], label: str) -> tuple[str, ...]:
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
    parser = argparse.ArgumentParser(description="Compare annual and prior-live NAIL anchors")
    parser.add_argument("--replay-mart-path", type=Path, default=DEFAULT_REPLAY_MART_PATH)
    parser.add_argument("--rebuild-replay-mart", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run = run_study(
        replay_mart_path=args.replay_mart_path,
        rebuild_replay_mart=args.rebuild_replay_mart,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "selected_previous_live_weight": run.selected_live_weight,
                "recommended_previous_live_weight": run.recommended_live_weight,
                "report": str(run.run_dir / "report.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
