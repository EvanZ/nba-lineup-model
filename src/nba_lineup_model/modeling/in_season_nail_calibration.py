"""Historical calibration for the possession-based in-season NAIL updater."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit

from nba_lineup_model.modeling.contextual_features import lineup_side_context_features
from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    BacktestModel,
    _load_state,
    _schedule_predictor,
    _transformed_target_profiles,
)
from nba_lineup_model.modeling.frozen_prior_evaluation import _recover_home_intercept
from nba_lineup_model.modeling.in_season_nail import (
    GameContribution,
    build_game_contribution,
)
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.models.baselines import signed_entity_matrix, vocabulary_mapping
from nba_lineup_model.season.schema import validate_season
from nba_lineup_model.season.stats_v3_lake import (
    AwsCliImmutableObjectStore,
    ImmutableObjectStore,
    _join_s3_uri,
)
from nba_lineup_model.web_api.inference import _compiled_linear_x3_coefficients

DEFAULT_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_SELECTION_SEASONS = ("2023-24", "2024-25")
DEFAULT_PRIOR_GRID = (
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    6000.0,
    8000.0,
    10000.0,
    12000.0,
    16000.0,
)
DEFAULT_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_ANALYTICAL_DIR = Path("data/analytical")
DEFAULT_GAME_CATALOG_PATH = Path("data/catalog/games.parquet")
DEFAULT_RUN_DIR = Path(
    "artifacts/models/forward_nail_rapm_v1212_residualized_lambda/2025-26/"
    "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/models/in_season_nail_updater_calibration")
DEFAULT_REPLAY_MART_PATH = Path("artifacts/cache/in_season_nail_replay_mart/recent-seasons.joblib")
DEFAULT_CALIBRATION_LAKE_URI = (
    "s3://nba-gestalt-data/lake/analytical/in_season_nail_updater_calibration"
)
CORE_MODEL_VERSION = "NAIL-RAPM v1.2.1.3"
UPDATER_VERSION = "in-season-possession-v0.1.0"
WIN_PROBABILITY_MARGIN_SCALE = 6.5


@dataclass(frozen=True)
class HistoricalReplayGame:
    """One chronologically scored game and its postgame evidence."""

    game_id: str
    game_date: date
    game_time_utc: datetime
    actual_home_margin: float
    frozen_home_margin: float
    margin_design: sparse.csr_matrix
    contribution: GameContribution

    def __post_init__(self) -> None:
        design = sparse.csr_matrix(self.margin_design, dtype=float)
        if design.shape != (1, len(self.contribution.player_ids)):
            raise ValueError("Game margin design must match the contribution vocabulary")
        if not np.isfinite(self.actual_home_margin) or not np.isfinite(self.frozen_home_margin):
            raise ValueError("Historical game margins must be finite")
        object.__setattr__(self, "margin_design", design)


@dataclass(frozen=True)
class HistoricalReplaySeason:
    season: str
    player_ids: tuple[int, ...]
    frozen_preseason_nail: np.ndarray
    games: tuple[HistoricalReplayGame, ...]
    source_run_id: str

    def __post_init__(self) -> None:
        validate_season(self.season)
        ratings = np.asarray(self.frozen_preseason_nail, dtype=float)
        if ratings.shape != (len(self.player_ids),) or not np.isfinite(ratings).all():
            raise ValueError("Frozen preseason ratings must match the player vocabulary")
        if len({game.game_id for game in self.games}) != len(self.games):
            raise ValueError("Historical replay games must be unique")
        if tuple(sorted(self.games, key=_game_sort_key)) != self.games:
            raise ValueError("Historical replay games must be chronological")


@dataclass(frozen=True)
class InSeasonNailCalibration:
    selected_prior_equivalent_possessions: float
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    selection_seasons: tuple[str, ...]
    holdout_seasons: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationRun:
    run_dir: Path
    run_id: str
    selected_prior_equivalent_possessions: float


def write_historical_replay_mart(
    replay_seasons: Sequence[HistoricalReplaySeason],
    path: Path | str,
) -> Path:
    """Persist compact sparse replay inputs so grid searches do not rescan stints."""

    replays = tuple(replay_seasons)
    if not replays:
        raise ValueError("Historical replay mart cannot be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{uuid4().hex}")
    joblib.dump(replays, temporary, compress=3)
    digest = _file_sha256(temporary)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "mart_sha256": digest,
        "seasons": [
            {
                "season": replay.season,
                "source_run_id": replay.source_run_id,
                "player_count": len(replay.player_ids),
                "game_count": len(replay.games),
            }
            for replay in replays
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + f".tmp-{uuid4().hex}")
    manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    manifest_temporary.replace(manifest_path)
    return output


def read_historical_replay_mart(
    path: Path | str,
    *,
    expected_seasons: Sequence[str] | None = None,
    expected_source_run_id: str | None = None,
) -> tuple[HistoricalReplaySeason, ...]:
    """Read and validate a compact sparse historical replay mart."""

    source = Path(path)
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported historical replay-mart schema")
    if _file_sha256(source) != manifest.get("mart_sha256"):
        raise ValueError("Historical replay-mart checksum changed")
    loaded = joblib.load(source)
    if not isinstance(loaded, tuple) or not all(
        isinstance(replay, HistoricalReplaySeason) for replay in loaded
    ):
        raise ValueError("Historical replay mart has an invalid payload")
    replays: tuple[HistoricalReplaySeason, ...] = loaded
    seasons = tuple(replay.season for replay in replays)
    manifest_seasons = tuple(str(row["season"]) for row in manifest.get("seasons", []))
    if seasons != manifest_seasons:
        raise ValueError("Historical replay-mart manifest seasons do not match")
    if expected_seasons is not None and seasons != tuple(expected_seasons):
        raise ValueError("Historical replay mart does not cover the requested seasons")
    if expected_source_run_id is not None and any(
        replay.source_run_id != expected_source_run_id for replay in replays
    ):
        raise ValueError("Historical replay mart does not use the requested source run")
    return replays


def build_historical_replay_seasons(
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    run_dir: Path | str = DEFAULT_RUN_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
) -> tuple[HistoricalReplaySeason, ...]:
    """Build leakage-safe replay inputs from the already-materialized local history."""

    targets = tuple(validate_season(season) for season in seasons)
    if not targets or len(targets) != len(set(targets)) or targets != tuple(sorted(targets)):
        raise ValueError("Calibration seasons must be unique and chronological")
    run_path = Path(run_dir)
    candidate = BacktestModel(
        MODEL_NAME,
        CORE_MODEL_VERSION,
        profile_builder=partial(
            build_contextual_player_profiles,
            padding_contract=MEDVEDOVSKY_2020_PROFILE_PADDING,
            use_last_observed_profile=True,
        ),
        uses_schedule_control=True,
    )
    state = _load_state(run_path, candidate=candidate, target_seasons=targets)
    panel = pd.read_parquet(player_season_panel_path)
    exposure_cohort = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(targets[-1])],
        through_season=targets[-1],
        analytical_dir=analytical_dir,
    )
    schedule_features = build_back_to_back_game_features(pd.read_parquet(game_catalog_path))
    analytical_root = Path(analytical_dir)
    outputs: list[HistoricalReplaySeason] = []
    for target in targets:
        outputs.append(
            _build_historical_replay_season(
                target,
                state=state,
                candidate=candidate,
                panel=panel,
                exposure_cohort=exposure_cohort,
                schedule_features=schedule_features,
                analytical_dir=analytical_root,
            )
        )
    return tuple(outputs)


def calibrate_prior_strength(
    replay_seasons: Sequence[HistoricalReplaySeason],
    *,
    prior_grid: Sequence[float] = DEFAULT_PRIOR_GRID,
    selection_seasons: Sequence[str] = DEFAULT_SELECTION_SEASONS,
) -> InSeasonNailCalibration:
    """Select equivalent-possession precision on seasons preceding the holdout."""

    replays = tuple(replay_seasons)
    if not replays:
        raise ValueError("At least one historical replay season is required")
    candidates = tuple(float(value) for value in prior_grid)
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("Prior grid must contain finite positive values")
    if len(candidates) != len(set(candidates)):
        raise ValueError("Prior grid cannot contain duplicates")
    available = {replay.season for replay in replays}
    selection = tuple(validate_season(season) for season in selection_seasons)
    if not selection or not set(selection).issubset(available):
        raise ValueError("Selection seasons must be a non-empty replay subset")
    holdout = tuple(replay.season for replay in replays if replay.season not in selection)

    predictions = pd.concat(
        [
            _replay_prior_candidate(replay, prior_equivalent_possessions=candidate)
            for replay in replays
            for candidate in candidates
        ]
        + [_replay_frozen_control(replay) for replay in replays],
        ignore_index=True,
    )
    predictions["split"] = np.where(predictions["season"].isin(selection), "selection", "holdout")
    metrics = _calibration_metrics(predictions)
    eligible = metrics.loc[
        metrics["split"].eq("selection")
        & metrics["scope"].eq("pooled")
        & metrics["prior_equivalent_possessions"].notna()
    ].sort_values(
        ["margin_rmse", "brier_score", "prior_equivalent_possessions"],
        kind="stable",
    )
    if eligible.empty:
        raise ValueError("Calibration produced no selectable prior candidates")
    selected = float(eligible.iloc[0]["prior_equivalent_possessions"])
    return InSeasonNailCalibration(
        selected_prior_equivalent_possessions=selected,
        predictions=predictions,
        metrics=metrics,
        selection_seasons=selection,
        holdout_seasons=holdout,
    )


def write_calibration_run(
    calibration: InSeasonNailCalibration,
    *,
    replay_seasons: Sequence[HistoricalReplaySeason],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> CalibrationRun:
    """Atomically persist the calibrated updater separately from core NAIL."""

    created_at = datetime.now(UTC)
    run_id = f"in-season-nail-calibration-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(output_dir)
    target = root / run_id
    staging = root / f".{run_id}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        predictions_path = staging / "game_predictions.parquet"
        metrics_path = staging / "metrics.parquet"
        calibration.predictions.to_parquet(predictions_path, index=False)
        calibration.metrics.to_parquet(metrics_path, index=False)
        parameters = {
            "schema_version": 1,
            "core_model_version": CORE_MODEL_VERSION,
            "state_updater_version": UPDATER_VERSION,
            "selected_prior_equivalent_possessions": (
                calibration.selected_prior_equivalent_possessions
            ),
            "selection_seasons": calibration.selection_seasons,
            "holdout_seasons": calibration.holdout_seasons,
            "win_probability_link": {
                "function": "logistic",
                "margin_scale": WIN_PROBABILITY_MARGIN_SCALE,
                "selection_role": "fixed comparative metric only",
            },
        }
        (staging / "updater_parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True) + "\n"
        )
        sources = [
            {
                "season": replay.season,
                "source_run_id": replay.source_run_id,
                "player_count": len(replay.player_ids),
                "game_count": len(replay.games),
                "first_game_date": replay.games[0].game_date.isoformat(),
                "last_game_date": replay.games[-1].game_date.isoformat(),
            }
            for replay in replay_seasons
        ]
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "sources": sources,
            "files": {
                path.name: _file_sha256(path)
                for path in (
                    predictions_path,
                    metrics_path,
                    staging / "updater_parameters.json",
                )
            },
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
    return CalibrationRun(
        run_dir=target,
        run_id=run_id,
        selected_prior_equivalent_possessions=(calibration.selected_prior_equivalent_possessions),
    )


def publish_calibration_run(
    run_dir: Path | str,
    *,
    lake_uri: str = DEFAULT_CALIBRATION_LAKE_URI,
    store: ImmutableObjectStore | None = None,
) -> dict[str, object]:
    """Publish one reviewed calibration run as immutable, verified S3 objects."""

    source = Path(run_dir)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run_id = str(manifest.get("run_id") or "")
    if not run_id or source.name != run_id:
        raise ValueError("Calibration manifest run_id must match its directory")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Calibration manifest has no file checksums")
    active_store = store or AwsCliImmutableObjectStore()
    root_uri = _join_s3_uri(lake_uri, f"run_id={run_id}")
    objects: list[dict[str, object]] = []
    for filename, expected_sha in sorted(files.items()):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid calibration artifact filename: {filename!r}")
        path = source / filename
        payload = path.read_bytes()
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Calibration artifact checksum mismatch: {filename}")
        uri = _join_s3_uri(root_uri, filename)
        created = active_store.put_bytes(uri, payload)
        objects.append(
            {
                "filename": filename,
                "sha256": actual_sha,
                "byte_count": len(payload),
                "uri": uri,
                "created": created,
            }
        )
    manifest_payload = manifest_path.read_bytes()
    manifest_uri = _join_s3_uri(root_uri, "manifest.json")
    manifest_created = active_store.put_bytes(manifest_uri, manifest_payload)
    return {
        "run_id": run_id,
        "root_uri": root_uri,
        "manifest_uri": manifest_uri,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_created": manifest_created,
        "objects": objects,
    }


def publish_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish a reviewed calibration run to S3")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--lake-uri", default=DEFAULT_CALIBRATION_LAKE_URI)
    args = parser.parse_args(argv)
    print(json.dumps(publish_calibration_run(args.run_dir, lake_uri=args.lake_uri), indent=2))


def run_calibration(
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    selection_seasons: Sequence[str] = DEFAULT_SELECTION_SEASONS,
    prior_grid: Sequence[float] = DEFAULT_PRIOR_GRID,
    run_dir: Path | str = DEFAULT_RUN_DIR,
    player_season_panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    replay_mart_path: Path | str = DEFAULT_REPLAY_MART_PATH,
    rebuild_replay_mart: bool = False,
) -> CalibrationRun:
    targets = tuple(validate_season(season) for season in seasons)
    run_metadata = json.loads((Path(run_dir) / "metadata.json").read_text())
    source_run_id = str(run_metadata["run_id"])
    mart_path = Path(replay_mart_path)
    mart_manifest_path = mart_path.with_suffix(mart_path.suffix + ".manifest.json")
    if mart_path.is_file() and mart_manifest_path.is_file() and not rebuild_replay_mart:
        replays = read_historical_replay_mart(
            mart_path,
            expected_seasons=targets,
            expected_source_run_id=source_run_id,
        )
    else:
        replays = build_historical_replay_seasons(
            seasons=targets,
            run_dir=run_dir,
            player_season_panel_path=player_season_panel_path,
            analytical_dir=analytical_dir,
            game_catalog_path=game_catalog_path,
        )
        write_historical_replay_mart(replays, mart_path)
    calibration = calibrate_prior_strength(
        replays,
        prior_grid=prior_grid,
        selection_seasons=selection_seasons,
    )
    return write_calibration_run(
        calibration,
        replay_seasons=replays,
        output_dir=output_dir,
    )


def _build_historical_replay_season(
    target: str,
    *,
    state: object,
    candidate: BacktestModel,
    panel: pd.DataFrame,
    exposure_cohort: pd.DataFrame,
    schedule_features: pd.DataFrame,
    analytical_dir: Path,
) -> HistoricalReplaySeason:
    source = _previous_season(target)
    stints = read_rapm_stints(target, analytical_dir=analytical_dir)
    source_stints = read_rapm_stints(source, analytical_dir=analytical_dir)
    profiles = _transformed_target_profiles(
        target,
        candidate=candidate,
        panel=panel,
        exposure_cohort=exposure_cohort,
        analytical_dir=analytical_dir,
        stints=stints,
    )
    context_model = state.context_models[source]
    schedule_model = state.schedule_models[source]
    home_features = lineup_side_context_features(
        stints["home_player_ids"].tolist(),
        profiles,
        feature_set=context_model.feature_set,
        rebound_model=getattr(context_model, "rebound_model", None),
        usage_model=getattr(context_model, "usage_model", None),
    )
    away_features = lineup_side_context_features(
        stints["away_player_ids"].tolist(),
        profiles,
        feature_set=context_model.feature_set,
        rebound_model=getattr(context_model, "rebound_model", None),
        usage_model=getattr(context_model, "usage_model", None),
    )
    context_offset = context_model.predict_side_pairs(home_features, away_features)
    schedule_offset = _schedule_predictor(schedule_model, schedule_features)(stints)
    source_coefficients = state.coefficients.loc[
        state.coefficients["season"].eq(source), ["player_id", "rapm"]
    ]
    home_intercept = _recover_home_intercept(source_stints, source_coefficients)

    participants = tuple(
        sorted(
            {
                int(player_id)
                for column in ("home_player_ids", "away_player_ids")
                for lineup in stints[column]
                for player_id in lineup
            }
        )
    )
    scalar = state.priors.loc[state.priors["season"].eq(target), ["player_id", "prior_rapm"]].copy()
    scalar["player_id"] = scalar["player_id"].astype(int)
    scalar = pd.DataFrame({"player_id": participants}).merge(
        scalar, on="player_id", how="left", validate="one_to_one"
    )
    # Match the promoted frozen evaluator: a player absent from the published
    # scalar-prior vector enters at zero while their pre-target profile remains
    # available to the frozen context model.
    scalar["prior_rapm"] = scalar["prior_rapm"].fillna(0.0)
    complete = _compiled_linear_x3_coefficients(
        scalar.rename(columns={"prior_rapm": "rapm"}), profiles, context_model
    )
    complete_map = dict(zip(complete["player_id"].astype(int), complete["rapm"], strict=True))
    scalar_map = dict(zip(scalar["player_id"].astype(int), scalar["prior_rapm"], strict=True))
    complete_prior = np.asarray([complete_map[player_id] for player_id in participants])
    player_columns = vocabulary_mapping(participants)
    design = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        player_columns,
        multiple=True,
    )
    scalar_prior = np.asarray([scalar_map[player_id] for player_id in participants])
    frozen_stint_prediction = (
        np.asarray(design @ scalar_prior).reshape(-1)
        + float(home_intercept)
        + np.asarray(context_offset, dtype=float)
        + np.asarray(schedule_offset, dtype=float)
    )
    fixed_context = frozen_stint_prediction - np.asarray(design @ complete_prior).reshape(-1)
    if not np.isfinite(fixed_context).all():
        raise ValueError(f"Frozen context offset is not finite for {target}")

    games: list[HistoricalReplayGame] = []
    for game_id, indices in stints.groupby("game_id", sort=False).groups.items():
        game_rows = stints.loc[indices].copy()
        positions = stints.index.get_indexer(indices)
        game_design = design[positions]
        weights = game_rows["possessions"].to_numpy(dtype=float)
        game_time = pd.Timestamp(game_rows["game_time_utc"].iloc[0]).to_pydatetime()
        source_hashes = tuple(game_rows["play_by_play_sha256"].astype(str).unique())
        if len(source_hashes) != 1:
            raise ValueError(f"Game {game_id} has multiple source checksums")
        contribution = build_game_contribution(
            game_rows,
            season=target,
            game_id=str(game_id),
            game_time_utc=game_time,
            finalized_at=game_time + timedelta(hours=6),
            source_sha256=source_hashes[0],
            source_uri=(f"local://data/analytical/rapm_stints/{target}/regular/{game_id}"),
            core_model_version=CORE_MODEL_VERSION,
            context_model_version=f"{state.run_id}:{source}",
            player_ids=participants,
            preseason_nail=complete_prior,
            fixed_context_offset=fixed_context[positions],
        )
        margin_design = sparse.csr_matrix(game_design.T @ (weights / 100.0))
        actual_margin = float(game_rows["home_margin"].sum())
        frozen_margin = float(np.dot(frozen_stint_prediction[positions], weights) / 100.0)
        game_date = pd.Timestamp(game_rows["game_date"].iloc[0]).date()
        games.append(
            HistoricalReplayGame(
                game_id=str(game_id),
                game_date=game_date,
                game_time_utc=game_time,
                actual_home_margin=actual_margin,
                frozen_home_margin=frozen_margin,
                margin_design=margin_design,
                contribution=contribution,
            )
        )
    return HistoricalReplaySeason(
        season=target,
        player_ids=participants,
        frozen_preseason_nail=complete_prior,
        games=tuple(sorted(games, key=_game_sort_key)),
        source_run_id=state.run_id,
    )


def _previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start:04d}-{str(start + 1)[-2:]}"


def _replay_prior_candidate(
    replay: HistoricalReplaySeason,
    *,
    prior_equivalent_possessions: float,
) -> pd.DataFrame:
    player_count = len(replay.player_ids)
    gram = sparse.csr_matrix((player_count, player_count), dtype=float)
    score = np.zeros(player_count, dtype=float)
    rows: list[dict[str, object]] = []
    games_by_date: dict[date, list[HistoricalReplayGame]] = {}
    for game in replay.games:
        games_by_date.setdefault(game.game_date, []).append(game)
    for _game_date, games in games_by_date.items():
        precision = gram + sparse.eye(player_count, format="csr") * prior_equivalent_possessions
        adjustment = (
            np.zeros(player_count, dtype=float)
            if gram.nnz == 0
            else np.asarray(spsolve(precision, score), dtype=float)
        )
        for game in games:
            update = float((game.margin_design @ adjustment).item())
            rows.append(
                _prediction_row(
                    replay,
                    game,
                    prior_equivalent_possessions=prior_equivalent_possessions,
                    predicted_home_margin=game.frozen_home_margin + update,
                )
            )
        for game in games:
            gram = gram + game.contribution.xtwx
            score += game.contribution.xtwy
    return pd.DataFrame(rows)


def _replay_frozen_control(replay: HistoricalReplaySeason) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _prediction_row(
                replay,
                game,
                prior_equivalent_possessions=None,
                predicted_home_margin=game.frozen_home_margin,
            )
            for game in replay.games
        ]
    )


def _prediction_row(
    replay: HistoricalReplaySeason,
    game: HistoricalReplayGame,
    *,
    prior_equivalent_possessions: float | None,
    predicted_home_margin: float,
) -> dict[str, object]:
    probability = float(expit(predicted_home_margin / WIN_PROBABILITY_MARGIN_SCALE))
    return {
        "season": replay.season,
        "game_id": game.game_id,
        "game_date": game.game_date,
        "game_time_utc": game.game_time_utc,
        "prior_equivalent_possessions": (
            np.nan if prior_equivalent_possessions is None else prior_equivalent_possessions
        ),
        "model": (
            "frozen_preseason_control" if prior_equivalent_possessions is None else UPDATER_VERSION
        ),
        "actual_home_margin": game.actual_home_margin,
        "predicted_home_margin": predicted_home_margin,
        "actual_home_win": game.actual_home_margin > 0,
        "predicted_home_win_probability": probability,
    }


def _calibration_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["split", "model", "prior_equivalent_possessions"]
    for keys, frame in predictions.groupby(group_columns, dropna=False, sort=False):
        split, model, prior = keys
        for scope, subset in [
            ("pooled", frame),
            *[(season, rows_) for season, rows_ in frame.groupby("season", sort=False)],
        ]:
            actual_margin = subset["actual_home_margin"].to_numpy(dtype=float)
            predicted_margin = subset["predicted_home_margin"].to_numpy(dtype=float)
            actual_win = subset["actual_home_win"].to_numpy(dtype=float)
            probability = np.clip(
                subset["predicted_home_win_probability"].to_numpy(dtype=float),
                1e-9,
                1.0 - 1e-9,
            )
            rows.append(
                {
                    "split": split,
                    "scope": scope,
                    "model": model,
                    "prior_equivalent_possessions": prior,
                    "season_count": int(subset["season"].nunique()),
                    "game_count": len(subset),
                    "margin_mae": float(np.mean(np.abs(actual_margin - predicted_margin))),
                    "margin_rmse": float(
                        np.sqrt(np.mean(np.square(actual_margin - predicted_margin)))
                    ),
                    "winner_accuracy": float(
                        np.mean((predicted_margin > 0) == (actual_margin > 0))
                    ),
                    "brier_score": float(np.mean(np.square(probability - actual_win))),
                    "log_loss": float(
                        -np.mean(
                            actual_win * np.log(probability)
                            + (1.0 - actual_win) * np.log(1.0 - probability)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _game_sort_key(game: HistoricalReplayGame) -> tuple[date, datetime, str]:
    return game.game_date, game.game_time_utc, game.game_id


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate possession-based in-season NAIL prior strength"
    )
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--selection-seasons", nargs="+", default=list(DEFAULT_SELECTION_SEASONS))
    parser.add_argument("--prior-grid", nargs="+", type=float, default=list(DEFAULT_PRIOR_GRID))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--analytical-dir", type=Path, default=DEFAULT_ANALYTICAL_DIR)
    parser.add_argument("--game-catalog-path", type=Path, default=DEFAULT_GAME_CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-mart-path", type=Path, default=DEFAULT_REPLAY_MART_PATH)
    parser.add_argument("--rebuild-replay-mart", action="store_true")
    args = parser.parse_args(argv)
    run = run_calibration(
        seasons=tuple(args.seasons),
        selection_seasons=tuple(args.selection_seasons),
        prior_grid=tuple(args.prior_grid),
        run_dir=args.run_dir,
        player_season_panel_path=args.panel_path,
        analytical_dir=args.analytical_dir,
        game_catalog_path=args.game_catalog_path,
        output_dir=args.output_dir,
        replay_mart_path=args.replay_mart_path,
        rebuild_replay_mart=args.rebuild_replay_mart,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "selected_prior_equivalent_possessions": (
                    run.selected_prior_equivalent_possessions
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
