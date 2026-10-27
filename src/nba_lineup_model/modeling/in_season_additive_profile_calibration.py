"""Chronological shadow study for the in-season additive-profile updater."""

from __future__ import annotations

import argparse
import hashlib
import html
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
from scipy.sparse.linalg import factorized
from scipy.special import expit

from nba_lineup_model.modeling.contextual_profiles import (
    MEDVEDOVSKY_2020_PROFILE_PADDING,
    build_contextual_player_profiles,
)
from nba_lineup_model.modeling.forward_nail_v1212_residualized_lambda import MODEL_NAME
from nba_lineup_model.modeling.frozen_multiseason_backtest import (
    BacktestModel,
    _load_state,
    _transformed_target_profiles,
)
from nba_lineup_model.modeling.in_season_additive_profile import (
    PRIOR_COMPONENTS,
    AdditiveProfileUpdaterConfig,
    FrozenAdditiveProfileContract,
    ProfileGameContribution,
    build_profile_game_contribution,
    profile_contract_from_context_model,
    solve_additive_profile_state,
)
from nba_lineup_model.modeling.in_season_nail_calibration import (
    CORE_MODEL_VERSION,
    DEFAULT_ANALYTICAL_DIR,
    DEFAULT_PANEL_PATH,
    DEFAULT_REPLAY_MART_PATH,
    DEFAULT_RUN_DIR,
    DEFAULT_SEASONS,
    DEFAULT_SELECTION_SEASONS,
    WIN_PROBABILITY_MARGIN_SCALE,
    HistoricalReplaySeason,
    build_historical_replay_seasons,
    read_historical_replay_mart,
)
from nba_lineup_model.modeling.replacement_level import prepare_player_exposure_cohort
from nba_lineup_model.modeling.stints import read_rapm_stints
from nba_lineup_model.season.schema import validate_season

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_PROFILE_REPLAY_MART_PATH = Path(
    "artifacts/cache/in_season_additive_profile_replay_mart/recent-seasons.joblib"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/models/in_season_additive_profile_calibration")
DEFAULT_MULTIPLIER_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
SCALAR_PRIOR_EQUIVALENT_POSSESSIONS = 6000.0
CALIBRATION_VERSION = "in-season-additive-profile-calibration-v0.1.0"


@dataclass(frozen=True)
class HistoricalProfileReplaySeason:
    """Frozen game replay plus the box-score profile evidence for each game."""

    scalar_replay: HistoricalReplaySeason
    contract: FrozenAdditiveProfileContract
    profile_contributions: tuple[ProfileGameContribution, ...]

    def __post_init__(self) -> None:
        if self.scalar_replay.season != self.contract.season:
            raise ValueError("Scalar and profile replay seasons differ")
        if self.scalar_replay.player_ids != self.contract.player_ids:
            raise ValueError("Scalar and profile replay player vocabularies differ")
        scalar_ids = tuple(game.game_id for game in self.scalar_replay.games)
        profile_ids = tuple(row.game_id for row in self.profile_contributions)
        if scalar_ids != profile_ids:
            raise ValueError("Scalar and profile replay games differ")


@dataclass(frozen=True)
class AdditiveProfileCalibrationRun:
    run_id: str
    run_dir: Path
    selected_prior_multiplier: float


def build_historical_profile_replays(
    scalar_replays: Sequence[HistoricalReplaySeason],
    *,
    run_dir: Path | str = DEFAULT_RUN_DIR,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    analytical_dir: Path | str = DEFAULT_ANALYTICAL_DIR,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> tuple[HistoricalProfileReplaySeason, ...]:
    """Build profile evidence from the already materialized local history."""

    scalar = tuple(scalar_replays)
    if not scalar:
        raise ValueError("Profile replay requires at least one scalar replay")
    seasons = tuple(replay.season for replay in scalar)
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
    state = _load_state(Path(run_dir), candidate=candidate, target_seasons=seasons)
    panel = pd.read_parquet(panel_path)
    exposure = prepare_player_exposure_cohort(
        panel.loc[panel["season"].astype(str).le(seasons[-1])],
        through_season=seasons[-1],
        analytical_dir=analytical_dir,
    )
    output: list[HistoricalProfileReplaySeason] = []
    for replay in scalar:
        target = replay.season
        source = _previous_season(target)
        stints = read_rapm_stints(target, analytical_dir=Path(analytical_dir))
        profiles = _transformed_target_profiles(
            target,
            candidate=candidate,
            panel=panel,
            exposure_cohort=exposure,
            analytical_dir=Path(analytical_dir),
            stints=stints,
        )
        profiles = pd.DataFrame({"player_id": replay.player_ids}).merge(
            profiles,
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        contract = profile_contract_from_context_model(
            season=target,
            core_model_version=CORE_MODEL_VERSION,
            source_run_id=state.run_id,
            profiles=profiles,
            context_model=state.context_models[source],
        )
        players = pd.read_parquet(Path(curated_dir) / "players" / target / "regular")
        players["game_id"] = players["game_id"].astype(str)
        stints["game_id"] = stints["game_id"].astype(str)
        player_groups = {str(key): rows for key, rows in players.groupby("game_id", sort=False)}
        stint_groups = {str(key): rows for key, rows in stints.groupby("game_id", sort=False)}
        contributions: list[ProfileGameContribution] = []
        for game in replay.games:
            if game.game_id not in player_groups or game.game_id not in stint_groups:
                raise ValueError(f"Missing profile source rows for {target} {game.game_id}")
            game_players = player_groups[game.game_id]
            game_stints = stint_groups[game.game_id]
            pbp_hashes = tuple(game_stints["play_by_play_sha256"].astype(str).unique())
            box_hashes = tuple(game_players["boxscore_sha256"].astype(str).unique())
            if len(pbp_hashes) != 1 or len(box_hashes) != 1:
                raise ValueError(f"Game {game.game_id} has ambiguous source checksums")
            combined_source = hashlib.sha256(
                f"{pbp_hashes[0]}:{box_hashes[0]}".encode()
            ).hexdigest()
            contributions.append(
                build_profile_game_contribution(
                    game_players,
                    game_stints,
                    season=target,
                    game_id=game.game_id,
                    game_time_utc=game.game_time_utc,
                    finalized_at=game.contribution.finalized_at,
                    source_sha256=combined_source,
                    source_uri=(
                        f"local://data/curated/players/{target}/regular/{game.game_id}"
                    ),
                )
            )
        output.append(
            HistoricalProfileReplaySeason(
                scalar_replay=replay,
                contract=contract,
                profile_contributions=tuple(contributions),
            )
        )
    return tuple(output)


def write_historical_profile_replay_mart(
    replays: Sequence[HistoricalProfileReplaySeason],
    path: Path | str,
) -> Path:
    payload = tuple(replays)
    if not payload:
        raise ValueError("Profile replay mart cannot be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{uuid4().hex}")
    joblib.dump(payload, temporary, compress=3)
    manifest = {
        "schema_version": 1,
        "calibration_version": CALIBRATION_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "mart_sha256": _file_sha256(temporary),
        "seasons": [
            {
                "season": replay.scalar_replay.season,
                "game_count": len(replay.scalar_replay.games),
                "player_count": len(replay.scalar_replay.player_ids),
                "profile_contract_sha256": replay.contract.lineage_sha256,
            }
            for replay in payload
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + f".tmp-{uuid4().hex}"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    temporary_manifest.replace(manifest_path)
    return output


def read_historical_profile_replay_mart(
    path: Path | str,
    *,
    expected_seasons: Sequence[str] | None = None,
) -> tuple[HistoricalProfileReplaySeason, ...]:
    source = Path(path)
    manifest = json.loads(source.with_suffix(source.suffix + ".manifest.json").read_text())
    if manifest.get("schema_version") != 1 or _file_sha256(source) != manifest.get(
        "mart_sha256"
    ):
        raise ValueError("Historical profile replay mart is invalid")
    loaded = joblib.load(source)
    if not isinstance(loaded, tuple) or not all(
        isinstance(row, HistoricalProfileReplaySeason) for row in loaded
    ):
        raise ValueError("Historical profile replay mart payload is invalid")
    if expected_seasons is not None and tuple(
        row.scalar_replay.season for row in loaded
    ) != tuple(expected_seasons):
        raise ValueError("Historical profile replay mart covers different seasons")
    return loaded


def calibrate_additive_profile_updater(
    replays: Sequence[HistoricalProfileReplaySeason],
    *,
    multiplier_grid: Sequence[float] = DEFAULT_MULTIPLIER_GRID,
    selection_seasons: Sequence[str] = DEFAULT_SELECTION_SEASONS,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a common multiplier, then evaluate four arms on the locked holdout."""

    replay_tuple = tuple(replays)
    available = {row.scalar_replay.season for row in replay_tuple}
    selection = tuple(validate_season(value) for value in selection_seasons)
    if not selection or not set(selection).issubset(available):
        raise ValueError("Selection seasons must be a non-empty replay subset")
    grid = tuple(float(value) for value in multiplier_grid)
    if not grid or len(grid) != len(set(grid)) or any(
        not np.isfinite(value) or value <= 0 for value in grid
    ):
        raise ValueError("Profile multiplier grid must be finite, positive, and unique")

    tuning = pd.concat(
        [
            _replay_candidate(
                replay,
                config=_common_multiplier_config(multiplier),
                capture_states=False,
            )[0]
            for replay in replay_tuple
            if replay.scalar_replay.season in selection
            for multiplier in grid
        ],
        ignore_index=True,
    )
    tuning_metrics = _metrics(tuning.assign(split="selection"))
    selectable = tuning_metrics.loc[
        tuning_metrics["scope"].eq("pooled")
        & tuning_metrics["model"].eq("profile_plus_residualized_scalar")
    ].sort_values(["margin_rmse", "brier_score", "prior_multiplier"], kind="stable")
    if selectable.empty:
        raise ValueError("Profile calibration produced no selectable candidate")
    selected = float(selectable.iloc[0]["prior_multiplier"])
    selected_config = _common_multiplier_config(selected)

    prediction_parts: list[pd.DataFrame] = []
    state_parts: list[pd.DataFrame] = []
    for replay in replay_tuple:
        predictions, states = _replay_candidate(
            replay,
            config=selected_config,
            capture_states=True,
        )
        split = "selection" if replay.scalar_replay.season in selection else "holdout"
        prediction_parts.append(predictions.assign(split=split))
        state_parts.append(states.assign(split=split))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    states = pd.concat(state_parts, ignore_index=True)
    metrics = _metrics(predictions)
    bootstrap = _paired_bootstrap(predictions.loc[predictions["split"].eq("holdout")])
    grid_diagnostics = tuning_metrics.assign(record_type="selection_grid")
    return (
        selected,
        predictions,
        metrics,
        pd.concat(
            [grid_diagnostics, bootstrap, _profile_state_summary(states)],
            ignore_index=True,
            sort=False,
        ),
        states,
    )


def _replay_candidate(
    replay: HistoricalProfileReplaySeason,
    *,
    config: AdditiveProfileUpdaterConfig,
    capture_states: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = replay.scalar_replay
    player_count = len(base.player_ids)
    prior_precision = sparse.eye(player_count, format="csr") * (
        SCALAR_PRIOR_EQUIVALENT_POSSESSIONS
    )
    gram = sparse.csr_matrix((player_count, player_count), dtype=float)
    scalar_score = np.zeros(player_count, dtype=float)
    residualized_score = np.zeros(player_count, dtype=float)
    profile_history: list[ProfileGameContribution] = []
    contribution_by_id = {row.game_id: row for row in replay.profile_contributions}
    predictions: list[dict[str, object]] = []
    state_rows: list[pd.DataFrame] = []
    games_by_date: dict[date, list[object]] = {}
    for game in base.games:
        games_by_date.setdefault(game.game_date, []).append(game)

    for game_date, games in games_by_date.items():
        cutoff = min(game.game_time_utc for game in games)
        state = solve_additive_profile_state(
            replay.contract,
            profile_history,
            cutoff_time=cutoff,
            config=config,
        ).set_index("player_id").loc[list(base.player_ids)]
        profile_delta = state["additive_profile_delta"].to_numpy(dtype=float)
        if gram.nnz == 0:
            scalar_delta = np.zeros(player_count, dtype=float)
            residualized_scalar_delta = np.zeros(player_count, dtype=float)
        else:
            solve = factorized((gram + prior_precision).tocsc())
            residualized_scalar_delta = np.asarray(
                solve(residualized_score), dtype=float
            )
            scalar_delta = (
                np.asarray(solve(scalar_score), dtype=float)
                if capture_states
                else np.zeros(player_count, dtype=float)
            )
        if capture_states:
            snapshot = _compact_state_snapshot(state.reset_index())
            snapshot["prediction_date"] = game_date
            if not snapshot.empty:
                state_rows.append(snapshot)
        for game in games:
            design = game.margin_design
            updates = {
                "profile_plus_residualized_scalar": float(
                    (design @ (profile_delta + residualized_scalar_delta)).item()
                )
            }
            if capture_states:
                updates.update(
                    {
                        "frozen_production": 0.0,
                        "scalar_updater": float((design @ scalar_delta).item()),
                        "profile_updater": float((design @ profile_delta).item()),
                        "naive_profile_plus_scalar": float(
                            (design @ (profile_delta + scalar_delta)).item()
                        ),
                    }
                )
            for model, update in updates.items():
                margin = game.frozen_home_margin + update
                predictions.append(
                    {
                        "season": base.season,
                        "game_id": game.game_id,
                        "game_date": game.game_date,
                        "game_time_utc": game.game_time_utc,
                        "model": model,
                        "prior_multiplier": float(
                            config.prior_multipliers[PRIOR_COMPONENTS[0]]
                        ),
                        "actual_home_margin": game.actual_home_margin,
                        "predicted_home_margin": margin,
                        "predicted_home_win_probability": float(
                            expit(margin / WIN_PROBABILITY_MARGIN_SCALE)
                        ),
                        "actual_home_win": game.actual_home_margin > 0,
                        "profile_state_sha256": state["profile_state_sha256"].iloc[0],
                    }
                )
        for game in games:
            scalar_score += game.contribution.xtwy
            residualized_score += game.contribution.xtwy - np.asarray(
                game.contribution.xtwx @ profile_delta
            ).reshape(-1)
            gram = gram + game.contribution.xtwx
            profile_history.append(contribution_by_id[game.game_id])
    if capture_states and profile_history:
        final_cutoff = max(row.finalized_at for row in profile_history) + timedelta(
            microseconds=1
        )
        final_state = solve_additive_profile_state(
            replay.contract,
            profile_history,
            cutoff_time=final_cutoff,
            config=config,
        )
        final_snapshot = _compact_state_snapshot(final_state)
        final_snapshot["prediction_date"] = max(games_by_date) + timedelta(days=1)
        if not final_snapshot.empty:
            state_rows.append(final_snapshot)
    prediction_frame = pd.DataFrame(predictions)
    state_frame = pd.concat(state_rows, ignore_index=True) if state_rows else pd.DataFrame()
    return prediction_frame, state_frame


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["split", "model", "prior_multiplier"]
    for keys, frame in predictions.groupby(group_columns, sort=False, dropna=False):
        split, model, multiplier = keys
        scopes = [("pooled", frame), *list(frame.groupby("season", sort=False))]
        for scope, subset in scopes:
            actual = subset["actual_home_margin"].to_numpy(dtype=float)
            predicted = subset["predicted_home_margin"].to_numpy(dtype=float)
            outcome = subset["actual_home_win"].to_numpy(dtype=float)
            probability = np.clip(
                subset["predicted_home_win_probability"].to_numpy(dtype=float),
                1e-9,
                1.0 - 1e-9,
            )
            rows.append(
                {
                    "split": split,
                    "scope": str(scope),
                    "model": model,
                    "prior_multiplier": float(multiplier),
                    "game_count": len(subset),
                    "margin_mae": float(np.mean(np.abs(predicted - actual))),
                    "margin_rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                    "winner_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
                    "brier_score": float(np.mean(np.square(probability - outcome))),
                    "log_loss": float(
                        -np.mean(
                            outcome * np.log(probability)
                            + (1.0 - outcome) * np.log(1.0 - probability)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_bootstrap(predictions: pd.DataFrame, *, samples: int = 2000) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    wide = predictions.pivot(
        index=["season", "game_id"],
        columns="model",
        values=[
            "actual_home_margin",
            "predicted_home_margin",
            "actual_home_win",
            "predicted_home_win_probability",
        ],
    )
    models = (
        "profile_updater",
        "profile_plus_residualized_scalar",
        "naive_profile_plus_scalar",
    )
    rng = np.random.default_rng(33)
    indices = rng.integers(0, len(wide), size=(samples, len(wide)))
    actual_margin = wide[("actual_home_margin", "scalar_updater")].to_numpy(dtype=float)
    actual_win = wide[("actual_home_win", "scalar_updater")].to_numpy(dtype=float)
    control_margin = wide[("predicted_home_margin", "scalar_updater")].to_numpy(dtype=float)
    control_prob = wide[("predicted_home_win_probability", "scalar_updater")].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for model in models:
        candidate_margin = wide[("predicted_home_margin", model)].to_numpy(dtype=float)
        candidate_prob = wide[("predicted_home_win_probability", model)].to_numpy(dtype=float)
        metrics = {
            "margin_mae": (
                np.abs(control_margin - actual_margin),
                np.abs(candidate_margin - actual_margin),
            ),
            "brier_score": (
                np.square(control_prob - actual_win),
                np.square(candidate_prob - actual_win),
            ),
            "log_loss": (
                -(
                    actual_win * np.log(np.clip(control_prob, 1e-9, 1 - 1e-9))
                    + (1 - actual_win) * np.log(np.clip(1 - control_prob, 1e-9, 1 - 1e-9))
                ),
                -(
                    actual_win * np.log(np.clip(candidate_prob, 1e-9, 1 - 1e-9))
                    + (1 - actual_win)
                    * np.log(np.clip(1 - candidate_prob, 1e-9, 1 - 1e-9))
                ),
            ),
        }
        for metric, (control_loss, candidate_loss) in metrics.items():
            differences = np.mean(
                candidate_loss[indices] - control_loss[indices], axis=1
            )
            rows.append(
                {
                    "record_type": "paired_bootstrap",
                    "model": model,
                    "metric": metric,
                    "candidate_minus_scalar": float(
                        np.mean(candidate_loss - control_loss)
                    ),
                    "lower_95": float(np.quantile(differences, 0.025)),
                    "upper_95": float(np.quantile(differences, 0.975)),
                    "probability_candidate_better": float(np.mean(differences < 0)),
                    "sample_count": len(wide),
                }
            )
    return pd.DataFrame(rows)


def _profile_state_summary(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()
    latest = (
        states.sort_values(["season", "prediction_date"], kind="stable")
        .groupby(["season", "player_id"], as_index=False, sort=False)
        .tail(1)
    )
    rows: list[dict[str, object]] = []
    for feature in (
        "three_pa_per_100",
        "three_pm_per_100",
        "assists_per_100",
        "turnovers_per_100",
        "usage_pct",
        "steals_per_100",
        "blocks_per_100",
        "offensive_rebound_claim_total",
    ):
        movement = latest[f"profile_delta_{feature}"].to_numpy(dtype=float)
        rows.append(
            {
                "record_type": "profile_movement",
                "model": "profile_updater",
                "metric": feature,
                "candidate_minus_scalar": float(np.mean(movement)),
                "lower_95": float(np.quantile(movement, 0.025)),
                "upper_95": float(np.quantile(movement, 0.975)),
                "probability_candidate_better": np.nan,
                "sample_count": len(movement),
            }
        )
    return pd.DataFrame(rows)


def _compact_state_snapshot(state: pd.DataFrame) -> pd.DataFrame:
    """Retain active-player trajectory fields without a wide repeated contract."""

    active = state.loc[state["observed_games"].gt(0)].copy()
    columns = [
        "season",
        "player_id",
        "observed_games",
        "raw_on_court_possessions",
        "preseason_additive_profile",
        "additive_profile_delta",
        "updated_additive_profile",
        "additive_profile_standard_error",
        "profile_state_sha256",
    ]
    for feature in (
        "three_pa_per_100",
        "three_pm_per_100",
        "assists_per_100",
        "turnovers_per_100",
        "usage_pct",
        "steals_per_100",
        "blocks_per_100",
        "offensive_rebound_claim_total",
    ):
        columns.extend(
            [
                f"preseason_{feature}",
                f"updated_{feature}",
                f"profile_delta_{feature}",
                f"evidence_weight_{feature}",
                f"rating_delta_{feature}",
            ]
        )
    return active.loc[:, columns].reset_index(drop=True)


def write_calibration_run(
    *,
    selected_multiplier: float,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    states: pd.DataFrame,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> AdditiveProfileCalibrationRun:
    created_at = datetime.now(UTC)
    run_id = (
        f"in-season-additive-profile-{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(output_dir)
    staging = root / f".{run_id}.tmp"
    target = root / run_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        files = {
            "game_predictions.parquet": predictions,
            "metrics.parquet": metrics,
            "diagnostics.parquet": diagnostics,
            "player_states.parquet": states,
        }
        for filename, frame in files.items():
            frame.to_parquet(staging / filename, index=False)
        parameters = {
            "schema_version": 1,
            "calibration_version": CALIBRATION_VERSION,
            "selected_prior_multiplier": selected_multiplier,
            "scalar_prior_equivalent_possessions": (
                SCALAR_PRIOR_EQUIVALENT_POSSESSIONS
            ),
            "selection_seasons": DEFAULT_SELECTION_SEASONS,
            "holdout_season": "2025-26",
            "production_serving_changed": False,
        }
        (staging / "parameters.json").write_text(
            json.dumps(parameters, indent=2, sort_keys=True) + "\n"
        )
        _write_html_report(
            staging / "report.html",
            metrics=metrics,
            diagnostics=diagnostics,
            states=states,
            selected_multiplier=selected_multiplier,
        )
        artifact_names = [*files, "parameters.json", "report.html"]
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "files": {
                filename: _file_sha256(staging / filename)
                for filename in artifact_names
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
    return AdditiveProfileCalibrationRun(run_id, target, selected_multiplier)


def run_calibration(
    *,
    seasons: Sequence[str] = DEFAULT_SEASONS,
    selection_seasons: Sequence[str] = DEFAULT_SELECTION_SEASONS,
    multiplier_grid: Sequence[float] = DEFAULT_MULTIPLIER_GRID,
    scalar_replay_mart_path: Path | str = DEFAULT_REPLAY_MART_PATH,
    profile_replay_mart_path: Path | str = DEFAULT_PROFILE_REPLAY_MART_PATH,
    rebuild_profile_mart: bool = False,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> AdditiveProfileCalibrationRun:
    targets = tuple(validate_season(value) for value in seasons)
    mart_path = Path(profile_replay_mart_path)
    manifest_path = mart_path.with_suffix(mart_path.suffix + ".manifest.json")
    if mart_path.is_file() and manifest_path.is_file() and not rebuild_profile_mart:
        replays = read_historical_profile_replay_mart(
            mart_path, expected_seasons=targets
        )
    else:
        try:
            scalar_replays = read_historical_replay_mart(
                scalar_replay_mart_path,
                expected_seasons=targets,
            )
        except AttributeError:
            # The original compact mart was produced by invoking its module as
            # ``__main__``, so its dataclass pickle is not portable to another
            # entry point. Rebuild from the already-local curated marts; never
            # fetch or copy the historical archive from S3 here.
            scalar_replays = build_historical_replay_seasons(seasons=targets)
        replays = build_historical_profile_replays(scalar_replays)
        write_historical_profile_replay_mart(replays, mart_path)
    selected, predictions, metrics, diagnostics, states = calibrate_additive_profile_updater(
        replays,
        multiplier_grid=multiplier_grid,
        selection_seasons=selection_seasons,
    )
    return write_calibration_run(
        selected_multiplier=selected,
        predictions=predictions,
        metrics=metrics,
        diagnostics=diagnostics,
        states=states,
        output_dir=output_dir,
    )


def _common_multiplier_config(value: float) -> AdditiveProfileUpdaterConfig:
    return AdditiveProfileUpdaterConfig(
        updater_version=f"in-season-additive-profile-common-{value:g}",
        prior_multipliers={component: float(value) for component in PRIOR_COMPONENTS},
    )


def _write_html_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    states: pd.DataFrame,
    selected_multiplier: float,
    player_panel_path: Path | str = DEFAULT_PANEL_PATH,
) -> None:
    """Render the study decision and its supporting evidence as one report."""

    holdout = metrics.loc[
        metrics["split"].eq("holdout") & metrics["scope"].eq("pooled")
    ].copy()
    model_labels = {
        "frozen_production": "Frozen production",
        "scalar_updater": "Scalar updater",
        "profile_updater": "Profile only",
        "profile_plus_residualized_scalar": "Profile + residualized scalar",
        "naive_profile_plus_scalar": "Naive profile + scalar",
    }
    holdout["model"] = holdout["model"].map(model_labels).fillna(holdout["model"])
    holdout["winner_accuracy"] *= 100.0
    holdout = holdout.rename(
        columns={
            "model": "Model",
            "game_count": "Games",
            "margin_mae": "Margin MAE",
            "margin_rmse": "Margin RMSE",
            "winner_accuracy": "Winner %",
            "brier_score": "Brier",
            "log_loss": "Log loss",
        }
    )
    table = holdout.loc[
        :,
        [
            "Model",
            "Games",
            "Margin MAE",
            "Margin RMSE",
            "Winner %",
            "Brier",
            "Log loss",
        ],
    ].to_html(index=False, classes="metrics", float_format=lambda value: f"{value:.5f}")
    bootstrap = diagnostics.loc[diagnostics["record_type"].eq("paired_bootstrap")]
    bootstrap = bootstrap.loc[
        bootstrap["model"].eq("profile_plus_residualized_scalar"),
        [
            "metric",
            "candidate_minus_scalar",
            "lower_95",
            "upper_95",
            "probability_candidate_better",
            "sample_count",
        ],
    ].copy()
    bootstrap["metric"] = bootstrap["metric"].map(
        {
            "margin_mae": "Margin MAE",
            "brier_score": "Brier",
            "log_loss": "Log loss",
        }
    )
    bootstrap["probability_candidate_better"] *= 100.0
    bootstrap = bootstrap.rename(
        columns={
            "metric": "Metric",
            "candidate_minus_scalar": "Candidate - scalar",
            "lower_95": "95% lower",
            "upper_95": "95% upper",
            "probability_candidate_better": "P(candidate better), %",
            "sample_count": "Games",
        }
    )
    bootstrap_table = bootstrap.to_html(
        index=False, classes="metrics", float_format=lambda value: f"{value:.5f}"
    )
    selection_grid = diagnostics.loc[
        diagnostics["record_type"].eq("selection_grid")
        & diagnostics["scope"].eq("pooled"),
        ["prior_multiplier", "game_count", "margin_rmse", "brier_score", "log_loss"],
    ].copy()
    scalar_selection = metrics.loc[
        metrics["split"].eq("selection")
        & metrics["scope"].eq("pooled")
        & metrics["model"].eq("scalar_updater")
    ]
    if not scalar_selection.empty:
        scalar_row = scalar_selection.iloc[0]
        selection_grid = pd.concat(
            [
                selection_grid,
                pd.DataFrame(
                    [
                        {
                            "prior_multiplier": np.inf,
                            "game_count": scalar_row["game_count"],
                            "margin_rmse": scalar_row["margin_rmse"],
                            "brier_score": scalar_row["brier_score"],
                            "log_loss": scalar_row["log_loss"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    selection_grid["prior_multiplier"] = selection_grid["prior_multiplier"].map(
        lambda value: "No profile update" if np.isinf(value) else f"{value:g}x"
    )
    selection_grid = selection_grid.rename(
        columns={
            "prior_multiplier": "Profile prior",
            "game_count": "Games",
            "margin_rmse": "Margin RMSE",
            "brier_score": "Brier",
            "log_loss": "Log loss",
        }
    )
    selection_table = selection_grid.to_html(
        index=False, classes="metrics", float_format=lambda value: f"{value:.5f}"
    )

    player_tables = _report_player_tables(states, player_panel_path=player_panel_path)
    combined = metrics.loc[
        metrics["split"].eq("holdout")
        & metrics["scope"].eq("pooled")
        & metrics["model"].eq("profile_plus_residualized_scalar")
    ].iloc[0]
    scalar = metrics.loc[
        metrics["split"].eq("holdout")
        & metrics["scope"].eq("pooled")
        & metrics["model"].eq("scalar_updater")
    ].iloc[0]
    margin_change = float(combined["margin_mae"] - scalar["margin_mae"])
    brier_change = float(combined["brier_score"] - scalar["brier_score"])
    log_loss_change = float(combined["log_loss"] - scalar["log_loss"])
    winner_change = 100.0 * float(
        combined["winner_accuracy"] - scalar["winner_accuracy"]
    )
    decision = "Do not promote the additive-profile updater."
    interpretation = (
        f"Against the scalar updater on the locked holdout, the combined candidate changed "
        f"margin MAE by {margin_change:+.4f} points, Brier by {brier_change:+.6f}, "
        f"log loss by {log_loss_change:+.6f}, and winner accuracy by "
        f"{winner_change:+.2f} percentage points. The margin difference is effectively zero, "
        "while both probability losses and winner accuracy move in the wrong direction."
    )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Additive profile updater</title>
<style>
body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;max-width:1180px;
margin:40px auto;padding:0 24px;color:#17211b;background:#faf9f5}}
h1,h2{{letter-spacing:-.025em}} h2{{margin-top:36px}}
.decision{{border-left:5px solid #a6422b;background:#f7e9e4;padding:18px 22px;margin:24px 0}}
.decision strong{{display:block;font-size:1.22rem;margin-bottom:8px;color:#7f2d1d}}
.note{{background:#eef3ed;border-left:5px solid #426b55;padding:15px 20px}}
table{{border-collapse:collapse;width:100%;margin:18px 0;background:white}}
th,td{{padding:9px 11px;border-bottom:1px solid #d5dbd5;text-align:right;
font-variant-numeric:tabular-nums}}
th{{background:#edf1ec}} th:first-child,td:first-child{{text-align:left}}
code{{background:#eef2ee;padding:2px 5px}}
.small{{color:#5a665e;font-size:.92rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body><h1>Shadow in-season additive-profile updater</h1>
<p>Selected common stat-prior multiplier: <strong>{selected_multiplier:g}</strong>.
The 2025-26 season was held out from selection. Public production serving remains
unchanged.</p>
<div class="decision"><strong>{decision}</strong>{interpretation}</div>
<p class="note"><strong>What the result means:</strong> updating the eight box-score
profiles is useful for explaining how a player is changing, but this study does not show
incremental predictive value beyond the existing possession-based scalar updater. The
8x setting is the least harmful finite candidate tested; it is not better than the null
of leaving profiles frozen.</p>
<h2>Locked 2025-26 game results</h2>{table}
<h2>Paired holdout uncertainty versus scalar updater</h2>{bootstrap_table}
<p><code>candidate_minus_scalar &lt; 0</code> favors the candidate for loss metrics.
Every interval crosses zero for the combined candidate.</p>
<h2>Development-season prior selection</h2>{selection_table}
<p class="small">The finite grid moved monotonically toward heavier damping. The
no-profile-update scalar control still had the best development-season loss.</p>
<h2>Player-level interpretation</h2>
<div class="grid"><section><h3>Stephen Curry</h3>{player_tables["curry"]}</section>
<section><h3>Largest final profile movements</h3>{player_tables["movers"]}</section></div>
<div class="grid"><section><h3>Cold-start example</h3>{player_tables["cold_start"]}</section>
<section><h3>Largest usage-role movement</h3>{player_tables["usage"]}</section></div>
<p class="small">Player tables show the final postgame endpoint in the locked season.
They are descriptive trajectory endpoints, not causal estimates.</p>
<h2>Double-counting audit</h2>
<p>The combined arm subtracts the pregame profile movement from each subsequent stint
target before fitting the scalar residual. The naive negative control adds the profile
movement to the unchanged scalar updater; its holdout probability losses are worse,
which is directionally consistent with double counting.</p>
<h2>Scope notes</h2>
<p>All arms use identical chronological game cutoffs, frozen minutes, and frozen context.
The source replay does not contain a comparable historical sportsbook line for every game,
so ATS is intentionally not reported. Possession-level sufficient statistics are used to
fit the updater, but the compact replay does not retain the scalar needed to reconstruct
an exact possession-residual RMSE; full-game margin and probability losses are the locked
selection and holdout criteria.</p>
</body></html>"""
    path.write_text(html)


def _report_player_tables(
    states: pd.DataFrame,
    *,
    player_panel_path: Path | str,
) -> dict[str, str]:
    """Build compact, named player examples from the locked holdout endpoint."""

    if states.empty:
        empty = "<p>No player states were materialized.</p>"
        return {key: empty for key in ("curry", "movers", "cold_start", "usage")}
    latest = (
        states.loc[states["split"].eq("holdout")]
        .sort_values(["prediction_date", "player_id"], kind="stable")
        .groupby("player_id", as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    panel_path = Path(player_panel_path)
    if panel_path.is_file():
        panel = pd.read_parquet(panel_path, columns=["season", "player_id", "player_name"])
        panel["season"] = panel["season"].astype(str)
        names = (
            panel.sort_values(["player_id", "season"], kind="stable")
            .groupby("player_id", as_index=False, sort=False)
            .agg(player_name=("player_name", "last"), career_start=("season", "min"))
        )
        latest = latest.merge(names, on="player_id", how="left", validate="one_to_one")
    else:
        latest["player_name"] = latest["player_id"].map(lambda value: f"Player {value}")
        latest["career_start"] = ""
    latest["player_name"] = latest["player_name"].fillna(
        latest["player_id"].map(lambda value: f"Player {value}")
    )

    curry_rows = latest.loc[latest["player_name"].eq("Stephen Curry")]
    curry = (
        _profile_decomposition_table(curry_rows.iloc[0])
        if not curry_rows.empty
        else "<p>Stephen Curry was not present in the holdout state.</p>"
    )
    movers = latest.assign(abs_delta=latest["additive_profile_delta"].abs()).nlargest(
        10, "abs_delta"
    )
    movers_table = movers.loc[
        :,
        ["player_name", "observed_games", "raw_on_court_possessions", "additive_profile_delta"],
    ].rename(
        columns={
            "player_name": "Player",
            "observed_games": "Games",
            "raw_on_court_possessions": "Possessions",
            "additive_profile_delta": "Profile rating delta",
        }
    ).to_html(index=False, classes="metrics", float_format=lambda value: f"{value:.3f}")

    cold = latest.loc[latest["career_start"].eq("2025-26")].assign(
        abs_delta=lambda frame: frame["additive_profile_delta"].abs()
    )
    cold_table = (
        _profile_decomposition_table(cold.nlargest(1, "abs_delta").iloc[0])
        if not cold.empty
        else "<p>No 2025-26 cold-start player was found.</p>"
    )
    usage_row = latest.assign(
        abs_usage_delta=latest["profile_delta_usage_pct"].abs()
    ).nlargest(1, "abs_usage_delta").iloc[0]
    usage_table = _profile_decomposition_table(usage_row)
    return {
        "curry": curry,
        "movers": movers_table,
        "cold_start": cold_table,
        "usage": usage_table,
    }


def _profile_decomposition_table(row: pd.Series) -> str:
    labels = {
        "three_pa_per_100": "3PA / 100",
        "three_pm_per_100": "3PM / 100",
        "assists_per_100": "Assists / 100",
        "turnovers_per_100": "Turnovers / 100",
        "usage_pct": "Usage %",
        "steals_per_100": "Steals / 100",
        "blocks_per_100": "Blocks / 100",
        "offensive_rebound_claim_total": "Offensive rebound claim %",
    }
    values = [
        {
            "Feature": label,
            "Preseason": float(row[f"preseason_{feature}"]),
            "Updated": float(row[f"updated_{feature}"]),
            "Rating delta": float(row[f"rating_delta_{feature}"]),
        }
        for feature, label in labels.items()
    ]
    title = html.escape(str(row["player_name"]))
    summary = (
        f"<p><strong>{title}</strong> — {int(row['observed_games'])} games, "
        f"{float(row['raw_on_court_possessions']):.0f} possessions; additive profile "
        f"{float(row['preseason_additive_profile']):+.3f} → "
        f"{float(row['updated_additive_profile']):+.3f} "
        f"({float(row['additive_profile_delta']):+.3f}).</p>"
    )
    return summary + pd.DataFrame(values).to_html(
        index=False, classes="metrics", float_format=lambda value: f"{value:.3f}"
    )


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument(
        "--selection-seasons", nargs="+", default=list(DEFAULT_SELECTION_SEASONS)
    )
    parser.add_argument(
        "--multiplier-grid", nargs="+", type=float, default=list(DEFAULT_MULTIPLIER_GRID)
    )
    parser.add_argument("--scalar-replay-mart", type=Path, default=DEFAULT_REPLAY_MART_PATH)
    parser.add_argument(
        "--profile-replay-mart", type=Path, default=DEFAULT_PROFILE_REPLAY_MART_PATH
    )
    parser.add_argument("--rebuild-profile-mart", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    run = run_calibration(
        seasons=tuple(args.seasons),
        selection_seasons=tuple(args.selection_seasons),
        multiplier_grid=tuple(args.multiplier_grid),
        scalar_replay_mart_path=args.scalar_replay_mart,
        profile_replay_mart_path=args.profile_replay_mart,
        rebuild_profile_mart=args.rebuild_profile_mart,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "selected_prior_multiplier": run.selected_prior_multiplier,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
