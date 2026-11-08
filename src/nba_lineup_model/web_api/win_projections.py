"""Frozen preseason team-strength and schedule win projections."""

from __future__ import annotations

from nba_lineup_model.storage import website_data as data_io

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.schedule_controls import build_back_to_back_game_features
from nba_lineup_model.season.schedule import SeasonScheduleCache
from nba_lineup_model.season.slate_time import PACIFIC, pacific_date
from nba_lineup_model.web_api.inference import (
    DEFAULT_ARTIFACTS_DIR,
    MODEL_ARTIFACT,
    TEAM_DISPLAY_NAMES,
    LineupEvaluator,
    _compiled_linear_x3_coefficients,
    _player_rating_center,
)
from nba_lineup_model.web_api.market_win_totals import (
    BETMGM_WIN_TOTALS_2026_27,
    BETMGM_WIN_TOTALS_2026_27_METADATA,
)

DEFAULT_CATALOG_PATH = Path("data/catalog/games.parquet")
CALIBRATION_SEASON = "2024-25"
HOLDOUT_SEASON = "2025-26"
PROJECTION_SEASON = "2026-27"
UNASSIGNED_REGULAR_GAMES = 2
WIN_TOTAL_INTERVAL_TRIALS = 10_000
WIN_TOTAL_INTERVAL_SEED = 20_260_910
DEFAULT_WIN_PROJECTION_CACHE_DIR = Path("artifacts/web/win_projections")
DEFAULT_SCHEDULE_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_SCHEDULE_API_URL = "http://127.0.0.1:8010/api/win-projections"
DEFAULT_SCHEDULE_S3_URI = "s3://nba-gestalt-data/nba-gestalt/schedule"
DEFAULT_SCHEDULE_DAYS_AHEAD = 7
POLYMARKET_EVENT_URL = "https://gamma-api.polymarket.com/events/slug"
POLYMARKET_MIDPOINTS_URL = "https://clob.polymarket.com/midpoints"
NBA_TEAM_ALIASES = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UT": "UTA",
    "UTAH": "UTA",
    "WSH": "WAS",
}
INJURY_STATUS_PLAY_PROBABILITY = {
    "Out": 0.0,
    "Doubtful": 0.25,
    "Questionable": 0.5,
    "Probable": 0.75,
    "Available": 1.0,
}


def build_win_projection_payload(
    *,
    evaluator: LineupEvaluator,
    minutes_payload: dict[str, object],
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> dict[str, object]:
    """Return an auditable 2026-27 baseline win projection.

    Player NAIL values and minute allocations remain frozen.  The only fitted
    win layer is a one-parameter logistic calibration trained on 2024-25,
    then tested without refitting on 2025-26.  The production scale is refit
    after that holdout on the two completed preseason snapshots.
    """

    catalog = data_io.parquet(catalog_path)
    calibration = _frozen_calibration(evaluator=evaluator, catalog=catalog)
    current_strength = _current_team_strength(minutes_payload)
    controls = evaluator.schedule_controls
    if controls is None:
        raise ValueError("Published NAIL artifact lacks schedule controls")
    games = _season_games(catalog, PROJECTION_SEASON)
    game_rows = _score_games(
        games,
        team_strength=current_strength,
        beta=calibration["production_beta"],
        home_court=float(controls.home_court),
        back_to_back=float(controls.back_to_back),
    )
    teams = _team_win_totals(
        game_rows,
        team_strength=current_strength,
        beta=calibration["production_beta"],
        home_court=float(controls.home_court),
    )
    teams["betmgm_win_total"] = teams["team"].map(BETMGM_WIN_TOTALS_2026_27)
    if teams["betmgm_win_total"].isna().any():
        missing = teams.loc[teams["betmgm_win_total"].isna(), "team"].tolist()
        raise ValueError(f"Missing BetMGM win totals for current teams: {missing}")
    intervals = simulate_baseline_win_total_intervals(
        game_rows,
        team_strength=current_strength,
        beta=calibration["production_beta"],
        home_court=float(controls.home_court),
    )
    teams = teams.merge(intervals, on="team", how="left", validate="one_to_one")
    return {
        "model": "W0 minute-weighted NAIL",
        "season": PROJECTION_SEASON,
        "scheduled_game_count": int(len(games)),
        "scheduled_games_per_team": int(len(games) * 2 / len(teams)),
        "unassigned_regular_games_per_team": UNASSIGNED_REGULAR_GAMES,
        "win_probability_scale": float(calibration["production_beta"]),
        "home_court": float(controls.home_court),
        "back_to_back": float(controls.back_to_back),
        "win_total_interval_trials": WIN_TOTAL_INTERVAL_TRIALS,
        "calibration": calibration,
        "market_win_totals": BETMGM_WIN_TOTALS_2026_27_METADATA,
        "teams": teams.to_dict(orient="records"),
        "scheduled_games": game_rows.assign(
            game_date=lambda frame: pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m-%d"),
            home_back_to_back=lambda frame: frame["home_back_to_back"].astype(int),
            away_back_to_back=lambda frame: frame["away_back_to_back"].astype(int),
        )
        .loc[
            :,
            [
                "game_id",
                "game_date",
                "home_team_tricode",
                "away_team_tricode",
                "home_back_to_back",
                "away_back_to_back",
            ],
        ]
        .rename(
            columns={
                "home_team_tricode": "home_team",
                "away_team_tricode": "away_team",
            }
        )
        .to_dict(orient="records"),
        "contract": (
            "Team strength is the minute-weighted sum of frozen player NAIL forecasts. "
            "Scheduled games use the official home-court and back-to-back controls. "
            "The NBA has published 80 regular-season games per team; two unassigned "
            "games are estimated against a neutral opponent with one home and one away setting."
        ),
    }


def build_current_season_minutes_payload(
    *,
    evaluator: LineupEvaluator,
    preseason_minutes_payload: dict[str, object],
) -> dict[str, object]:
    """Overlay the promoted rating and role state onto the frozen roster payload.

    The result remains a forward-looking allocation: current participation and
    conditional-minutes estimates choose and weight a 15-player rotation, then
    each team is normalized to 240 minutes.  Daily injury reports still replace
    participation assumptions when an individual game is materialized.
    """

    current = evaluator.current_season_rankings.copy()
    if current.empty:
        return dict(preseason_minutes_payload)
    required = {
        "player_id",
        "player_name",
        "team",
        "rapm",
        "current_expected_minutes_if_available",
        "current_participation_estimate",
    }
    missing = sorted(required - set(current))
    if missing:
        raise ValueError("Current serving state lacks minutes fields: " + ", ".join(missing))

    baseline_rows = {
        int(row["player_id"]): row
        for row in preseason_minutes_payload.get("players", [])
        if isinstance(row, dict) and row.get("player_id") is not None
    }
    rows: list[dict[str, object]] = []
    for row in current.itertuples(index=False):
        player_id = int(row.player_id)
        baseline = baseline_rows.get(player_id, {})
        availability = _finite_or_default(
            row.current_participation_estimate,
            baseline.get("availability_probability", 1.0),
        )
        conditional = _finite_or_default(
            row.current_expected_minutes_if_available,
            baseline.get("conditional_minutes_per_game", 0.0),
        )
        availability = min(1.0, max(0.0, availability))
        conditional = min(48.0, max(0.0, conditional))
        rows.append(
            {
                **baseline,
                "player_id": player_id,
                "player_name": str(row.player_name),
                "position": str(getattr(row, "position", baseline.get("position", ""))),
                "age": _none_if_not_finite(getattr(row, "age", baseline.get("age"))),
                "team": str(row.team),
                "prior_season_minutes": float(baseline.get("prior_season_minutes", 0.0)),
                "is_rating_fallback": False,
                "has_prior_availability_state": bool(
                    baseline.get("has_prior_availability_state", False)
                ),
                "has_prior_minutes_state": bool(
                    baseline.get("has_prior_minutes_state", False)
                ),
                "availability_probability": availability,
                "conditional_minutes_per_game": conditional,
                "raw_expected_total_minutes": 82.0 * availability * conditional,
                "projected_nail": float(row.rapm),
                "rating_source": "in-season updater",
                "preseason_projected_nail": _finite_or_default(
                    getattr(row, "preseason_nail", None),
                    baseline.get("projected_nail", 0.0),
                ),
                "preseason_conditional_minutes_per_game": _finite_or_default(
                    getattr(row, "preseason_expected_minutes_if_available", None),
                    baseline.get("conditional_minutes_per_game", 0.0),
                ),
            }
        )

    players = pd.DataFrame.from_records(rows)
    if players.empty:
        raise ValueError("Current serving state produced no player minutes")
    players["is_baseline_rotation_candidate"] = False
    players["baseline_minutes_per_game"] = 0.0
    players["baseline_minute_share"] = 0.0
    rotation_size = int(preseason_minutes_payload.get("initial_rotation_size", 15))
    for _team, index in players.groupby("team", sort=True).groups.items():
        ordered = players.loc[index].sort_values(
            ["raw_expected_total_minutes", "player_name", "player_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        selected = ordered.head(rotation_size).index
        total = float(players.loc[selected, "raw_expected_total_minutes"].sum())
        if total <= 0.0:
            raise ValueError(f"Current minutes state produced no rotation for {_team}")
        players.loc[selected, "is_baseline_rotation_candidate"] = True
        players.loc[selected, "baseline_minute_share"] = (
            players.loc[selected, "raw_expected_total_minutes"] / total
        )
        players.loc[selected, "baseline_minutes_per_game"] = (
            240.0 * players.loc[selected, "raw_expected_total_minutes"] / total
        )

    players = players.sort_values(
        ["team", "baseline_minutes_per_game", "player_name", "player_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    metadata = evaluator.current_season_metadata
    artifact_sha256 = metadata.get("artifact_sha256", {})
    artifact_sources = metadata.get("artifact_sources", {})
    return {
        **{
            key: value
            for key, value in preseason_minutes_payload.items()
            if key not in {"players", "teams", "contract", "model", "win_projection"}
        },
        "season": PROJECTION_SEASON,
        "completed_season": preseason_minutes_payload.get("completed_season", HOLDOUT_SEASON),
        "model": "In-season NAIL + conditional-minutes updater",
        "state_kind": "in_season_updated",
        "as_of": metadata.get("evidence_through_date"),
        "serving_bundle_sha256": metadata.get("bundle_sha256"),
        "projection_provenance": {
            "schema_version": 1,
            "serving_bundle_sha256": metadata.get("bundle_sha256"),
            "rating_state_sha256": artifact_sha256.get("rating_state"),
            "minutes_state_sha256": artifact_sha256.get("minutes_state"),
            "rating_source_uri": artifact_sources.get("rating_state"),
            "minutes_source_uri": artifact_sources.get("minutes_state"),
            "rating_as_of": metadata.get("evidence_through_date"),
            "as_of_cutoff_time": metadata.get("as_of_cutoff_time"),
        },
        "initial_rotation_size": rotation_size,
        "regulation_team_minutes": 240.0,
        "contract": (
            "Current NAIL ratings and current role estimates are normalized to 240 "
            "minutes per team. Official game injury reports supersede the participation "
            "estimate when Schedule projections are frozen."
        ),
        "teams": sorted(players["team"].astype(str).unique().tolist()),
        "players": players.to_dict(orient="records"),
    }


def build_current_season_win_projection_payload(
    *,
    evaluator: LineupEvaluator,
    minutes_payload: dict[str, object],
    preseason_forecast: dict[str, object],
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> dict[str, object]:
    """Project final wins from actual results plus the remaining current schedule."""

    metadata = evaluator.current_season_metadata
    if not metadata:
        return dict(preseason_forecast)
    through_date = str(metadata["evidence_through_date"])
    records_payload = read_current_season_records(
        state_dir=state_dir,
        season=PROJECTION_SEASON,
        through_date=through_date,
    )
    records = pd.DataFrame(records_payload["teams"])
    if records.empty:
        records = pd.DataFrame(
            {
                "team": minutes_payload["teams"],
                "games": 0,
                "wins": 0,
                "losses": 0,
            }
        )
    else:
        records = records.loc[:, ["team", "games", "wins", "losses"]]

    catalog = data_io.parquet(catalog_path)
    all_games = _season_games(catalog, PROJECTION_SEASON)
    completed_game_ids = {
        str(game_id) for game_id in records_payload.get("completed_game_ids", [])
    }
    remaining_games = all_games.loc[
        ~all_games["game_id"].astype(str).isin(completed_game_ids)
    ].copy()
    strength = _current_team_strength(minutes_payload)
    beta = float(preseason_forecast["win_probability_scale"])
    schedule_controls = metadata.get("schedule_controls")
    home_court = float(
        schedule_controls["home_court"]
        if isinstance(schedule_controls, dict)
        and schedule_controls.get("home_court") is not None
        else preseason_forecast["home_court"]
    )
    back_to_back = float(preseason_forecast["back_to_back"])
    game_rows = _score_games(
        remaining_games,
        team_strength=strength,
        beta=beta,
        home_court=home_court,
        back_to_back=back_to_back,
    )
    remaining = _team_win_totals(
        game_rows,
        team_strength=strength,
        beta=beta,
        home_court=home_court,
    ).rename(
        columns={
            "scheduled_wins": "remaining_scheduled_wins",
            "scheduled_games": "remaining_scheduled_games",
        }
    )
    teams = remaining.merge(records, on="team", how="left", validate="one_to_one")
    for column in ("games", "wins", "losses"):
        teams[column] = teams[column].fillna(0).astype(int)
    teams["scheduled_wins"] = teams["wins"] + teams["remaining_scheduled_wins"]
    teams["scheduled_games"] = teams["games"] + teams["remaining_scheduled_games"]
    teams["projected_wins"] = teams["scheduled_wins"] + teams["unassigned_wins"]
    teams["projected_losses"] = 82.0 - teams["projected_wins"]
    teams = teams.rename(
        columns={
            "games": "completed_games",
            "wins": "actual_wins",
            "losses": "actual_losses",
        }
    )
    market = {
        str(row["team"]): row.get("betmgm_win_total")
        for row in preseason_forecast.get("teams", [])
        if isinstance(row, dict)
    }
    teams["betmgm_win_total"] = teams["team"].map(market)
    intervals = simulate_current_win_total_intervals(
        game_rows,
        team_strength=strength,
        records=teams.set_index("team")[["completed_games", "actual_wins"]],
        beta=beta,
        home_court=home_court,
    )
    teams = teams.drop(
        columns=["win_total_p1", "win_total_p5", "win_total_p50", "win_total_p95", "win_total_p99"],
        errors="ignore",
    ).merge(intervals, on="team", how="left", validate="one_to_one")
    teams = teams.sort_values(
        ["projected_wins", "team"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    return {
        **{
            key: value
            for key, value in preseason_forecast.items()
            if key not in {"teams", "scheduled_games", "contract", "model"}
        },
        "model": "Current NAIL + current minutes",
        "model_track": "updated_ratings_updated_minutes",
        "state_kind": "in_season_updated",
        "as_of": through_date,
        "serving_bundle_sha256": metadata.get("bundle_sha256"),
        "home_court": home_court,
        "schedule_controls": schedule_controls,
        "actual_results_source": records_payload.get("source"),
        "scheduled_game_count": int(len(all_games)),
        "scheduled_games_per_team": int(len(all_games) * 2 / len(strength)),
        "remaining_scheduled_game_count": int(len(remaining_games)),
        "teams": teams.to_dict(orient="records"),
        # ``scheduled_games`` is the public season calendar contract.  Schedule
        # consumers need completed games here as well as future games; the
        # simulation itself still uses only ``game_rows`` above.
        "scheduled_games": _public_scheduled_games(all_games),
        "remaining_scheduled_games": _public_scheduled_games(game_rows),
        "contract": (
            f"Actual wins through {through_date} are fixed. Remaining games use the "
            "latest promoted player ratings and role estimates, plus the selected "
            "causal home-court state, frozen back-to-back, and win-calibration layers."
        ),
    }


def win_projection_cache_path(model_artifact: str, run_id: str) -> Path:
    """Return the immutable release path for a baseline W0 payload."""

    return DEFAULT_WIN_PROJECTION_CACHE_DIR / model_artifact / f"{run_id}.json"


def materialize_win_projection_cache(
    *,
    evaluator: LineupEvaluator,
    output_path: Path | str | None = None,
) -> Path:
    """Build the all-team baseline envelope once for a release bundle."""

    from nba_lineup_model.web_api.inference import MODEL_ARTIFACT
    from nba_lineup_model.web_api.preseason_minutes import (
        build_forward_conditional_preseason_minutes_payload,
    )

    minutes = build_forward_conditional_preseason_minutes_payload(
        preseason_rankings=evaluator.preseason_rankings,
    )
    payload = {
        "minutes": minutes,
        "win_projection": build_win_projection_payload(
            evaluator=evaluator,
            minutes_payload=minutes,
        ),
    }
    path = Path(output_path) if output_path is not None else win_projection_cache_path(
        MODEL_ARTIFACT, evaluator.run_id
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def read_win_projection_cache(*, model_artifact: str, run_id: str) -> dict[str, object]:
    """Read the release-materialized W0 payload without a request-path replay."""

    path = win_projection_cache_path(model_artifact, run_id)
    if not data_io.is_file(path):
        raise FileNotFoundError(f"Missing materialized win projection cache: {path}")
    payload = json.loads(data_io.read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid win projection cache payload: {path}")
    return payload


def simulate_baseline_win_total_intervals(
    games: pd.DataFrame,
    *,
    team_strength: pd.Series,
    beta: float,
    home_court: float,
    trials: int = WIN_TOTAL_INTERVAL_TRIALS,
    seed: int = WIN_TOTAL_INTERVAL_SEED,
) -> pd.DataFrame:
    """Simulate final win-total quantiles for every baseline team schedule.

    Each team gets an independent deterministic draw stream. This mirrors the
    public envelope's within-team Bernoulli contract while making the published
    P5--P95 table values reproducible in the release bundle.
    """

    if trials <= 0:
        raise ValueError("Win-total interval trials must be positive")
    required = {"home_team_tricode", "away_team_tricode", "home_win_probability"}
    missing = sorted(required - set(games))
    if missing:
        raise ValueError("Win-total intervals require: " + ", ".join(missing))

    probabilities: dict[str, list[float]] = {str(team): [] for team in team_strength.index}
    for row in games.loc[
        :, ["home_team_tricode", "away_team_tricode", "home_win_probability"]
    ].itertuples(index=False):
        home_team = str(row.home_team_tricode)
        away_team = str(row.away_team_tricode)
        home_probability = float(row.home_win_probability)
        probabilities[home_team].append(home_probability)
        probabilities[away_team].append(1.0 - home_probability)

    for team, strength in team_strength.items():
        probabilities[str(team)].extend(
            [
                float(_sigmoid(beta * (float(strength) + home_court))),
                float(_sigmoid(beta * (float(strength) - home_court))),
            ]
        )

    rows: list[dict[str, float | str]] = []
    for team in sorted(probabilities):
        team_probabilities = np.asarray(probabilities[team], dtype=float)
        if len(team_probabilities) != 82:
            raise ValueError(
                f"Expected 82 modeled games for {team}; found {len(team_probabilities)}"
            )
        team_seed = seed + sum(
            (index + 1) * ord(character) for index, character in enumerate(team)
        )
        generator = np.random.default_rng(team_seed)
        totals = (
            generator.random((trials, len(team_probabilities))) < team_probabilities
        ).sum(axis=1)
        ordered = np.sort(totals)
        rows.append(
            {
                "team": team,
                "win_total_p1": float(_empirical_quantile(ordered, 0.01)),
                "win_total_p5": float(_empirical_quantile(ordered, 0.05)),
                "win_total_p50": float(_empirical_quantile(ordered, 0.50)),
                "win_total_p95": float(_empirical_quantile(ordered, 0.95)),
                "win_total_p99": float(_empirical_quantile(ordered, 0.99)),
            }
        )
    return pd.DataFrame.from_records(rows)


def simulate_current_win_total_intervals(
    games: pd.DataFrame,
    *,
    team_strength: pd.Series,
    records: pd.DataFrame,
    beta: float,
    home_court: float,
    trials: int = WIN_TOTAL_INTERVAL_TRIALS,
    seed: int = WIN_TOTAL_INTERVAL_SEED,
) -> pd.DataFrame:
    """Simulate final totals conditional on already observed wins and losses."""

    if trials <= 0:
        raise ValueError("Win-total interval trials must be positive")
    probabilities: dict[str, list[float]] = {str(team): [] for team in team_strength.index}
    for row in games.loc[
        :, ["home_team_tricode", "away_team_tricode", "home_win_probability"]
    ].itertuples(index=False):
        home_team = str(row.home_team_tricode)
        away_team = str(row.away_team_tricode)
        home_probability = float(row.home_win_probability)
        probabilities[home_team].append(home_probability)
        probabilities[away_team].append(1.0 - home_probability)
    for team, strength in team_strength.items():
        probabilities[str(team)].extend(
            [
                float(_sigmoid(beta * (float(strength) + home_court))),
                float(_sigmoid(beta * (float(strength) - home_court))),
            ]
        )

    rows: list[dict[str, float | str]] = []
    for team in sorted(probabilities):
        completed_games = int(records.loc[team, "completed_games"]) if team in records.index else 0
        actual_wins = int(records.loc[team, "actual_wins"]) if team in records.index else 0
        team_probabilities = np.asarray(probabilities[team], dtype=float)
        modeled_final_games = completed_games + len(team_probabilities)
        if modeled_final_games != 82:
            raise ValueError(
                f"Expected 82 observed-plus-modeled games for {team}; "
                f"found {modeled_final_games}"
            )
        team_seed = seed + sum(
            (index + 1) * ord(character) for index, character in enumerate(team)
        )
        generator = np.random.default_rng(team_seed)
        totals = actual_wins + (
            generator.random((trials, len(team_probabilities))) < team_probabilities
        ).sum(axis=1)
        ordered = np.sort(totals)
        rows.append(
            {
                "team": team,
                "win_total_p1": float(_empirical_quantile(ordered, 0.01)),
                "win_total_p5": float(_empirical_quantile(ordered, 0.05)),
                "win_total_p50": float(_empirical_quantile(ordered, 0.50)),
                "win_total_p95": float(_empirical_quantile(ordered, 0.95)),
                "win_total_p99": float(_empirical_quantile(ordered, 0.99)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _public_scheduled_games(games: pd.DataFrame) -> list[dict[str, object]]:
    return (
        games.assign(
            game_date=lambda frame: pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m-%d"),
            home_back_to_back=lambda frame: frame["home_back_to_back"].astype(int),
            away_back_to_back=lambda frame: frame["away_back_to_back"].astype(int),
        )
        .loc[
            :,
            [
                "game_id",
                "game_date",
                "home_team_tricode",
                "away_team_tricode",
                "home_back_to_back",
                "away_back_to_back",
            ],
        ]
        .rename(
            columns={
                "home_team_tricode": "home_team",
                "away_team_tricode": "away_team",
            }
        )
        .to_dict(orient="records")
    )


def _finite_or_default(value: object, default: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not np.isfinite(parsed):
        parsed = float(default)
    return parsed


def _none_if_not_finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _empirical_quantile(ordered_values: np.ndarray, probability: float) -> int:
    """Match the public envelope's nearest-upper empirical percentile rule."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Empirical percentile must lie between zero and one")
    index = int(np.ceil(probability * (len(ordered_values) - 1)))
    return int(ordered_values[index])


def _json_default(value: Any) -> object:
    """Serialize NumPy scalars without changing the cache's JSON contract."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _frozen_calibration(*, evaluator: LineupEvaluator, catalog: pd.DataFrame) -> dict[str, float]:
    calibration = _historical_game_rows(
        evaluator=evaluator, catalog=catalog, season=CALIBRATION_SEASON
    )
    calibration_beta = fit_win_probability_scale(calibration["edge"], calibration["home_win"])
    holdout = _historical_game_rows(evaluator=evaluator, catalog=catalog, season=HOLDOUT_SEASON)
    holdout_metrics = score_win_probabilities(
        holdout["edge"], holdout["home_win"], beta=calibration_beta
    )
    combined = pd.concat([calibration, holdout], ignore_index=True)
    production_beta = fit_win_probability_scale(combined["edge"], combined["home_win"])
    return {
        "calibration_season": CALIBRATION_SEASON,
        "holdout_season": HOLDOUT_SEASON,
        "calibration_beta": calibration_beta,
        "holdout_brier": holdout_metrics["brier"],
        "holdout_log_loss": holdout_metrics["log_loss"],
        "holdout_accuracy": holdout_metrics["accuracy"],
        "production_beta": production_beta,
    }


def _historical_game_rows(
    *, evaluator: LineupEvaluator,
    catalog: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    strength = _historical_team_strength(evaluator=evaluator, season=season)
    controls = _historical_schedule_controls(evaluator=evaluator, target_season=season)
    games = _season_games(catalog, season)
    predicted = _score_games(
        games,
        team_strength=strength,
        beta=1.0,
        home_court=controls["home_court"],
        back_to_back=controls["back_to_back"],
    )
    outcomes = _actual_home_outcomes(season)
    output = predicted.merge(outcomes, on="game_id", how="inner", validate="one_to_one")
    if len(output) != len(games):
        raise ValueError(f"Missing completed game outcomes for {season}")
    return output


def _historical_team_strength(*, evaluator: LineupEvaluator, season: str) -> pd.Series:
    run_dir = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / evaluator.season / evaluator.run_id
    priors = data_io.parquet(run_dir / "season_player_priors.parquet")
    priors = priors.loc[priors["season"].eq(season), ["player_id", "prior_rapm"]].rename(
        columns={"prior_rapm": "rapm"}
    )
    profiles = evaluator.historical_profiles.loc[
        evaluator.historical_profiles["season"].eq(season)
    ].copy()
    prior_season = _previous_season(season)
    context_model = evaluator.season_context_models[prior_season]
    exposures = evaluator.exposure_cohort.loc[
        evaluator.exposure_cohort["season"].eq(prior_season),
        ["player_id", "on_court_possessions"],
    ].rename(columns={"on_court_possessions": "possessions"})
    uncentered = _compiled_linear_x3_coefficients(priors, profiles, context_model)
    center = _player_rating_center(uncentered, exposures)
    ratings = _compiled_linear_x3_coefficients(priors, profiles, context_model, center=center)
    rating_by_player = ratings.set_index("player_id")["rapm"]
    candidates = data_io.parquet(
        Path("data/curated/opening_rosters") / season / "part-00000.parquet"
    ).copy()
    previous_minutes = _prior_minutes(_previous_season(season))
    candidates["player_id"] = candidates["player_id"].astype(int)
    candidates["score"] = candidates["player_id"].map(previous_minutes).fillna(0.0)
    candidates["rating"] = candidates["player_id"].map(rating_by_player).fillna(0.0)
    candidates["share"] = candidates.groupby("team", sort=False)["score"].transform(
        lambda values: _smoothed_shares(values, epsilon=0.32)
    )
    strength = candidates.assign(weight=lambda frame: 5.0 * frame["share"]).groupby(
        "team", sort=True
    ).apply(lambda frame: float((frame["weight"] * frame["rating"]).sum()), include_groups=False)
    return strength - float(strength.mean())


def _current_team_strength(minutes_payload: dict[str, object]) -> pd.Series:
    players = pd.DataFrame(minutes_payload["players"])
    strength = players.assign(
        weight=lambda frame: frame["baseline_minutes_per_game"].astype(float) / 48.0
    ).groupby("team", sort=True).apply(
        lambda frame: float((frame["weight"] * frame["projected_nail"]).sum()),
        include_groups=False,
    )
    return strength - float(strength.mean())


def _historical_schedule_controls(
    *, evaluator: LineupEvaluator, target_season: str
) -> dict[str, float]:
    run_dir = DEFAULT_ARTIFACTS_DIR / MODEL_ARTIFACT / evaluator.season / evaluator.run_id
    context = data_io.parquet(run_dir / "season_context_metadata.parquet")
    schedule = data_io.parquet(run_dir / "season_schedule_control_metadata.parquet")
    completed = context.loc[context["season"].astype(str).lt(target_season)].copy()
    completed_schedule = schedule.loc[schedule["season"].astype(str).lt(target_season)].copy()
    if completed.empty or completed_schedule.empty:
        raise ValueError(f"No completed schedule-control states before {target_season}")
    return {
        "home_court": float(
            np.average(
                completed["context_home_intercept"],
                weights=completed["context_training_weight_sum"],
            )
        ),
        "back_to_back": float(
            np.average(
                completed_schedule["schedule_control_raw_weight"],
                weights=completed_schedule["schedule_training_stint_count"],
            )
        ),
    }


def _season_games(catalog: pd.DataFrame, season: str) -> pd.DataFrame:
    games = catalog.loc[
        catalog["season"].eq(season) & catalog["season_type"].eq("regular"),
        [
            "game_id",
            "season",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_team_tricode",
            "away_team_tricode",
        ],
    ].copy()
    features = build_back_to_back_game_features(games)
    return games.merge(
        features.loc[:, ["game_id", "home_back_to_back", "away_back_to_back"]],
        on="game_id", how="left", validate="one_to_one"
    )


def _score_games(
    games: pd.DataFrame,
    *, team_strength: pd.Series,
    beta: float,
    home_court: float,
    back_to_back: float,
) -> pd.DataFrame:
    output = games.copy()
    output["home_strength"] = output["home_team_tricode"].map(team_strength)
    output["away_strength"] = output["away_team_tricode"].map(team_strength)
    if output[["home_strength", "away_strength"]].isna().any().any():
        missing = sorted(
            set(output.loc[output["home_strength"].isna(), "home_team_tricode"])
            | set(output.loc[output["away_strength"].isna(), "away_team_tricode"])
        )
        raise ValueError("Team-strength map is missing schedule teams: " + ", ".join(missing))
    output["edge"] = (
        output["home_strength"] - output["away_strength"] + home_court
        + back_to_back * (output["home_back_to_back"] - output["away_back_to_back"])
    )
    output["home_win_probability"] = _sigmoid(beta * output["edge"].to_numpy(dtype=float))
    return output


def _team_win_totals(
    games: pd.DataFrame,
    *, team_strength: pd.Series,
    beta: float,
    home_court: float,
) -> pd.DataFrame:
    home = games.loc[:, ["home_team_tricode", "home_win_probability"]].rename(
        columns={"home_team_tricode": "team", "home_win_probability": "scheduled_wins"}
    )
    away = games.loc[:, ["away_team_tricode", "home_win_probability"]].rename(
        columns={"away_team_tricode": "team"}
    )
    away["scheduled_wins"] = 1.0 - away.pop("home_win_probability")
    scheduled = pd.concat([home, away], ignore_index=True).groupby("team", as_index=False).agg(
        scheduled_wins=("scheduled_wins", "sum"), scheduled_games=("scheduled_wins", "size")
    )
    strengths = pd.DataFrame(
        {"team": team_strength.index, "team_strength": team_strength.values}
    )
    output = strengths.merge(
        scheduled, on="team", how="left", validate="one_to_one"
    )
    neutral_home = _sigmoid(beta * (output["team_strength"].to_numpy(dtype=float) + home_court))
    neutral_away = _sigmoid(beta * (output["team_strength"].to_numpy(dtype=float) - home_court))
    raw_unassigned_wins = UNASSIGNED_REGULAR_GAMES * (neutral_home + neutral_away) / 2.0
    # The two unresolved Cup games must still contribute exactly 30 league wins.
    # Preserve teams' relative neutral-opponent differences, then center their
    # aggregate expectation to one win per team.
    output["unassigned_wins"] = (
        raw_unassigned_wins
        - float(raw_unassigned_wins.mean())
        + UNASSIGNED_REGULAR_GAMES / 2.0
    )
    output["projected_wins"] = output["scheduled_wins"] + output["unassigned_wins"]
    output["projected_losses"] = 82.0 - output["projected_wins"]
    return output.sort_values(
        ["projected_wins", "team"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def fit_win_probability_scale(edges: pd.Series, outcomes: pd.Series) -> float:
    """Fit one no-intercept logistic scale by stable Newton updates."""

    x = np.asarray(edges, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if (
        len(x) == 0
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isin(y, [0.0, 1.0]).all()
    ):
        raise ValueError("Win calibration requires finite edges and binary outcomes")
    beta = 0.1
    for _ in range(100):
        p = _sigmoid(beta * x)
        gradient = float(np.sum((p - y) * x))
        hessian = float(np.sum(p * (1.0 - p) * np.square(x)))
        if hessian <= 1e-12:
            break
        updated = max(0.0, beta - gradient / hessian)
        if abs(updated - beta) <= 1e-10:
            beta = updated
            break
        beta = updated
    return float(beta)


def score_win_probabilities(
    edges: pd.Series, outcomes: pd.Series, *, beta: float
) -> dict[str, float]:
    p = _sigmoid(float(beta) * np.asarray(edges, dtype=float))
    y = np.asarray(outcomes, dtype=float)
    return {
        "brier": float(np.mean(np.square(p - y))),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1.0))),
    }


def build_frozen_season_win_backtest(
    *,
    evaluator: LineupEvaluator,
    catalog: pd.DataFrame,
    target_seasons: list[str],
) -> pd.DataFrame:
    """Create strict preseason W0 win totals for completed target seasons.

    The probability scale for season ``t`` uses only completed game results
    from ``t - 1``.  Player forecasts, minute shares, and schedule controls
    are already frozen by :func:`_historical_game_rows`.
    """

    frames: list[pd.DataFrame] = []
    for target_season in target_seasons:
        calibration_season = _previous_season(target_season)
        calibration = _historical_game_rows(
            evaluator=evaluator, catalog=catalog, season=calibration_season
        )
        beta = fit_win_probability_scale(calibration["edge"], calibration["home_win"])
        target = _historical_game_rows(
            evaluator=evaluator, catalog=catalog, season=target_season
        )
        totals = _completed_team_win_totals(target, beta=beta)
        if len(totals) != 30 or not totals["scheduled_games"].eq(82).all():
            raise ValueError(f"Incomplete completed-season schedule: {target_season}")
        totals.insert(0, "season", target_season)
        totals["calibration_season"] = calibration_season
        totals["win_probability_scale"] = beta
        frames.append(totals)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["season", "team"], kind="stable"
    ).reset_index(drop=True)


def score_season_win_forecasts(
    forecasts: pd.DataFrame, *, prediction_column: str
) -> dict[str, float]:
    """Return team-season point-forecast error measures."""

    required = {prediction_column, "actual_wins"}
    missing = sorted(required - set(forecasts))
    if missing:
        raise ValueError("Season forecast scores require: " + ", ".join(missing))
    error = forecasts[prediction_column].astype(float) - forecasts["actual_wins"].astype(float)
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(error.mean()),
    }


def _actual_home_outcomes(season: str) -> pd.DataFrame:
    """Read completed regular-season results from the official schedule cache.

    The schedule endpoint is authoritative for final scores and covers the
    complete slate.  Player box-score partitions are intentionally not used
    here because historical player coverage can be incomplete even where the
    schedule result is known.
    """

    response = SeasonScheduleCache().read(season)
    if response is None:
        raise FileNotFoundError(f"No cached official NBA schedule for {season}")

    records: list[dict[str, object]] = []
    game_dates = response.payload.get("leagueSchedule", {}).get("gameDates", [])
    for game_date in game_dates:
        for game in game_date.get("games", []):
            game_id = str(game.get("gameId", ""))
            if not game_id.startswith("002") or int(game.get("gameStatus", 0)) != 3:
                continue
            home_score = game.get("homeTeam", {}).get("score")
            away_score = game.get("awayTeam", {}).get("score")
            if home_score is None or away_score is None:
                raise ValueError(f"Final schedule game lacks a score: {game_id}")
            if home_score == away_score:
                raise ValueError(f"Tied regular-season outcome in {season}: {game_id}")
            records.append({"game_id": game_id, "home_win": float(home_score > away_score)})

    outcomes = pd.DataFrame.from_records(records)
    if outcomes.empty:
        raise ValueError(f"No completed regular-season outcomes in schedule cache: {season}")
    if outcomes["game_id"].duplicated().any():
        raise ValueError(f"Duplicate regular-season outcome in schedule cache: {season}")
    return outcomes


def _completed_team_win_totals(games: pd.DataFrame, *, beta: float) -> pd.DataFrame:
    required = {"home_team_tricode", "away_team_tricode", "edge", "home_win"}
    missing = sorted(required - set(games))
    if missing:
        raise ValueError("Completed team win totals require: " + ", ".join(missing))
    probabilities = _sigmoid(beta * games["edge"].to_numpy(dtype=float))
    home = pd.DataFrame(
        {
            "team": games["home_team_tricode"].to_numpy(),
            "projected_wins": probabilities,
            "actual_wins": games["home_win"].to_numpy(dtype=float),
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team_tricode"].to_numpy(),
            "projected_wins": 1.0 - probabilities,
            "actual_wins": 1.0 - games["home_win"].to_numpy(dtype=float),
        }
    )
    totals = pd.concat([home, away], ignore_index=True).groupby("team", as_index=False).agg(
        projected_wins=("projected_wins", "sum"),
        actual_wins=("actual_wins", "sum"),
        scheduled_games=("projected_wins", "size"),
    )
    return totals


def _prior_minutes(season: str) -> pd.Series:
    from nba_lineup_model.rotation.l1_minute_share_persistence import read_regular_game_minutes

    return read_regular_game_minutes(season).groupby("player_id")["minutes"].sum()


def _smoothed_shares(scores: pd.Series, *, epsilon: float) -> pd.Series:
    values = scores.to_numpy(dtype=float)
    if float(values.sum()) > 0.0:
        normalized = values / values.sum()
    else:
        normalized = np.full(len(values), 1.0 / len(values))
    return pd.Series((1.0 - epsilon) * normalized + epsilon / len(values), index=scores.index)


def _previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def read_schedule_day(
    game_date: str,
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    live_score_state_dir: Path | str | None = None,
) -> dict[str, object]:
    """Read the latest board with immutable pregame and final audit records."""

    state_root = Path(state_dir)
    from nba_lineup_model.season.serving_selection import INDEX_FILENAME, read_selected_day
    if data_io.is_file(state_root / INDEX_FILENAME):
        return read_selected_day(game_date, root=state_root)
    path = state_root / "latest" / f"{game_date}.json"
    if not data_io.is_file(path):
        return {
            "available": False,
            "game_date": game_date,
            "source": "NBA official schedule and liveData",
            "games": [],
        }
    payload = json.loads(data_io.read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid schedule archive payload: {path}")
    games = payload.get("games", [])
    if not isinstance(games, list):
        raise ValueError(f"Invalid schedule games payload: {path}")
    observation_root = (
        Path(live_score_state_dir)
        if live_score_state_dir is not None
        else state_root
    )
    observation_path = observation_root / "latest" / f"{game_date}.json"
    if observation_root != state_root and data_io.is_file(observation_path):
        observations = json.loads(data_io.read_text(observation_path))
        observed_games = (
            observations.get("games")
            if isinstance(observations, dict)
            else None
        )
        if not isinstance(observed_games, list):
            raise ValueError(f"Invalid live-score games payload: {observation_path}")
        by_id = {
            str(game.get("game_id")): game
            for game in observed_games
            if isinstance(game, dict) and game.get("game_id")
        }
        for game in games:
            if not isinstance(game, dict) or not game.get("game_id"):
                continue
            observed = by_id.get(str(game["game_id"]))
            if observed is None:
                continue
            for team_key in ("away_team", "home_team"):
                if observed.get(team_key) != game.get(team_key):
                    raise ValueError(
                        f"Live-score {team_key} mismatch for {game['game_id']}"
                    )
            for key in ("away_score", "home_score", "status"):
                if key in observed:
                    game[key] = copy.deepcopy(observed[key])
        if observations.get("live_score_collected_at"):
            payload["live_score_collected_at"] = observations[
                "live_score_collected_at"
            ]
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", "")).strip()
        if not game_id:
            continue
        frozen_path = state_root / "frozen" / f"{game_id}.json"
        frozen = None
        if data_io.is_file(frozen_path):
            frozen = json.loads(data_io.read_text(frozen_path))
        elif str(game.get("status", {}).get("state")) != "pre":
            frozen = _select_immutable_pretip_candidate(
                state_root=state_root,
                game=game,
            )
        if isinstance(frozen, dict):
            game["pregame"] = _renderable_frozen_pregame(
                frozen,
                served_game=game,
                game_id=game_id,
            )
        final_path = state_root / "final" / f"{game_id}.summary.json"
        final_audit: dict[str, object] | None = None
        if data_io.is_file(final_path):
            stored_final_audit = json.loads(data_io.read_text(final_path))
            if isinstance(stored_final_audit, dict):
                final_audit = stored_final_audit
        elif (
            observation_root != state_root
            and str(game.get("status", {}).get("state")) == "post"
            and isinstance(game.get("pregame"), dict)
        ):
            observed_final_path = (
                observation_root / "final" / f"{game_id}.summary.json"
            )
            if data_io.is_file(observed_final_path):
                observed_final_audit = json.loads(data_io.read_text(observed_final_path))
                if not isinstance(observed_final_audit, dict):
                    raise ValueError(
                        f"Invalid live-score final audit payload: {observed_final_path}"
                    )
                for identity_key in ("game_id", "away_team", "home_team"):
                    if str(observed_final_audit.get(identity_key)) != str(
                        game.get(identity_key)
                    ):
                        raise ValueError(
                            f"Live-score final audit {identity_key} mismatch for {game_id}"
                        )
                retrospective = observed_final_audit.get("retrospective")
                archived_at = datetime.fromisoformat(
                    str(observed_final_audit["archived_at"]).replace("Z", "+00:00")
                )
                final_audit = _build_final_game_archive(
                    game,
                    pregame=game["pregame"],
                    retrospective=(
                        retrospective if isinstance(retrospective, dict) else {}
                    ),
                    archived_at=archived_at,
                )
        if final_audit is not None:
            game["final_audit"] = final_audit
            retrospective = final_audit.get("retrospective")
            if isinstance(retrospective, dict):
                game["retrospective"] = retrospective
    payload["available"] = True
    manifest_path = state_root / "manifest.json"
    if data_io.is_file(manifest_path):
        manifest = json.loads(data_io.read_text(manifest_path))
        if isinstance(manifest, dict):
            payload["forecast_version"] = {
                "model_track": manifest.get("model_track"),
                "evaluation_cell": manifest.get("evaluation_cell"),
                "mode": "strict_forward_replay",
                "through_date": manifest.get("through_date"),
                "identity": manifest.get("identity"),
            }
    return payload


def _select_immutable_pretip_candidate(
    *,
    state_root: Path,
    game: dict[str, object],
) -> dict[str, object] | None:
    """Render the exact pre-tip candidate when a read-only release turns live."""

    game_id = str(game["game_id"])
    scheduled_start = datetime.fromisoformat(
        str(game["start_time"]).replace("Z", "+00:00")
    )
    candidates: list[tuple[datetime, dict[str, object]]] = []
    for path in sorted(data_io.glob(state_root / "pregame-candidates" / game_id, "*.json")):
        candidate = json.loads(data_io.read_text(path))
        if not isinstance(candidate, dict):
            raise ValueError(f"Invalid pregame candidate: {path}")
        digest = str(candidate.get("candidate_sha256", ""))
        unsigned = {
            key: value
            for key, value in candidate.items()
            if key != "candidate_sha256"
        }
        actual_digest = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if not digest or digest != actual_digest:
            raise ValueError(f"Pregame candidate checksum changed: {path}")
        if str(candidate.get("game_id")) != game_id:
            raise ValueError(f"Pregame candidate game mismatch: {path}")
        captured_at = datetime.fromisoformat(
            str(candidate["captured_at"]).replace("Z", "+00:00")
        )
        if captured_at < scheduled_start:
            candidates.append((captured_at, candidate))
    if not candidates:
        return None
    captured_at, selected = max(candidates, key=lambda row: row[0])
    selected_game = selected.get("game")
    if not isinstance(selected_game, dict):
        raise ValueError(f"Pregame candidate has no game payload for {game_id}")
    return {
        **copy.deepcopy(selected_game),
        "pregame_candidate_sha256": selected["candidate_sha256"],
        "pregame_captured_at": captured_at.isoformat().replace("+00:00", "Z"),
        "frozen_at": scheduled_start.isoformat().replace("+00:00", "Z"),
        "freeze_policy": "latest checksum-verified candidate captured before tipoff",
        "late_freeze": False,
    }


def _renderable_frozen_pregame(
    frozen: dict[str, object],
    *,
    served_game: dict[str, object],
    game_id: str,
) -> dict[str, object]:
    """Expose verified derived attribution without mutating the freeze.

    Older frozen candidates contain the exact prediction and minutes but were
    captured before the attribution ledger became part of the schema. The
    daily serving artifact enriches those same inputs after verifying the
    matching serving bundle. Use that verified derivation for the rendered
    pregame view only when its provenance binds it to this exact freeze.
    """

    rendered = copy.deepcopy(frozen)
    frozen_projection = frozen.get("projection")
    if not isinstance(frozen_projection, dict):
        return rendered
    if isinstance(frozen_projection.get("ledger"), dict):
        return rendered

    served_projection = served_game.get("projection")
    served_minutes = served_game.get("projected_minutes")
    if not isinstance(served_projection, dict) or not isinstance(
        served_projection.get("ledger"), dict
    ):
        return rendered
    if not isinstance(served_minutes, dict):
        return rendered
    provenance = served_projection.get("input_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Attributed projection has no provenance for {game_id}")
    expected_freeze_sha = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if provenance.get("source_frozen_pregame_sha256") != expected_freeze_sha:
        raise ValueError(
            f"Attributed projection is not bound to frozen pregame {game_id}"
        )
    frozen_margin = frozen_projection.get("home_margin")
    served_margin = served_projection.get("home_margin")
    if frozen_margin is None or served_margin is None or abs(
        float(frozen_margin) - float(served_margin)
    ) > 1e-10:
        raise ValueError(
            f"Attributed projection changes frozen pregame margin for {game_id}"
        )
    rendered["projection"] = copy.deepcopy(served_projection)
    rendered["projected_minutes"] = copy.deepcopy(served_minutes)
    return rendered


def enrich_player_game_trajectory(
    trajectory: list[dict[str, object]],
    *,
    player_id: int,
    player_name: str,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
) -> list[dict[str, object]]:
    """Attach immutable pregame, result, and box-score facts to trajectory games.

    The updater trajectory is the authoritative chronological rating record.
    This helper only decorates its team-game rows from the already archived
    schedule finals, so player pages never depend on a live provider request.
    """

    enriched = [dict(point) for point in trajectory]
    game_ids = {
        str(point["game_id"])
        for point in enriched
        if point.get("game_id") not in {None, ""}
    }
    if not game_ids:
        return enriched

    final_dir = Path(state_dir) / "final"
    if not data_io.is_dir(final_dir):
        return enriched

    summaries: dict[str, tuple[dict[str, object], Path]] = {}
    game_dates = {
        str(point["game_date"])
        for point in enriched
        if point.get("game_id") in game_ids and point.get("game_date")
    }
    for game_date in game_dates:
        latest_path = Path(state_dir) / "latest" / f"{game_date}.json"
        try:
            latest = json.loads(data_io.read_text(latest_path))
        except (OSError, json.JSONDecodeError):
            continue
        games = latest.get("games") if isinstance(latest, dict) else None
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, dict):
                continue
            game_id = str(game.get("game_id", ""))
            if game_id not in game_ids:
                continue
            path = final_dir / f"{game_id}.summary.json"
            try:
                summary = json.loads(data_io.read_text(path))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(summary, dict):
                summaries[game_id] = (summary, path)

    # Old archives may predate the daily latest-file contract. Keep a bounded
    # compatibility fallback only when direct game-date lookup missed a row.
    missing_game_ids = game_ids - set(summaries)
    for path in data_io.glob(final_dir, "*.summary.json") if missing_game_ids else ():
        try:
            summary = json.loads(data_io.read_text(path))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict):
            continue
        game_id = str(summary.get("game_id", ""))
        if game_id in missing_game_ids:
            summaries[game_id] = (summary, path)
            missing_game_ids.remove(game_id)
        if not missing_game_ids:
            break

    for point in enriched:
        game_id = str(point.get("game_id", ""))
        match = summaries.get(game_id)
        if match is None:
            continue
        summary, summary_path = match
        _decorate_player_game_point(
            point,
            summary=summary,
            summary_path=summary_path,
            player_id=player_id,
            player_name=player_name,
        )
    return enriched


def enrich_current_rotation_history(
    profile: dict[str, object],
    trajectory: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Overlay the promoted in-season minutes state onto the current history row.

    Completed seasons remain immutable.  The current season keeps its frozen
    preseason estimates as explicit reference fields while its observed totals
    and updater estimates advance with the nightly promoted serving bundle.
    """

    raw_history = profile.get("rotation_history")
    history = [dict(row) for row in raw_history if isinstance(row, dict)] if isinstance(raw_history, list) else []
    if profile.get("rating_state") != "in_season_updated":
        return history

    season = str(profile.get("rating_season", ""))
    current = next((row for row in history if str(row.get("season")) == season), None)
    if current is None:
        # Current observations exist independently of a preseason history row
        # (late signings, returners, and cold starts). Missing historical prior
        # fields stay unknown; never discard the observed current totals.
        current = {
            "season": season,
            "season_start_year": int(season.split("-")[0]),
            "player_id": profile.get("player_id"),
            "player_name": profile.get("player_name"),
            "age": profile.get("age"),
            "predicted_available_share": None,
            "predicted_minutes_per_available_game": profile.get("preseason_expected_minutes_if_available"),
            "projected_total_minutes": None,
            "is_preseason_forecast": False,
        }
        history.append(current)

    current.setdefault(
        "preseason_predicted_available_share",
        current.get("predicted_available_share"),
    )
    current.setdefault(
        "preseason_predicted_minutes_per_available_game",
        current.get("predicted_minutes_per_available_game"),
    )
    current.setdefault(
        "preseason_projected_total_minutes",
        current.get("projected_total_minutes"),
    )

    known_games = _integer_or_none(profile.get("participation_observations"))
    available_games = _integer_or_none(profile.get("minutes_role_observations"))
    current_availability = _float_or_none(profile.get("current_participation_estimate"))
    current_minutes = _float_or_none(
        profile.get("current_expected_minutes_if_available")
    )
    game_points = [
        point
        for point in trajectory
        if point.get("game_id") not in {None, ""}
    ]
    actual_minutes = [_float_or_none(point.get("actual_minutes")) for point in game_points]
    has_complete_minutes = bool(game_points) and all(
        value is not None for value in actual_minutes
    )
    actual_total = (
        sum(value for value in actual_minutes if value is not None)
        if has_complete_minutes
        else None
    )

    current.update(
        {
            "actual_available_games": available_games,
            "known_roster_games": known_games,
            "actual_availability_share": (
                available_games / known_games
                if available_games is not None
                and known_games is not None
                and known_games > 0
                else None
            ),
            "actual_minutes_per_available_game": (
                actual_total / available_games
                if actual_total is not None
                and available_games is not None
                and available_games > 0
                else None
            ),
            "actual_total_minutes": actual_total,
            "predicted_available_share": (
                current_availability
                if current_availability is not None
                else current.get("predicted_available_share")
            ),
            "predicted_minutes_per_available_game": (
                current_minutes
                if current_minutes is not None
                else current.get("predicted_minutes_per_available_game")
            ),
            "is_preseason_forecast": False,
            "is_current_season_update": True,
            "evidence_through_date": profile.get("rating_evidence_through_date"),
        }
    )
    return history


def _decorate_player_game_point(
    point: dict[str, object],
    *,
    summary: dict[str, object],
    summary_path: Path,
    player_id: int,
    player_name: str,
) -> None:
    team = str(point.get("team", ""))
    home_team = str(summary.get("home_team", ""))
    away_team = str(summary.get("away_team", ""))
    if team not in {home_team, away_team}:
        return
    is_home = team == home_team
    opponent = away_team if is_home else home_team
    final = summary.get("final")
    if isinstance(final, dict):
        team_score = _integer_or_none(
            final.get("home_score" if is_home else "away_score")
        )
        opponent_score = _integer_or_none(
            final.get("away_score" if is_home else "home_score")
        )
        point.update(
            {
                "home_away": "home" if is_home else "away",
                "opponent": opponent,
                "team_score": team_score,
                "opponent_score": opponent_score,
                "result": (
                    "W"
                    if team_score is not None
                    and opponent_score is not None
                    and team_score > opponent_score
                    else "L"
                    if team_score is not None
                    and opponent_score is not None
                    and team_score < opponent_score
                    else None
                ),
            }
        )

    pregame = summary.get("pregame")
    projected_player = _find_archived_player(
        pregame.get("projected_minutes") if isinstance(pregame, dict) else None,
        player_id=player_id,
    )
    if projected_player is not None:
        point["projected_minutes"] = _float_or_none(
            projected_player.get("projected_minutes")
        )
        point["pregame_status"] = (
            str(projected_player["status"])
            if projected_player.get("status") not in {None, ""}
            else point.get("pregame_status")
        )
        point["pregame_reason"] = (
            str(projected_player["reason"])
            if projected_player.get("reason") not in {None, ""}
            else point.get("pregame_reason")
        )

    retrospective = summary.get("retrospective")
    actual_player = _find_archived_player(
        retrospective.get("actual_minutes")
        if isinstance(retrospective, dict)
        else None,
        player_id=player_id,
    )
    if actual_player is not None:
        point["actual_minutes"] = _float_or_none(actual_player.get("actual_minutes"))
    elif point.get("played") is False:
        point["actual_minutes"] = 0.0

    game_id = str(summary.get("game_id", summary_path.name.split(".", 1)[0]))
    boxscore_path = summary_path.with_name(f"{game_id}.nba.json")
    if not data_io.is_file(boxscore_path):
        return
    try:
        nba_boxscore = json.loads(data_io.read_text(boxscore_path))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(nba_boxscore, dict):
        return
    box_score = _player_box_score(nba_boxscore, player_id=player_id)
    if box_score is not None:
        point["box_score"] = box_score


def _find_archived_player(
    player_groups: object,
    *,
    player_id: int,
) -> dict[str, object] | None:
    if not isinstance(player_groups, dict):
        return None
    for side in ("away", "home"):
        rows = player_groups.get(side)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                archived_player_id = int(row.get("player_id"))
            except (TypeError, ValueError):
                continue
            if archived_player_id == player_id:
                return row
    return None


def _player_box_score(
    nba_boxscore: dict[str, object],
    *,
    player_id: int,
) -> dict[str, object] | None:
    game = nba_boxscore.get("game")
    if not isinstance(game, dict):
        return None
    for side in ("homeTeam", "awayTeam"):
        team = game.get(side)
        players = team.get("players") if isinstance(team, dict) else None
        if not isinstance(players, list):
            continue
        for player in players:
            if (
                not isinstance(player, dict)
                or _integer_or_none(player.get("personId")) != player_id
            ):
                continue
            statistics = player.get("statistics")
            if not isinstance(statistics, dict):
                return None
            field_goals_made = _integer_or_none(statistics.get("fieldGoalsMade"))
            field_goals_attempted = _integer_or_none(
                statistics.get("fieldGoalsAttempted")
            )
            threes_made = _integer_or_none(statistics.get("threePointersMade"))
            threes_attempted = _integer_or_none(
                statistics.get("threePointersAttempted")
            )
            return {
                "points": _integer_or_none(statistics.get("points")),
                "rebounds": _integer_or_none(statistics.get("reboundsTotal")),
                "assists": _integer_or_none(statistics.get("assists")),
                "turnovers": _integer_or_none(statistics.get("turnovers")),
                "steals": _integer_or_none(statistics.get("steals")),
                "blocks": _integer_or_none(statistics.get("blocks")),
                "two_points_made": (
                    field_goals_made - threes_made
                    if field_goals_made is not None and threes_made is not None
                    else None
                ),
                "two_points_attempted": (
                    field_goals_attempted - threes_attempted
                    if field_goals_attempted is not None and threes_attempted is not None
                    else None
                ),
                "three_points_made": threes_made,
                "three_points_attempted": threes_attempted,
                "free_throws_made": _integer_or_none(statistics.get("freeThrowsMade")),
                "free_throws_attempted": _integer_or_none(
                    statistics.get("freeThrowsAttempted")
                ),
                "plus_minus": _signed_integer_or_none(
                    statistics.get("plusMinusPoints")
                ),
            }
    return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _made_attempted(value: object) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    parts = str(value).strip().split("-", maxsplit=1)
    if len(parts) != 2:
        return None, None
    return _integer_or_none(parts[0]), _integer_or_none(parts[1])


def _signed_integer_or_none(value: object) -> int | None:
    """Accept NBA integral floats without truncating malformed fractional stats."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _schedule_final_audits(
    *,
    state_dir: Path | str,
    live_score_state_dir: Path | str | None,
    season: str,
) -> list[dict[str, object]]:
    """Join one canonical final observation to each environment forecast."""

    state_root = Path(state_dir)
    from nba_lineup_model.season.serving_selection import INDEX_FILENAME, read_selected_day
    if data_io.is_file(state_root / INDEX_FILENAME):
        index = json.loads(data_io.read_text(state_root / INDEX_FILENAME))
        return sorted([
            game["final_audit"]
            for game_date in sorted(index["dates"])
            for game in read_selected_day(game_date, root=state_root)["games"]
            if isinstance(game.get("final_audit"), dict)
            and game["final_audit"].get("season") == season
        ], key=lambda audit: str(audit["game_id"]))
    observation_root = (
        Path(live_score_state_dir)
        if live_score_state_dir is not None
        else state_root
    )
    audits: dict[str, dict[str, object]] = {}
    state_final_dir = state_root / "final"
    if data_io.is_dir(state_final_dir):
        for path in sorted(data_io.glob(state_final_dir, "*.summary.json")):
            audit = json.loads(data_io.read_text(path))
            if not isinstance(audit, dict) or audit.get("season") != season:
                continue
            game_id = str(audit.get("game_id", ""))
            if game_id.startswith("002"):
                audits[game_id] = audit

    if observation_root != state_root:
        missing_by_date: dict[str, set[str]] = {}
        observation_final_dir = observation_root / "final"
        if data_io.is_dir(observation_final_dir):
            for path in sorted(data_io.glob(observation_final_dir, "*.summary.json")):
                observed_audit = json.loads(data_io.read_text(path))
                if (
                    not isinstance(observed_audit, dict)
                    or observed_audit.get("season") != season
                ):
                    continue
                game_id = str(observed_audit.get("game_id", ""))
                if not game_id.startswith("002") or game_id in audits:
                    continue
                game_date = str(observed_audit.get("game_date", ""))
                if not game_date:
                    raise ValueError(
                        f"Canonical final audit {game_id} has no game date"
                    )
                missing_by_date.setdefault(game_date, set()).add(game_id)

        for game_date, game_ids in sorted(missing_by_date.items()):
            board = read_schedule_day(
                game_date,
                state_dir=state_root,
                live_score_state_dir=observation_root,
            )
            games = board.get("games")
            if not isinstance(games, list):
                raise ValueError(f"Invalid joined schedule board for {game_date}")
            joined = {
                str(game.get("game_id")): game.get("final_audit")
                for game in games
                if isinstance(game, dict) and game.get("game_id")
            }
            for game_id in sorted(game_ids):
                audit = joined.get(game_id)
                if not isinstance(audit, dict):
                    raise ValueError(
                        f"Canonical final {game_id} has no joined forecast audit"
                    )
                audits[game_id] = audit

    return [audits[game_id] for game_id in sorted(audits)]


def read_current_season_records(
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    live_score_state_dir: Path | str | None = None,
    season: str = PROJECTION_SEASON,
    through_date: str | None = None,
) -> dict[str, object]:
    """Aggregate current regular-season records from immutable final audits."""

    records: dict[str, dict[str, object]] = {}
    completed_game_ids: list[str] = []
    latest_game_date: str | None = None
    latest_archive_time: str | None = None
    final_dir = Path(live_score_state_dir or state_dir) / "final"
    from nba_lineup_model.season.serving_selection import INDEX_FILENAME, selected_result_records
    if data_io.is_dir(final_dir) or data_io.is_file(Path(state_dir) / INDEX_FILENAME):
        payloads = (
            selected_result_records(Path(state_dir), season=season)
            if data_io.is_file(Path(state_dir) / INDEX_FILENAME)
            else [json.loads(data_io.read_text(path)) for path in sorted(data_io.glob(final_dir, "*.summary.json"))]
        )
        for payload in payloads:
            if not isinstance(payload, dict) or payload.get("season") != season:
                continue
            game_id = str(payload.get("game_id", ""))
            if not game_id.startswith("002"):
                continue
            final = payload.get("final")
            if not isinstance(final, dict):
                continue
            status = final.get("status")
            if isinstance(status, dict) and status.get("completed") is not True:
                continue
            home_score = _integer_or_none(final.get("home_score"))
            away_score = _integer_or_none(final.get("away_score"))
            if home_score is None or away_score is None or home_score == away_score:
                continue
            home_team = str(payload.get("home_team", ""))
            away_team = str(payload.get("away_team", ""))
            if not home_team or not away_team or home_team == away_team:
                continue
            start_time = str(payload.get("start_time", ""))
            try:
                game_date = pacific_date(start_time).isoformat()
            except ValueError:
                continue
            if through_date is not None and game_date > through_date:
                continue
            completed_game_ids.append(game_id)
            winner = home_team if home_score > away_score else away_team
            for team in (away_team, home_team):
                record = records.setdefault(
                    team,
                    {
                        "team": team,
                        "games": 0,
                        "wins": 0,
                        "losses": 0,
                        "last_game_date": game_date,
                    },
                )
                record["games"] = int(record["games"]) + 1
                if team == winner:
                    record["wins"] = int(record["wins"]) + 1
                else:
                    record["losses"] = int(record["losses"]) + 1
                record["last_game_date"] = max(str(record["last_game_date"]), game_date)
            latest_game_date = max(latest_game_date or game_date, game_date)
            archived_at = payload.get("archived_at")
            if isinstance(archived_at, str):
                latest_archive_time = max(latest_archive_time or archived_at, archived_at)
    return {
        "season": season,
        "as_of": latest_game_date,
        "through_date": through_date,
        "updated_at": latest_archive_time,
        "source": "Immutable completed-game archive (NBA official liveData)",
        "completed_game_ids": sorted(set(completed_game_ids)),
        "teams": sorted(records.values(), key=lambda item: str(item["team"])),
    }


def read_schedule_performance(
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    live_score_state_dir: Path | str | None = None,
    season: str = PROJECTION_SEASON,
) -> dict[str, object]:
    """Score frozen pregame forecasts against final regular-season results."""

    manifest_path = Path(state_dir) / "manifest.json"
    manifest = (
        json.loads(data_io.read_text(manifest_path)) if data_io.is_file(manifest_path) else {}
    )
    if not isinstance(manifest, dict):
        manifest = {}
    model_track = manifest.get("model_track")
    declared_model_track = str(model_track or "").strip()
    completed: list[tuple[str, dict[str, object]]] = []
    latest_archive_time: str | None = None
    for payload in _schedule_final_audits(
        state_dir=state_dir,
        live_score_state_dir=live_score_state_dir,
        season=season,
    ):
        # Explicit selected references supersede historical display-label filters.
        if declared_model_track and not data_io.is_file(Path(state_dir) / "forecast-selections.json"):
            pregame = payload.get("pregame")
            pregame = pregame if isinstance(pregame, dict) else {}
            forecast_version = pregame.get("forecast_version")
            forecast_version = (
                forecast_version if isinstance(forecast_version, dict) else {}
            )
            projection = pregame.get("projection")
            projection = projection if isinstance(projection, dict) else {}
            belongs_to_updated_track = (
                forecast_version.get("model_track") == declared_model_track
                or projection.get("model_track") == declared_model_track
                or (
                    declared_model_track == "updated_ratings_updated_minutes"
                    and (
                        payload.get("record_type")
                        == "strict_forward_replay_final_game_audit"
                        or (
                            projection.get("state_kind")
                            == "in_season_updated"
                            and bool(
                                projection.get("serving_bundle_sha256")
                            )
                        )
                    )
                )
            )
            if not belongs_to_updated_track:
                continue
        if not str(payload.get("game_id", "")).startswith("002"):
            continue
        final = payload.get("final")
        grading = payload.get("grading")
        if not isinstance(final, dict) or not isinstance(grading, dict):
            continue
        status = final.get("status")
        if isinstance(status, dict) and status.get("completed") is not True:
            continue
        start_time = str(payload.get("start_time", ""))
        try:
            game_date = pacific_date(start_time).isoformat()
        except ValueError:
            continue
        completed.append((game_date, payload))
        archived_at = payload.get("archived_at")
        if isinstance(archived_at, str):
            latest_archive_time = max(latest_archive_time or archived_at, archived_at)

    as_of = max((game_date for game_date, _payload in completed), default=None)
    windows: list[dict[str, object]] = []
    if as_of is not None:
        end_date = datetime.strptime(as_of, "%Y-%m-%d").date()
        for key, label, days in (
            ("last_7_days", "7 days", 7),
            ("last_30_days", "30 days", 30),
            ("season", "Season", None),
        ):
            start_date = None if days is None else (end_date - timedelta(days=days - 1))
            games = [
                payload
                for game_date, payload in completed
                if start_date is None or game_date >= start_date.isoformat()
            ]
            windows.append(
                {
                    "key": key,
                    "label": label,
                    "start_date": start_date.isoformat() if start_date is not None else None,
                    "end_date": as_of,
                    **_score_schedule_performance_games(games),
                }
            )
    return {
        "season": season,
        "as_of": as_of,
        "updated_at": latest_archive_time,
        "source": (
            "Strict-forward updated-ratings + updated-minutes replay"
            if declared_model_track == "updated_ratings_updated_minutes"
            else (
                "Causal HCA replacement candidate over immutable pregame inputs"
                if declared_model_track == "causal_hca_replacement_candidate_v2"
                else "Immutable pregame snapshots and completed-game archive"
            )
        ),
        "model_track": model_track,
        "evaluation_cell": manifest.get("evaluation_cell"),
        "replay_through_date": manifest.get("through_date"),
        "windows": windows,
    }


def _score_schedule_performance_games(
    games: list[dict[str, object]],
) -> dict[str, object]:
    model_probability_rows: list[tuple[float, float]] = []
    market_probability_rows: list[tuple[float, float]] = []
    polymarket_probability_rows: list[tuple[float, float]] = []
    retrospective_probability_rows: list[tuple[float, float]] = []
    spread_rows: list[tuple[float, float, float]] = []
    retrospective_spread_rows: list[tuple[float, float]] = []
    ats_wins = 0
    ats_losses = 0
    ats_pushes = 0
    market_source_counts = {"draftkings": 0, "market_consensus": 0}
    ats_edge_buckets = [
        {
            "key": "under_2",
            "label": "<2 pts",
            "minimum_edge": 0.0,
            "maximum_edge": 2.0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        },
        {
            "key": "2_to_under_4",
            "label": "2–<4 pts",
            "minimum_edge": 2.0,
            "maximum_edge": 4.0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        },
        {
            "key": "4_plus",
            "label": "4+ pts",
            "minimum_edge": 4.0,
            "maximum_edge": None,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
        },
    ]
    for game in games:
        final = game.get("final")
        grading = game.get("grading")
        pregame = game.get("pregame")
        if not isinstance(final, dict) or not isinstance(grading, dict):
            continue
        if not isinstance(pregame, dict):
            pregame = {}
        actual_margin = _float_or_none(final.get("actual_home_margin"))
        model_margin = _float_or_none(grading.get("projected_home_margin"))
        model_probability = _float_or_none(grading.get("projected_home_win_probability"))
        retrospective = game.get("retrospective")
        retrospective = retrospective if isinstance(retrospective, dict) else {}
        retrospective_margin = _float_or_none(retrospective.get("home_margin"))
        retrospective_probability = _float_or_none(
            retrospective.get("home_win_probability")
        )
        market_grading = grading.get("market")
        market_grading = market_grading if isinstance(market_grading, dict) else {}
        market = pregame.get("market")
        market = market if isinstance(market, dict) else {}
        market_source = _market_benchmark_source(market)
        is_market_benchmark = market_source is not None
        if market_source is not None:
            market_source_counts[market_source] += 1
        market_probability = (
            _float_or_none(market.get("home_win_probability"))
            if is_market_benchmark
            else None
        )
        if market_probability is None and is_market_benchmark:
            market_probability = _no_vig_home_probability(
                market.get("home_moneyline"), market.get("away_moneyline")
            )
        prediction_market = pregame.get("prediction_market")
        prediction_market = (
            prediction_market if isinstance(prediction_market, dict) else {}
        )
        polymarket_probability = _float_or_none(
            prediction_market.get("home_probability")
        )
        if actual_margin is not None:
            outcome = 1.0 if actual_margin > 0.0 else 0.0
            if model_probability is not None:
                model_probability_rows.append((outcome, model_probability))
            if market_probability is not None:
                market_probability_rows.append((outcome, market_probability))
            if polymarket_probability is not None:
                polymarket_probability_rows.append((outcome, polymarket_probability))
        if actual_margin is not None and retrospective_probability is not None:
            outcome = 1.0 if actual_margin > 0.0 else 0.0
            retrospective_probability_rows.append((outcome, retrospective_probability))
        market_margin = _float_or_none(market_grading.get("market_implied_home_margin"))
        if (
            is_market_benchmark
            and actual_margin is not None
            and model_margin is not None
            and market_margin is not None
        ):
            spread_rows.append((actual_margin, model_margin, market_margin))
        if actual_margin is not None and retrospective_margin is not None:
            retrospective_spread_rows.append((actual_margin, retrospective_margin))
        ats_result = (
            market_grading.get("model_ats_result") if is_market_benchmark else None
        )
        if ats_result == "win":
            ats_wins += 1
        elif ats_result == "loss":
            ats_losses += 1
        elif ats_result == "push":
            ats_pushes += 1
        if ats_result in {"win", "loss", "push"}:
            model_edge = _float_or_none(market_grading.get("model_edge_points"))
            if model_edge is None and model_margin is not None and market_margin is not None:
                model_edge = model_margin - market_margin
            if model_edge is not None:
                absolute_edge = abs(model_edge)
                bucket_index = (
                    0 if absolute_edge < 2.0 else 1 if absolute_edge < 4.0 else 2
                )
                result_key = {"win": "wins", "loss": "losses", "push": "pushes"}[
                    ats_result
                ]
                ats_edge_buckets[bucket_index][result_key] += 1

    probability = {
        "n": len(model_probability_rows),
        "gestalt": {
            "n": len(model_probability_rows),
            **_probability_metrics(model_probability_rows, 1),
        },
        "market_consensus": {
            "n": len(market_probability_rows),
            **_probability_metrics(market_probability_rows, 1),
        },
        "polymarket": {
            "n": len(polymarket_probability_rows),
            **_probability_metrics(polymarket_probability_rows, 1),
        },
        "actual_minutes": {
            "n": len(retrospective_probability_rows),
            **_probability_metrics(retrospective_probability_rows, 1),
        },
    }
    model_mae = (
        float(np.mean([abs(actual - model) for actual, model, _market in spread_rows]))
        if spread_rows
        else None
    )
    market_mae = (
        float(np.mean([abs(actual - market) for actual, _model, market in spread_rows]))
        if spread_rows
        else None
    )
    graded_ats = ats_wins + ats_losses
    for bucket in ats_edge_buckets:
        bucket_wins = int(bucket["wins"])
        bucket_losses = int(bucket["losses"])
        bucket_pushes = int(bucket["pushes"])
        bucket_graded = bucket_wins + bucket_losses
        bucket["n"] = bucket_graded + bucket_pushes
        bucket["cover_rate"] = bucket_wins / bucket_graded if bucket_graded else None
        bucket["units_at_minus_110"] = bucket_wins * (100.0 / 110.0) - bucket_losses
    return {
        "games": len(games),
        "market_sources": market_source_counts,
        "probability": probability,
        "spread": {
            "n": len(spread_rows),
            "gestalt_mae": model_mae,
            "market_consensus_mae": market_mae,
            "gestalt_advantage": (
                market_mae - model_mae
                if market_mae is not None and model_mae is not None
                else None
            ),
            "actual_minutes_n": len(retrospective_spread_rows),
            "actual_minutes_mae": (
                float(
                    np.mean(
                        [
                            abs(actual - retrospective)
                            for actual, retrospective in retrospective_spread_rows
                        ]
                    )
                )
                if retrospective_spread_rows
                else None
            ),
        },
        "ats": {
            "n": ats_wins + ats_losses + ats_pushes,
            "wins": ats_wins,
            "losses": ats_losses,
            "pushes": ats_pushes,
            "cover_rate": ats_wins / graded_ats if graded_ats else None,
            "units_at_minus_110": ats_wins * (100.0 / 110.0) - ats_losses,
            "edge_buckets": ats_edge_buckets,
        },
    }


def _market_benchmark_source(market: dict[str, object]) -> str | None:
    provider_key = str(market.get("provider_key") or "").lower().replace(" ", "")
    provider = str(market.get("provider") or "").lower().replace(" ", "")
    if provider_key == "market_consensus":
        return "market_consensus"
    if provider_key == "draftkings" or provider == "draftkings":
        return "draftkings"
    return None


def _probability_metrics(
    rows: list[tuple[float, ...]], probability_index: int
) -> dict[str, float | None]:
    if not rows:
        return {"accuracy": None, "brier": None}
    outcomes = np.asarray([row[0] for row in rows], dtype=float)
    probabilities = np.asarray([row[probability_index] for row in rows], dtype=float)
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(np.mean(predictions == (outcomes == 1.0))),
        "brier": float(np.mean(np.square(probabilities - outcomes))),
    }


def _no_vig_home_probability(home_moneyline: object, away_moneyline: object) -> float | None:
    home = _american_implied_probability(home_moneyline)
    away = _american_implied_probability(away_moneyline)
    if home is None or away is None or home + away <= 0.0:
        return None
    return home / (home + away)


def _american_implied_probability(value: object) -> float | None:
    odds = _float_or_none(value)
    if odds is None or odds == 0.0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0.0 else -odds / (-odds + 100.0)






def mirror_missing_frozen_pregame_snapshots(
    *,
    game_date: str,
    source_state_dir: Path | str,
    target_state_dir: Path | str,
) -> list[str]:
    """Mirror valid immutable pregames into a second schedule-state tree."""

    source_root = Path(source_state_dir)
    target_root = Path(target_state_dir)
    latest_path = source_root / "latest" / f"{game_date}.json"
    if not data_io.is_file(latest_path):
        return []
    latest = json.loads(data_io.read_text(latest_path))
    games = latest.get("games", []) if isinstance(latest, dict) else []
    mirrored: list[str] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", "")).strip()
        if not game_id:
            continue
        source_path = source_root / "frozen" / f"{game_id}.json"
        target_path = target_root / "frozen" / f"{game_id}.json"
        if data_io.is_file(target_path) or not data_io.is_file(source_path):
            continue
        frozen = json.loads(data_io.read_text(source_path))
        if not isinstance(frozen, dict):
            raise ValueError(f"Invalid frozen pregame record: {source_path}")
        if str(frozen.get("game_id", "")) != game_id:
            raise ValueError(f"Frozen pregame game ID mismatch: {source_path}")
        status = frozen.get("status")
        state = status.get("state") if isinstance(status, dict) else None
        if state != "pre" or bool(frozen.get("late_freeze")):
            raise ValueError(
                f"Refusing to mirror a non-pregame frozen record: {source_path}"
            )
        _write_json_exclusive(target_path, frozen)
        if data_io.is_file(target_path):
            mirrored.append(game_id)
    return mirrored


def _fetch_latest_official_injury_report(
    *, report_date: str, now: datetime
) -> tuple[dict[str, object], bytes]:
    eastern = now.astimezone(ZoneInfo("America/New_York"))
    requested_date = datetime.strptime(report_date, "%Y-%m-%d").date()
    if requested_date > eastern.date():
        raise ValueError(f"Official injury report is not available yet for {report_date}")
    if requested_date == eastern.date():
        cursor = eastern.replace(
            minute=(eastern.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
    else:
        cursor = datetime.combine(
            requested_date,
            datetime.max.time().replace(hour=23, minute=45, second=0, microsecond=0),
            tzinfo=ZoneInfo("America/New_York"),
        )
    for _ in range(96):
        if cursor.date() != requested_date:
            break
        suffix = cursor.strftime("%I_%M%p")
        url = (
            "https://ak-static.cms.nba.com/referee/injury/"
            f"Injury-Report_{report_date}_{suffix}.pdf"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nba-gestalt/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                pdf = response.read()
            report = _parse_official_injury_report(pdf)
            report.update(
                {
                    "source_url": url,
                    "published_at": cursor.isoformat(),
                }
            )
            return report, pdf
        except urllib.error.HTTPError as error:
            if error.code not in {403, 404}:
                raise
        cursor -= timedelta(minutes=15)
    raise FileNotFoundError(f"No official NBA injury report found for {report_date}")


def _parse_official_injury_report(pdf: bytes) -> dict[str, object]:
    from pypdf import PdfReader

    statuses: list[dict[str, object]] = []
    team_reports: dict[str, dict[str, object]] = {}
    game_team_reports: dict[str, dict[str, dict[str, object]]] = {}
    current_game_date: str | None = None
    current_matchup: tuple[str, str] | None = None
    current_team: str | None = None
    for page in PdfReader(io.BytesIO(pdf)).pages:
        chunks: list[tuple[float, float, str]] = []

        def visitor(
            text: str,
            _cm: object,
            text_matrix: list[float],
            _font: object,
            _size: float,
            _chunks: list[tuple[float, float, str]] = chunks,
        ) -> None:
            clean = " ".join(text.split())
            if clean:
                _chunks.append(
                    (round(float(text_matrix[5]), 1), float(text_matrix[4]), clean)
                )

        page.extract_text(visitor_text=visitor)
        rows: dict[float, list[tuple[float, str]]] = {}
        for y, x, text in chunks:
            rows.setdefault(y, []).append((x, text))
        ordered_rows = [(y, sorted(parts)) for y, parts in sorted(rows.items())]
        player_row_ys: list[float] = []
        for y, ordered_parts in ordered_rows:
            status = next(
                (
                    text
                    for x, text in ordered_parts
                    if 580.0 <= x < 666.0
                    and text in INJURY_STATUS_PLAY_PROBABILITY
                ),
                None,
            )
            raw_name = " ".join(
                text for x, text in ordered_parts if 420.0 <= x < 580.0
            )
            if status is not None and "," in raw_name:
                player_row_ys.append(y)

        reason_by_player_y: dict[float, str] = {}
        first_reason_boundary: float | None = None
        for index, player_y in enumerate(player_row_ys):
            previous_y = player_row_ys[index - 1] if index > 0 else None
            next_y = (
                player_row_ys[index + 1]
                if index + 1 < len(player_row_ys)
                else None
            )
            before = (
                (player_y - previous_y) / 2.0
                if previous_y is not None
                else (next_y - player_y) / 2.0
                if next_y is not None
                else 16.0
            )
            after = (
                (next_y - player_y) / 2.0
                if next_y is not None
                else (player_y - previous_y) / 2.0
                if previous_y is not None
                else 16.0
            )
            lower = player_y - before
            upper = player_y + after
            if first_reason_boundary is None:
                first_reason_boundary = lower
            reason_parts: list[str] = []
            for reason_y, reason_row in ordered_rows:
                if not lower <= reason_y < upper:
                    continue
                row_text = " ".join(text for _x, text in reason_row)
                upper_row_text = row_text.upper()
                if (
                    "NOT YET SUBMITTED" in upper_row_text
                    or "NO INJURIES TO REPORT" in upper_row_text
                ):
                    continue
                reason_parts.extend(
                    text
                    for x, text in reason_row
                    if x >= 666.0 and text.upper() != "REASON"
                )
            if reason_parts:
                reason_by_player_y[player_y] = " ".join(reason_parts)

        if statuses and first_reason_boundary is not None:
            leading_reason_parts: list[str] = []
            for reason_y, reason_row in ordered_rows:
                if reason_y >= first_reason_boundary:
                    break
                row_text = " ".join(text for _x, text in reason_row)
                upper_row_text = row_text.upper()
                if (
                    "NOT YET SUBMITTED" in upper_row_text
                    or "NO INJURIES TO REPORT" in upper_row_text
                ):
                    continue
                leading_reason_parts.extend(
                    text
                    for x, text in reason_row
                    if x >= 666.0 and text.upper() != "REASON"
                )
            if leading_reason_parts:
                statuses[-1]["reason"] = " ".join(
                    part
                    for part in [
                        statuses[-1].get("reason"),
                        " ".join(leading_reason_parts),
                    ]
                    if part
                )

        for _y, ordered_parts in ordered_rows:
            game_date_text = "".join(
                text for x, text in ordered_parts if 20.0 <= x < 119.0
            )
            game_date_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", game_date_text)
            if game_date_match is not None:
                current_game_date = datetime.strptime(
                    game_date_match.group(0), "%m/%d/%Y"
                ).date().isoformat()
            matchup_text = "".join(
                text for x, text in ordered_parts if 190.0 <= x < 250.0
            )
            matchup = re.search(r"([A-Z]{2,4})@([A-Z]{2,4})", matchup_text)
            if matchup is not None:
                current_matchup = (
                    NBA_TEAM_ALIASES.get(matchup.group(1), matchup.group(1)),
                    NBA_TEAM_ALIASES.get(matchup.group(2), matchup.group(2)),
                )
                current_team = None
                if current_game_date is not None:
                    game_key = f"{current_game_date}|{'@'.join(current_matchup)}"
                    scoped_reports = game_team_reports.setdefault(game_key, {})
                    for team in current_matchup:
                        # The NBA PDF omits a team's row entirely when that team
                        # has no players to list. A report is pending only when
                        # the document explicitly says NOT YET SUBMITTED.
                        scoped_reports.setdefault(
                            team,
                            {
                                "team": team,
                                "matchup": "@".join(current_matchup),
                                "game_date": current_game_date,
                                "status": "submitted",
                            },
                        )
            team_tokens = [
                text for x, text in ordered_parts if 250.0 <= x < 420.0
            ]
            has_player_or_reason = any(x >= 420.0 for x, _text in ordered_parts)
            matched_team_position: int | None = None
            if current_matchup is not None and team_tokens:
                normalized_team_tokens = re.sub(
                    r"[^a-z0-9]", "", " ".join(team_tokens).lower()
                )
                normalized_team_tokens = normalized_team_tokens.replace(
                    "losangeles", "la", 1
                )
                for position, candidate_team in enumerate(current_matchup):
                    expected_name = TEAM_DISPLAY_NAMES.get(
                        candidate_team, candidate_team
                    )
                    normalized_expected_name = re.sub(
                        r"[^a-z0-9]", "", expected_name.lower()
                    ).replace("losangeles", "la", 1)
                    if normalized_team_tokens == normalized_expected_name:
                        matched_team_position = position
                        break
            if matched_team_position is not None and has_player_or_reason:
                # A team with no listed players is omitted from the NBA PDF.
                # Match either team explicitly rather than requiring the away
                # team to appear before the home team.
                team = current_matchup[matched_team_position]
                current_team = team
                row_text = " ".join(text for _x, text in ordered_parts).upper()
                team_report = {
                    "team": team,
                    "matchup": "@".join(current_matchup),
                    "game_date": current_game_date,
                    "status": (
                        "not_yet_submitted"
                        if "NOT YET SUBMITTED" in row_text
                        else "submitted"
                    ),
                }
                # Preserve the earliest occurrence for backward-compatible
                # consumers, but retain every date/matchup for game-scoped use.
                team_reports.setdefault(team, team_report)
                if current_game_date is not None:
                    game_key = f"{current_game_date}|{'@'.join(current_matchup)}"
                    game_team_reports.setdefault(game_key, {})[team] = team_report
            status = next(
                (
                    text
                    for x, text in ordered_parts
                    if 580.0 <= x < 666.0 and text in INJURY_STATUS_PLAY_PROBABILITY
                ),
                None,
            )
            names = [text for x, text in ordered_parts if 420.0 <= x < 580.0]
            raw_name = " ".join(names)
            if status is None or not raw_name or "," not in raw_name:
                continue
            name_parts = [part.strip() for part in raw_name.split(",") if part.strip()]
            if len(name_parts) < 2:
                continue
            player_name = " ".join([name_parts[-1], *name_parts[:-1]])
            reason = reason_by_player_y.get(_y)
            player_status: dict[str, object] = {
                "player_name": player_name,
                "status": status,
                "reason": reason,
                "play_probability": INJURY_STATUS_PLAY_PROBABILITY[status],
            }
            if current_game_date is not None:
                player_status["game_date"] = current_game_date
            if current_matchup is not None:
                player_status["matchup"] = "@".join(current_matchup)
            if current_team is not None:
                player_status["team"] = current_team
            statuses.append(player_status)
    unassigned_scoped_players = [
        str(row["player_name"])
        for row in statuses
        if row.get("matchup") is not None and row.get("team") is None
    ]
    if unassigned_scoped_players:
        raise ValueError(
            "Official NBA injury report has game-scoped players without team "
            "attribution: " + ", ".join(unassigned_scoped_players)
        )
    if not statuses:
        raise ValueError("Official NBA injury report contained no player statuses")
    return {
        "players": statuses,
        "team_reports": team_reports,
        "game_team_reports": game_team_reports,
    }


def _normalized_player_name(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    ascii_name = re.sub(r"\s+(?:Jr\.?|Sr\.?|II|III|IV)$", "", ascii_name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


def _game_rotation(
    team: str,
    *,
    players: list[dict[str, object]],
    injury_report: dict[str, object],
    rotation_size: int,
    game_date: str | None = None,
    matchup: str | None = None,
) -> dict[str, object]:
    team_players = [row for row in players if str(row.get("team")) == team]
    if not team_players:
        raise ValueError(f"Minutes payload is missing the {team} roster")
    roster_by_name = {
        _normalized_player_name(row.get("player_name")): row for row in team_players
    }
    game_key = f"{game_date}|{matchup}" if game_date and matchup else None
    game_team_reports = injury_report.get("game_team_reports")
    supports_game_scope = isinstance(game_team_reports, dict)
    scoped_team_reports = (
        game_team_reports.get(game_key)
        if game_key is not None and supports_game_scope
        else None
    )
    has_game_scope = isinstance(scoped_team_reports, dict)
    official_rows = [
        row
        for row in injury_report.get("players", [])
        if isinstance(row, dict)
        and (
            not supports_game_scope
            or (
                has_game_scope
                and row.get("game_date") == game_date
                and row.get("matchup") == matchup
                and row.get("team") == team
            )
            or (
                not has_game_scope
                and row.get("team") == team
                and row.get("status") != "Available"
            )
        )
    ]
    official = {
        _normalized_player_name(row.get("player_name")): row
        for row in official_rows
    }
    if has_game_scope:
        team_report = scoped_team_reports.get(team)
    elif supports_game_scope:
        team_report = None
    else:
        team_report = injury_report.get("team_reports", {}).get(team)
    if isinstance(team_report, dict):
        report_status = str(team_report.get("status", "not_in_report"))
    else:
        report_status = "not_in_report"
    report_pending = report_status != "submitted"
    allocations: list[dict[str, object]] = []
    for player in team_players:
        status_row = official.get(_normalized_player_name(player.get("player_name")))
        if status_row:
            status = str(status_row.get("status"))
            public_play_probability: float | None = INJURY_STATUS_PLAY_PROBABILITY.get(
                status, 1.0
            )
        elif report_pending:
            status = "Report pending"
            public_play_probability = None
        else:
            status = "Available"
            public_play_probability = 1.0
        allocation_probability = (
            public_play_probability if public_play_probability is not None else 1.0
        )
        conditional_minutes = float(player.get("conditional_minutes_per_game", 0.0))
        allocations.append(
            {
                "player_id": int(player["player_id"]),
                "player_name": str(player["player_name"]),
                "status": status,
                "play_probability": public_play_probability,
                "conditional_minutes": conditional_minutes,
                "weight": allocation_probability * conditional_minutes,
                "projected_nail": float(player.get("projected_nail", 0.0)),
                "reason": status_row.get("reason") if status_row else None,
            }
        )
    selected = sorted(
        allocations,
        key=lambda row: (-float(row["weight"]), str(row["player_name"])),
    )[:rotation_size]
    total_weight = sum(float(row["weight"]) for row in selected)
    if total_weight <= 0.0:
        raise ValueError(f"Injury assumptions produced no available minutes for {team}")
    for row in allocations:
        row["projected_minutes"] = (
            240.0 * float(row["weight"]) / total_weight if row in selected else 0.0
        )
        row["expected_contribution"] = (
            float(row["projected_minutes"])
            / 48.0
            * float(row["projected_nail"])
        )
    team_strength = sum(float(row["expected_contribution"]) for row in allocations)
    assumptions = []
    for normalized_name, status_row in official.items():
        player = roster_by_name.get(normalized_name)
        if player is None:
            continue
        assumptions.append(
            {
                "player_id": int(player["player_id"]),
                "player_name": str(player["player_name"]),
                "team": team,
                "status": status_row.get("status"),
                "reason": status_row.get("reason"),
                "play_probability": status_row.get("play_probability"),
                "projected_minutes": next(
                    float(row["projected_minutes"])
                    for row in allocations
                    if row["player_id"] == int(player["player_id"])
                ),
            }
        )
    public_players = [
        {key: value for key, value in row.items() if key != "weight"}
        for row in sorted(
            allocations,
            key=lambda item: (-float(item["projected_minutes"]), str(item["player_name"])),
        )
        if float(row["projected_minutes"]) > 0.0
        or row["status"] not in {"Available", "Report pending"}
        or _normalized_player_name(row["player_name"]) in official
    ]
    return {
        "team_strength": team_strength,
        "players": public_players,
        "injury_assumptions": sorted(
            assumptions, key=lambda row: (str(row["team"]), str(row["player_name"]))
        ),
        "report_status": report_status,
    }








def _nested(mapping: object, *keys: str) -> object:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _line_value(mapping: object, side: str, moment: str) -> float | None:
    value = _nested(mapping, side, moment, "line")
    if value is None:
        return None
    try:
        return float(str(value).lower().lstrip("ou+"))
    except ValueError:
        return None


def _odds_value(mapping: object, side: str, moment: str) -> int | None:
    value = _nested(mapping, side, moment, "odds")
    try:
        return int(str(value)) if value is not None else None
    except ValueError:
        return None


def _integer_or_none(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _fetch_polymarket_moneyline_snapshot(
    schedule_payload: dict[str, object],
    *,
    game_date: str,
    collected_at: datetime,
) -> dict[str, object]:
    """Read each listed NBA moneyline and its current CLOB midpoint.

    Polymarket's game-event slugs are deterministic from the NBA team codes and
    Eastern game date. The event contains many props and alternate lines, so we
    accept only the market whose slug exactly matches the event slug. Every
    selected field and unavailable lookup is retained in the timestamped
    snapshot written by the schedule collector.
    """

    captured_at = collected_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    specifications: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    seen_games: set[str] = set()
    matchups = schedule_payload.get("games")
    if isinstance(matchups, list):
        normalized_matchups = [
            (str(row.get("away_team", "")), str(row.get("home_team", "")))
            for row in matchups
            if isinstance(row, dict)
        ]
    else:
        raise ValueError("Prediction markets require canonical NBA schedule games")
    for away_team, home_team in normalized_matchups:
        game_key = _prediction_market_game_key(away_team, home_team)
        if not home_team or not away_team or game_key in seen_games:
            continue
        seen_games.add(game_key)
        event_slug = f"nba-{away_team.lower()}-{home_team.lower()}-{game_date}"
        try:
            event = _fetch_json(f"{POLYMARKET_EVENT_URL}/{event_slug}", timeout=5)
        except (OSError, TimeoutError, ValueError) as error:
            unavailable.append(
                {
                    "game": game_key,
                    "event_slug": event_slug,
                    "reason": f"event lookup failed: {error}",
                }
            )
            continue
        market = _polymarket_moneyline_market(event, event_slug=event_slug)
        if market is None:
            unavailable.append(
                {
                    "game": game_key,
                    "event_slug": event_slug,
                    "reason": "moneyline market unavailable",
                }
            )
            continue
        outcomes = _json_string_list(market.get("outcomes"))
        asset_ids = _polymarket_asset_ids(market)
        fallback_prices = _json_float_list(market.get("outcomePrices"))
        if len(outcomes) != 2 or len(asset_ids) != 2:
            unavailable.append(
                {
                    "game": game_key,
                    "event_slug": event_slug,
                    "reason": "moneyline outcomes or asset IDs are invalid",
                }
            )
            continue
        specifications.append(
            {
                "game": game_key,
                "away_team": away_team,
                "home_team": home_team,
                "event": event,
                "event_slug": event_slug,
                "market": market,
                "outcomes": outcomes,
                "asset_ids": asset_ids,
                "fallback_prices": fallback_prices,
            }
        )

    midpoint_requests = [
        {"token_id": asset_id}
        for specification in specifications
        for asset_id in specification["asset_ids"]
    ]
    midpoint_error: str | None = None
    midpoints: dict[str, object] = {}
    if midpoint_requests:
        try:
            midpoints = _post_json(POLYMARKET_MIDPOINTS_URL, midpoint_requests)
        except (OSError, TimeoutError, ValueError) as error:
            midpoint_error = str(error)

    games: dict[str, object] = {}
    for specification in specifications:
        asset_ids = specification["asset_ids"]
        fallback_prices = specification["fallback_prices"]
        midpoint_prices = [_float_or_none(midpoints.get(str(asset_id))) for asset_id in asset_ids]
        if all(price is not None for price in midpoint_prices):
            raw_prices = [float(price) for price in midpoint_prices]
            pricing_method = "CLOB midpoint"
        elif len(fallback_prices) == 2:
            raw_prices = fallback_prices
            pricing_method = "Gamma outcome price fallback"
        else:
            unavailable.append(
                {
                    "game": specification["game"],
                    "event_slug": specification["event_slug"],
                    "reason": "midpoint and fallback prices are unavailable",
                }
            )
            continue
        raw_sum = float(sum(raw_prices))
        if raw_sum <= 0.0 or any(price < 0.0 or price > 1.0 for price in raw_prices):
            unavailable.append(
                {
                    "game": specification["game"],
                    "event_slug": specification["event_slug"],
                    "reason": "moneyline prices are outside the probability range",
                }
            )
            continue
        market = specification["market"]
        event = specification["event"]
        outcomes = specification["outcomes"]
        games[str(specification["game"])] = {
            "provider": "Polymarket",
            "event_id": str(event.get("id", "")),
            "market_id": str(market.get("id", "")),
            "event_slug": specification["event_slug"],
            "market_slug": market.get("slug"),
            "url": f"https://polymarket.com/event/{specification['event_slug']}",
            "captured_at": captured_at,
            "pricing_method": pricing_method,
            "away_outcome": outcomes[0],
            "home_outcome": outcomes[1],
            "away_asset_id": asset_ids[0],
            "home_asset_id": asset_ids[1],
            "away_probability_raw": raw_prices[0],
            "home_probability_raw": raw_prices[1],
            "raw_probability_sum": raw_sum,
            "away_probability": raw_prices[0] / raw_sum,
            "home_probability": raw_prices[1] / raw_sum,
            "volume": _float_or_none(market.get("volumeNum")),
            "liquidity": _float_or_none(market.get("liquidityNum")),
            "bid_ask_spread": _float_or_none(market.get("spread")),
        }
    return {
        "provider": "Polymarket",
        "captured_at": captured_at,
        "pricing_method": "CLOB midpoint; Gamma outcome price fallback",
        "requested_game_count": len(seen_games),
        "available_game_count": len(games),
        "midpoint_error": midpoint_error,
        "games": games,
        "unavailable": unavailable,
    }


def _prediction_market_game_key(away_team: str, home_team: str) -> str:
    return f"{away_team}-{home_team}"


def _polymarket_moneyline_market(
    event: dict[str, object], *, event_slug: str
) -> dict[str, object] | None:
    for market in event.get("markets", []):
        if not isinstance(market, dict):
            continue
        if market.get("slug") != event_slug:
            continue
        if market.get("closed") is True or market.get("active") is False:
            continue
        return market
    return None


def _polymarket_asset_ids(market: dict[str, object]) -> list[str]:
    if str(market.get("version", "v1")) == "v2":
        value = market.get("positionIds")
        if isinstance(value, list):
            return [str(item) for item in value]
    return _json_string_list(market.get("clobTokenIds"))


def _json_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _json_float_list(value: object) -> list[float]:
    return [
        parsed
        for item in _json_string_list(value)
        if (parsed := _float_or_none(item)) is not None
    ]


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_json(url: str, *, timeout: float = 20) -> dict[str, object]:
    from nba_lineup_model.season.source_policy import assert_allowed_url
    assert_allowed_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "nba-gestalt/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON endpoint did not return an object: {url}")
    return payload


def _post_json(url: str, payload: object) -> dict[str, object]:
    from nba_lineup_model.season.source_policy import assert_allowed_url
    assert_allowed_url(url)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "nba-gestalt/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"JSON endpoint did not return an object: {url}")
    return result


def _ensure_s3_injury_report(
    *,
    injury_pdf: bytes,
    state_root: Path,
    s3_uri: str,
) -> dict[str, str]:
    """Publish one content-addressed injury PDF and retain only a tiny marker."""

    digest = hashlib.sha256(injury_pdf).hexdigest()
    destination = (
        f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}"
        f"/injury-reports/sha256={digest}.pdf"
    )
    marker_path = state_root / "artifacts" / "injury-reports" / f"{digest}.json"
    artifact = {
        "sha256": digest,
        "s3_uri": destination,
    }
    if data_io.is_file(marker_path):
        existing = json.loads(data_io.read_text(marker_path))
        if existing == artifact:
            return artifact

    pending_path = state_root / "pending" / "injury-reports" / f"{digest}.pdf"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_bytes(injury_pdf)
    _aws_s3_cp_exclusive(pending_path, destination)
    _write_json(marker_path, artifact)
    pending_path.unlink()
    return artifact


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with data_io.open_file(path, "x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        return


def _aws_s3_cp(source: Path, destination: str) -> None:
    subprocess.run(["aws", "s3", "cp", str(source), destination, "--only-show-errors"], check=True)


def _aws_s3_cp_exclusive(source: Path, destination: str) -> None:
    """Create an immutable S3 key, accepting only byte-identical retries."""

    if not destination.startswith("s3://"):
        raise ValueError(f"S3 destination is invalid: {destination}")
    bucket, separator, key = destination.removeprefix("s3://").partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"S3 destination is invalid: {destination}")
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(source),
            destination,
            "--no-overwrite",
            "--only-show-errors",
        ],
        check=True,
    )
    existing = subprocess.run(
        ["aws", "s3", "cp", destination, "-", "--only-show-errors"],
        capture_output=True,
        check=True,
    ).stdout
    if existing != data_io.read_bytes(source):
        raise RuntimeError(
            f"Refusing to overwrite immutable S3 object with different bytes: {destination}"
        )






def _build_final_game_archive(
    game: dict[str, object],
    *,
    pregame: dict[str, object],
    retrospective: dict[str, object],
    archived_at: datetime | None = None,
) -> dict[str, object]:
    """Build the immutable record used to audit a pregame forecast."""

    home_team = str(game["home_team"])
    away_team = str(game["away_team"])
    home_score = _integer_or_none(game.get("home_score"))
    away_score = _integer_or_none(game.get("away_score"))
    if home_score is None or away_score is None:
        raise ValueError(f"Final NBA game {game.get('game_id')} is missing a score")
    actual_home_margin = home_score - away_score

    projection = pregame.get("projection", {})
    if not isinstance(projection, dict) or projection.get("home_margin") is None:
        raise ValueError(f"Frozen NBA game {game.get('game_id')} is missing its projection")
    projected_home_margin = float(projection["home_margin"])
    model_side = home_team if projected_home_margin > 0.0 else away_team
    model_side_correct = (
        actual_home_margin > 0.0 if model_side == home_team else actual_home_margin < 0.0
    )

    market = pregame.get("market")
    market_grading: dict[str, object] | None = None
    if isinstance(market, dict) and market.get("home_spread") is not None:
        home_spread = float(market["home_spread"])
        away_spread = (
            float(market["away_spread"])
            if market.get("away_spread") is not None
            else -home_spread
        )
        market_implied_home_margin = -home_spread
        model_edge = projected_home_margin - market_implied_home_margin
        selection = home_team if model_edge > 0.0 else away_team
        selection_spread = home_spread if selection == home_team else away_spread
        cover_margin = (
            actual_home_margin + home_spread
            if selection == home_team
            else -actual_home_margin + away_spread
        )
        outcome = (
            "push"
            if np.isclose(cover_margin, 0.0)
            else ("win" if cover_margin > 0.0 else "loss")
        )
        market_side = home_team if market_implied_home_margin > 0.0 else away_team
        market_side_correct = (
            actual_home_margin > 0.0
            if market_side == home_team
            else actual_home_margin < 0.0
        )
        market_grading = {
            "provider": market.get("provider"),
            "provider_key": market.get("provider_key"),
            "bookmaker_keys": market.get("bookmaker_keys"),
            "bookmaker_count": market.get("bookmaker_count"),
            "captured_home_spread": home_spread,
            "captured_away_spread": away_spread,
            "captured_total": market.get("total"),
            "captured_home_moneyline": market.get("home_moneyline"),
            "captured_away_moneyline": market.get("away_moneyline"),
            "captured_home_win_probability": market.get("home_win_probability"),
            "captured_away_win_probability": market.get("away_win_probability"),
            "market_implied_home_margin": market_implied_home_margin,
            "market_favorite": market_side,
            "market_favorite_won": bool(market_side_correct),
            "model_edge_points": model_edge,
            "model_ats_selection": selection,
            "model_ats_selection_spread": selection_spread,
            "model_ats_cover_margin": cover_margin,
            "model_ats_result": outcome,
        }

    frozen_at = pregame.get("frozen_at")
    return {
        "schema_version": 1,
        "record_type": "immutable_final_game_audit",
        "archived_at": (archived_at or datetime.now(UTC)).isoformat().replace("+00:00", "Z"),
        "season": PROJECTION_SEASON,
        "game_date": pacific_date(game.get("start_time")).isoformat(),
        "game_id": game.get("game_id"),
        "start_time": game.get("start_time"),
        "away_team": away_team,
        "home_team": home_team,
        "final": {
            "away_score": away_score,
            "home_score": home_score,
            "actual_home_margin": actual_home_margin,
            "winner": home_team if actual_home_margin > 0.0 else away_team,
            "status": game.get("status"),
        },
        "pregame": pregame,
        "grading": {
            "pregame_frozen_at": frozen_at,
            "model": projection.get("model"),
            "projected_home_margin": projected_home_margin,
            "model_fair_home_spread": -projected_home_margin,
            "model_fair_line_team": model_side,
            "model_fair_line": -abs(projected_home_margin),
            "projected_home_win_probability": projection.get("home_win_probability"),
            "model_favorite": model_side,
            "model_favorite_won": bool(model_side_correct),
            "margin_absolute_error": abs(actual_home_margin - projected_home_margin),
            "market": market_grading,
        },
        "retrospective": retrospective,
    }




def _parse_boxscore_minutes(value: object) -> float:
    text = str(value or "0").strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _actual_minutes_team_strength(rows: list[dict[str, object]], team: str) -> float:
    total = sum(float(row["actual_minutes"]) for row in rows)
    if total <= 0.0:
        raise ValueError(f"Final NBA box score has no actual minutes for {team}")
    return sum(
        5.0 * float(row["actual_minutes"]) / total * float(row["projected_nail"])
        for row in rows
    )


def _public_actual_minutes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in row.items() if key != "projected_nail"}
        for row in sorted(
            rows,
            key=lambda row: (-float(row["actual_minutes"]), str(row["player_name"])),
        )
    ]


def _default_live_score_date(
    *,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    now: datetime | None = None,
) -> str:
    """Prefer yesterday's unfinished slate while its live window is still open."""

    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    pacific_date_today = current_time.astimezone(PACIFIC).date()
    candidate_dates = [
        (pacific_date_today - timedelta(days=1)).isoformat(),
        pacific_date_today.isoformat(),
    ]
    for game_date in candidate_dates:
        path = Path(state_dir) / "latest" / f"{game_date}.json"
        if not data_io.is_file(path):
            continue
        payload = json.loads(data_io.read_text(path))
        if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
            continue
        games = [game for game in payload["games"] if isinstance(game, dict)]
        states = [str(game.get("status", {}).get("state", "pre")) for game in games]
        starts = [
            datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
            for game in games
            if game.get("start_time")
        ]
        if not states or not starts or all(state == "post" for state in states):
            continue
        if min(starts) - timedelta(minutes=30) <= current_time <= max(starts) + timedelta(
            hours=6
        ):
            return game_date
    return pacific_date_today.isoformat()






def main() -> int:
    """Materialize the immutable baseline win-total interval cache."""

    parser = argparse.ArgumentParser(
        description="Build the release-time NBA GESTALT win projection cache"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    evaluator = LineupEvaluator.from_latest_artifact()
    path = materialize_win_projection_cache(evaluator=evaluator, output_path=args.output)
    print(f"Materialized win projection cache: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
