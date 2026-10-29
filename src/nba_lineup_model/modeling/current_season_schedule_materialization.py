"""Materialize a dev Schedule archive from strict-forward updater evaluations.

The production Schedule archive remains the immutable record of what was shown
at the time.  This module creates a separate, content-addressed development
view whose default forecast is the verified ``updated_updated`` cell: updated
NAIL ratings plus updated conditional minutes, both evaluated using only games
available before the forecasted game.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.current_season_serving_materialization import (
    _publication_exclusions,
)
from nba_lineup_model.web_api.win_projections import (
    _build_final_game_archive,
    _game_rotation,
)

DEFAULT_SOURCE_ROOT = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_RATING_ROOT = Path("/var/lib/nba-gestalt/in-season-nail")
DEFAULT_MINUTES_ROOT = Path("/var/lib/nba-gestalt/in-season-minutes")
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/schedule")
MODEL_TRACK = "updated_ratings_updated_minutes"
EVALUATION_CELL = "updated_updated"


def materialize_current_season_schedule(
    *,
    source_root: Path | str = DEFAULT_SOURCE_ROOT,
    rating_root: Path | str = DEFAULT_RATING_ROOT,
    minutes_root: Path | str = DEFAULT_MINUTES_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    current_forecast_response: dict[str, Any] | None = None,
) -> Path:
    """Publish and promote one isolated strict-forward Schedule view."""

    source_root = Path(source_root)
    rating_root = Path(rating_root)
    minutes_root = Path(minutes_root)
    output_root = Path(output_root)
    rating_publication_path = rating_root / "latest.json"
    minutes_publication_path = minutes_root / "latest.json"
    rating_publication = _load_object(rating_publication_path)
    minutes_publication = _load_object(minutes_publication_path)
    _validate_publications(
        rating_publication,
        minutes_publication,
        rating_root=rating_root,
        minutes_root=minutes_root,
    )

    through_date = str(minutes_publication["through_date"])
    if current_forecast_response is not None:
        forecast = current_forecast_response.get("win_projection")
        if not isinstance(forecast, dict) or str(forecast.get("as_of")) != through_date:
            raise ValueError(
                "Current forecast cutoff does not match the updater publications"
            )
        _validate_forecast_schedule_contract(forecast)
    run_id = str(minutes_publication["dagster_run_id"])
    run_root = minutes_root / "runs" / f"through_date={through_date}" / run_id
    game_path = run_root / "four_cell_game_evaluation.csv"
    player_path = run_root / "player_game_evaluation.parquet"
    if not game_path.is_file() or not player_path.is_file():
        raise FileNotFoundError(f"Minutes evaluation artifacts are missing: {run_root}")
    _verify_published_artifact(game_path, minutes_publication, "four_cell_game_evaluation")
    _verify_published_artifact(player_path, minutes_publication, "player_game_evaluation")

    identity_payload = {
        "schema_version": 1,
        "model_track": MODEL_TRACK,
        "evaluation_cell": EVALUATION_CELL,
        "season": minutes_publication["season"],
        "through_date": through_date,
        "source_schedule_sha256": _schedule_source_sha256(source_root),
        "ratings_manifest_sha256": _sha256(rating_publication_path),
        "minutes_manifest_sha256": _sha256(minutes_publication_path),
        "game_evaluation_sha256": _sha256(game_path),
        "player_evaluation_sha256": _sha256(player_path),
        "current_serving_bundle_sha256": (
            _forecast_bundle_sha256(current_forecast_response)
            if current_forecast_response is not None
            else None
        ),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    releases_root = output_root / "releases"
    release_path = releases_root / identity[:16]
    if not release_path.is_dir():
        releases_root.mkdir(parents=True, exist_ok=True)
        staging = releases_root / f".{identity[:16]}.tmp-{uuid.uuid4().hex}"
        try:
            _build_release(
                staging=staging,
                source_root=source_root,
                game_path=game_path,
                player_path=player_path,
                rating_publication=rating_publication,
                rating_publication_path=rating_publication_path,
                minutes_publication=minutes_publication,
                minutes_publication_path=minutes_publication_path,
                identity=identity,
                identity_payload=identity_payload,
                current_forecast_response=current_forecast_response,
            )
            staging.replace(release_path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    _promote(release_path, output_root / "current")
    return release_path


def _build_release(
    *,
    staging: Path,
    source_root: Path,
    game_path: Path,
    player_path: Path,
    rating_publication: dict[str, Any],
    rating_publication_path: Path,
    minutes_publication: dict[str, Any],
    minutes_publication_path: Path,
    identity: str,
    identity_payload: dict[str, Any],
    current_forecast_response: dict[str, Any] | None,
) -> None:
    staging.mkdir(parents=True)
    _copy_tree_files(source_root / "latest", staging / "latest", "*.json")
    _copy_tree_files(source_root / "frozen", staging / "frozen", "*.json")
    _copy_tree_files(source_root / "final", staging / "final", "*.json")
    _copy_tree_files(source_root / "odds-api", staging / "odds-api", "*.json")

    games = pd.read_csv(game_path, dtype={"game_id": "string"})
    players = pd.read_parquet(player_path)
    _require_columns(
        games,
        {
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "updated_updated_margin",
            "updated_updated_probability",
        },
        label="game evaluation",
    )
    _require_columns(
        players,
        {
            "game_id",
            "team",
            "player_id",
            "player_name",
            "updated_conditional_minutes",
            "updated_projected_minutes",
        },
        label="player evaluation",
    )
    games["game_id"] = games["game_id"].map(_normalize_game_id)
    players["game_id"] = players["game_id"].map(_normalize_game_id)
    _validate_player_evaluation(players)
    player_groups = {
        game_id: frame.copy()
        for game_id, frame in players.groupby("game_id", sort=False)
    }
    summaries = _summary_paths_by_game_id(source_root / "final")
    replayed: list[dict[str, Any]] = []
    replayed_game_ids: set[str] = set()
    skipped: list[dict[str, str]] = []

    for row in games.to_dict(orient="records"):
        game_id = str(row["game_id"])
        source_summary_path = summaries.get(game_id)
        if source_summary_path is None:
            skipped.append({"game_id": game_id, "reason": "missing final summary"})
            continue
        source_summary = _load_object(source_summary_path)
        source_pregame = source_summary.get("pregame")
        if not isinstance(source_pregame, dict):
            skipped.append({"game_id": game_id, "reason": "missing pregame snapshot"})
            continue
        game_players = player_groups.get(game_id)
        if game_players is None or game_players.empty:
            skipped.append({"game_id": game_id, "reason": "missing player evaluation"})
            continue

        replay_pregame = _build_replay_pregame(
            source_pregame,
            source_summary=source_summary,
            game_row=row,
            players=game_players,
            rating_publication=rating_publication,
            rating_publication_path=rating_publication_path,
            minutes_publication=minutes_publication,
            minutes_publication_path=minutes_publication_path,
        )
        espn_game_id = str(source_summary["espn_game_id"])
        _write_json(staging / "frozen" / f"{espn_game_id}.json", replay_pregame)
        _write_json(
            staging / "as-published" / "frozen" / f"{espn_game_id}.json",
            source_pregame,
        )
        _write_json(
            staging / "as-published" / "final" / f"{espn_game_id}.summary.json",
            source_summary,
        )

        final_payload = source_summary.get("final")
        if not isinstance(final_payload, dict):
            skipped.append({"game_id": game_id, "reason": "missing final score"})
            continue
        game = {
            "game_id": game_id,
            "espn_game_id": espn_game_id,
            "start_time": source_summary.get("start_time"),
            "away_team": source_summary.get("away_team"),
            "home_team": source_summary.get("home_team"),
            "away_score": final_payload.get("away_score"),
            "home_score": final_payload.get("home_score"),
            "status": final_payload.get("status"),
        }
        archived_at = _parse_datetime(source_summary.get("archived_at"))
        replay_summary = _build_final_game_archive(
            game,
            pregame=replay_pregame,
            retrospective=(
                source_summary["retrospective"]
                if isinstance(source_summary.get("retrospective"), dict)
                else {}
            ),
            archived_at=archived_at,
        )
        replay_summary.update(
            {
                "record_type": "strict_forward_replay_final_game_audit",
                "replay_identity": identity,
                "original_published_summary_sha256": _sha256(source_summary_path),
            }
        )
        _write_json(
            staging / "final" / f"{espn_game_id}.summary.json", replay_summary
        )
        replayed.append(
            {
                "game_id": game_id,
                "espn_game_id": espn_game_id,
                "game_date": str(row["game_date"]),
                "away_team": str(row["away_team"]),
                "home_team": str(row["home_team"]),
                "original_pregame_sha256": _object_sha256(source_pregame),
                "replay_pregame_sha256": _sha256(
                    staging / "frozen" / f"{espn_game_id}.json"
                ),
            }
        )
        replayed_game_ids.add(game_id)

    if skipped:
        reasons = "; ".join(
            f"{row['game_id']}: {row['reason']}" for row in skipped[:10]
        )
        raise ValueError(
            f"Strict-forward Schedule replay skipped {len(skipped)} games: {reasons}"
        )

    boundary_reforecasts: list[dict[str, Any]] = []
    if current_forecast_response is not None:
        boundary_reforecasts = _reforecast_uncovered_frozen_games(
            staging=staging,
            source_root=source_root,
            replayed_game_ids=replayed_game_ids,
            forecast_response=current_forecast_response,
            through_date=str(minutes_publication["through_date"]),
            identity=identity,
        )

    manifest = {
        **identity_payload,
        "identity": identity,
        "record_type": "current_season_schedule_strict_forward_replay",
        "generated_at": datetime.now().astimezone().isoformat(),
        "rating_manifest": rating_publication,
        "minutes_manifest": minutes_publication,
        "coverage": {
            "evaluation_games": int(len(games)),
            "replayed_games": len(replayed),
            "cutoff_state_reforecast_games": len(boundary_reforecasts),
            "replayed_final_games": len(replayed)
            + sum(bool(row["completed"]) for row in boundary_reforecasts),
            "skipped_games": len(skipped),
        },
        "games": replayed,
        "cutoff_state_reforecasts": boundary_reforecasts,
        "skipped": skipped,
        "fallback_policy": (
            "Games not covered by the strict-forward evaluation retain their original "
            "published Schedule snapshot."
        ),
    }
    _write_json(staging / "manifest.json", manifest)


def _reforecast_uncovered_frozen_games(
    *,
    staging: Path,
    source_root: Path,
    replayed_game_ids: set[str],
    forecast_response: dict[str, Any],
    through_date: str,
    identity: str,
) -> list[dict[str, Any]]:
    forecast = forecast_response.get("win_projection")
    if not isinstance(forecast, dict):
        raise ValueError("Current forecast response has no win_projection object")
    minute_players = forecast_response.get("players")
    if not isinstance(minute_players, list):
        raise ValueError("Current forecast response has no player minutes")
    scheduled_games = [
        row for row in forecast.get("scheduled_games", []) if isinstance(row, dict)
    ]
    schedule_by_game_id = {
        _normalize_game_id(row.get("game_id")): row
        for row in scheduled_games
        if row.get("game_id") is not None
    }
    rotation_size = int(forecast_response.get("initial_rotation_size", 15))
    beta = float(forecast["win_probability_scale"])
    home_court = float(forecast["home_court"])
    back_to_back = float(forecast["back_to_back"])
    output: list[dict[str, Any]] = []

    for source_path in sorted((source_root / "frozen").glob("*.json")):
        source_pregame = _load_object(source_path)
        game_id = _normalize_game_id(source_pregame.get("game_id"))
        if not game_id or game_id in replayed_game_ids:
            continue
        game_date = str(source_pregame.get("game_date") or "")
        if not game_date:
            start_time = str(source_pregame.get("start_time") or "")
            if not start_time:
                continue
            from nba_lineup_model.season.slate_time import pacific_date

            game_date = pacific_date(start_time).isoformat()
        if game_date <= through_date:
            raise ValueError(
                f"Frozen game {game_id} is at or before the updater cutoff but was not evaluated"
            )
        scheduled = schedule_by_game_id.get(game_id)
        if scheduled is None:
            scheduled = next(
                (
                    row
                    for row in scheduled_games
                    if str(row.get("game_date")) == game_date
                    and str(row.get("away_team")) == str(source_pregame.get("away_team"))
                    and str(row.get("home_team")) == str(source_pregame.get("home_team"))
                ),
                None,
            )
        if scheduled is None:
            raise ValueError(f"Current forecast is missing frozen game {game_id}")
        replay = _reforecast_pregame_from_current_state(
            source_pregame,
            forecast=forecast,
            minute_players=minute_players,
            scheduled=scheduled,
            rotation_size=rotation_size,
            beta=beta,
            home_court=home_court,
            back_to_back=back_to_back,
            through_date=through_date,
        )
        espn_game_id = str(source_pregame["espn_game_id"])
        _write_json(
            staging / "as-published" / "frozen" / f"{espn_game_id}.json",
            source_pregame,
        )
        _write_json(staging / "frozen" / f"{espn_game_id}.json", replay)

        source_summary_path = source_root / "final" / f"{espn_game_id}.summary.json"
        completed = source_summary_path.is_file()
        if completed:
            source_summary = _load_object(source_summary_path)
            _write_json(
                staging / "as-published" / "final" / source_summary_path.name,
                source_summary,
            )
            final_payload = source_summary["final"]
            game = {
                "game_id": game_id,
                "espn_game_id": espn_game_id,
                "start_time": source_summary.get("start_time"),
                "away_team": source_summary.get("away_team"),
                "home_team": source_summary.get("home_team"),
                "away_score": final_payload.get("away_score"),
                "home_score": final_payload.get("home_score"),
                "status": final_payload.get("status"),
            }
            replay_summary = _build_final_game_archive(
                game,
                pregame=replay,
                retrospective=(
                    source_summary["retrospective"]
                    if isinstance(source_summary.get("retrospective"), dict)
                    else {}
                ),
                archived_at=_parse_datetime(source_summary.get("archived_at")),
            )
            replay_summary.update(
                {
                    "record_type": "strict_forward_replay_final_game_audit",
                    "replay_method": "cutoff_state_reforecast",
                    "replay_identity": identity,
                    "original_published_summary_sha256": _sha256(source_summary_path),
                }
            )
            _write_json(staging / "final" / source_summary_path.name, replay_summary)

        output.append(
            {
                "game_id": game_id,
                "espn_game_id": espn_game_id,
                "game_date": game_date,
                "away_team": str(source_pregame["away_team"]),
                "home_team": str(source_pregame["home_team"]),
                "completed": completed,
                "evidence_through_date": through_date,
                "original_pregame_sha256": _object_sha256(source_pregame),
                "replay_pregame_sha256": _sha256(
                    staging / "frozen" / f"{espn_game_id}.json"
                ),
            }
        )
    return output


def _reforecast_pregame_from_current_state(
    source_pregame: dict[str, Any],
    *,
    forecast: dict[str, Any],
    minute_players: list[dict[str, Any]],
    scheduled: dict[str, Any],
    rotation_size: int,
    beta: float,
    home_court: float,
    back_to_back: float,
    through_date: str,
) -> dict[str, Any]:
    replay = copy.deepcopy(source_pregame)
    game_date = str(scheduled["game_date"])
    away_team = str(source_pregame["away_team"])
    home_team = str(source_pregame["home_team"])
    matchup = f"{away_team}@{home_team}"
    archived_injury = source_pregame.get("injury_report")
    archived_injury = archived_injury if isinstance(archived_injury, dict) else {}
    team_reports = archived_injury.get("team_reports")
    team_reports = team_reports if isinstance(team_reports, dict) else {}
    assumptions = [
        row
        for row in archived_injury.get("assumptions", [])
        if isinstance(row, dict)
    ]
    injury_report = {
        "players": [
            {
                **row,
                "game_date": game_date,
                "matchup": matchup,
            }
            for row in assumptions
        ],
        "team_reports": team_reports,
        "game_team_reports": {f"{game_date}|{matchup}": team_reports},
    }
    away_rotation = _game_rotation(
        away_team,
        players=minute_players,
        injury_report=injury_report,
        rotation_size=rotation_size,
        game_date=game_date,
        matchup=matchup,
    )
    home_rotation = _game_rotation(
        home_team,
        players=minute_players,
        injury_report=injury_report,
        rotation_size=rotation_size,
        game_date=game_date,
        matchup=matchup,
    )
    margin = (
        float(home_rotation["team_strength"])
        - float(away_rotation["team_strength"])
        + home_court
        + back_to_back
        * (
            float(scheduled.get("home_back_to_back", 0))
            - float(scheduled.get("away_back_to_back", 0))
        )
    )
    original_projection = replay.get("projection")
    original_projection = (
        original_projection if isinstance(original_projection, dict) else {}
    )
    replay["projection"] = {
        **original_projection,
        "model": forecast.get("model"),
        "model_track": MODEL_TRACK,
        "state_kind": "in_season_updated",
        "rating_as_of": through_date,
        "serving_bundle_sha256": forecast.get("serving_bundle_sha256"),
        "home_margin": margin,
        "home_win_probability": float(1.0 / (1.0 + math.exp(-beta * margin))),
    }
    replay["projected_minutes"] = {
        "away": away_rotation["players"],
        "home": home_rotation["players"],
    }
    replay["injury_report"] = {
        **archived_injury,
        "team_reports": {
            away_team: {"status": away_rotation["report_status"]},
            home_team: {"status": home_rotation["report_status"]},
        },
        "assumptions": [
            *away_rotation["injury_assumptions"],
            *home_rotation["injury_assumptions"],
        ],
    }
    replay["forecast_version"] = {
        "model_track": MODEL_TRACK,
        "mode": "cutoff_state_reforecast",
        "evidence_through_date": through_date,
        "serving_bundle_sha256": forecast.get("serving_bundle_sha256"),
        "original_published_snapshot_sha256": _object_sha256(source_pregame),
    }
    return replay


def _build_replay_pregame(
    source_pregame: dict[str, Any],
    *,
    source_summary: dict[str, Any],
    game_row: dict[str, Any],
    players: pd.DataFrame,
    rating_publication: dict[str, Any],
    rating_publication_path: Path,
    minutes_publication: dict[str, Any],
    minutes_publication_path: Path,
) -> dict[str, Any]:
    replay = copy.deepcopy(source_pregame)
    original_projection = replay.get("projection")
    original_projection = (
        original_projection if isinstance(original_projection, dict) else {}
    )
    replay["projection"] = {
        **original_projection,
        "model": "NAIL in-season updater",
        "model_track": MODEL_TRACK,
        "state_kind": "in_season_strict_forward_replay",
        "home_margin": float(game_row["updated_updated_margin"]),
        "home_win_probability": float(game_row["updated_updated_probability"]),
        "evidence_through_date": _previous_evidence_date(
            players, str(game_row["game_date"])
        ),
    }
    replay["projected_minutes"] = _updated_minutes_groups(
        replay.get("projected_minutes"),
        players=players,
        away_team=str(source_summary["away_team"]),
        home_team=str(source_summary["home_team"]),
    )
    injury_report = replay.get("injury_report")
    if isinstance(injury_report, dict) and isinstance(
        injury_report.get("assumptions"), list
    ):
        projected_by_player = {
            int(player["player_id"]): float(player["projected_minutes"])
            for side in ("away", "home")
            for player in replay["projected_minutes"][side]
            if player.get("player_id") is not None
        }
        for assumption in injury_report["assumptions"]:
            if not isinstance(assumption, dict):
                continue
            player_id = _integer(assumption.get("player_id"))
            if player_id in projected_by_player:
                assumption["projected_minutes"] = projected_by_player[player_id]

    replay["forecast_version"] = {
        "model_track": MODEL_TRACK,
        "evaluation_cell": EVALUATION_CELL,
        "mode": "strict_forward_replay",
        "original_published_snapshot_sha256": _object_sha256(source_pregame),
        "rating_updater_publication": _publication_reference(
            rating_publication, rating_publication_path
        ),
        "strict_forward_evaluation_publication": _publication_reference(
            minutes_publication, minutes_publication_path
        ),
    }
    return replay


def _updated_minutes_groups(
    original: object,
    *,
    players: pd.DataFrame,
    away_team: str,
    home_team: str,
) -> dict[str, list[dict[str, Any]]]:
    original = original if isinstance(original, dict) else {}
    return {
        "away": _updated_team_minutes(original.get("away"), players, away_team),
        "home": _updated_team_minutes(original.get("home"), players, home_team),
    }


def _validate_player_evaluation(players: pd.DataFrame) -> None:
    """Reject updater artifacts that contain impossible individual minutes."""

    minutes = pd.to_numeric(players["updated_projected_minutes"], errors="raise")
    invalid = ~np.isfinite(minutes) | minutes.lt(-1e-9) | minutes.gt(48.0 + 1e-6)
    if not invalid.any():
        return
    rows = players.loc[
        invalid,
        ["game_id", "team", "player_id", "player_name", "updated_projected_minutes"],
    ].head(10)
    raise ValueError(
        "Player evaluation contains physically impossible projected minutes: "
        + rows.to_dict(orient="records").__repr__()
    )


def _updated_team_minutes(
    original_rows: object, players: pd.DataFrame, team: str
) -> list[dict[str, Any]]:
    originals = [copy.deepcopy(row) for row in original_rows or [] if isinstance(row, dict)]
    updates = {
        int(row.player_id): row
        for row in players.loc[players["team"].astype(str).eq(team)].itertuples()
    }
    seen: set[int] = set()
    output: list[dict[str, Any]] = []
    for original in originals:
        player_id = _integer(original.get("player_id"))
        update = updates.get(player_id)
        if update is not None:
            original["conditional_minutes"] = float(update.updated_conditional_minutes)
            original["projected_minutes"] = float(update.updated_projected_minutes)
            original["minutes_model"] = "in_season_conditional_minutes"
            seen.add(player_id)
        output.append(original)
    for player_id, update in updates.items():
        if player_id in seen or float(update.updated_projected_minutes) <= 0.0:
            continue
        output.append(
            {
                "player_id": player_id,
                "player_name": str(update.player_name),
                "status": "Available",
                "play_probability": 1.0,
                "conditional_minutes": float(update.updated_conditional_minutes),
                "projected_minutes": float(update.updated_projected_minutes),
                "reason": None,
                "minutes_model": "in_season_conditional_minutes",
            }
        )
    return sorted(
        output,
        key=lambda row: (-float(row.get("projected_minutes") or 0.0), str(row.get("player_name"))),
    )


def _publication_reference(
    publication: dict[str, Any], publication_path: Path
) -> dict[str, Any]:
    return {
        "manifest_sha256": _sha256(publication_path),
        "through_date": publication.get("through_date"),
        "dagster_run_id": publication.get("dagster_run_id"),
        "generated_at": publication.get("generated_at"),
        "artifacts": publication.get("artifacts"),
        "coverage": publication.get("coverage"),
    }


def _forecast_bundle_sha256(
    forecast_response: dict[str, Any] | None,
) -> str | None:
    if not isinstance(forecast_response, dict):
        return None
    forecast = forecast_response.get("win_projection")
    if not isinstance(forecast, dict):
        return None
    value = forecast.get("serving_bundle_sha256")
    return str(value) if value else None


def _build_current_forecast_response(
    *, source_root: Path, catalog_path: Path
) -> dict[str, Any]:
    """Build the same promoted current-state payload served by the API."""

    from nba_lineup_model.web_api.inference import MODEL_ARTIFACT, LineupEvaluator
    from nba_lineup_model.web_api.win_projections import (
        build_current_season_minutes_payload,
        build_current_season_win_projection_payload,
        read_win_projection_cache,
    )

    evaluator = LineupEvaluator.from_latest_artifact()
    cached = read_win_projection_cache(
        model_artifact=MODEL_ARTIFACT,
        run_id=evaluator.run_id,
    )
    preseason_minutes = cached.get("minutes")
    preseason_forecast = cached.get("win_projection")
    if not isinstance(preseason_minutes, dict) or not isinstance(
        preseason_forecast, dict
    ):
        raise ValueError("Published win-projection cache is incomplete")
    minutes = build_current_season_minutes_payload(
        evaluator=evaluator,
        preseason_minutes_payload=preseason_minutes,
    )
    forecast = build_current_season_win_projection_payload(
        evaluator=evaluator,
        minutes_payload=minutes,
        preseason_forecast=preseason_forecast,
        state_dir=source_root,
        catalog_path=catalog_path,
    )
    return {**minutes, "win_projection": forecast}


def _previous_evidence_date(players: pd.DataFrame, game_date: str) -> str | None:
    # Evaluation rows are emitted before this game's observation advances the
    # state.  The explicit game date plus strict-forward evaluation contract is
    # more reliable than inferring a prior date from a player's sparse games.
    return (datetime.fromisoformat(game_date).date() - timedelta(days=1)).isoformat()


def _validate_publications(
    ratings: dict[str, Any],
    minutes: dict[str, Any],
    *,
    rating_root: Path,
    minutes_root: Path,
) -> None:
    for label, publication, root, summary_key in (
        ("rating", ratings, rating_root, "excluded_contributions"),
        ("minutes", minutes, minutes_root, "excluded_games"),
    ):
        coverage = publication.get("coverage")
        if not isinstance(coverage, dict) or int(coverage.get("excluded", -1)) < 0:
            raise ValueError(f"{label.title()} publication coverage is invalid")
        _publication_exclusions(
            publication,
            root=root,
            summary_key=summary_key,
        )
    if ratings.get("season") != minutes.get("season"):
        raise ValueError("Rating and minutes publication seasons do not match")
    if ratings.get("through_date") != minutes.get("through_date"):
        raise ValueError("Rating and minutes publication dates do not match")
    if not minutes.get("dagster_run_id"):
        raise ValueError("Minutes publication has no Dagster run ID")


def _validate_forecast_schedule_contract(forecast: dict[str, Any]) -> None:
    """Fail closed before promotion if the public season calendar is truncated."""

    scheduled_game_count = _integer(forecast.get("scheduled_game_count"))
    scheduled_games = forecast.get("scheduled_games")
    if scheduled_game_count < 1:
        raise ValueError("Current forecast has no scheduled_game_count")
    if not isinstance(scheduled_games, list):
        raise ValueError("Current forecast has no scheduled_games calendar")
    if len(scheduled_games) != scheduled_game_count:
        raise ValueError(
            "Current forecast calendar is incomplete: "
            f"expected {scheduled_game_count} games, found {len(scheduled_games)}"
        )
    game_ids = [
        _normalize_game_id(row.get("game_id"))
        for row in scheduled_games
        if isinstance(row, dict)
    ]
    if len(game_ids) != scheduled_game_count or any(not game_id for game_id in game_ids):
        raise ValueError("Current forecast calendar contains invalid game rows")
    if len(set(game_ids)) != scheduled_game_count:
        raise ValueError("Current forecast calendar contains duplicate game IDs")


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, label: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label.title()} is missing columns: {', '.join(missing)}")


def _verify_published_artifact(
    path: Path, publication: dict[str, Any], key: str
) -> None:
    artifacts = publication.get("artifacts")
    uri = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not isinstance(uri, str):
        raise ValueError(f"Minutes publication is missing {key}")
    published_digest = Path(uri).stem.rsplit("-", 1)[-1]
    if published_digest != _sha256(path):
        raise ValueError(f"Local {key} does not match its publication checksum")


def _summary_paths_by_game_id(final_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in final_root.glob("*.summary.json"):
        try:
            payload = _load_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        game_id = _normalize_game_id(payload.get("game_id"))
        if game_id:
            output[game_id] = path
    return output


def _copy_tree_files(source: Path, destination: Path, pattern: str) -> None:
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob(pattern):
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def _promote(release_path: Path, current_path: Path) -> None:
    current_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = current_path.with_name(f".{current_path.name}.tmp-{uuid.uuid4().hex}")
    temporary.symlink_to(release_path.resolve(), target_is_directory=True)
    os.replace(temporary, current_path)


def _normalize_game_id(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(10) if text.isdigit() else text


def _integer(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schedule_source_sha256(root: Path) -> str:
    """Fingerprint every schedule input copied into a materialized release."""

    digest = hashlib.sha256()
    file_count = 0
    for directory, pattern in (
        ("latest", "*.json"),
        ("frozen", "*.json"),
        ("final", "*.json"),
        ("odds-api", "*.json"),
    ):
        for path in sorted((root / directory).glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256(path)))
            file_count += 1
    if file_count == 0:
        raise ValueError(f"Schedule source contains no materializable files: {root}")
    return digest.hexdigest()


def _object_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--rating-root", type=Path, default=DEFAULT_RATING_ROOT)
    parser.add_argument("--minutes-root", type=Path, default=DEFAULT_MINUTES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--catalog-path", type=Path, default=Path("data/catalog/games.parquet"))
    parser.add_argument(
        "--include-current-forecast",
        action="store_true",
        help="Reforecast frozen games immediately after the updater cutoff",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    current_forecast_response = (
        _build_current_forecast_response(
            source_root=args.source_root,
            catalog_path=args.catalog_path,
        )
        if args.include_current_forecast
        else None
    )
    path = materialize_current_season_schedule(
        source_root=args.source_root,
        rating_root=args.rating_root,
        minutes_root=args.minutes_root,
        output_root=args.output_root,
        current_forecast_response=current_forecast_response,
    )
    print(path)


if __name__ == "__main__":
    main()
