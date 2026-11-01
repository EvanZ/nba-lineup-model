"""Official-NBA schedule boards and bounded live-score updates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache
from nba_lineup_model.season.game_prediction_artifact import (
    build_actual_minutes_ledger,
    build_spread_ledger,
    complete_projected_minutes_attribution,
    require_projection_reconciliation,
)
from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.web_api.the_odds_api import (
    CONSENSUS_BOOKMAKERS,
    normalize_market_consensus,
    read_latest_odds_snapshot,
)
from nba_lineup_model.web_api.win_projections import (
    DEFAULT_SCHEDULE_API_URL,
    DEFAULT_SCHEDULE_DAYS_AHEAD,
    DEFAULT_SCHEDULE_S3_URI,
    DEFAULT_SCHEDULE_STATE_DIR,
    INJURY_STATUS_PLAY_PROBABILITY,
    PROJECTION_SEASON,
    _aws_s3_cp,
    _aws_s3_cp_exclusive,
    _build_final_game_archive,
    _ensure_s3_injury_report,
    _fetch_json,
    _fetch_latest_official_injury_report,
    _fetch_polymarket_moneyline_snapshot,
    _game_rotation,
    _integer_or_none,
    _prediction_market_game_key,
    _write_json,
    _write_json_exclusive,
)

DEFAULT_GAME_CATALOG_PATH = Path("/var/lib/nba-gestalt/catalog/games.parquet")
NBA_DURATION_RE = re.compile(
    r"^PT(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?$"
)
OFFICIAL_CONSENSUS_BOOKMAKERS = tuple(
    key for key in CONSENSUS_BOOKMAKERS if key.casefold() != "espnbet"
)
PREGAME_CANDIDATE_LEAD = timedelta(minutes=45)


def refresh_official_schedule_window(
    *,
    start_date: str,
    days_ahead: int = DEFAULT_SCHEDULE_DAYS_AHEAD,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    odds_state_dir: Path | str | None = None,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    projection_url: str = DEFAULT_SCHEDULE_API_URL,
    s3_uri: str | None = DEFAULT_SCHEDULE_S3_URI,
    now: datetime | None = None,
) -> dict[str, object]:
    """Refresh rolling boards using only canonical NBA schedule identities."""

    if days_ahead < 0:
        raise ValueError("Schedule days ahead must be non-negative")
    collected_at = (now or datetime.now(UTC)).astimezone(UTC)
    first_date = date.fromisoformat(start_date)
    final_date = first_date + timedelta(days=days_ahead)
    catalog_path = Path(game_catalog_path).resolve()
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    games_by_date = _catalog_games(
        catalog_path, start_date=first_date, end_date=final_date
    )
    report_date = min(first_date, collected_at.astimezone(PACIFIC).date()).isoformat()
    injury_report, injury_pdf = _fetch_latest_official_injury_report(
        report_date=report_date,
        now=collected_at,
    )
    state_root = Path(state_dir)
    injury_artifact: dict[str, str] = {
        "sha256": hashlib.sha256(injury_pdf).hexdigest(),
    }
    if s3_uri:
        injury_artifact = _ensure_s3_injury_report(
            injury_pdf=injury_pdf,
            state_root=state_root,
            s3_uri=s3_uri,
        )
    forecast_response = _fetch_json(projection_url)
    odds_snapshot = read_latest_odds_snapshot(
        state_dir=odds_state_dir if odds_state_dir is not None else state_root
    )
    bookmaker_lines = normalize_market_consensus(
        odds_snapshot,
        bookmaker_keys=OFFICIAL_CONSENSUS_BOOKMAKERS,
    )
    scoreboard = _scoreboard_games(state_root)
    days: list[dict[str, object]] = []
    total_games = 0
    total_frozen = 0
    total_candidates = 0
    missing_candidates: list[str] = []
    for game_date, catalog_games in games_by_date.items():
        prediction_market = _fetch_polymarket_moneyline_snapshot(
            {"games": catalog_games},
            game_date=game_date,
            collected_at=collected_at,
        )
        payload = _normalize_official_board(
            games=catalog_games,
            scoreboard=scoreboard,
            forecast_response=forecast_response,
            injury_report=injury_report,
            bookmaker_lines=bookmaker_lines,
            prediction_markets=prediction_market.get("games", {}),
            game_date=game_date,
            collected_at=collected_at,
            catalog_sha256=catalog_sha,
            injury_artifact=injury_artifact,
        )
        payload["injury_report_artifact"] = injury_artifact
        publication = _publish_board(
            payload,
            state_root=state_root,
            s3_uri=s3_uri,
            collected_at=collected_at,
        )
        total_games += len(catalog_games)
        frozen = int(publication["newly_frozen_game_count"])
        candidates = int(publication["newly_published_candidate_count"])
        total_frozen += frozen
        total_candidates += candidates
        missing_candidates.extend(publication["missing_pregame_candidate_game_ids"])
        days.append(
            {
                "game_date": game_date,
                "game_count": len(catalog_games),
                "newly_frozen_game_count": frozen,
                "newly_published_candidate_count": int(
                    publication["newly_published_candidate_count"]
                ),
                "missing_pregame_candidate_game_ids": publication[
                    "missing_pregame_candidate_game_ids"
                ],
            }
        )
    return {
        "start_date": start_date,
        "days_ahead": days_ahead,
        "report_date": report_date,
        "archived_day_count": len(days),
        "game_count": total_games,
        "newly_frozen_game_count": total_frozen,
        "newly_published_candidate_count": total_candidates,
        "missing_pregame_candidate_game_ids": sorted(set(missing_candidates)),
        "days": days,
    }


def refresh_official_live_score_state(
    *,
    game_date: str,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    s3_uri: str | None = None,
) -> dict[str, object]:
    """Overlay the NBA CDN scoreboard and archive official final box scores."""

    state_root = Path(state_dir)
    latest_path = state_root / "latest" / f"{game_date}.json"
    payload = _read_object(latest_path)
    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError(f"Invalid schedule games payload: {latest_path}")
    scoreboard = _scoreboard_games(state_root)
    newly_completed: list[dict[str, object]] = []
    observed_at_datetime = datetime.now(UTC)
    observed_at = observed_at_datetime.isoformat().replace("+00:00", "Z")
    newly_frozen = 0
    missing_pregame_candidates: list[str] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", ""))
        event = scoreboard.get(game_id)
        if event is None:
            continue
        _overlay_scoreboard(game, event)
        state = str(game["status"].get("state", "pre"))
        frozen_path = state_root / "frozen" / f"{game_id}.json"
        if state != "pre" and not frozen_path.exists():
            start = datetime.fromisoformat(
                str(game["start_time"]).replace("Z", "+00:00")
            )
            if _freeze_latest_pregame_candidate(
                game_id=game_id,
                scheduled_start=start,
                state_root=state_root,
                frozen_at=observed_at_datetime,
            ):
                newly_frozen += 1
                if s3_uri:
                    prefix = (
                        f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}"
                        f"/game_date={game_date}/frozen/{game_id}.json"
                    )
                    _aws_s3_cp_exclusive(frozen_path, prefix)
            else:
                missing_pregame_candidates.append(game_id)
        if frozen_path.is_file():
            generated = copy.deepcopy(game)
            frozen = _read_object(frozen_path)
            _apply_frozen_pregame(game, frozen)
            _require_or_attach_frozen_attribution(
                game,
                frozen=frozen,
                generated=generated,
            )
        summary_path = state_root / "final" / f"{game_id}.summary.json"
        if game["status"]["completed"] and not summary_path.is_file():
            if not frozen_path.is_file():
                continue
            game["final_observed_at"] = observed_at
            newly_completed.append(game)
    uploaded = 0
    if newly_completed:
        projection_payload = _fetch_json(DEFAULT_SCHEDULE_API_URL)
        cache = RawJsonCache(state_root / "pending" / "nba-live-data")
        with NbaCdnClient(cache=cache) as client:
            for game in newly_completed:
                game_id = str(game["game_id"])
                response = client.fetch_boxscore(game_id, use_cache=False)
                boxscore = response.payload
                frozen_path = state_root / "frozen" / f"{game_id}.json"
                if not frozen_path.is_file():
                    raise FileNotFoundError(
                        f"Cannot finalize NBA game {game_id} without its frozen pregame"
                    )
                frozen = _read_object(frozen_path)
                pregame = copy.deepcopy(frozen)
                for key in ("projection", "projected_minutes"):
                    if key in game:
                        pregame[key] = copy.deepcopy(game[key])
                retrospective = _build_official_actual_minutes_retrospective(
                    game,
                    pregame=pregame,
                    boxscore=boxscore,
                    projection_payload=projection_payload,
                )
                game["retrospective"] = retrospective
                final_record = _build_final_game_archive(
                    game,
                    pregame=pregame,
                    retrospective=retrospective,
                )
                summary_path = state_root / "final" / f"{game_id}.summary.json"
                boxscore_path = state_root / "final" / f"{game_id}.nba.json"
                _write_json_exclusive(summary_path, final_record)
                _write_json_exclusive(boxscore_path, boxscore)
                if s3_uri:
                    prefix = (
                        f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}"
                        f"/game_date={game_date}/final"
                    )
                    for path in (summary_path, boxscore_path):
                        _aws_s3_cp_exclusive(path, f"{prefix}/{path.name}")
                        uploaded += 1
    payload["live_score_collected_at"] = datetime.now(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    _write_json(latest_path, payload)
    starts = [
        datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
        for game in games
        if isinstance(game, dict) and game.get("start_time")
    ]
    states = [
        str(game.get("status", {}).get("state", "pre"))
        for game in games
        if isinstance(game, dict)
    ]
    return {
        "game_count": len(states),
        "any_live": "in" in states,
        "all_final": bool(states) and all(state == "post" for state in states),
        "earliest_start": min(starts) if starts else None,
        "latest_start": max(starts) if starts else None,
        "newly_frozen_game_count": newly_frozen,
        "missing_pregame_candidate_game_ids": sorted(missing_pregame_candidates),
        "newly_finalized_game_count": len(newly_completed),
        "uploaded_final_artifact_count": uploaded,
    }


def reconcile_official_final_artifacts_to_s3(
    *,
    game_date: str,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
    s3_uri: str = DEFAULT_SCHEDULE_S3_URI,
) -> dict[str, object]:
    state_root = Path(state_dir)
    payload = _read_object(state_root / "latest" / f"{game_date}.json")
    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError("Official schedule board has no games")
    final_count = 0
    verified = 0
    missing: list[str] = []
    for game in games:
        if not isinstance(game, dict) or not game.get("status", {}).get("completed"):
            continue
        final_count += 1
        game_id = str(game["game_id"])
        for suffix in ("summary.json", "nba.json"):
            path = state_root / "final" / f"{game_id}.{suffix}"
            if not path.is_file():
                missing.append(path.name)
                continue
            prefix = (
                f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}"
                f"/game_date={game_date}/final"
            )
            _aws_s3_cp_exclusive(path, f"{prefix}/{path.name}")
            verified += 1
    if missing:
        raise FileNotFoundError("Missing official final artifacts: " + ", ".join(missing))
    return {
        "game_date": game_date,
        "final_game_count": final_count,
        "verified_artifact_count": verified,
    }


def _catalog_games(
    path: Path, *, start_date: date, end_date: date
) -> dict[str, list[dict[str, Any]]]:
    frame = pd.read_parquet(path)
    required = {
        "game_id",
        "season",
        "season_type",
        "game_time_utc",
        "home_team_tricode",
        "away_team_tricode",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Official NBA catalog is missing: " + ", ".join(missing))
    frame = frame.loc[
        frame["season"].astype(str).eq(PROJECTION_SEASON)
        & frame["season_type"].astype(str).eq("regular")
    ].copy()
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    frame["game_date_pacific"] = (
        frame["game_time_utc"].dt.tz_convert(PACIFIC).dt.date
    )
    frame = frame.loc[frame["game_date_pacific"].between(start_date, end_date)]
    output: dict[str, list[dict[str, Any]]] = {}
    for row in frame.sort_values(["game_time_utc", "game_id"]).itertuples(index=False):
        output.setdefault(row.game_date_pacific.isoformat(), []).append(
            {
                "game_id": str(row.game_id),
                "start_time": pd.Timestamp(row.game_time_utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "home_team": str(row.home_team_tricode),
                "away_team": str(row.away_team_tricode),
            }
        )
    return output


def _scoreboard_games(state_root: Path) -> dict[str, dict[str, Any]]:
    cache = RawJsonCache(state_root / "pending" / "nba-live-data")
    with NbaCdnClient(cache=cache) as client:
        response = client.fetch_todays_scoreboard(use_cache=False)
    scoreboard = response.payload.get("scoreboard")
    games = scoreboard.get("games") if isinstance(scoreboard, dict) else None
    if not isinstance(games, list):
        raise ValueError("NBA CDN scoreboard has no game list")
    return {
        str(game["gameId"]): game
        for game in games
        if isinstance(game, dict) and game.get("gameId")
    }


def _normalize_official_board(
    *,
    games: list[dict[str, Any]],
    scoreboard: dict[str, dict[str, Any]],
    forecast_response: dict[str, object],
    injury_report: dict[str, object],
    bookmaker_lines: object,
    prediction_markets: object,
    game_date: str,
    collected_at: datetime,
    catalog_sha256: str,
    injury_artifact: dict[str, str],
) -> dict[str, object]:
    forecast = forecast_response.get("win_projection", forecast_response)
    minutes = forecast_response.get("minutes", forecast_response)
    if not isinstance(forecast, dict) or not isinstance(minutes, dict):
        raise ValueError("Projection endpoint has no forecast/minutes payload")
    market_lines = bookmaker_lines if isinstance(bookmaker_lines, dict) else {}
    markets = prediction_markets if isinstance(prediction_markets, dict) else {}
    minute_players = [row for row in minutes.get("players", []) if isinstance(row, dict)]
    projection_provenance = minutes.get("projection_provenance")
    projection_provenance = (
        dict(projection_provenance)
        if isinstance(projection_provenance, dict)
        else {}
    )
    projection_provenance.update(
        {
            "injury_report_sha256": injury_artifact["sha256"],
            "injury_report_published_at": injury_report.get("published_at"),
            "injury_report_source_url": injury_report.get("source_url"),
        }
    )
    scheduled = {
        str(row.get("game_id")): row
        for row in forecast.get("scheduled_games", [])
        if isinstance(row, dict)
    }
    output: list[dict[str, object]] = []
    for source in games:
        game_id = str(source["game_id"])
        home_team = str(source["home_team"])
        away_team = str(source["away_team"])
        controls = scheduled.get(game_id, {})
        home_rotation = _game_rotation(
            home_team,
            players=minute_players,
            injury_report=injury_report,
            rotation_size=int(minutes.get("initial_rotation_size", 15)),
            game_date=game_date,
            matchup=f"{away_team}@{home_team}",
        )
        away_rotation = _game_rotation(
            away_team,
            players=minute_players,
            injury_report=injury_report,
            rotation_size=int(minutes.get("initial_rotation_size", 15)),
            game_date=game_date,
            matchup=f"{away_team}@{home_team}",
        )
        home_court_adjustment = float(forecast["home_court"])
        back_to_back_coefficient = float(forecast["back_to_back"])
        home_back_to_back = bool(float(controls.get("home_back_to_back", 0)))
        away_back_to_back = bool(float(controls.get("away_back_to_back", 0)))
        projected_minutes = {
            "away": away_rotation["players"],
            "home": home_rotation["players"],
        }
        ratings = {
            int(row["player_id"]): float(row["projected_nail"])
            for row in minute_players
            if isinstance(row, dict)
            and row.get("player_id") is not None
            and row.get("projected_nail") is not None
        }
        projected_minutes = complete_projected_minutes_attribution(
            projected_minutes,
            injury_report=injury_report,
            away_team=away_team,
            home_team=home_team,
            ratings=ratings,
            game_id=game_id,
        )
        ledger = build_spread_ledger(
            game_id=game_id,
            away_team=away_team,
            home_team=home_team,
            projected_minutes=projected_minutes,
            home_court_adjustment=home_court_adjustment,
            back_to_back_coefficient=back_to_back_coefficient,
            home_back_to_back=home_back_to_back,
            away_back_to_back=away_back_to_back,
            provenance=projection_provenance,
        )
        margin = float(ledger["home_margin"])
        game = {
            "game_id": game_id,
            "start_time": source["start_time"],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": None,
            "away_score": None,
            "status": _scheduled_status(),
            "projection": {
                "model": forecast.get("model"),
                "model_track": forecast.get("model_track"),
                "state_kind": forecast.get("state_kind", "current_season"),
                "rating_as_of": forecast.get("as_of"),
                "serving_bundle_sha256": forecast.get("serving_bundle_sha256"),
                "input_provenance": projection_provenance,
                "home_margin": margin,
                "fair_home_spread": -margin,
                "home_win_probability": float(
                    1.0
                    / (
                        1.0
                        + np.exp(-float(forecast["win_probability_scale"]) * margin)
                    )
                ),
                "ledger": ledger,
            },
            "injury_report": {
                "source": "NBA official injury report",
                "source_url": injury_report.get("source_url"),
                "published_at": injury_report.get("published_at"),
                "artifact_sha256": injury_artifact["sha256"],
                "status_probabilities": INJURY_STATUS_PLAY_PROBABILITY,
                "team_reports": {
                    away_team: {"status": away_rotation["report_status"]},
                    home_team: {"status": home_rotation["report_status"]},
                },
                "assumptions": [
                    *away_rotation["injury_assumptions"],
                    *home_rotation["injury_assumptions"],
                ],
            },
            "projected_minutes": projected_minutes,
            "market": market_lines.get(f"{game_date}|{away_team}@{home_team}"),
            "prediction_market": markets.get(
                _prediction_market_game_key(away_team, home_team)
            ),
        }
        event = scoreboard.get(game_id)
        if event is not None:
            _overlay_scoreboard(game, event)
        output.append(game)
    return {
        "available": True,
        "game_date": game_date,
        "season": PROJECTION_SEASON,
        "collected_at": collected_at.isoformat().replace("+00:00", "Z"),
        "source": "NBA official schedule and liveData",
        "source_provenance": {
            "nba_schedule_sha256": catalog_sha256,
            "provider_fallbacks": [],
        },
        "prediction_market_source": "Polymarket public Gamma and CLOB APIs",
        "projection_state": {
            "kind": forecast.get("state_kind", "current_season"),
            "rating_as_of": forecast.get("as_of"),
            "serving_bundle_sha256": forecast.get("serving_bundle_sha256"),
        },
        "injury_report": {
            "source": "NBA official injury report",
            "source_url": injury_report.get("source_url"),
            "published_at": injury_report.get("published_at"),
            "status_probabilities": INJURY_STATUS_PLAY_PROBABILITY,
            "team_reports": injury_report.get("team_reports", {}),
        },
        "games": output,
    }


def _publish_board(
    payload: dict[str, object],
    *,
    state_root: Path,
    s3_uri: str | None,
    collected_at: datetime,
) -> dict[str, object]:
    game_date = str(payload["game_date"])
    latest_path = state_root / "latest" / f"{game_date}.json"
    previous = _read_object(latest_path) if latest_path.is_file() else {}
    prior_by_id = {
        str(game.get("game_id")): game
        for game in previous.get("games", [])
        if isinstance(game, dict)
    }
    newly_frozen = 0
    newly_published_candidates: list[Path] = []
    missing_candidates: list[str] = []
    for game in payload["games"]:
        if not isinstance(game, dict):
            continue
        game_id = str(game["game_id"])
        prior = prior_by_id.get(game_id)
        if isinstance(prior, dict) and isinstance(prior.get("retrospective"), dict):
            game["retrospective"] = prior["retrospective"]
        frozen_path = state_root / "frozen" / f"{game_id}.json"
        start = datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
        state = str(game.get("status", {}).get("state", "pre"))
        if (
            not frozen_path.exists()
            and state == "pre"
            and start - PREGAME_CANDIDATE_LEAD <= collected_at < start
        ):
            candidate_path, created = _publish_pregame_candidate(
                game,
                state_root=state_root,
                captured_at=collected_at,
            )
            if created:
                newly_published_candidates.append(candidate_path)
        if not frozen_path.exists() and (state != "pre" or collected_at >= start):
            if _freeze_latest_pregame_candidate(
                game_id=game_id,
                scheduled_start=start,
                state_root=state_root,
                frozen_at=collected_at,
            ):
                newly_frozen += 1
            elif not frozen_path.exists():
                missing_candidates.append(game_id)
        if frozen_path.is_file():
            generated = copy.deepcopy(game)
            frozen = _read_object(frozen_path)
            _apply_frozen_pregame(game, frozen)
            _require_or_attach_frozen_attribution(
                game,
                frozen=frozen,
                generated=generated,
            )
    _write_json(latest_path, payload)
    if s3_uri:
        timestamp = collected_at.strftime("%Y%m%dT%H%M%SZ")
        pending = state_root / "pending" / game_date / timestamp / "nba.json"
        _write_json(pending, payload)
        prefix = f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}/game_date={game_date}"
        _aws_s3_cp_exclusive(pending, f"{prefix}/snapshots/{timestamp}.nba.json")
        _aws_s3_cp(latest_path, f"{prefix}/latest.json")
        for candidate_path in newly_published_candidates:
            game_id = candidate_path.parent.name
            _aws_s3_cp_exclusive(
                candidate_path,
                f"{prefix}/pregame-candidates/{game_id}/{candidate_path.name}",
            )
        for game in payload["games"]:
            if not isinstance(game, dict):
                continue
            game_id = str(game["game_id"])
            frozen_path = state_root / "frozen" / f"{game_id}.json"
            if frozen_path.is_file():
                _aws_s3_cp_exclusive(frozen_path, f"{prefix}/frozen/{game_id}.json")
        shutil.rmtree(pending.parent)
    return {
        "newly_published_candidate_count": len(newly_published_candidates),
        "newly_frozen_game_count": newly_frozen,
        "missing_pregame_candidate_game_ids": sorted(missing_candidates),
    }


def _publish_pregame_candidate(
    game: dict[str, object],
    *,
    state_root: Path,
    captured_at: datetime,
) -> tuple[Path, bool]:
    """Write one immutable candidate projection observed strictly before tipoff."""

    game_id = str(game["game_id"])
    start = datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
    if captured_at >= start:
        raise ValueError(f"Pregame candidate for {game_id} was not captured before tipoff")
    if str(game.get("status", {}).get("state", "pre")) != "pre":
        raise ValueError(f"Pregame candidate for {game_id} is not in pregame state")
    unsigned = {
        "schema_version": 1,
        "game_id": game_id,
        "captured_at": captured_at.isoformat().replace("+00:00", "Z"),
        "scheduled_start_time": start.isoformat().replace("+00:00", "Z"),
        "game": game,
    }
    digest = _object_sha256(unsigned)
    candidate = {**unsigned, "candidate_sha256": digest}
    timestamp = captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
    path = (
        state_root
        / "pregame-candidates"
        / game_id
        / f"{timestamp}.{digest}.json"
    )
    created = not path.exists()
    _write_json_exclusive(path, candidate)
    return path, created


def _freeze_latest_pregame_candidate(
    *,
    game_id: str,
    scheduled_start: datetime,
    state_root: Path,
    frozen_at: datetime,
) -> bool:
    """Freeze the most recent verified candidate captured strictly before tipoff."""

    frozen_path = state_root / "frozen" / f"{game_id}.json"
    if frozen_path.exists():
        return False
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    candidate_root = state_root / "pregame-candidates" / game_id
    for path in sorted(candidate_root.glob("*.json")):
        candidate = _read_object(path)
        digest = str(candidate.get("candidate_sha256", ""))
        unsigned = {
            key: value for key, value in candidate.items() if key != "candidate_sha256"
        }
        if digest != _object_sha256(unsigned):
            raise ValueError(f"Pregame candidate checksum changed: {path}")
        if str(candidate.get("game_id")) != game_id:
            raise ValueError(f"Pregame candidate game mismatch: {path}")
        captured_at = datetime.fromisoformat(
            str(candidate["captured_at"]).replace("Z", "+00:00")
        )
        if captured_at < scheduled_start:
            candidates.append((captured_at, candidate))
    if not candidates:
        return False
    captured_at, selected = max(candidates, key=lambda row: row[0])
    selected_game = selected.get("game")
    if not isinstance(selected_game, dict):
        raise ValueError(f"Pregame candidate for {game_id} has no game payload")
    frozen = {
        **selected_game,
        "pregame_candidate_sha256": selected["candidate_sha256"],
        "pregame_captured_at": captured_at.isoformat().replace("+00:00", "Z"),
        "frozen_at": frozen_at.isoformat().replace("+00:00", "Z"),
        "freeze_policy": "latest valid pre-tip candidate",
        "late_freeze": frozen_at >= scheduled_start,
    }
    _write_json_exclusive(frozen_path, frozen)
    return True


def _apply_frozen_pregame(
    game: dict[str, object], frozen: dict[str, Any]
) -> None:
    """Keep observed score state current while retaining immutable pregame inputs."""

    for key in (
        "projection",
        "injury_report",
        "projected_minutes",
        "market",
        "prediction_market",
        "pregame_candidate_sha256",
        "pregame_captured_at",
        "frozen_at",
        "freeze_policy",
        "late_freeze",
    ):
        if key in frozen:
            game[key] = frozen[key]


def _require_or_attach_frozen_attribution(
    game: dict[str, object],
    *,
    frozen: dict[str, Any],
    generated: dict[str, object],
) -> None:
    """Serve every frozen game with the exact matching-bundle attribution."""

    game_id = str(game["game_id"])
    projection = game.get("projection")
    if not isinstance(projection, dict):
        raise ValueError(f"Frozen projection is missing for {game_id}")
    if isinstance(projection.get("ledger"), dict):
        require_projection_reconciliation(projection, game_id=game_id)
        return

    generated_projection = generated.get("projection")
    if not isinstance(generated_projection, dict):
        raise ValueError(f"Generated projection is missing for {game_id}")
    frozen_bundle = str(projection.get("serving_bundle_sha256") or "")
    generated_bundle = str(
        generated_projection.get("serving_bundle_sha256") or ""
    )
    if not frozen_bundle or frozen_bundle != generated_bundle:
        raise ValueError(
            f"Cannot attribute frozen game {game_id}: serving bundles differ"
        )

    generated_minutes = generated.get("projected_minutes")
    if not isinstance(generated_minutes, dict):
        raise ValueError(f"Generated projected minutes are missing for {game_id}")
    ratings: dict[int, float] = {}
    for side in ("away", "home"):
        rows = generated_minutes.get(side)
        if not isinstance(rows, list):
            raise ValueError(f"Generated {side} rotation is missing for {game_id}")
        for row in rows:
            if not isinstance(row, dict) or row.get("player_id") is None:
                continue
            ratings[int(row["player_id"])] = float(row["projected_nail"])
    frozen_minutes = game.get("projected_minutes")
    if not isinstance(frozen_minutes, dict):
        raise ValueError(f"Frozen projected minutes are missing for {game_id}")
    enriched = complete_projected_minutes_attribution(
        frozen_minutes,
        injury_report=(
            frozen.get("injury_report")
            if isinstance(frozen.get("injury_report"), dict)
            else None
        ),
        away_team=str(game["away_team"]),
        home_team=str(game["home_team"]),
        ratings=ratings,
        game_id=game_id,
    )
    generated_ledger = generated_projection.get("ledger")
    if not isinstance(generated_ledger, dict):
        raise ValueError(f"Generated attribution is missing for {game_id}")
    back_to_back = generated_ledger.get("back_to_back")
    if not isinstance(back_to_back, dict):
        raise ValueError(f"Generated context is missing for {game_id}")
    provenance = generated_projection.get("input_provenance")
    provenance = dict(provenance) if isinstance(provenance, dict) else {}
    provenance.update(
        {
            "source_frozen_pregame_sha256": _object_sha256(frozen),
            "attribution_contract": "exact matching serving bundle",
        }
    )
    ledger = build_spread_ledger(
        game_id=game_id,
        away_team=str(game["away_team"]),
        home_team=str(game["home_team"]),
        projected_minutes=enriched,
        home_court_adjustment=float(generated_ledger["home_court_adjustment"]),
        back_to_back_coefficient=float(back_to_back["coefficient"]),
        home_back_to_back=bool(back_to_back["home_team_is_back_to_back"]),
        away_back_to_back=bool(back_to_back["away_team_is_back_to_back"]),
        provenance=provenance,
    )
    frozen_margin = float(projection["home_margin"])
    if not math.isclose(
        float(ledger["home_margin"]), frozen_margin, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(
            f"Frozen attribution does not reproduce the spread for {game_id}"
        )
    projection = copy.deepcopy(projection)
    projection["input_provenance"] = provenance
    projection["fair_home_spread"] = -frozen_margin
    projection["ledger"] = ledger
    require_projection_reconciliation(projection, game_id=game_id)
    game["projection"] = projection
    game["projected_minutes"] = enriched


def _object_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _overlay_scoreboard(game: dict[str, object], event: dict[str, Any]) -> None:
    home = event.get("homeTeam") if isinstance(event.get("homeTeam"), dict) else {}
    away = event.get("awayTeam") if isinstance(event.get("awayTeam"), dict) else {}
    if str(home.get("teamTricode", "")) != str(game["home_team"]) or str(
        away.get("teamTricode", "")
    ) != str(game["away_team"]):
        raise ValueError(f"NBA scoreboard team mismatch for {game['game_id']}")
    status_code = int(event.get("gameStatus") or 1)
    game["home_score"] = _integer_or_none(home.get("score"))
    game["away_score"] = _integer_or_none(away.get("score"))
    game["status"] = {
        "state": {1: "pre", 2: "in", 3: "post"}.get(status_code, "pre"),
        "completed": status_code == 3,
        "description": str(event.get("gameStatusText") or "Scheduled"),
        "detail": str(event.get("gameStatusText") or "Scheduled"),
        "display_clock": event.get("gameClock"),
        "period": _integer_or_none(event.get("period")),
    }


def _build_official_actual_minutes_retrospective(
    game: dict[str, object],
    *,
    pregame: dict[str, object],
    boxscore: dict[str, object],
    projection_payload: dict[str, object],
) -> dict[str, object]:
    players = [
        row for row in projection_payload.get("players", []) if isinstance(row, dict)
    ]
    ratings = {
        int(row["player_id"]): float(row.get("projected_nail", 0.0))
        for row in players
        if row.get("player_id")
    }
    projection = pregame.get("projection")
    if not isinstance(projection, dict) or projection.get("home_margin") is None:
        raise ValueError("Frozen pregame has no projected margin")
    pregame_ledger = projection.get("ledger")
    if not isinstance(pregame_ledger, dict):
        raise ValueError("Frozen pregame has no attribution ledger")
    for side in ("away", "home"):
        team_ledger = pregame_ledger.get(side)
        if not isinstance(team_ledger, dict):
            raise ValueError(f"Frozen pregame has no {side} attribution")
        for row in team_ledger.get("players", []):
            if not isinstance(row, dict) or row.get("player_id") is None:
                continue
            ratings[int(row["player_id"])] = float(row["frozen_nail"])
    actual: dict[str, list[dict[str, object]]] = {}
    nba_game = boxscore.get("game")
    if not isinstance(nba_game, dict) or str(nba_game.get("gameId")) != str(
        game["game_id"]
    ):
        raise ValueError("NBA box score does not match the finalized game")
    for side in ("homeTeam", "awayTeam"):
        team_row = nba_game.get(side)
        if not isinstance(team_row, dict):
            raise ValueError(f"NBA box score has no {side}")
        team = str(team_row["teamTricode"])
        rows: list[dict[str, object]] = []
        for player in team_row.get("players", []):
            if not isinstance(player, dict) or player.get("personId") is None:
                continue
            statistics = player.get("statistics")
            minutes = _duration_minutes(
                statistics.get("minutes") if isinstance(statistics, dict) else None
            )
            if minutes <= 0:
                continue
            player_id = int(player["personId"])
            rating = ratings.get(player_id)
            if rating is None:
                raise ValueError(
                    f"Official minutes player {player_id} has no rating-before state"
                )
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": " ".join(
                        filter(
                            None,
                            (
                                str(player.get("firstName") or "").strip(),
                                str(player.get("familyName") or "").strip(),
                            ),
                        )
                    ),
                    "actual_minutes": minutes,
                }
            )
        actual[team] = rows
    home = str(game["home_team"])
    away = str(game["away_team"])
    frozen_margin = float(projection["home_margin"])
    ledger = build_actual_minutes_ledger(
        game_id=str(game["game_id"]),
        away_team=away,
        home_team=home,
        actual_minutes={"away": actual[away], "home": actual[home]},
        ratings=ratings,
        pregame_ledger=pregame_ledger,
        provenance={
            "pregame_ledger_sha256": _object_sha256(pregame_ledger),
            "rating_contract": "the same frozen rating-before state as the pregame ledger",
            "minutes_source": "official NBA liveData box score",
        },
    )
    retrospective_margin = float(ledger["home_margin"])
    return {
        "computed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "method": "NAIL rating-before state reweighted by official NBA actual minutes",
        "home_margin": retrospective_margin,
        "home_win_probability": _rescale_probability(
            frozen_margin=frozen_margin,
            frozen_probability=projection.get("home_win_probability"),
            retrospective_margin=retrospective_margin,
        ),
        "context_offset": float(ledger["home_court_adjustment"])
        + float(ledger["back_to_back"]["adjustment"]),
        "home_team_strength": float(ledger["home"]["team_strength"]),
        "away_team_strength": float(ledger["away"]["team_strength"]),
        "ledger": ledger,
        "actual_minutes": {
            "away": copy.deepcopy(ledger["away"]["players"]),
            "home": copy.deepcopy(ledger["home"]["players"]),
        },
        "source": "NBA official liveData box score",
    }


def _duration_minutes(value: object) -> float:
    text = str(value or "")
    match = NBA_DURATION_RE.fullmatch(text)
    if match is None:
        return 0.0
    return (
        float(match.group("hours") or 0) * 60.0
        + float(match.group("minutes") or 0)
        + float(match.group("seconds") or 0) / 60.0
    )


def _rescale_probability(
    *,
    frozen_margin: float,
    frozen_probability: object,
    retrospective_margin: float,
) -> float | None:
    try:
        probability = float(frozen_probability)
    except (TypeError, ValueError):
        return None
    if not 0.0 < probability < 1.0 or abs(frozen_margin) < 1e-9:
        return probability if abs(retrospective_margin - frozen_margin) < 1e-9 else None
    beta = math.log(probability / (1.0 - probability)) / frozen_margin
    return 1.0 / (1.0 + math.exp(-beta * retrospective_margin))


def _scheduled_status() -> dict[str, object]:
    return {
        "state": "pre",
        "completed": False,
        "description": "Scheduled",
        "detail": "Scheduled",
        "display_clock": None,
        "period": 0,
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def refresh_schedule_main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh NBA GESTALT schedule boards from official NBA sources"
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--days-ahead", type=int, default=DEFAULT_SCHEDULE_DAYS_AHEAD)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_SCHEDULE_STATE_DIR)
    parser.add_argument("--odds-state-dir", type=Path)
    parser.add_argument(
        "--game-catalog-path", type=Path, default=DEFAULT_GAME_CATALOG_PATH
    )
    parser.add_argument("--projection-url", default=DEFAULT_SCHEDULE_API_URL)
    parser.add_argument("--s3-uri", default=DEFAULT_SCHEDULE_S3_URI)
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()
    result = refresh_official_schedule_window(
        start_date=args.date,
        days_ahead=args.days_ahead,
        state_dir=args.state_dir,
        odds_state_dir=args.odds_state_dir,
        game_catalog_path=args.game_catalog_path,
        projection_url=args.projection_url,
        s3_uri=None if args.no_s3 else args.s3_uri,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    refresh_schedule_main()
