"""Official-NBA schedule boards and bounded live-score updates."""

from __future__ import annotations

import argparse
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
from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.web_api.the_odds_api import (
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


def refresh_official_schedule_window(
    *,
    start_date: str,
    days_ahead: int = DEFAULT_SCHEDULE_DAYS_AHEAD,
    state_dir: Path | str = DEFAULT_SCHEDULE_STATE_DIR,
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
    injury_artifact = (
        _ensure_s3_injury_report(
            injury_pdf=injury_pdf,
            state_root=state_root,
            s3_uri=s3_uri,
        )
        if s3_uri
        else None
    )
    forecast_response = _fetch_json(projection_url)
    odds_snapshot = read_latest_odds_snapshot(state_dir=state_root)
    bookmaker_lines = normalize_market_consensus(odds_snapshot)
    scoreboard = _scoreboard_games(state_root)
    days: list[dict[str, object]] = []
    total_games = 0
    total_frozen = 0
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
        )
        if injury_artifact is not None:
            payload["injury_report_artifact"] = injury_artifact
        frozen = _publish_board(
            payload,
            state_root=state_root,
            s3_uri=s3_uri,
            collected_at=collected_at,
        )
        total_games += len(catalog_games)
        total_frozen += frozen
        days.append(
            {
                "game_date": game_date,
                "game_count": len(catalog_games),
                "newly_frozen_game_count": frozen,
            }
        )
    return {
        "start_date": start_date,
        "days_ahead": days_ahead,
        "report_date": report_date,
        "archived_day_count": len(days),
        "game_count": total_games,
        "newly_frozen_game_count": total_frozen,
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
    observed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", ""))
        event = scoreboard.get(game_id)
        if event is None:
            continue
        was_completed = bool(
            isinstance(game.get("status"), dict) and game["status"].get("completed")
        )
        _overlay_scoreboard(game, event)
        if game["status"]["completed"] and not was_completed:
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
                pregame = _read_object(frozen_path)
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
) -> dict[str, object]:
    forecast = forecast_response.get("win_projection", forecast_response)
    minutes = forecast_response.get("minutes", forecast_response)
    if not isinstance(forecast, dict) or not isinstance(minutes, dict):
        raise ValueError("Projection endpoint has no forecast/minutes payload")
    market_lines = bookmaker_lines if isinstance(bookmaker_lines, dict) else {}
    markets = prediction_markets if isinstance(prediction_markets, dict) else {}
    minute_players = [row for row in minutes.get("players", []) if isinstance(row, dict)]
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
        margin = (
            float(home_rotation["team_strength"])
            - float(away_rotation["team_strength"])
            + float(forecast["home_court"])
            + float(forecast["back_to_back"])
            * (
                float(controls.get("home_back_to_back", 0))
                - float(controls.get("away_back_to_back", 0))
            )
        )
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
                "home_margin": margin,
                "home_win_probability": float(
                    1.0
                    / (
                        1.0
                        + np.exp(-float(forecast["win_probability_scale"]) * margin)
                    )
                ),
            },
            "injury_report": {
                "source": "NBA official injury report",
                "source_url": injury_report.get("source_url"),
                "published_at": injury_report.get("published_at"),
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
            "projected_minutes": {
                "away": away_rotation["players"],
                "home": home_rotation["players"],
            },
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
) -> int:
    game_date = str(payload["game_date"])
    latest_path = state_root / "latest" / f"{game_date}.json"
    previous = _read_object(latest_path) if latest_path.is_file() else {}
    prior_by_id = {
        str(game.get("game_id")): game
        for game in previous.get("games", [])
        if isinstance(game, dict)
    }
    newly_frozen = 0
    for game in payload["games"]:
        if not isinstance(game, dict):
            continue
        game_id = str(game["game_id"])
        prior = prior_by_id.get(game_id)
        if isinstance(prior, dict) and isinstance(prior.get("retrospective"), dict):
            game["retrospective"] = prior["retrospective"]
        frozen_path = state_root / "frozen" / f"{game_id}.json"
        if frozen_path.exists():
            continue
        start = datetime.fromisoformat(str(game["start_time"]).replace("Z", "+00:00"))
        state = str(game.get("status", {}).get("state", "pre"))
        if collected_at < start - timedelta(minutes=30) and state == "pre":
            continue
        candidate = (
            prior
            if isinstance(prior, dict)
            and prior.get("status", {}).get("state") == "pre"
            else game
        )
        frozen = {
            **candidate,
            "frozen_at": collected_at.isoformat().replace("+00:00", "Z"),
            "freeze_policy": "last half-hour pregame capture",
            "late_freeze": candidate is game and state != "pre",
        }
        _write_json_exclusive(frozen_path, frozen)
        newly_frozen += 1
    _write_json(latest_path, payload)
    if s3_uri:
        timestamp = collected_at.strftime("%Y%m%dT%H%M%SZ")
        pending = state_root / "pending" / game_date / timestamp / "nba.json"
        _write_json(pending, payload)
        prefix = f"{s3_uri.rstrip('/')}/season={PROJECTION_SEASON}/game_date={game_date}"
        _aws_s3_cp_exclusive(pending, f"{prefix}/snapshots/{timestamp}.nba.json")
        _aws_s3_cp(latest_path, f"{prefix}/latest.json")
        for game in payload["games"]:
            if not isinstance(game, dict):
                continue
            game_id = str(game["game_id"])
            frozen_path = state_root / "frozen" / f"{game_id}.json"
            if frozen_path.is_file():
                _aws_s3_cp_exclusive(frozen_path, f"{prefix}/frozen/{game_id}.json")
        shutil.rmtree(pending.parent)
    return newly_frozen


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
    ratings = {int(row["player_id"]): row for row in players if row.get("player_id")}
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
            model = ratings.get(player_id)
            if model is None:
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
                    "projected_nail": float(model.get("projected_nail", 0.0)),
                    "rating_matched": True,
                }
            )
        actual[team] = rows
    home = str(game["home_team"])
    away = str(game["away_team"])
    projected = pregame.get("projected_minutes")
    projection = pregame.get("projection")
    if not isinstance(projected, dict):
        raise ValueError("Frozen pregame has no projected minutes")
    if not isinstance(projection, dict) or projection.get("home_margin") is None:
        raise ValueError("Frozen pregame has no projected margin")
    frozen_margin = float(projection["home_margin"])
    projected_home = _projected_strength(projected.get("home"), ratings)
    projected_away = _projected_strength(projected.get("away"), ratings)
    context_offset = frozen_margin - (projected_home - projected_away)
    actual_home = _actual_strength(actual[home])
    actual_away = _actual_strength(actual[away])
    retrospective_margin = actual_home - actual_away + context_offset
    return {
        "computed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "method": "NAIL rating-before state reweighted by official NBA actual minutes",
        "home_margin": retrospective_margin,
        "home_win_probability": _rescale_probability(
            frozen_margin=frozen_margin,
            frozen_probability=projection.get("home_win_probability"),
            retrospective_margin=retrospective_margin,
        ),
        "context_offset": context_offset,
        "home_team_strength": actual_home,
        "away_team_strength": actual_away,
        "actual_minutes": {
            "away": _public_actual(actual[away]),
            "home": _public_actual(actual[home]),
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


def _actual_strength(rows: list[dict[str, object]]) -> float:
    total = sum(float(row["actual_minutes"]) for row in rows)
    if total <= 0:
        raise ValueError("Official NBA box score has no actual minutes")
    return sum(
        5.0
        * float(row["actual_minutes"])
        / total
        * float(row["projected_nail"])
        for row in rows
    )


def _projected_strength(value: object, ratings: dict[int, dict[str, object]]) -> float:
    rows = [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
    total = sum(float(row.get("projected_minutes", 0.0)) for row in rows)
    if total <= 0:
        raise ValueError("Frozen pregame has no positive projected minutes")
    strength = 0.0
    for row in rows:
        player_id = int(row["player_id"])
        model = ratings.get(player_id)
        if model is None:
            raise ValueError(f"Projected player {player_id} has no rating-before state")
        strength += (
            5.0
            * float(row.get("projected_minutes", 0.0))
            / total
            * float(model.get("projected_nail", 0.0))
        )
    return strength


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


def _public_actual(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in row.items() if key != "projected_nail"}
        for row in sorted(
            rows,
            key=lambda row: (-float(row["actual_minutes"]), str(row["player_name"])),
        )
    ]


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
        game_catalog_path=args.game_catalog_path,
        projection_url=args.projection_url,
        s3_uri=None if args.no_s3 else args.s3_uri,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    refresh_schedule_main()
