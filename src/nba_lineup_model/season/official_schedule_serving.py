"""Build schedule-serving state from the verified official-NBA daily chain.

The public Schedule page has two kinds of information:

* immutable GESTALT artifacts (pregame projections and captured markets); and
* observed basketball facts (schedule identity, scores, box scores, and minutes).

This module preserves the former while replacing the latter with the official
NBA artifacts accepted by ADR-0009.  The migration is deliberately fail-closed:
there is no presentation-feed fallback and every completed game must join by
the canonical NBA game ID.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.season.daily_data_artifact import resolve_daily_data_pointer
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.game_prediction_artifact import (
    build_actual_minutes_ledger,
    build_spread_ledger,
    complete_projected_minutes_attribution,
    require_projection_reconciliation,
)

DEFAULT_DATA_ROOT = Path("/var/lib/nba-gestalt-dev/daily-data-nba-v2")
DEFAULT_MODEL_ROOT = Path("/var/lib/nba-gestalt-dev/daily-model-nba-v1")
DEFAULT_LEGACY_SCHEDULE_ROOT = Path("/var/lib/nba-gestalt/schedule-serving/current")
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/official-schedule")
DEFAULT_GAME_CATALOG_PATH = Path("/var/lib/nba-gestalt/catalog/games.parquet")
FORBIDDEN_BOOKMAKER_KEYS = frozenset({"espnbet"})


def materialize_official_schedule_state(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    frozen_artifact_root: Path | str = DEFAULT_LEGACY_SCHEDULE_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    projection_controls: Mapping[str, object] | None = None,
) -> Path:
    """Publish one content-addressed completed-game schedule state.

    ``frozen_artifact_root`` supplies only GESTALT-owned pregame projections,
    injury assumptions, and captured market observations.  Schedule identity,
    team identity, scores, actual minutes, and player box scores always come
    from the verified daily data chain.
    """

    data_root = Path(data_root).resolve()
    model_root = Path(model_root).resolve()
    frozen_root = Path(frozen_artifact_root).resolve()
    output_root = Path(output_root).resolve()
    game_catalog_path = Path(game_catalog_path).resolve()
    controls = _projection_controls(projection_controls)
    data_chain = _artifact_chain(data_root, kind="data")
    model_chain = _artifact_chain(model_root, kind="model")
    if not data_chain or not model_chain:
        raise ValueError("Official data and model chains must both be non-empty")
    if [row[0] for row in data_chain] != [row[0] for row in model_chain]:
        raise ValueError("Official data and model chains cover different Pacific dates")

    from nba_lineup_model.season.source_policy import require_clean_artifact_sources
    require_clean_artifact_sources(frozen_root)
    frozen_by_game = _pregame_artifacts_by_nba_game_id(frozen_root)
    completed: list[dict[str, Any]] = []
    official_boxscores: dict[str, dict[str, Any]] = {}
    source_inventory: dict[str, dict[str, str]] = {}
    pregame_inventory: dict[str, str] = {}
    player_page_ratings: dict[int, float] = {}
    data_tip_sha = data_chain[-1][2]["bundle_sha256"]
    model_tip_sha = model_chain[-1][2]["bundle_sha256"]
    through_date = data_chain[-1][0]
    upcoming = _official_upcoming_games(
        game_catalog_path,
        season=str(data_chain[-1][2]["identity"]["season"]),
        after=through_date,
        pregames=frozen_by_game,
    )

    with tempfile.TemporaryDirectory(prefix="official-schedule-components-") as directory:
        _component_root = Path(directory)
        for (game_date, data_bundle, data_manifest), (
            model_date,
            model_bundle,
            model_manifest,
        ) in zip(data_chain, model_chain, strict=True):
            if model_date != game_date:
                raise ValueError("Official data/model date mismatch")
            data_identity = data_manifest["identity"]
            model_identity = model_manifest["identity"]
            if model_identity["input_data_sha256"] != data_manifest["bundle_sha256"]:
                raise ValueError(
                    f"Model artifact for {game_date} does not consume its data artifact"
                )
            final_games = _read_json_component(
                data_bundle, data_identity["components"], "final_games"
            )
            boxscores = _read_json_component(
                data_bundle, data_identity["components"], "nba_boxscores"
            )
            actual_minutes = pd.read_parquet(
                _component_path(
                    data_bundle, data_identity["components"], "actual_minutes"
                )
            )
            current_ratings = pd.read_parquet(
                _component_path(
                    model_bundle, model_identity["components"], "player_ratings"
                )
            )
            rating_trajectory = pd.read_parquet(
                _component_path(
                    model_bundle, model_identity["components"], "rating_trajectory"
                )
            )
            game_projections = pd.read_parquet(
                _component_path(
                    model_bundle, model_identity["components"], "game_projections"
                )
            )
            for row in current_ratings.itertuples(index=False):
                player_page_ratings.setdefault(
                    int(row.player_id), float(row.frozen_preseason_nail)
                )
            day_trajectory = rating_trajectory.loc[
                rating_trajectory["game_date_pacific"].astype(str).eq(
                    game_date.isoformat()
                )
            ].copy()
            projection_by_game = {
                str(row.game_id): row
                for row in game_projections.itertuples(index=False)
            }
            boxscore_by_game = {
                str(row["game_id"]): row for row in _require_list(boxscores, "boxscores")
            }
            final_rows = _require_list(final_games, "final games")
            if sorted(str(row["game_id"]) for row in final_rows) != sorted(
                data_identity["game_ids"]
            ):
                raise ValueError(f"Final-game component mismatch for {game_date}")
            final_rows.sort(key=lambda row: (str(row.get("start_time")), str(row["game_id"])))
            for final_game in final_rows:
                game_id = str(final_game["game_id"])
                pregame_entry = frozen_by_game.get(game_id)
                if pregame_entry is None:
                    raise ValueError(f"Frozen GESTALT pregame is missing for NBA game {game_id}")
                source_pregame, source_summary = pregame_entry
                game_trajectory = day_trajectory.loc[
                    day_trajectory["game_id"].astype(str).eq(game_id)
                ]
                game_ratings = dict(player_page_ratings)
                game_ratings.update(
                    game_trajectory.set_index("player_id")["rating_before"]
                    .astype(float)
                    .to_dict()
                )
                game_projection = projection_by_game.get(game_id)
                if game_projection is None:
                    raise ValueError(
                        f"Daily model game projection is missing for {game_id}"
                    )
                pregame = _attach_game_prediction_attribution(
                    source_pregame,
                    ratings=game_ratings,
                    home_court=float(controls["home_court"]),
                    back_to_back=float(controls["back_to_back"]),
                    win_probability_scale=float(controls["win_probability_scale"]),
                    home_back_to_back=bool(game_projection.home_back_to_back),
                    away_back_to_back=bool(game_projection.away_back_to_back),
                    provenance={
                        "model_artifact_sha256": str(model_manifest["bundle_sha256"]),
                        "rating_trajectory_sha256": str(
                            model_identity["components"]["rating_trajectory"]["sha256"]
                        ),
                        "frozen_projected_minutes_sha256": _object_sha256(
                            source_pregame.get("projected_minutes")
                        ),
                        "source_pregame_sha256": _object_sha256(source_pregame),
                        "rating_contract": (
                            "player-page rating_before for this game; otherwise the "
                            "latest prior player-page rating or frozen preseason baseline"
                        ),
                    },
                )
                pregame_inventory[game_id] = _object_sha256(pregame)
                boxscore_record = boxscore_by_game.get(game_id)
                if not isinstance(boxscore_record, dict):
                    raise ValueError(f"Official NBA box score is missing for {game_id}")
                boxscore = boxscore_record.get("payload")
                if not isinstance(boxscore, dict):
                    raise ValueError(f"Official NBA box score payload is invalid for {game_id}")
                actual = actual_minutes.loc[
                    actual_minutes["game_id"].astype(str).eq(game_id)
                ].copy()
                retrospective = _official_retrospective(
                    final_game=final_game,
                    pregame=pregame,
                    actual_minutes=actual,
                    ratings=game_ratings,
                )
                source_rows = data_identity["game_sources"][game_id]
                summary = _official_final_summary(
                    final_game=final_game,
                    pregame=pregame,
                    retrospective=retrospective,
                    source_rows=source_rows,
                    data_sha=str(data_manifest["bundle_sha256"]),
                    model_sha=str(model_manifest["bundle_sha256"]),
                    archived_at=source_summary.get("archived_at"),
                )
                completed.append(summary)
                official_boxscores[game_id] = boxscore
                source_inventory[game_id] = {
                    name: str(row["sha256"]) for name, row in source_rows.items()
                }
                for trajectory_row in game_trajectory.itertuples(index=False):
                    player_page_ratings[int(trajectory_row.player_id)] = float(
                        trajectory_row.rating_after
                    )
        for upcoming_game in upcoming:
            game_id = str(upcoming_game["game_id"])
            pregame = upcoming_game.get("_pregame")
            if not isinstance(pregame, dict):
                raise ValueError(f"Upcoming game {game_id} has no pregame artifact")
            pregame_inventory[game_id] = _object_sha256(pregame)

        identity = {
            "schema_version": 2,
            "artifact_kind": "official_schedule_serving",
            "prediction_attribution_contract": (
                "complete_displayed_player_universe_v2"
            ),
            "season": str(data_chain[-1][2]["identity"]["season"]),
            "through_date": through_date.isoformat(),
            "game_ids": sorted(source_inventory),
            "game_count": len(source_inventory),
            "data_tip_sha256": str(data_tip_sha),
            "model_tip_sha256": str(model_tip_sha),
            "game_catalog_sha256": hashlib.sha256(
                game_catalog_path.read_bytes()
            ).hexdigest(),
            "upcoming_game_ids": sorted(str(row["game_id"]) for row in upcoming),
            "pregame_sha256_by_game": {
                game_id: pregame_inventory[game_id] for game_id in sorted(pregame_inventory)
            },
            "source_inventory": source_inventory,
            "projection_controls": controls,
        }
        identity_sha = hashlib.sha256(_canonical_json(identity)).hexdigest()
        release_path = output_root / "releases" / f"release_sha256={identity_sha}"
        if not release_path.exists():
            release_path.parent.mkdir(parents=True, exist_ok=True)
            staging = release_path.parent / f".release.tmp-{uuid.uuid4().hex}"
            try:
                _write_release(
                    staging,
                    completed=completed,
                    official_boxscores=official_boxscores,
                    upcoming=upcoming,
                    identity=identity,
                    identity_sha=identity_sha,
                )
                staging.replace(release_path)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        _atomic_symlink(release_path, output_root / "current")
        return release_path


def _write_release(
    root: Path,
    *,
    completed: list[dict[str, Any]],
    official_boxscores: Mapping[str, Mapping[str, Any]],
    upcoming: list[dict[str, Any]],
    identity: Mapping[str, Any],
    identity_sha: str,
) -> None:
    root.mkdir(parents=True)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for summary in completed:
        game_id = str(summary["game_id"])
        pregame = summary["pregame"]
        _write_json(root / "frozen" / f"{game_id}.json", pregame)
        _write_json(root / "final" / f"{game_id}.summary.json", summary)
        _write_json(root / "final" / f"{game_id}.nba.json", official_boxscores[game_id])
        game = {
            key: copy.deepcopy(pregame.get(key))
            for key in (
                "game_id",
                "start_time",
                "home_team",
                "away_team",
                "projection",
                "injury_report",
                "projected_minutes",
                "market",
                "prediction_market",
            )
        }
        final = summary["final"]
        game.update(
            {
                "home_score": final["home_score"],
                "away_score": final["away_score"],
                "status": final["status"],
                "final_observed_at": summary["archived_at"],
                "retrospective": summary["retrospective"],
            }
        )
        by_date.setdefault(str(summary["game_date"]), []).append(game)
    for game in upcoming:
        game_id = str(game["game_id"])
        pregame = game.pop("_pregame")
        if pregame.get("frozen_at"):
            _write_json(root / "frozen" / f"{game_id}.json", pregame)
        by_date.setdefault(str(game.pop("_game_date")), []).append(game)
    for game_date, games in by_date.items():
        games.sort(key=lambda row: (str(row.get("start_time")), str(row["game_id"])))
        _write_json(
            root / "latest" / f"{game_date}.json",
            {
                "available": True,
                "game_date": game_date,
                "season": identity["season"],
                "collected_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "source": "NBA official schedule and liveData",
                "games": games,
            },
        )
    _write_json(
        root / "manifest.json",
        {
            **dict(identity),
            "identity": identity_sha,
            "model_track": "updated_ratings_updated_minutes",
            "evaluation_cell": "official_nba_daily_chain",
            "provider_fallbacks": [],
        },
    )


def _pregame_artifacts_by_nba_game_id(
    root: Path,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    output: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for summary_path in sorted((root / "final").glob("*.summary.json")):
        summary = _read_object(summary_path)
        game_id = str(summary.get("game_id", "")).strip()
        pregame = summary.get("pregame")
        if not game_id or not isinstance(pregame, dict):
            continue
        cleaned = _sanitize_frozen_pregame(pregame, game_id=game_id)
        if game_id in output:
            raise ValueError(f"Duplicate frozen NBA game ID: {game_id}")
        output[game_id] = (cleaned, summary)
    for pregame_path in sorted((root / "frozen").glob("*.json")):
        pregame = _read_object(pregame_path)
        game_id = str(pregame.get("game_id", "")).strip()
        if not game_id or game_id in output:
            continue
        output[game_id] = (
            _sanitize_frozen_pregame(pregame, game_id=game_id),
            {},
        )
    for latest_path in sorted((root / "latest").glob("*.json")):
        latest = _read_object(latest_path)
        games = latest.get("games")
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, dict):
                continue
            game_id = str(game.get("game_id", "")).strip()
            if not game_id or game_id in output:
                continue
            output[game_id] = (
                _sanitize_frozen_pregame(game, game_id=game_id),
                {},
            )
    return output


def _official_upcoming_games(
    catalog_path: Path,
    *,
    season: str,
    after: date,
    pregames: Mapping[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Official NBA game catalog is missing: {catalog_path}")
    frame = pd.read_parquet(catalog_path)
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
        raise ValueError("Official game catalog is missing: " + ", ".join(missing))
    selected = frame.loc[
        frame["season"].astype(str).eq(season)
        & frame["season_type"].astype(str).eq("regular")
    ].copy()
    selected["game_time_utc"] = pd.to_datetime(selected["game_time_utc"], utc=True)
    selected["game_date_pacific"] = (
        selected["game_time_utc"].dt.tz_convert("America/Los_Angeles").dt.date
    )
    selected = selected.loc[selected["game_date_pacific"] > after]
    rows: list[dict[str, Any]] = []
    for row in selected.sort_values(["game_time_utc", "game_id"]).itertuples(index=False):
        game_id = str(row.game_id)
        entry = pregames.get(game_id)
        if entry is None:
            continue
        pregame = copy.deepcopy(entry[0])
        home_team = str(row.home_team_tricode)
        away_team = str(row.away_team_tricode)
        if {
            str(pregame.get("home_team")),
            str(pregame.get("away_team")),
        } != {home_team, away_team}:
            raise ValueError(f"Frozen matchup disagrees with NBA catalog for {game_id}")
        start_time = pd.Timestamp(row.game_time_utc).isoformat().replace("+00:00", "Z")
        pregame.update(
            {
                "game_id": game_id,
                "start_time": start_time,
                "home_team": home_team,
                "away_team": away_team,
            }
        )
        rows.append(
            {
                **copy.deepcopy(pregame),
                "home_score": None,
                "away_score": None,
                "status": {
                    "state": "pre",
                    "completed": False,
                    "description": "Scheduled",
                    "detail": "Scheduled",
                    "display_clock": None,
                    "period": 0,
                },
                "_game_date": row.game_date_pacific.isoformat(),
                "_pregame": pregame,
            }
        )
    return rows


def _sanitize_frozen_pregame(
    pregame: Mapping[str, Any], *, game_id: str
) -> dict[str, Any]:
    source = copy.deepcopy(dict(pregame))
    market = _sanitize_market(source.pop("market", None))
    cleaned = _drop_provider_fields(source)
    cleaned["game_id"] = game_id
    cleaned.pop("broadcasts", None)
    cleaned["market"] = market
    cleaned["source_provenance"] = {
        "schedule_identity": "NBA official schedule",
        "basketball_facts": "NBA official liveData",
        "market": "The Odds API" if cleaned.get("market") else None,
        "injuries": "NBA official injury report",
        "provider_fallbacks": [],
    }
    required = {"start_time", "home_team", "away_team", "projection"}
    missing = sorted(key for key in required if cleaned.get(key) in (None, ""))
    if missing:
        raise ValueError(
            f"Frozen GESTALT pregame {game_id} lacks required fields: {', '.join(missing)}"
        )
    return cleaned


def _sanitize_market(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    market = copy.deepcopy(value)
    books = [
        row
        for row in market.get("bookmakers", [])
        if isinstance(row, dict)
        and row.get("key") not in (None, "")
        and str(row.get("key", "")).casefold() not in FORBIDDEN_BOOKMAKER_KEYS
    ]
    market["bookmakers"] = books
    market["bookmaker_keys"] = [str(row["key"]) for row in books]
    market["bookmaker_count"] = len(books)
    spread_books = [row for row in books if row.get("home_spread") is not None]
    probability_books = [
        row for row in books if row.get("home_win_probability") is not None
    ]
    market["spread_bookmaker_count"] = len(spread_books)
    market["probability_bookmaker_count"] = len(probability_books)
    if spread_books:
        market["home_spread"] = float(
            pd.Series([float(row["home_spread"]) for row in spread_books]).median()
        )
        market["away_spread"] = -float(market["home_spread"])
    if probability_books:
        home_probability = float(
            pd.Series(
                [float(row["home_win_probability"]) for row in probability_books]
            ).median()
        )
        market["home_win_probability"] = home_probability
        market["away_win_probability"] = 1.0 - home_probability
    return _drop_provider_fields(market)


def _drop_provider_fields(value: Any) -> Any:
    """Remove the retired provider's identifiers and labels recursively."""

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if "espn" in str(key).casefold():
                continue
            cleaned = _drop_provider_fields(item)
            if isinstance(cleaned, str) and "espn" in cleaned.casefold():
                continue
            output[str(key)] = cleaned
        return output
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if not (isinstance(item, str) and "espn" in item.casefold())
            for cleaned in [_drop_provider_fields(item)]
        ]
    return value


def _official_retrospective(
    *,
    final_game: Mapping[str, Any],
    pregame: Mapping[str, Any],
    actual_minutes: pd.DataFrame,
    ratings: Mapping[int, float],
) -> dict[str, Any]:
    game_id = str(final_game["game_id"])
    required_minutes = {"team", "player_id", "player_name", "minutes"}
    missing_minutes = sorted(required_minutes - set(actual_minutes.columns))
    if missing_minutes or actual_minutes.empty:
        raise ValueError(
            f"Official actual minutes for {game_id} are invalid; missing={missing_minutes}"
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in actual_minutes.itertuples(index=False):
        player_id = int(row.player_id)
        if player_id not in ratings:
            raise ValueError(
                f"Official minutes player {player_id} lacks pregame rating for {game_id}"
            )
        rows.setdefault(str(row.team), []).append(
            {
                "player_id": player_id,
                "player_name": str(row.player_name),
                "actual_minutes": float(row.minutes),
            }
        )
    home_team = str(final_game["home_team"])
    away_team = str(final_game["away_team"])
    projection = pregame.get("projection")
    if not isinstance(projection, dict) or projection.get("home_margin") is None:
        raise ValueError(f"Frozen home-margin projection is missing for {game_id}")
    pregame_ledger = projection.get("ledger")
    if not isinstance(pregame_ledger, Mapping):
        raise ValueError(f"Frozen pregame ledger is missing for {game_id}")
    frozen_margin = float(projection["home_margin"])
    actual_by_side = {
        "away": rows.get(away_team, []),
        "home": rows.get(home_team, []),
    }
    ledger = build_actual_minutes_ledger(
        game_id=game_id,
        away_team=away_team,
        home_team=home_team,
        actual_minutes=actual_by_side,
        ratings=ratings,
        pregame_ledger=pregame_ledger,
        provenance={
            "pregame_ledger_sha256": _object_sha256(pregame_ledger),
            "rating_contract": "the same frozen rating-before state as the pregame ledger",
            "minutes_source": "verified official NBA actual-minutes artifact",
        },
    )
    official_margin = float(ledger["home_margin"])
    frozen_probability = projection.get("home_win_probability")
    probability = _rescale_probability(
        frozen_margin=frozen_margin,
        frozen_probability=frozen_probability,
        official_margin=official_margin,
    )
    return {
        "computed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "method": "NAIL rating-before state reweighted by official NBA actual minutes",
        "home_margin": official_margin,
        "home_win_probability": probability,
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


def _projection_controls(
    supplied: Mapping[str, object] | None,
) -> dict[str, float]:
    """Load the immutable season controls used by the public projection model."""

    if supplied is None:
        from nba_lineup_model.web_api.inference import MODEL_ARTIFACT, LineupEvaluator
        from nba_lineup_model.web_api.win_projections import read_win_projection_cache

        evaluator = LineupEvaluator.from_latest_artifact()
        payload = read_win_projection_cache(
            model_artifact=MODEL_ARTIFACT, run_id=evaluator.run_id
        )
        supplied = payload.get("win_projection")
    if not isinstance(supplied, Mapping):
        raise ValueError("Projection controls are unavailable")
    output: dict[str, float] = {}
    for key in ("home_court", "back_to_back", "win_probability_scale"):
        value = float(supplied[key])
        if not math.isfinite(value):
            raise ValueError(f"Projection control is not finite: {key}")
        output[key] = value
    return output


def _attach_game_prediction_attribution(
    source_pregame: Mapping[str, Any],
    *,
    ratings: Mapping[int, float],
    home_court: float,
    back_to_back: float,
    win_probability_scale: float,
    home_back_to_back: bool,
    away_back_to_back: bool,
    provenance: Mapping[str, object],
) -> dict[str, Any]:
    """Materialize the player-page ratings and frozen minutes as one artifact."""

    pregame = copy.deepcopy(dict(source_pregame))
    game_id = str(pregame["game_id"])
    projected = pregame.get("projected_minutes")
    if not isinstance(projected, Mapping):
        raise ValueError(f"Frozen projected minutes are missing for {game_id}")
    enriched = complete_projected_minutes_attribution(
        projected,
        injury_report=(
            pregame.get("injury_report")
            if isinstance(pregame.get("injury_report"), Mapping)
            else None
        ),
        away_team=str(pregame["away_team"]),
        home_team=str(pregame["home_team"]),
        ratings=ratings,
        game_id=game_id,
    )
    ledger = build_spread_ledger(
        game_id=game_id,
        away_team=str(pregame["away_team"]),
        home_team=str(pregame["home_team"]),
        projected_minutes=enriched,
        home_court_adjustment=home_court,
        back_to_back_coefficient=back_to_back,
        home_back_to_back=home_back_to_back,
        away_back_to_back=away_back_to_back,
        provenance=provenance,
    )
    projection = pregame.get("projection")
    projection = copy.deepcopy(projection) if isinstance(projection, dict) else {}
    margin = float(ledger["home_margin"])
    projection.update(
        {
            "model_track": "updated_ratings_updated_minutes",
            "state_kind": "in_season_strict_forward_replay",
            "input_provenance": dict(provenance),
            "home_margin": margin,
            "fair_home_spread": -margin,
            "home_win_probability": float(
                1.0 / (1.0 + math.exp(-win_probability_scale * margin))
            ),
            "ledger": ledger,
        }
    )
    require_projection_reconciliation(projection, game_id=game_id)
    pregame["projection"] = projection
    pregame["projected_minutes"] = enriched
    return pregame


def _rescale_probability(
    *, frozen_margin: float, frozen_probability: object, official_margin: float
) -> float | None:
    try:
        probability = float(frozen_probability)
    except (TypeError, ValueError):
        return None
    if not 0.0 < probability < 1.0 or abs(frozen_margin) < 1e-9:
        return probability if abs(official_margin - frozen_margin) < 1e-9 else None
    beta = math.log(probability / (1.0 - probability)) / frozen_margin
    return 1.0 / (1.0 + math.exp(-beta * official_margin))


def _official_final_summary(
    *,
    final_game: Mapping[str, Any],
    pregame: Mapping[str, Any],
    retrospective: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    data_sha: str,
    model_sha: str,
    archived_at: object,
) -> dict[str, Any]:
    home_score = int(final_game["home_score"])
    away_score = int(final_game["away_score"])
    actual_margin = home_score - away_score
    projection = pregame.get("projection")
    if not isinstance(projection, dict) or projection.get("home_margin") is None:
        raise ValueError(f"Frozen projection is missing for {final_game['game_id']}")
    projected_margin = float(projection["home_margin"])
    home_team = str(final_game["home_team"])
    away_team = str(final_game["away_team"])
    model_side = home_team if projected_margin > 0 else away_team
    market_grading = _market_grading(
        market=pregame.get("market"),
        home_team=home_team,
        away_team=away_team,
        actual_home_margin=actual_margin,
        projected_home_margin=projected_margin,
    )
    return {
        "schema_version": 2,
        "record_type": "immutable_official_nba_final_game_audit",
        "archived_at": str(archived_at or datetime.now(UTC).isoformat()).replace(
            "+00:00", "Z"
        ),
        "season": str(final_game["season"]),
        "game_date": str(final_game["game_date_pacific"]),
        "game_id": str(final_game["game_id"]),
        "start_time": final_game["start_time"],
        "away_team": away_team,
        "home_team": home_team,
        "final": {
            "away_score": away_score,
            "home_score": home_score,
            "actual_home_margin": actual_margin,
            "winner": home_team if actual_margin > 0 else away_team,
            "status": {
                "state": "post",
                "completed": True,
                "description": "Final",
                "detail": "Final",
                "display_clock": "0:00",
                "period": None,
            },
        },
        "pregame": copy.deepcopy(dict(pregame)),
        "grading": {
            "pregame_frozen_at": pregame.get("frozen_at"),
            "model": projection.get("model"),
            "projected_home_margin": projected_margin,
            "model_fair_home_spread": -projected_margin,
            "model_fair_line_team": model_side,
            "model_fair_line": -abs(projected_margin),
            "projected_home_win_probability": projection.get(
                "home_win_probability"
            ),
            "model_favorite": model_side,
            "model_favorite_won": (
                actual_margin > 0 if model_side == home_team else actual_margin < 0
            ),
            "margin_absolute_error": abs(actual_margin - projected_margin),
            "market": market_grading,
        },
        "retrospective": copy.deepcopy(dict(retrospective)),
        "source_provenance": {
            "data_artifact_sha256": data_sha,
            "model_artifact_sha256": model_sha,
            "official_sources": {
                name: {"uri": str(row["uri"]), "sha256": str(row["sha256"])}
                for name, row in source_rows.items()
            },
            "provider_fallbacks": [],
        },
    }


def _market_grading(
    *,
    market: object,
    home_team: str,
    away_team: str,
    actual_home_margin: int,
    projected_home_margin: float,
) -> dict[str, Any] | None:
    if not isinstance(market, dict) or market.get("home_spread") is None:
        return None
    home_spread = float(market["home_spread"])
    away_spread = float(market.get("away_spread", -home_spread))
    implied_margin = -home_spread
    edge = projected_home_margin - implied_margin
    selection = home_team if edge > 0 else away_team
    selection_spread = home_spread if selection == home_team else away_spread
    cover_margin = (
        actual_home_margin + home_spread
        if selection == home_team
        else -actual_home_margin + away_spread
    )
    result = "push" if abs(cover_margin) < 1e-9 else "win" if cover_margin > 0 else "loss"
    market_side = home_team if implied_margin > 0 else away_team
    return {
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
        "market_implied_home_margin": implied_margin,
        "market_favorite": market_side,
        "market_favorite_won": (
            actual_home_margin > 0 if market_side == home_team else actual_home_margin < 0
        ),
        "model_edge_points": edge,
        "model_ats_selection": selection,
        "model_ats_selection_spread": selection_spread,
        "model_ats_cover_margin": cover_margin,
        "model_ats_result": result,
    }


def _artifact_chain(
    root: Path, *, kind: str
) -> list[tuple[date, Path, dict[str, Any]]]:
    pointer = _read_object(root / "latest.json")
    if kind == "data":
        collection = resolve_daily_data_pointer(root, pointer)
        rows: list[tuple[date, Path, dict[str, Any]]] = []
        for selected_date, selected in collection["identity"]["dates"].items():
            bundle = (root / str(selected["bundle_path"])).resolve()
            manifest = verify_daily_data_artifact(bundle)
            identity = manifest["identity"]
            game_date = date.fromisoformat(str(identity["game_date_pacific"]))
            if game_date.isoformat() != selected_date:
                raise ValueError("Selected daily data date does not match its artifact")
            rows.append((game_date, bundle, manifest))
        return rows
    if kind != "model":
        raise ValueError(f"Unsupported artifact chain kind: {kind}")
    bundle = (root / str(pointer["bundle_path"])).resolve()
    rows: list[tuple[date, Path, dict[str, Any]]] = []
    verify = verify_daily_model_artifact
    parent_key = "parent_model_sha256"
    directory_key = "model_sha256"
    while True:
        manifest = verify(bundle)
        identity = manifest["identity"]
        game_date = date.fromisoformat(str(identity["game_date_pacific"]))
        rows.append((game_date, bundle, manifest))
        parent_sha = identity[parent_key]
        if parent_sha is None:
            break
        prior_date = game_date - timedelta(days=1)
        bundle = (
            root
            / f"season={identity['season']}"
            / f"game_date_pacific={prior_date.isoformat()}"
            / f"{directory_key}={parent_sha}"
        ).resolve()
    rows.reverse()
    return rows


def _component_path(
    bundle: Path, components: Mapping[str, Any], name: str
) -> Path:
    entry = components.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Artifact component is missing: {name}")
    path = bundle / str(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != str(entry["sha256"]):
        raise ValueError(f"Artifact component checksum changed: {name}")
    return path


def _read_json_component(
    bundle: Path, components: Mapping[str, Any], name: str
) -> Any:
    return json.loads(_component_path(bundle, components, name).read_text())


def _require_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Official {label} component is invalid")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.parent / f".{link.name}.tmp-{uuid.uuid4().hex}"
    temporary.symlink_to(target)
    temporary.replace(link)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--frozen-artifact-root", type=Path, default=DEFAULT_LEGACY_SCHEDULE_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--game-catalog-path", type=Path, default=DEFAULT_GAME_CATALOG_PATH
    )
    args = parser.parse_args()
    print(
        materialize_official_schedule_state(
            data_root=args.data_root,
            model_root=args.model_root,
            frozen_artifact_root=args.frozen_artifact_root,
            output_root=args.output_root,
            game_catalog_path=args.game_catalog_path,
        )
    )


if __name__ == "__main__":
    main()
