"""S3-backed Matchup Lab using the accepted historical and preseason models."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import threading
from functools import lru_cache
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .inference import (
    DISPLAY_SEASON,
    MODEL_ARTIFACT,
    PRESEASON_PREVIEW_SEASON,
    LineupEvaluationError,
    LineupEvaluator,
    SeasonLineupState,
    _is_compiled_linear_x3,
    _mean_reverted_schedule_controls,
    _normalize_search_text,
    _records,
)
from .player_season_history import SOURCES, _lock, _reader
from .rankings import SOURCE_ID, historical_rankings
from .schedule import BUCKET
from .team_navigation import selected

router = APIRouter(prefix="/api/lab")
_logger = logging.getLogger(__name__)
_model_lock = threading.Lock()
RUN_ID = "forward-nail-rapm-v1212-residualized-lambda-2025-26-20260827T221311Z-2b0c1a25"
ARCHIVE = "nba-gestalt/releases/gestalt-web-d739c28/artifacts"
MODEL_ROOT = f"{ARCHIVE}/models/{MODEL_ARTIFACT}/{DISPLAY_SEASON}/{RUN_ID}/"
MODEL_SHA256 = "16155585a285da2cc7d7bc061cffac2e3a9e4694e8f8876b307c222ac793e174"


def source(key, version_id, etag, size_bytes):
    return {
        "uri": f"s3://{BUCKET}/{key}",
        "version_id": version_id,
        "etag": f'"{etag}"',
        "size_bytes": size_bytes,
    }


REALIZED_PROFILES = {
    "profiles": source(
        f"{ARCHIVE}/web/historical_realized_profiles/{MODEL_ARTIFACT}/{RUN_ID}.parquet",
        "dsfE9QPvHIEKX_TeXPOdItutgG1HMaRf",
        "8953c7b365c49adcde6fc4c87d0e38ff",
        2153861,
    )
}
SCHEDULE_CONTROLS = {
    "model": source(
        MODEL_ROOT + "season_model_metadata.parquet",
        "6JB95.2A.Ze1mwn01_oW7yVf4xQr3XdE",
        "1105b3356fc0744c6289933e43273308",
        60152,
    ),
    "schedule": source(
        MODEL_ROOT + "season_schedule_control_metadata.parquet",
        "Cr4HwqBW8b3WFFYbbwohKvIaJ73mmE02",
        "11d603cba3b2cd309fbcf2e2e157c335",
        10534,
    ),
}


@lru_cache(maxsize=1)
def preseason():
    season = PRESEASON_PREVIEW_SEASON
    prefix = f"lake/artifacts/player_ratings/season={season}/model_version=v1/priors/"
    doc, _ = selected(
        _reader().client,
        prefix + "latest.json",
        prefix,
        {"schema_version": 1, "season": season, "model_version": "v1"},
    )
    if doc["artifact_kind"] != "player_rating_priors" or doc["model_run_id"] != RUN_ID:
        raise ValueError("Unrecognized saved Lab preseason state")
    if doc["context"]["source_season"] != DISPLAY_SEASON:
        raise ValueError("Saved Lab preseason context uses a different fit")
    return doc


@lru_cache(maxsize=1)
def historical_seasons():
    reader = _reader()
    with _lock:
        reader.check_sources(REALIZED_PROFILES)
        reader.authenticate()
        rows = reader.connection.execute(
            "SELECT DISTINCT season FROM read_parquet(?) ORDER BY season DESC",
            [REALIZED_PROFILES["profiles"]["uri"]],
        ).fetchall()
        reader.check_sources(REALIZED_PROFILES)
    return tuple(row[0] for row in rows)


@lru_cache(maxsize=8)
def player_pool(season):
    if season == PRESEASON_PREVIEW_SEASON:
        players = pd.DataFrame(preseason()["players"])
    else:
        # Use the published NAIL scale and its matching descriptive splits.
        players = pd.DataFrame(historical_rankings(season, SOURCE_ID)["players"])
    if players.empty or players["player_id"].duplicated().any():
        raise ValueError("Invalid Lab player pool")
    if not players["rapm"].map(math.isfinite).all():
        raise ValueError("Invalid saved player rating")
    return players


@lru_cache(maxsize=8)
def lineup_state(season):
    players = player_pool(season)
    if season == PRESEASON_PREVIEW_SEASON:
        profiles = pd.DataFrame(preseason()["profiles"]).drop(columns="season")
    else:
        reader = _reader()
        with _lock:
            reader.check_sources(REALIZED_PROFILES)
            reader.authenticate()
            profiles = reader.connection.execute(
                "SELECT * EXCLUDE (season) FROM read_parquet(?) WHERE season = ?",
                [REALIZED_PROFILES["profiles"]["uri"], season],
            ).fetchdf()
            reader.check_sources(REALIZED_PROFILES)
    if profiles["player_id"].duplicated().any() or set(profiles["player_id"]) != set(
        players["player_id"]
    ):
        raise ValueError("Saved Lab profiles do not cover the selected player pool")
    return SeasonLineupState(season, players[["player_id", "rapm"]], profiles, players)


@lru_cache(maxsize=1)
def model_data():
    reader = _reader()
    with reader.client.get_object(Bucket=BUCKET, Key=MODEL_ROOT + "season_context_models.joblib")[
        "Body"
    ] as body:
        raw = body.read()
    # Joblib is executable serialization. Load only this registered published object.
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError("Lab context model checksum mismatch")
    contexts = joblib.load(io.BytesIO(raw))
    if set(contexts) != set(historical_seasons()) or not all(
        _is_compiled_linear_x3(context) for context in contexts.values()
    ):
        raise ValueError("Unexpected saved Lab context models")
    with _lock:
        reader.check_sources(SCHEDULE_CONTROLS)
        reader.authenticate()
        model = reader.connection.execute(
            """SELECT season, context_home_intercept, context_training_weight_sum
               FROM read_parquet(?)""",
            [SCHEDULE_CONTROLS["model"]["uri"]],
        ).fetchdf()
        schedule = reader.connection.execute(
            """SELECT season, schedule_control_raw_weight, schedule_training_stint_count
               FROM read_parquet(?)""",
            [SCHEDULE_CONTROLS["schedule"]["uri"]],
        ).fetchdf()
        reader.check_sources(SCHEDULE_CONTROLS)
    return contexts, _mean_reverted_schedule_controls(model, schedule)


class LabMatchupRequest(BaseModel):
    unit_player_ids: list[int] = Field(min_length=5, max_length=5)
    opponent_player_ids: list[int] = Field(min_length=5, max_length=5)
    unit_season: str
    opponent_season: str
    environment: Literal["unit", "neutral", "opponent"] = "unit"
    court: Literal["neutral", "unit_home", "opponent_home"] = "neutral"
    unit_back_to_back: bool = False
    opponent_back_to_back: bool = False


def unavailable(error):
    _logger.warning("Saved Lab data unavailable: %s", type(error).__name__)
    return HTTPException(
        status_code=503,
        detail="Saved Lab data is unavailable.",
        headers={"Cache-Control": "no-store"},
    )


def require_season(season):
    if season != PRESEASON_PREVIEW_SEASON and season not in historical_seasons():
        raise HTTPException(status_code=404, detail="Lab scoring is unavailable for this season.")


def read_pool(season):
    try:
        require_season(season)
        return player_pool(season)
    except HTTPException:
        raise
    except Exception as error:
        raise unavailable(error) from None


@router.get("/seasons")
def lab_seasons():
    try:
        seasons = historical_seasons()
        preseason()
    except Exception as error:
        raise unavailable(error) from None
    return {"seasons": [PRESEASON_PREVIEW_SEASON, *seasons]}


@lru_cache(maxsize=128)
def rating_paths(player_ids, through_season):
    reader = _reader()
    source = {"ratings": SOURCES["ratings"]}
    with _lock:
        reader.check_sources(source)
        reader.authenticate()
        history = reader.connection.execute(
            """SELECT player_id, season, rapm
               FROM read_parquet(?)
               WHERE player_id IN (SELECT unnest(?)) AND season <= ?
               ORDER BY season""",
            [source["ratings"]["uri"], list(player_ids), through_season],
        ).fetchall()
        reader.check_sources(source)
    paths = {player_id: [] for player_id in player_ids}
    for player_id, season, rating in history:
        paths[player_id].append({"season": season, "rating": rating})
    return paths


def player_results(frame, season):
    players = _records(frame)
    if players:
        try:
            paths = rating_paths(tuple(sorted(row["player_id"] for row in players)), season)
        except Exception as error:
            raise unavailable(error) from None
        for row in players:
            row["rating_history"] = paths[row["player_id"]]
    return {"players": players}


@router.get("/players")
def lab_players(
    q: str = Query(min_length=1, max_length=128),
    season: str = DISPLAY_SEASON,
    team: str | None = None,
    limit: int = Query(default=12, ge=1, le=25),
):
    players = read_pool(season)
    normalized = _normalize_search_text(q)
    if not normalized:
        return {"players": []}
    names = players["player_name"].map(_normalize_search_text)
    matches = players.loc[names.str.contains(normalized, regex=False)]
    if team:
        matches = matches.loc[matches["team"].eq(team)]
    return player_results(matches.head(limit), season)


@router.get("/random-lineup")
def lab_random_lineup(
    season: str = DISPLAY_SEASON,
    team: str | None = None,
    exclude_player_id: Annotated[tuple[int, ...], Query()] = (),
    count: int = Query(default=5, ge=1, le=5),
):
    players = read_pool(season)
    eligible = players.loc[~players["player_id"].isin(exclude_player_id)]
    if team:
        eligible = eligible.loc[eligible["team"].eq(team)]
        eligible = eligible.nlargest(
            max(count, math.ceil(len(eligible) / 2)), "possessions", keep="all"
        )
    else:
        eligible = eligible.loc[eligible["possessions"].ge(eligible["possessions"].quantile(0.75))]
    if len(eligible) < count:
        raise HTTPException(status_code=422, detail="Too few eligible players to fill this lineup.")
    return player_results(
        eligible.sample(n=count).sort_values("player_name", kind="stable"), season
    )


@router.post("/matchup")
def lab_matchup(request: LabMatchupRequest):
    try:
        states = {}
        for season in dict.fromkeys((request.unit_season, request.opponent_season)):
            require_season(season)
            states[season] = lineup_state(season)
        with _model_lock:
            contexts, schedule_controls = model_data()
        unit = states[request.unit_season]
        evaluator = LineupEvaluator(
            season=DISPLAY_SEASON,
            run_id=RUN_ID,
            coefficients=unit.coefficients,
            profiles=unit.profiles,
            players=unit.players,
            context_model=contexts[DISPLAY_SEASON],
            response_cache={},
            schedule_controls=schedule_controls,
            season_states=states,
            season_context_models=contexts,
        )
        return evaluator.evaluate(**request.model_dump())
    except HTTPException:
        raise
    except LineupEvaluationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise unavailable(error) from None
