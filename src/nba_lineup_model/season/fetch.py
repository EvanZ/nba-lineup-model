from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nba_lineup_model.ingest.nba_cdn import (
    DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
    CachedResponse,
    NbaCdnClient,
    NbaCdnEndpoint,
    NbaCdnError,
    RawJsonCache,
)
from nba_lineup_model.season.schema import (
    CatalogGame,
    GameCatalog,
    GameFetchRecord,
    validate_season,
)


@dataclass(frozen=True)
class RawArtifactEvidence:
    """Validated cache evidence for one exact raw response body."""

    response: CachedResponse
    sha256: str
    byte_count: int


def fetch_game_raw(
    game: CatalogGame,
    *,
    run_id: str,
    raw_dir: Path | str = Path("data/raw"),
    refresh: bool = False,
    attempt_number: int = 1,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
    started_at: datetime | None = None,
    client: NbaCdnClient | None = None,
    min_request_interval_seconds: float = 0.0,
    request_interval_jitter_seconds: float = 0.0,
    access_denial_cooldown_seconds: float = DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
) -> GameFetchRecord:
    """Fetch and validate both raw game documents without Prefect dependencies."""

    started_at = started_at or datetime.now(UTC)
    cache = client.cache if client is not None else RawJsonCache(raw_dir)
    initial_play_by_play = None
    initial_boxscore = None
    if not refresh:
        initial_play_by_play = _safe_artifact_evidence(
            cache,
            NbaCdnEndpoint.PLAY_BY_PLAY,
            game.game_id,
        )
        initial_boxscore = _safe_artifact_evidence(
            cache,
            NbaCdnEndpoint.BOXSCORE,
            game.game_id,
        )
        if initial_play_by_play is not None and initial_boxscore is not None:
            return _terminal_record(
                game,
                run_id=run_id,
                started_at=started_at,
                attempt_number=attempt_number,
                prefect_flow_run_id=prefect_flow_run_id,
                prefect_task_run_id=prefect_task_run_id,
                status="skipped",
                refresh=False,
                play_by_play_cache_hit=True,
                boxscore_cache_hit=True,
                play_by_play=initial_play_by_play,
                boxscore=initial_boxscore,
                skip_reason="already_cached",
            )

    owns_client = client is None
    client = client or NbaCdnClient(
        cache=cache,
        min_request_interval_seconds=min_request_interval_seconds,
        request_interval_jitter_seconds=request_interval_jitter_seconds,
        access_denial_cooldown_seconds=access_denial_cooldown_seconds,
    )
    try:
        if refresh or initial_play_by_play is None:
            client.fetch_play_by_play(game.game_id, use_cache=False)
        if refresh or initial_boxscore is None:
            client.fetch_boxscore(game.game_id, use_cache=False)
    finally:
        if owns_client:
            client.close()

    play_by_play = require_artifact_evidence(
        cache,
        NbaCdnEndpoint.PLAY_BY_PLAY,
        game.game_id,
    )
    boxscore = require_artifact_evidence(
        cache,
        NbaCdnEndpoint.BOXSCORE,
        game.game_id,
    )
    return _terminal_record(
        game,
        run_id=run_id,
        started_at=started_at,
        attempt_number=attempt_number,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
        status="succeeded",
        refresh=refresh,
        play_by_play_cache_hit=initial_play_by_play is not None and not refresh,
        boxscore_cache_hit=initial_boxscore is not None and not refresh,
        play_by_play=play_by_play,
        boxscore=boxscore,
    )


def failed_fetch_record(
    game: CatalogGame,
    *,
    run_id: str,
    started_at: datetime,
    error: Exception,
    raw_dir: Path | str = Path("data/raw"),
    refresh: bool = False,
    attempt_number: int = 1,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
) -> GameFetchRecord:
    """Build a terminal failure record while retaining valid partial artifacts."""

    cache = RawJsonCache(raw_dir)
    play_by_play = _safe_artifact_evidence(
        cache,
        NbaCdnEndpoint.PLAY_BY_PLAY,
        game.game_id,
    )
    boxscore = _safe_artifact_evidence(
        cache,
        NbaCdnEndpoint.BOXSCORE,
        game.game_id,
    )
    finished_at = datetime.now(UTC)
    return GameFetchRecord(
        run_id=run_id,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
        attempt_number=attempt_number,
        game_id=game.game_id,
        season=game.season,
        season_type=game.season_type,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        status="failed",
        refresh=refresh,
        play_by_play_cache_hit=_was_cache_hit(play_by_play, started_at, refresh),
        boxscore_cache_hit=_was_cache_hit(boxscore, started_at, refresh),
        play_by_play_sha256=play_by_play.sha256 if play_by_play is not None else None,
        boxscore_sha256=boxscore.sha256 if boxscore is not None else None,
        play_by_play_bytes=play_by_play.byte_count if play_by_play is not None else None,
        boxscore_bytes=boxscore.byte_count if boxscore is not None else None,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def artifact_evidence(
    cache: RawJsonCache,
    endpoint: NbaCdnEndpoint,
    game_id: str,
) -> RawArtifactEvidence | None:
    """Read and hash one valid cached artifact."""

    metadata_path = cache.metadata_path_for(endpoint, game_id)
    if not metadata_path.exists():
        raise NbaCdnError(f"Cached NBA response is missing provenance metadata: {metadata_path}")
    response = cache.read(endpoint, game_id)
    if response is None:
        return None
    _validate_game_payload(response, endpoint, game_id)
    raw_body = cache.path_for(endpoint, game_id).read_bytes()
    return RawArtifactEvidence(
        response=response,
        sha256=hashlib.sha256(raw_body).hexdigest(),
        byte_count=len(raw_body),
    )


def require_artifact_evidence(
    cache: RawJsonCache,
    endpoint: NbaCdnEndpoint,
    game_id: str,
) -> RawArtifactEvidence:
    """Require a valid raw artifact after a successful client fetch."""

    evidence = artifact_evidence(cache, endpoint, game_id)
    if evidence is None:
        raise NbaCdnError(f"NBA {endpoint.value} response was not cached for {game_id}")
    return evidence


def is_transient_fetch_error(error: BaseException) -> bool:
    """Return whether a failure should be retried by an orchestration layer."""

    current: BaseException | None = error
    while current is not None:
        if isinstance(current, NbaCdnError):
            return current.transient
        current = current.__cause__
    return False


def select_catalog_games(
    catalog: GameCatalog,
    *,
    season: str,
    season_types: list[str] | None = None,
    game_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[CatalogGame]:
    """Select deterministic, final-only game work from the canonical catalog."""

    season = validate_season(season)
    selected_types = set(season_types or [])
    selected_ids = set(game_ids or [])
    games = [
        game
        for game in catalog.games
        if game.season == season
        and game.game_status == "final"
        and (not selected_types or game.season_type in selected_types)
        and (not selected_ids or game.game_id in selected_ids)
    ]
    games.sort(key=lambda game: (game.game_date, game.game_id))
    if limit is not None:
        if limit < 1:
            raise ValueError("Fetch limit must be positive")
        games = games[:limit]
    return games


def _terminal_record(
    game: CatalogGame,
    *,
    run_id: str,
    started_at: datetime,
    attempt_number: int,
    prefect_flow_run_id: str | None,
    prefect_task_run_id: str | None,
    status: str,
    refresh: bool,
    play_by_play_cache_hit: bool,
    boxscore_cache_hit: bool,
    play_by_play: RawArtifactEvidence,
    boxscore: RawArtifactEvidence,
    skip_reason: str | None = None,
) -> GameFetchRecord:
    finished_at = datetime.now(UTC)
    return GameFetchRecord(
        run_id=run_id,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
        attempt_number=attempt_number,
        game_id=game.game_id,
        season=game.season,
        season_type=game.season_type,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        status=status,
        refresh=refresh,
        play_by_play_cache_hit=play_by_play_cache_hit,
        boxscore_cache_hit=boxscore_cache_hit,
        play_by_play_sha256=play_by_play.sha256,
        boxscore_sha256=boxscore.sha256,
        play_by_play_bytes=play_by_play.byte_count,
        boxscore_bytes=boxscore.byte_count,
        skip_reason=skip_reason,
    )


def _safe_artifact_evidence(
    cache: RawJsonCache,
    endpoint: NbaCdnEndpoint,
    game_id: str,
) -> RawArtifactEvidence | None:
    try:
        return artifact_evidence(cache, endpoint, game_id)
    except (OSError, ValueError, NbaCdnError):
        return None


def _validate_game_payload(
    response: CachedResponse,
    endpoint: NbaCdnEndpoint,
    game_id: str,
) -> None:
    game = response.payload.get("game")
    if not isinstance(game, dict):
        raise NbaCdnError(f"NBA {endpoint.value} response is missing its game object")
    if game.get("gameId") != game_id:
        raise NbaCdnError(f"NBA {endpoint.value} response gameId does not match {game_id}")
    if endpoint is NbaCdnEndpoint.PLAY_BY_PLAY and not isinstance(
        game.get("actions"),
        list,
    ):
        raise NbaCdnError("NBA playbyplay response is missing its actions")
    if endpoint is NbaCdnEndpoint.BOXSCORE and (
        not isinstance(game.get("homeTeam"), dict) or not isinstance(game.get("awayTeam"), dict)
    ):
        raise NbaCdnError("NBA boxscore response is missing its teams")


def _was_cache_hit(
    evidence: RawArtifactEvidence | None,
    started_at: datetime,
    refresh: bool,
) -> bool:
    if evidence is None or refresh:
        return False
    return evidence.response.fetched_at < started_at
