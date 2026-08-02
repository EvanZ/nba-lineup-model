from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from prefect import flow, get_run_logger, runtime, task
from prefect.client.schemas.objects import State, TaskRun
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.tasks import Task
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nba_lineup_model.ingest.nba_cdn import NbaCdnEndpoint, NbaCdnError, RawJsonCache
from nba_lineup_model.ingest.nba_stats import (
    DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS,
    NbaStatsEndpoint,
    NbaStatsRawCache,
)
from nba_lineup_model.season.fetch import artifact_evidence, select_catalog_games
from nba_lineup_model.season.schema import CatalogGame, validate_season
from nba_lineup_model.season.stats import (
    StatsFetchRecord,
    append_stats_fetch_records,
    failed_stats_fetch_record,
    fetch_stats_endpoint_raw,
    is_transient_stats_fetch_error,
    stats_play_by_play_final_score,
)
from nba_lineup_model.season.storage import read_game_catalog

DEFAULT_STATS_HISTORY_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)
DEFAULT_STATS_ENDPOINTS = (
    NbaStatsEndpoint.PLAY_BY_PLAY_V3,
    NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
)


class StatsHistoryFetchSummary(BaseModel):
    """Terminal summary for one multi-season NBA Stats acquisition flow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    prefect_flow_run_id: str | None = None
    seasons: tuple[str, ...]
    selected_game_count: int = Field(ge=0)
    selected_endpoint_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    manifest_path: str
    failed_work: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> StatsHistoryFetchSummary:
        terminal_count = self.succeeded_count + self.skipped_count + self.failed_count
        if terminal_count != self.selected_endpoint_count:
            raise ValueError("Stats fetch terminal counts do not match selected endpoint work")
        if len(self.failed_work) != self.failed_count:
            raise ValueError("Stats fetch failed work does not match the failure count")
        return self


class StatsFetchTaskError(RuntimeError):
    """Carries a terminal endpoint record through Prefect retry handling."""

    def __init__(self, record: StatsFetchRecord, error: Exception) -> None:
        super().__init__(str(error))
        self.record = record
        self.original_error = error


def should_retry_stats_fetch(
    _task: Task[Any, Any],
    _task_run: TaskRun,
    state: State,
) -> bool:
    try:
        state.result()
    except BaseException as error:
        return is_transient_stats_fetch_error(error)
    return False


@task(
    name="fetch-nba-stats-endpoint",
    retries=3,
    retry_delay_seconds=[5, 30, 120],
    retry_jitter_factor=0.25,
    retry_condition_fn=should_retry_stats_fetch,
    persist_result=False,
)
def fetch_stats_endpoint_task(
    game: CatalogGame,
    endpoint: NbaStatsEndpoint,
    run_id: str,
    stats_raw_dir: str,
    refresh: bool,
    min_request_interval_seconds: float,
    request_interval_jitter_seconds: float,
    access_denial_cooldown_seconds: float,
) -> StatsFetchRecord:
    """Prefect wrapper for one independently resumable source response."""

    started_at = datetime.now(UTC)
    flow_run_id = _runtime_id(runtime.flow_run.id)
    task_run_id = _runtime_id(runtime.task_run.id)
    attempt_number = runtime.task_run.run_count or 1
    try:
        record = fetch_stats_endpoint_raw(
            game,
            endpoint,
            run_id=run_id,
            raw_dir=stats_raw_dir,
            refresh=refresh,
            attempt_number=attempt_number,
            prefect_flow_run_id=flow_run_id,
            prefect_task_run_id=task_run_id,
            started_at=started_at,
            min_request_interval_seconds=min_request_interval_seconds,
            request_interval_jitter_seconds=request_interval_jitter_seconds,
            access_denial_cooldown_seconds=access_denial_cooldown_seconds,
        )
        _log_downloaded_game(record, game, Path(stats_raw_dir))
        return record
    except Exception as error:
        record = failed_stats_fetch_record(
            game,
            endpoint,
            run_id=run_id,
            started_at=started_at,
            error=error,
            raw_dir=stats_raw_dir,
            refresh=refresh,
            attempt_number=attempt_number,
            prefect_flow_run_id=flow_run_id,
            prefect_task_run_id=task_run_id,
        )
        raise StatsFetchTaskError(record, error) from error


def _log_downloaded_game(
    record: StatsFetchRecord,
    game: CatalogGame,
    stats_raw_dir: Path,
) -> None:
    """Emit useful archive progress only for a newly retained source response."""

    if record.status != "succeeded":
        return

    detail = "final unavailable"
    if record.endpoint is NbaStatsEndpoint.PLAY_BY_PLAY_V3:
        response = NbaStatsRawCache(stats_raw_dir).read(record.endpoint, game.game_id)
        if response is not None:
            final_score = stats_play_by_play_final_score(response.payload)
            if final_score is not None:
                home_score, away_score = final_score
                detail = (
                    f"final {game.away_team_tricode} {away_score}-"
                    f"{home_score} {game.home_team_tricode}"
                )

    get_run_logger().info(
        "Archived %s | %s | %s @ %s | %s",
        game.game_id,
        game.game_date.isoformat(),
        game.away_team_tricode,
        game.home_team_tricode,
        detail,
    )


@flow(
    name="fetch-nba-stats-history",
    flow_run_name="stats-history-{run_id}",
    task_runner=ThreadPoolTaskRunner(max_workers=2),
    persist_result=False,
)
def fetch_stats_history_flow(
    seasons: list[str],
    *,
    catalog_path: str = "data/catalog/games.parquet",
    raw_dir: str = "data/raw",
    manifest_path: str = "data/manifests/stats_fetches.parquet",
    season_types: list[str] | None = None,
    game_ids: list[str] | None = None,
    endpoints: list[NbaStatsEndpoint] | None = None,
    limit: int | None = None,
    cdn_missing_only: bool = False,
    refresh: bool = False,
    run_id: str | None = None,
    min_request_interval_seconds: float = 1.0,
    request_interval_jitter_seconds: float = 0.25,
    access_denial_cooldown_seconds: float = (DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS),
) -> StatsHistoryFetchSummary:
    """Retain raw NBA Stats game responses across one or more seasons."""

    started_at = datetime.now(UTC)
    normalized_seasons = tuple(validate_season(season) for season in seasons)
    if not normalized_seasons:
        raise ValueError("Stats history fetch requires at least one season")
    if len(normalized_seasons) != len(set(normalized_seasons)):
        raise ValueError("Stats history fetch seasons must be unique")
    selected_endpoints = tuple(endpoints or DEFAULT_STATS_ENDPOINTS)
    if not selected_endpoints:
        raise ValueError("Stats history fetch requires at least one endpoint")
    if len(selected_endpoints) != len(set(selected_endpoints)):
        raise ValueError("Stats history fetch endpoints must be unique")
    if limit is not None and limit < 1:
        raise ValueError("Stats history fetch limit must be positive")

    run_id = run_id or _new_run_id(started_at)
    catalog = read_game_catalog(catalog_path)
    games: list[CatalogGame] = []
    for season in normalized_seasons:
        games.extend(
            select_catalog_games(
                catalog,
                season=season,
                season_types=season_types or ["regular"],
                game_ids=game_ids,
            )
        )
    games.sort(key=lambda game: (game.game_date, game.game_id))
    if limit is not None:
        games = games[:limit]

    cdn_cache = RawJsonCache(raw_dir)
    work = [
        (game, endpoint)
        for game in games
        for endpoint in selected_endpoints
        if not cdn_missing_only
        or _corresponding_cdn_artifact_is_missing(cdn_cache, endpoint, game.game_id)
    ]
    stats_raw_dir = str(Path(raw_dir) / "stats")
    submitted_at: dict[tuple[str, NbaStatsEndpoint], datetime] = {}
    futures: list[PrefectFuture[StatsFetchRecord]] = []
    for game, endpoint in work:
        submitted_at[(game.game_id, endpoint)] = datetime.now(UTC)
        futures.append(
            fetch_stats_endpoint_task.submit(
                game,
                endpoint,
                run_id,
                stats_raw_dir,
                refresh,
                min_request_interval_seconds,
                request_interval_jitter_seconds,
                access_denial_cooldown_seconds,
            )
        )
    if futures:
        wait(futures)

    records: list[StatsFetchRecord] = []
    for (game, endpoint), future in zip(work, futures, strict=True):
        try:
            records.append(future.result())
        except StatsFetchTaskError as error:
            records.append(error.record)
        except Exception as error:
            records.append(
                failed_stats_fetch_record(
                    game,
                    endpoint,
                    run_id=run_id,
                    started_at=submitted_at[(game.game_id, endpoint)],
                    error=error,
                    raw_dir=stats_raw_dir,
                    refresh=refresh,
                    prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
                    prefect_task_run_id=_runtime_id(future.task_run_id),
                )
            )

    if records:
        append_stats_fetch_records(records, manifest_path)
    finished_at = datetime.now(UTC)
    counts = Counter(record.status for record in records)
    return StatsHistoryFetchSummary(
        run_id=run_id,
        prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
        seasons=normalized_seasons,
        selected_game_count=len({game.game_id for game, _endpoint in work}),
        selected_endpoint_count=len(records),
        succeeded_count=counts["succeeded"],
        skipped_count=counts["skipped"],
        failed_count=counts["failed"],
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        manifest_path=str(Path(manifest_path)),
        failed_work=[
            f"{record.game_id}:{record.endpoint.value}"
            for record in records
            if record.status == "failed"
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retain raw NBA Stats V3 responses for historical regular seasons."
    )
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help="Season in YYYY-YY format; repeat to override the historical default",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        choices=[endpoint.value for endpoint in NbaStatsEndpoint],
        dest="endpoints",
        help=(
            "Endpoint to retain; repeat for multiple endpoints. Defaults to "
            "playbyplayv3 and boxscoretraditionalv3."
        ),
    )
    parser.add_argument("--catalog", default="data/catalog/games.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--manifest",
        default="data/manifests/stats_fetches.parquet",
    )
    parser.add_argument(
        "--season-type",
        action="append",
        dest="season_types",
        help="Season type to include; defaults to regular",
    )
    parser.add_argument("--game-id", action="append", dest="game_ids")
    parser.add_argument("--limit", type=int, help="Limit selected games across all seasons")
    parser.add_argument(
        "--cdn-missing-only",
        action="store_true",
        help="Fetch only endpoint artifacts missing from the liveData CDN cache",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--min-request-interval", type=float, default=1.0)
    parser.add_argument("--request-interval-jitter", type=float, default=0.25)
    parser.add_argument(
        "--access-denial-cooldown",
        type=float,
        default=DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS,
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.min_request_interval < 0:
        raise SystemExit("--min-request-interval cannot be negative")
    if args.request_interval_jitter < 0:
        raise SystemExit("--request-interval-jitter cannot be negative")
    if args.access_denial_cooldown <= 0:
        raise SystemExit("--access-denial-cooldown must be positive")
    endpoints = (
        [NbaStatsEndpoint(endpoint) for endpoint in args.endpoints] if args.endpoints else None
    )
    configured_flow = fetch_stats_history_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )
    summary = configured_flow(
        list(args.seasons or DEFAULT_STATS_HISTORY_SEASONS),
        catalog_path=args.catalog,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        season_types=args.season_types,
        game_ids=args.game_ids,
        endpoints=endpoints,
        limit=args.limit,
        cdn_missing_only=args.cdn_missing_only,
        refresh=args.refresh,
        run_id=args.run_id,
        min_request_interval_seconds=args.min_request_interval,
        request_interval_jitter_seconds=args.request_interval_jitter,
        access_denial_cooldown_seconds=args.access_denial_cooldown,
    )
    print(
        f"Stats history {summary.run_id}: seasons={len(summary.seasons)}, "
        f"games={summary.selected_game_count}, endpoints={summary.selected_endpoint_count}, "
        f"succeeded={summary.succeeded_count}, skipped={summary.skipped_count}, "
        f"failed={summary.failed_count}; manifest={summary.manifest_path}"
    )
    if summary.failed_count:
        print("Failed work: " + ", ".join(summary.failed_work))
        raise SystemExit(1)


def _corresponding_cdn_artifact_is_missing(
    cache: RawJsonCache,
    endpoint: NbaStatsEndpoint,
    game_id: str,
) -> bool:
    if endpoint is NbaStatsEndpoint.GAME_ROTATION:
        return True
    cdn_endpoint = (
        NbaCdnEndpoint.PLAY_BY_PLAY
        if endpoint is NbaStatsEndpoint.PLAY_BY_PLAY_V3
        else NbaCdnEndpoint.BOXSCORE
    )
    try:
        return artifact_evidence(cache, cdn_endpoint, game_id) is None
    except (OSError, ValueError, NbaCdnError):
        return True


def _new_run_id(started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"stats-history-{timestamp}-{uuid4().hex[:8]}"


def _runtime_id(value: object) -> str | None:
    return str(value) if value else None


if __name__ == "__main__":
    main()
