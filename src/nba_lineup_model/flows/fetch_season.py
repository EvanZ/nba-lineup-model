from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from prefect import flow, runtime, task
from prefect.client.schemas.objects import State, TaskRun
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.tasks import Task
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nba_lineup_model.ingest.nba_cdn import DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS
from nba_lineup_model.season.fetch import (
    failed_fetch_record,
    fetch_game_raw,
    is_transient_fetch_error,
    select_catalog_games,
)
from nba_lineup_model.season.schema import (
    CatalogGame,
    GameFetchRecord,
    validate_season,
)
from nba_lineup_model.season.storage import (
    append_fetch_records,
    read_game_catalog,
)


class FetchRunSummary(BaseModel):
    """Compact terminal summary returned by one Prefect season fetch flow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    prefect_flow_run_id: str | None = None
    season: str
    selected_game_count: int = Field(ge=1)
    succeeded_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    manifest_path: str
    failed_game_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> FetchRunSummary:
        terminal_count = self.succeeded_count + self.skipped_count + self.failed_count
        if terminal_count != self.selected_game_count:
            raise ValueError("Fetch summary terminal counts do not match selected games")
        if len(self.failed_game_ids) != self.failed_count:
            raise ValueError("Fetch summary failed IDs do not match the failure count")
        return self


class GameFetchTaskError(RuntimeError):
    """Carries a durable failure record through Prefect retry handling."""

    def __init__(self, record: GameFetchRecord, error: Exception) -> None:
        super().__init__(str(error))
        self.record = record
        self.original_error = error


def should_retry_fetch(
    _task: Task[Any, Any],
    _task_run: TaskRun,
    state: State,
) -> bool:
    """Retry only failures marked transient by the direct NBA client."""

    try:
        state.result()
    except BaseException as error:
        return is_transient_fetch_error(error)
    return False


@task(
    name="fetch-raw-game",
    retries=3,
    retry_delay_seconds=[5, 30, 120],
    retry_jitter_factor=0.25,
    retry_condition_fn=should_retry_fetch,
    persist_result=False,
)
def fetch_raw_game_task(
    game: CatalogGame,
    run_id: str,
    raw_dir: str,
    refresh: bool,
    min_request_interval_seconds: float,
    request_interval_jitter_seconds: float,
    access_denial_cooldown_seconds: float,
) -> GameFetchRecord:
    """Prefect task wrapper around the framework-independent raw fetcher."""

    started_at = datetime.now(UTC)
    flow_run_id = _runtime_id(runtime.flow_run.id)
    task_run_id = _runtime_id(runtime.task_run.id)
    attempt_number = runtime.task_run.run_count or 1
    try:
        return fetch_game_raw(
            game,
            run_id=run_id,
            raw_dir=raw_dir,
            refresh=refresh,
            attempt_number=attempt_number,
            prefect_flow_run_id=flow_run_id,
            prefect_task_run_id=task_run_id,
            started_at=started_at,
            min_request_interval_seconds=min_request_interval_seconds,
            request_interval_jitter_seconds=request_interval_jitter_seconds,
            access_denial_cooldown_seconds=access_denial_cooldown_seconds,
        )
    except Exception as error:
        record = failed_fetch_record(
            game,
            run_id=run_id,
            started_at=started_at,
            error=error,
            raw_dir=raw_dir,
            refresh=refresh,
            attempt_number=attempt_number,
            prefect_flow_run_id=flow_run_id,
            prefect_task_run_id=task_run_id,
        )
        raise GameFetchTaskError(record, error) from error


@flow(
    name="fetch-nba-season-raw",
    flow_run_name="fetch-{season}",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    persist_result=False,
)
def fetch_season_raw_flow(
    season: str,
    *,
    catalog_path: str = "data/catalog/games.parquet",
    raw_dir: str = "data/raw",
    manifest_path: str = "data/manifests/fetches.parquet",
    season_types: list[str] | None = None,
    game_ids: list[str] | None = None,
    limit: int | None = None,
    refresh: bool = False,
    run_id: str | None = None,
    min_request_interval_seconds: float = 1.0,
    request_interval_jitter_seconds: float = 0.25,
    access_denial_cooldown_seconds: float = DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
) -> FetchRunSummary:
    """Fetch raw game feeds concurrently for one cataloged NBA season."""

    started_at = datetime.now(UTC)
    season = validate_season(season)
    run_id = run_id or _new_run_id(season, started_at)
    games = select_catalog_games(
        read_game_catalog(catalog_path),
        season=season,
        season_types=season_types,
        game_ids=game_ids,
        limit=limit,
    )
    if not games:
        raise ValueError(f"No final catalog games matched the fetch selection for {season}")

    submitted_at: dict[str, datetime] = {}
    futures: list[PrefectFuture[GameFetchRecord]] = []
    for game in games:
        submitted_at[game.game_id] = datetime.now(UTC)
        futures.append(
            fetch_raw_game_task.submit(
                game,
                run_id,
                raw_dir,
                refresh,
                min_request_interval_seconds,
                request_interval_jitter_seconds,
                access_denial_cooldown_seconds,
            )
        )
    wait(futures)

    records: list[GameFetchRecord] = []
    for game, future in zip(games, futures, strict=True):
        try:
            records.append(future.result())
        except GameFetchTaskError as error:
            records.append(error.record)
        except Exception as error:
            records.append(
                failed_fetch_record(
                    game,
                    run_id=run_id,
                    started_at=submitted_at[game.game_id],
                    error=error,
                    raw_dir=raw_dir,
                    refresh=refresh,
                    prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
                    prefect_task_run_id=_runtime_id(future.task_run_id),
                )
            )

    append_fetch_records(records, manifest_path)
    finished_at = datetime.now(UTC)
    counts = Counter(record.status for record in records)
    return FetchRunSummary(
        run_id=run_id,
        prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
        season=season,
        selected_game_count=len(records),
        succeeded_count=counts["succeeded"],
        skipped_count=counts["skipped"],
        failed_count=counts["failed"],
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        manifest_path=str(Path(manifest_path)),
        failed_game_ids=[record.game_id for record in records if record.status == "failed"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch raw play-by-play and boxscore JSON for one NBA season."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument(
        "--catalog",
        default="data/catalog/games.parquet",
        help="Canonical game catalog path",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Raw response cache root")
    parser.add_argument(
        "--manifest",
        default="data/manifests/fetches.parquet",
        help="Append-oriented terminal fetch manifest",
    )
    parser.add_argument(
        "--season-type",
        action="append",
        dest="season_types",
        help="Limit to one season type; repeat for multiple types",
    )
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help="Limit to one game ID; repeat for multiple games",
    )
    parser.add_argument("--limit", type=int, help="Fetch only the first N selected games")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum concurrent game fetches",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch both feeds even when valid cache files exist",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=1.0,
        help="Minimum process-wide seconds between NBA CDN requests",
    )
    parser.add_argument(
        "--request-interval-jitter",
        type=float,
        default=0.25,
        help="Maximum random seconds added to each request interval",
    )
    parser.add_argument(
        "--access-denial-cooldown",
        type=float,
        default=DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
        help="Seconds to block new requests after an HTTP 403 or 429",
    )
    parser.add_argument("--run-id", help="Optional caller-owned run identifier")
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
    configured_flow = fetch_season_raw_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )
    summary = configured_flow(
        args.season,
        catalog_path=args.catalog,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        season_types=args.season_types,
        game_ids=args.game_ids,
        limit=args.limit,
        refresh=args.refresh,
        run_id=args.run_id,
        min_request_interval_seconds=args.min_request_interval,
        request_interval_jitter_seconds=args.request_interval_jitter,
        access_denial_cooldown_seconds=args.access_denial_cooldown,
    )
    print(
        f"{summary.season} raw fetch: selected={summary.selected_game_count}, "
        f"succeeded={summary.succeeded_count}, skipped={summary.skipped_count}, "
        f"failed={summary.failed_count}; manifest={summary.manifest_path}"
    )
    if summary.failed_count:
        raise SystemExit(1)


def _new_run_id(season: str, started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"fetch-{season}-{timestamp}-{uuid4().hex[:8]}"


def _runtime_id(value: object) -> str | None:
    return str(value) if value else None


if __name__ == "__main__":
    main()
