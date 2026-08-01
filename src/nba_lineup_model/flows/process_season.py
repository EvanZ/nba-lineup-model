from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from prefect import flow, runtime, task
from prefect.client.schemas.objects import State, TaskRun
from prefect.futures import PrefectFuture, as_completed
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.tasks import Task
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nba_lineup_model.season.fetch import select_catalog_games
from nba_lineup_model.season.process import (
    GameProcessOutcome,
    failed_process_outcome,
    latest_successful_builds,
    process_catalog_game,
    processing_code_fingerprint,
    quality_record_for_outcome,
    sample_processing_games,
)
from nba_lineup_model.season.schema import (
    CatalogGame,
    GameBuildRecord,
    GameQualityRecord,
)
from nba_lineup_model.season.storage import (
    append_build_records,
    merge_quality_records,
    read_build_ledger,
    read_game_catalog,
    read_quality_report,
)


class ProcessRunSummary(BaseModel):
    """Compact terminal summary returned by one season processing flow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    prefect_flow_run_id: str | None = None
    season: str
    code_version: str
    selected_game_count: int = Field(ge=1)
    succeeded_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    quality_pass_count: int = Field(ge=0)
    quality_warning_count: int = Field(ge=0)
    quality_fail_count: int = Field(ge=0)
    quality_error_count: int = Field(ge=0)
    quality_missing_count: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    ledger_path: str
    quality_games_path: str
    quality_summary_path: str
    failed_game_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> ProcessRunSummary:
        terminal_count = self.succeeded_count + self.skipped_count + self.failed_count
        if terminal_count != self.selected_game_count:
            raise ValueError("Process summary terminal counts do not match selected games")
        quality_count = (
            self.quality_pass_count
            + self.quality_warning_count
            + self.quality_fail_count
            + self.quality_error_count
            + self.quality_missing_count
        )
        if quality_count != self.selected_game_count:
            raise ValueError("Process summary quality counts do not match selected games")
        if len(self.failed_game_ids) != self.failed_count:
            raise ValueError("Process summary failed IDs do not match the failure count")
        return self


class GameProcessTaskError(RuntimeError):
    """Carries a terminal process outcome through Prefect task failure handling."""

    def __init__(self, outcome: GameProcessOutcome) -> None:
        message = outcome.record.error_message or "Game processing failed"
        super().__init__(message)
        self.outcome = outcome
        self.retryable = outcome.retryable


def should_retry_process(
    _task: Task[Any, Any],
    _task_run: TaskRun,
    state: State,
) -> bool:
    """Retry only game failures classified as local infrastructure errors."""

    try:
        state.result()
    except BaseException as error:
        current: BaseException | None = error
        while current is not None:
            if isinstance(current, GameProcessTaskError):
                return current.retryable
            current = current.__cause__
    return False


@task(
    name="process-season-game",
    task_run_name="process-{game.game_id}",
    retries=1,
    retry_delay_seconds=3,
    retry_condition_fn=should_retry_process,
    persist_result=False,
)
def process_game_task(
    game: CatalogGame,
    run_id: str,
    raw_dir: str,
    processed_dir: str,
    code_version: str,
    prior_success: GameBuildRecord | None,
    prior_quality: GameQualityRecord | None,
    force: bool,
) -> GameProcessOutcome:
    """Prefect task wrapper around the framework-independent game processor."""

    started_at = datetime.now(UTC)
    outcome = process_catalog_game(
        game,
        run_id=run_id,
        code_version=code_version,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        prior_success=prior_success,
        prior_quality=prior_quality,
        force=force,
        attempt_number=runtime.task_run.run_count or 1,
        prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
        prefect_task_run_id=_runtime_id(runtime.task_run.id),
        started_at=started_at,
    )
    if outcome.record.status == "failed":
        raise GameProcessTaskError(outcome)
    return outcome


@flow(
    name="process-nba-season",
    flow_run_name="process-{season}",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    persist_result=False,
)
def process_season_flow(
    season: str,
    *,
    catalog_path: str = "data/catalog/games.parquet",
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    ledger_path: str = "data/manifests/builds.parquet",
    quality_games_path: str = "data/quality/games.parquet",
    quality_summary_path: str = "data/quality/summary.parquet",
    season_types: list[str] | None = None,
    game_ids: list[str] | None = None,
    limit: int | None = None,
    sample_per_stratum: int | None = None,
    random_seed: int = 0,
    force: bool = False,
    checkpoint_size: int = 25,
    code_version: str | None = None,
    run_id: str | None = None,
) -> ProcessRunSummary:
    """Process and quality-gate final catalog games from the local raw cache."""

    if checkpoint_size < 1:
        raise ValueError("Checkpoint size must be positive")
    started_at = datetime.now(UTC)
    run_id = run_id or _new_run_id(season, started_at)
    code_version = code_version or processing_code_fingerprint()
    games = select_catalog_games(
        read_game_catalog(catalog_path),
        season=season,
        season_types=season_types,
        game_ids=game_ids,
        limit=limit,
    )
    if sample_per_stratum is not None:
        games = sample_processing_games(
            games,
            games_per_stratum=sample_per_stratum,
            random_seed=random_seed,
        )
    if not games:
        raise ValueError(f"No final catalog games matched the process selection for {season}")

    prior_successes = latest_successful_builds(read_build_ledger(ledger_path))
    prior_quality = {
        record.game_id: record
        for record in read_quality_report(quality_games_path).records
    }
    submitted_at: dict[str, datetime] = {}
    game_by_future: dict[int, CatalogGame] = {}
    futures: list[PrefectFuture[GameProcessOutcome]] = []
    for game in games:
        submitted_at[game.game_id] = datetime.now(UTC)
        future = process_game_task.submit(
            game,
            run_id,
            raw_dir,
            processed_dir,
            code_version,
            prior_successes.get(game.game_id),
            prior_quality.get(game.game_id),
            force,
        )
        futures.append(future)
        game_by_future[id(future)] = game

    outcomes: list[GameProcessOutcome] = []
    checkpoint: list[GameProcessOutcome] = []
    for future in as_completed(futures):
        game = game_by_future[id(future)]
        try:
            outcome = future.result()
        except GameProcessTaskError as error:
            outcome = error.outcome
        except Exception as error:
            outcome = failed_process_outcome(
                game,
                run_id=run_id,
                code_version=code_version,
                started_at=submitted_at[game.game_id],
                error=error,
                prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
                prefect_task_run_id=_runtime_id(future.task_run_id),
            )
        outcomes.append(outcome)
        checkpoint.append(outcome)
        if len(checkpoint) >= checkpoint_size:
            _write_checkpoint(
                checkpoint,
                ledger_path=ledger_path,
                quality_games_path=quality_games_path,
                quality_summary_path=quality_summary_path,
            )
            checkpoint.clear()

    if checkpoint:
        _write_checkpoint(
            checkpoint,
            ledger_path=ledger_path,
            quality_games_path=quality_games_path,
            quality_summary_path=quality_summary_path,
        )

    outcomes.sort(key=lambda outcome: outcome.record.game_id)
    statuses = Counter(outcome.record.status for outcome in outcomes)
    selected_ids = {game.game_id for game in games}
    current_quality = {
        record.game_id: record
        for record in read_quality_report(quality_games_path).records
        if record.game_id in selected_ids
    }
    quality_statuses = Counter(record.status for record in current_quality.values())
    finished_at = datetime.now(UTC)
    return ProcessRunSummary(
        run_id=run_id,
        prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
        season=season,
        code_version=code_version,
        selected_game_count=len(outcomes),
        succeeded_count=statuses["succeeded"],
        skipped_count=statuses["skipped"],
        failed_count=statuses["failed"],
        quality_pass_count=quality_statuses["pass"],
        quality_warning_count=quality_statuses["warning"],
        quality_fail_count=quality_statuses["fail"],
        quality_error_count=quality_statuses["error"],
        quality_missing_count=len(selected_ids - set(current_quality)),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        ledger_path=str(Path(ledger_path)),
        quality_games_path=str(Path(quality_games_path)),
        quality_summary_path=str(Path(quality_summary_path)),
        failed_game_ids=[
            outcome.record.game_id
            for outcome in outcomes
            if outcome.record.status == "failed"
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process cached NBA season games into validated per-game Parquet tables."
        )
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument(
        "--catalog",
        default="data/catalog/games.parquet",
        help="Canonical game catalog path",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Raw response cache root")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Per-game processed Parquet root",
    )
    parser.add_argument(
        "--ledger",
        default="data/manifests/builds.parquet",
        help="Append-oriented build ledger",
    )
    parser.add_argument(
        "--quality-games",
        default="data/quality/games.parquet",
        help="Canonical latest game-quality report",
    )
    parser.add_argument(
        "--quality-summary",
        default="data/quality/summary.parquet",
        help="Aggregate quality summary",
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
    parser.add_argument(
        "--audit-games",
        help=(
            "Audit games.parquet whose pass/warning rows define the process selection; "
            "cannot be combined with --game-id"
        ),
    )
    parser.add_argument(
        "--audit-offset",
        type=int,
        default=0,
        help="Skip this many audit-selected games before applying --limit",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        type=int,
        help="Process only the first N selected games",
    )
    selection.add_argument(
        "--sample-per-stratum",
        type=int,
        help="Sample N games per season-type/overtime stratum",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic stratified-sample seed",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent game processors",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=25,
        help="Terminal outcomes per durable metadata checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when source, code, quality, and outputs match",
    )
    parser.add_argument("--run-id", help="Optional caller-owned run identifier")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.checkpoint_size < 1:
        raise SystemExit("--checkpoint-size must be positive")
    if args.audit_games and args.game_ids:
        raise SystemExit("--audit-games cannot be combined with --game-id")
    if args.audit_offset < 0:
        raise SystemExit("--audit-offset must be non-negative")
    game_ids = args.game_ids
    if args.audit_games:
        game_ids = _eligible_audit_game_ids(args.audit_games, args.season)[args.audit_offset :]
    elif args.audit_offset:
        raise SystemExit("--audit-offset requires --audit-games")
    configured_flow = process_season_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )
    summary = configured_flow(
        args.season,
        catalog_path=args.catalog,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        ledger_path=args.ledger,
        quality_games_path=args.quality_games,
        quality_summary_path=args.quality_summary,
        season_types=args.season_types,
        game_ids=game_ids,
        limit=args.limit,
        sample_per_stratum=args.sample_per_stratum,
        random_seed=args.seed,
        force=args.force,
        checkpoint_size=args.checkpoint_size,
        run_id=args.run_id,
    )
    print(
        f"{summary.season} processing: selected={summary.selected_game_count}, "
        f"succeeded={summary.succeeded_count}, skipped={summary.skipped_count}, "
        f"failed={summary.failed_count}; quality pass={summary.quality_pass_count}, "
        f"warning={summary.quality_warning_count}, fail={summary.quality_fail_count}, "
        f"error={summary.quality_error_count}, missing={summary.quality_missing_count}; "
        f"ledger={summary.ledger_path}"
    )
    if summary.failed_count:
        raise SystemExit(1)


def _write_checkpoint(
    outcomes: list[GameProcessOutcome],
    *,
    ledger_path: str,
    quality_games_path: str,
    quality_summary_path: str,
) -> None:
    quality_records = [
        record
        for outcome in outcomes
        if (record := quality_record_for_outcome(outcome)) is not None
    ]
    if quality_records:
        merge_quality_records(
            quality_records,
            quality_games_path,
            summary_path=quality_summary_path,
        )
    append_build_records(
        [outcome.record for outcome in outcomes],
        ledger_path,
    )


def _new_run_id(season: str, started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"process-{season}-{timestamp}-{uuid4().hex[:8]}"


def _eligible_audit_game_ids(audit_games_path: str, season: str) -> list[str]:
    """Select only audit-approved regular-season game IDs for one season."""

    frame = pd.read_parquet(audit_games_path)
    required = {"game_id", "season", "season_type", "status"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Audit report missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame["season"].eq(season)
        & frame["season_type"].eq("regular")
        & frame["status"].isin(["pass", "warning"]),
        "game_id",
    ].astype(str)
    if selected.duplicated().any():
        raise ValueError("Audit report has duplicate eligible game IDs")
    if selected.empty:
        raise ValueError(f"Audit report has no eligible regular-season games for {season}")
    return selected.tolist()


def _runtime_id(value: object) -> str | None:
    return str(value) if value else None


if __name__ == "__main__":
    main()
