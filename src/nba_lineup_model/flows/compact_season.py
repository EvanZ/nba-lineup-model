from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from prefect import flow, runtime, task
from prefect.client.schemas.objects import State, TaskRun
from prefect.futures import PrefectFuture, as_completed
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.tasks import Task
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nba_lineup_model.season.compact import (
    CuratedGameSource,
    PartitionCompactionOutcome,
    compact_curated_partition,
    curation_code_fingerprint,
)
from nba_lineup_model.season.fetch import select_catalog_games
from nba_lineup_model.season.layout import CURATED_TABLES, CuratedPartition
from nba_lineup_model.season.process import latest_successful_builds
from nba_lineup_model.season.storage import (
    read_build_ledger,
    read_game_catalog,
    read_quality_report,
)


class CompactSeasonRunSummary(BaseModel):
    """Terminal summary for one curated season compaction flow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    run_id: str
    prefect_flow_run_id: str | None = None
    season: str
    curation_code_version: str
    selection_policy: Literal["complete_catalog", "quality_eligible_subset"] = (
        "complete_catalog"
    )
    catalog_game_count: int = Field(ge=1)
    selected_game_count: int = Field(ge=1)
    excluded_game_count: int = Field(ge=0)
    quality_pass_count: int = Field(ge=0)
    quality_warning_count: int = Field(ge=0)
    partition_count: int = Field(ge=1)
    succeeded_partition_count: int = Field(ge=0)
    skipped_partition_count: int = Field(ge=0)
    failed_partition_count: int = Field(ge=0)
    input_row_count: int = Field(ge=0)
    output_row_count: int = Field(ge=0)
    part_count: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    curated_dir: str
    run_manifest_path: str
    failed_partitions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> CompactSeasonRunSummary:
        if self.selected_game_count + self.excluded_game_count != self.catalog_game_count:
            raise ValueError("Compaction selection counts do not match the catalog")
        quality_count = self.quality_pass_count + self.quality_warning_count
        if quality_count != self.selected_game_count:
            raise ValueError("Compaction quality counts do not match selected games")
        terminal_count = (
            self.succeeded_partition_count
            + self.skipped_partition_count
            + self.failed_partition_count
        )
        if terminal_count != self.partition_count:
            raise ValueError("Compaction terminal counts do not match partitions")
        if len(self.failed_partitions) != self.failed_partition_count:
            raise ValueError("Compaction failed partition labels do not match")
        if self.input_row_count != self.output_row_count:
            raise ValueError("Successful curated partitions must conserve rows")
        return self


class CompactSeasonRunManifest(BaseModel):
    """Durable flow summary plus every partition outcome."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    summary: CompactSeasonRunSummary
    outcomes: tuple[PartitionCompactionOutcome, ...]

    @model_validator(mode="after")
    def validate_outcomes(self) -> CompactSeasonRunManifest:
        if len(self.outcomes) != self.summary.partition_count:
            raise ValueError("Run manifest outcomes do not match partition count")
        return self

    def write(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(self.model_dump_json(indent=2) + "\n")
        temporary_path.replace(output_path)
        return output_path


def should_retry_compaction(
    _task: Task[Any, Any],
    _task_run: TaskRun,
    state: State,
) -> bool:
    """Retry only local I/O failures."""

    try:
        state.result()
    except BaseException as error:
        current: BaseException | None = error
        while current is not None:
            if isinstance(current, OSError):
                return True
            current = current.__cause__
    return False


@task(
    name="compact-season-partition",
    task_run_name="compact-{partition.table}-{partition.season_type}",
    retries=1,
    retry_delay_seconds=3,
    retry_condition_fn=should_retry_compaction,
    persist_result=False,
)
def compact_partition_task(
    partition: CuratedPartition,
    sources: list[CuratedGameSource],
    run_id: str,
    processed_dir: str,
    curated_dir: str,
    games_per_part: int,
    force: bool,
    curation_code_version: str,
) -> PartitionCompactionOutcome:
    """Prefect wrapper around framework-independent partition compaction."""

    return compact_curated_partition(
        partition,
        sources,
        run_id=run_id,
        processed_dir=processed_dir,
        curated_dir=curated_dir,
        games_per_part=games_per_part,
        force=force,
        curation_code_version=curation_code_version,
        prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
        prefect_task_run_id=_runtime_id(runtime.task_run.id),
    )


@flow(
    name="compact-nba-season",
    flow_run_name="compact-{season}",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    persist_result=False,
)
def compact_season_flow(
    season: str,
    *,
    catalog_path: str = "data/catalog/games.parquet",
    processed_dir: str = "data/processed",
    curated_dir: str = "data/curated",
    ledger_path: str = "data/manifests/builds.parquet",
    quality_games_path: str = "data/quality/games.parquet",
    season_types: list[str] | None = None,
    game_ids: list[str] | None = None,
    games_per_part: int = 100,
    force: bool = False,
    quality_eligible_only: bool = False,
    curation_code_version: str | None = None,
    run_id: str | None = None,
) -> CompactSeasonRunSummary:
    """Compact all quality-gated final games into curated season partitions."""

    if games_per_part < 1:
        raise ValueError("games_per_part must be positive")
    started_at = datetime.now(UTC)
    run_id = run_id or _new_run_id(season, started_at)
    curation_code_version = (
        curation_code_version or curation_code_fingerprint()
    )
    games = select_catalog_games(
        read_game_catalog(catalog_path),
        season=season,
        season_types=season_types,
        game_ids=game_ids,
    )
    if not games:
        raise ValueError(f"No final catalog games matched compaction for {season}")

    successful_builds = latest_successful_builds(read_build_ledger(ledger_path))
    quality_by_game = {
        record.game_id: record
        for record in read_quality_report(quality_games_path).records
    }
    sources: list[CuratedGameSource] = []
    preflight_errors: list[str] = []
    for game in games:
        build = successful_builds.get(game.game_id)
        quality = quality_by_game.get(game.game_id)
        if build is None:
            preflight_errors.append(f"{game.game_id}: no successful build")
            continue
        if quality is None:
            preflight_errors.append(f"{game.game_id}: no quality record")
            continue
        try:
            sources.append(
                CuratedGameSource(
                    game=game,
                    build=build,
                    quality=quality,
                )
            )
        except ValueError as error:
            preflight_errors.append(f"{game.game_id}: {error}")
    if preflight_errors and not quality_eligible_only:
        examples = "; ".join(preflight_errors[:5])
        remainder = len(preflight_errors) - min(5, len(preflight_errors))
        suffix = f"; plus {remainder} more" if remainder else ""
        raise ValueError(
            f"Curated preflight failed for {len(preflight_errors)} games: "
            f"{examples}{suffix}"
        )
    if not sources:
        raise ValueError(f"No quality-eligible games are available for {season}")

    sources_by_type: dict[str, list[CuratedGameSource]] = {}
    for source in sources:
        sources_by_type.setdefault(source.game.season_type, []).append(source)

    future_partitions: dict[int, CuratedPartition] = {}
    futures: list[PrefectFuture[PartitionCompactionOutcome]] = []
    for season_type in sorted(sources_by_type):
        partition_sources = sources_by_type[season_type]
        for table_name in CURATED_TABLES:
            partition = CuratedPartition(
                table=table_name,
                season=season,
                season_type=season_type,
            )
            future = compact_partition_task.submit(
                partition,
                partition_sources,
                run_id,
                processed_dir,
                curated_dir,
                games_per_part,
                force,
                curation_code_version,
            )
            futures.append(future)
            future_partitions[id(future)] = partition

    outcomes: list[PartitionCompactionOutcome] = []
    for future in as_completed(futures):
        partition = future_partitions[id(future)]
        try:
            outcome = future.result()
        except Exception as error:
            outcome = PartitionCompactionOutcome(
                partition=partition,
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        outcomes.append(outcome)
    outcomes.sort(
        key=lambda outcome: (
            outcome.partition.season_type,
            outcome.partition.table,
        )
    )

    statuses = Counter(outcome.status for outcome in outcomes)
    quality_statuses = Counter(source.quality.status for source in sources)
    finished_at = datetime.now(UTC)
    run_manifest_path = (
        Path(curated_dir)
        / "_manifests"
        / season
        / f"{run_id}.json"
    )
    failed_partitions = [
        _partition_label(outcome.partition)
        for outcome in outcomes
        if outcome.status == "failed"
    ]
    summary = CompactSeasonRunSummary(
        run_id=run_id,
        prefect_flow_run_id=_runtime_id(runtime.flow_run.id),
        season=season,
        curation_code_version=curation_code_version,
        selection_policy=(
            "quality_eligible_subset" if quality_eligible_only else "complete_catalog"
        ),
        catalog_game_count=len(games),
        selected_game_count=len(sources),
        excluded_game_count=len(preflight_errors),
        quality_pass_count=quality_statuses["pass"],
        quality_warning_count=quality_statuses["warning"],
        partition_count=len(outcomes),
        succeeded_partition_count=statuses["succeeded"],
        skipped_partition_count=statuses["skipped"],
        failed_partition_count=statuses["failed"],
        input_row_count=sum(outcome.input_row_count for outcome in outcomes),
        output_row_count=sum(outcome.output_row_count for outcome in outcomes),
        part_count=sum(outcome.part_count for outcome in outcomes),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        curated_dir=str(Path(curated_dir)),
        run_manifest_path=str(run_manifest_path),
        failed_partitions=failed_partitions,
    )
    CompactSeasonRunManifest(
        summary=summary,
        outcomes=tuple(outcomes),
    ).write(run_manifest_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compact quality-gated per-game Parquet into curated season datasets."
        )
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument(
        "--catalog",
        default="data/catalog/games.parquet",
        help="Canonical game catalog path",
    )
    parser.add_argument(
        "--quality-eligible-only",
        action="store_true",
        help=(
            "Publish only successful pass/warning games; intended for the "
            "documented historical modeling subset"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Per-game processed Parquet root",
    )
    parser.add_argument(
        "--curated-dir",
        default="data/curated",
        help="Curated season-dataset root",
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
        "--season-type",
        action="append",
        dest="season_types",
        help="Compact one complete season type; repeat for multiple types",
    )
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help="Compact one selected game ID; repeat for multiple games",
    )
    parser.add_argument(
        "--games-per-part",
        type=int,
        default=100,
        help="Maximum source games per Parquet part",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent partition compactors",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild partitions even when inputs and manifests match",
    )
    parser.add_argument(
        "--quality-eligible-only",
        action="store_true",
        help=(
            "Publish only successful pass/warning games; intended for the "
            "documented historical modeling subset"
        ),
    )
    parser.add_argument("--run-id", help="Optional caller-owned run identifier")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.games_per_part < 1:
        raise SystemExit("--games-per-part must be positive")
    configured_flow = compact_season_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )
    summary = configured_flow(
        args.season,
        catalog_path=args.catalog,
        processed_dir=args.processed_dir,
        curated_dir=args.curated_dir,
        ledger_path=args.ledger,
        quality_games_path=args.quality_games,
        season_types=args.season_types,
        game_ids=args.game_ids,
        games_per_part=args.games_per_part,
        force=args.force,
        quality_eligible_only=args.quality_eligible_only,
        run_id=args.run_id,
    )
    print(
        f"{summary.season} compaction: games={summary.selected_game_count}, "
        f"partitions={summary.partition_count}, "
        f"succeeded={summary.succeeded_partition_count}, "
        f"skipped={summary.skipped_partition_count}, "
        f"failed={summary.failed_partition_count}, parts={summary.part_count}, "
        f"rows={summary.output_row_count}; manifest={summary.run_manifest_path}"
    )
    if summary.failed_partition_count:
        raise SystemExit(1)


def _new_run_id(season: str, started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"compact-{season}-{timestamp}-{uuid4().hex[:8]}"


def _runtime_id(value: object) -> str | None:
    return str(value) if value else None


def _partition_label(partition: CuratedPartition) -> str:
    return f"{partition.table}/{partition.season_type}"


if __name__ == "__main__":
    main()
