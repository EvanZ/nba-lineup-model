from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nba_lineup_model.flows.compact_season import compact_season_flow
from nba_lineup_model.flows.fetch_season import fetch_season_raw_flow
from nba_lineup_model.flows.process_season import process_season_flow
from nba_lineup_model.ingest.nba_cdn import DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS
from nba_lineup_model.modeling.train import train_regular_season_baselines
from nba_lineup_model.players.collect import collect_player_bios
from nba_lineup_model.season.schedule import (
    NbaScheduleClient,
    SeasonScheduleCache,
    catalog_from_schedule,
    replace_catalog_season,
)
from nba_lineup_model.season.schema import validate_season
from nba_lineup_model.season.storage import (
    read_game_catalog,
    write_game_catalog,
)
from nba_lineup_model.tracking import track_completed_run

DEFAULT_HISTORY_SEASONS = (
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
)
BACKFILL_STAGES = (
    "discover",
    "bios",
    "fetch",
    "process",
    "compact",
    "rapm",
)
BackfillStage = Literal[
    "discover",
    "bios",
    "fetch",
    "process",
    "compact",
    "rapm",
]


class HistoricalBackfillStageRecord(BaseModel):
    """One terminal stage within a resumable historical backfill."""

    model_config = ConfigDict(strict=True, extra="forbid")

    season: str
    stage: BackfillStage
    status: Literal["completed", "failed"]
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> HistoricalBackfillStageRecord:
        has_error = self.error_type is not None or self.error_message is not None
        if self.status == "failed" and not has_error:
            raise ValueError("Failed history stages require error details")
        if self.status == "completed" and has_error:
            raise ValueError("Completed history stages cannot retain errors")
        return self


class HistoricalBackfillManifest(BaseModel):
    """Checkpointed parent record for a serial multi-season backfill."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    created_at: datetime
    seasons: tuple[str, ...]
    from_stage: BackfillStage
    through_stage: BackfillStage
    season_type: Literal["regular"] = "regular"
    records: tuple[HistoricalBackfillStageRecord, ...] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> HistoricalBackfillManifest:
        if len(self.seasons) != len(set(self.seasons)):
            raise ValueError("Historical backfill seasons must be unique")
        if BACKFILL_STAGES.index(self.from_stage) > BACKFILL_STAGES.index(self.through_stage):
            raise ValueError("Historical backfill stage range is reversed")
        keys = [(record.season, record.stage) for record in self.records]
        if len(keys) != len(set(keys)):
            raise ValueError("Historical backfill stage records must be unique")
        return self


def run_historical_backfill(
    seasons: tuple[str, ...] = DEFAULT_HISTORY_SEASONS,
    *,
    from_stage: BackfillStage = "discover",
    through_stage: BackfillStage = "rapm",
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    raw_dir: Path | str = Path("data/raw"),
    processed_dir: Path | str = Path("data/processed"),
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    artifacts_dir: Path | str = Path("artifacts/models"),
    fetch_manifest_path: Path | str = Path("data/manifests/fetches.parquet"),
    build_ledger_path: Path | str = Path("data/manifests/builds.parquet"),
    quality_games_path: Path | str = Path("data/quality/games.parquet"),
    quality_summary_path: Path | str = Path("data/quality/summary.parquet"),
    run_manifest_dir: Path | str = Path("data/manifests/history_backfill"),
    max_workers: int = 2,
    checkpoint_size: int = 25,
    games_per_part: int = 100,
    min_request_interval_seconds: float = 1.0,
    request_interval_jitter_seconds: float = 0.25,
    access_denial_cooldown_seconds: float = DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
    refresh: bool = False,
    force: bool = False,
    run_id: str | None = None,
) -> tuple[HistoricalBackfillManifest, Path]:
    """Run existing resumable season stages serially across historical seasons."""

    normalized_seasons = tuple(validate_season(season) for season in seasons)
    if not normalized_seasons:
        raise ValueError("Historical backfill requires at least one season")
    if max_workers < 1 or checkpoint_size < 1 or games_per_part < 1:
        raise ValueError("Historical backfill worker and checkpoint sizes must be positive")
    if min_request_interval_seconds < 0:
        raise ValueError("Minimum request interval cannot be negative")
    if request_interval_jitter_seconds < 0:
        raise ValueError("Request interval jitter cannot be negative")
    if access_denial_cooldown_seconds <= 0:
        raise ValueError("Access-denial cooldown must be positive")
    if from_stage not in BACKFILL_STAGES or through_stage not in BACKFILL_STAGES:
        raise ValueError("Unknown historical backfill stage")
    if BACKFILL_STAGES.index(from_stage) > BACKFILL_STAGES.index(through_stage):
        raise ValueError("Historical backfill stage range is reversed")

    created_at = datetime.now(UTC)
    run_id = run_id or (f"history-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}")
    manifest_path = Path(run_manifest_dir) / f"{run_id}.json"
    if manifest_path.exists():
        manifest = HistoricalBackfillManifest.model_validate_json(manifest_path.read_text())
        expected_plan = (
            normalized_seasons,
            from_stage,
            through_stage,
        )
        actual_plan = (
            manifest.seasons,
            manifest.from_stage,
            manifest.through_stage,
        )
        if actual_plan != expected_plan:
            raise ValueError(f"Existing historical backfill run {run_id} has a different plan")
    else:
        manifest = HistoricalBackfillManifest(
            run_id=run_id,
            created_at=created_at,
            seasons=normalized_seasons,
            from_stage=from_stage,
            through_stage=through_stage,
        )
        _write_manifest(manifest, manifest_path)
    selected_stages = BACKFILL_STAGES[
        BACKFILL_STAGES.index(from_stage) : BACKFILL_STAGES.index(through_stage) + 1
    ]
    context = {
        "catalog_path": Path(catalog_path),
        "raw_dir": Path(raw_dir),
        "processed_dir": Path(processed_dir),
        "curated_dir": Path(curated_dir),
        "analytical_dir": Path(analytical_dir),
        "artifacts_dir": Path(artifacts_dir),
        "fetch_manifest_path": Path(fetch_manifest_path),
        "build_ledger_path": Path(build_ledger_path),
        "quality_games_path": Path(quality_games_path),
        "quality_summary_path": Path(quality_summary_path),
        "max_workers": max_workers,
        "checkpoint_size": checkpoint_size,
        "games_per_part": games_per_part,
        "min_request_interval_seconds": min_request_interval_seconds,
        "request_interval_jitter_seconds": request_interval_jitter_seconds,
        "access_denial_cooldown_seconds": access_denial_cooldown_seconds,
        "refresh": refresh,
        "force": force,
    }

    for season in normalized_seasons:
        for stage in selected_stages:
            completed = any(
                record.season == season and record.stage == stage and record.status == "completed"
                for record in manifest.records
            )
            if completed:
                continue
            started_at = datetime.now(UTC)
            try:
                details = _execute_stage(season, stage, context)
            except Exception as error:
                finished_at = datetime.now(UTC)
                record = HistoricalBackfillStageRecord(
                    season=season,
                    stage=stage,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=(finished_at - started_at).total_seconds(),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                manifest = _with_stage_record(manifest, record)
                _write_manifest(manifest, manifest_path)
                raise
            finished_at = datetime.now(UTC)
            record = HistoricalBackfillStageRecord(
                season=season,
                stage=stage,
                status="completed",
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=(finished_at - started_at).total_seconds(),
                details=details,
            )
            manifest = _with_stage_record(manifest, record)
            _write_manifest(manifest, manifest_path)
    return manifest, manifest_path


def _execute_stage(
    season: str,
    stage: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    if stage == "discover":
        return _discover_season(
            season,
            context["catalog_path"],
            context["raw_dir"],
            context["refresh"],
        )
    if stage == "bios":
        summary = collect_player_bios(
            season,
            raw_dir=context["raw_dir"],
            curated_dir=context["curated_dir"],
            refresh=context["refresh"],
        )
        return summary.model_dump(mode="json")
    if stage == "fetch":
        flow = fetch_season_raw_flow.with_options(
            task_runner=ThreadPoolTaskRunner(max_workers=context["max_workers"])
        )
        summary = flow(
            season,
            catalog_path=str(context["catalog_path"]),
            raw_dir=str(context["raw_dir"]),
            manifest_path=str(context["fetch_manifest_path"]),
            season_types=["regular"],
            refresh=context["refresh"],
            min_request_interval_seconds=context["min_request_interval_seconds"],
            request_interval_jitter_seconds=context["request_interval_jitter_seconds"],
            access_denial_cooldown_seconds=context["access_denial_cooldown_seconds"],
        )
        if summary.failed_count:
            raise RuntimeError(f"{season} raw fetch failed for {summary.failed_count} games")
        return summary.model_dump(mode="json")
    if stage == "process":
        flow = process_season_flow.with_options(
            task_runner=ThreadPoolTaskRunner(max_workers=context["max_workers"])
        )
        summary = flow(
            season,
            catalog_path=str(context["catalog_path"]),
            raw_dir=str(context["raw_dir"]),
            processed_dir=str(context["processed_dir"]),
            ledger_path=str(context["build_ledger_path"]),
            quality_games_path=str(context["quality_games_path"]),
            quality_summary_path=str(context["quality_summary_path"]),
            season_types=["regular"],
            force=context["force"],
            checkpoint_size=context["checkpoint_size"],
        )
        if summary.failed_count:
            raise RuntimeError(f"{season} processing failed for {summary.failed_count} games")
        return summary.model_dump(mode="json")
    if stage == "compact":
        flow = compact_season_flow.with_options(
            task_runner=ThreadPoolTaskRunner(max_workers=context["max_workers"])
        )
        summary = flow(
            season,
            catalog_path=str(context["catalog_path"]),
            processed_dir=str(context["processed_dir"]),
            curated_dir=str(context["curated_dir"]),
            ledger_path=str(context["build_ledger_path"]),
            quality_games_path=str(context["quality_games_path"]),
            season_types=["regular"],
            games_per_part=context["games_per_part"],
            force=context["force"],
        )
        if summary.failed_partition_count:
            raise RuntimeError(
                f"{season} compaction failed for {summary.failed_partition_count} partitions"
            )
        return summary.model_dump(mode="json")
    if stage == "rapm":
        manifest, run_dir = train_regular_season_baselines(
            season,
            curated_dir=context["curated_dir"],
            analytical_dir=context["analytical_dir"],
            artifacts_dir=context["artifacts_dir"],
        )
        tracking = track_completed_run(run_dir)
        return {
            "run_id": manifest.run_id,
            "run_dir": str(run_dir),
            "player_count": manifest.player_count,
            "game_count": manifest.game_count,
            "selected_rapm_lambda": manifest.selected_rapm_lambda,
            "mlflow_run_id": (tracking.mlflow_run_id if tracking is not None else None),
        }
    raise ValueError(f"Unknown historical backfill stage: {stage}")


def _discover_season(
    season: str,
    catalog_path: Path,
    raw_dir: Path,
    refresh: bool,
) -> dict[str, Any]:
    with NbaScheduleClient(cache=SeasonScheduleCache(raw_dir)) as client:
        response = client.fetch(season, use_cache=not refresh)
    discovered = catalog_from_schedule(response)
    catalog = discovered
    if catalog_path.exists():
        catalog = replace_catalog_season(
            read_game_catalog(catalog_path),
            discovered,
        )
    write_game_catalog(catalog, catalog_path)
    counts = Counter(game.season_type for game in discovered.games)
    return {
        "game_count": len(discovered.games),
        "regular_game_count": counts["regular"],
        "season_type_counts": dict(sorted(counts.items())),
        "catalog_game_count": len(catalog.games),
        "catalog_path": str(catalog_path),
        "source_url": response.url,
    }


def _write_manifest(
    manifest: HistoricalBackfillManifest,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def _with_stage_record(
    manifest: HistoricalBackfillManifest,
    record: HistoricalBackfillStageRecord,
) -> HistoricalBackfillManifest:
    records = tuple(
        existing
        for existing in manifest.records
        if (existing.season, existing.stage) != (record.season, record.stage)
    )
    return manifest.model_copy(update={"records": (*records, record)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable regular-season discovery, ingestion, curation, and "
            "RAPM training across historical NBA seasons."
        )
    )
    parser.add_argument(
        "--season",
        action="append",
        dest="seasons",
        help=(
            "Historical season in YYYY-YY format; repeat to override the "
            "2019-20 through 2024-25 default"
        ),
    )
    parser.add_argument(
        "--from-stage",
        choices=BACKFILL_STAGES,
        default="discover",
    )
    parser.add_argument(
        "--through-stage",
        choices=BACKFILL_STAGES,
        default="rapm",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--checkpoint-size", type=int, default=25)
    parser.add_argument("--games-per-part", type=int, default=100)
    parser.add_argument("--min-request-interval", type=float, default=1.0)
    parser.add_argument("--request-interval-jitter", type=float, default=0.25)
    parser.add_argument(
        "--access-denial-cooldown",
        type=float,
        default=DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seasons = tuple(args.seasons or DEFAULT_HISTORY_SEASONS)
    manifest, path = run_historical_backfill(
        seasons,
        from_stage=args.from_stage,
        through_stage=args.through_stage,
        max_workers=args.max_workers,
        checkpoint_size=args.checkpoint_size,
        games_per_part=args.games_per_part,
        min_request_interval_seconds=args.min_request_interval,
        request_interval_jitter_seconds=args.request_interval_jitter,
        access_denial_cooldown_seconds=args.access_denial_cooldown,
        refresh=args.refresh,
        force=args.force,
        run_id=args.run_id,
    )
    print(
        f"Historical backfill {manifest.run_id}: seasons={len(manifest.seasons)}, "
        f"stages={len(manifest.records)}, status=completed; manifest={path}"
    )


if __name__ == "__main__":
    main()
