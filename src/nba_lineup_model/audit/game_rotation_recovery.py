"""Reprocess failed games that have cached NBA Game Rotation evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from prefect.task_runners import ThreadPoolTaskRunner

from nba_lineup_model.audit.game_rotation_probe import (
    RotationEvidence,
    game_rotation_evidence,
    latest_failed_builds,
)
from nba_lineup_model.flows.process_season import process_season_flow
from nba_lineup_model.ingest.nba_stats import NbaStatsEndpoint, NbaStatsRawCache
from nba_lineup_model.season.schema import CatalogGame, GameBuildRecord
from nba_lineup_model.season.storage import (
    read_build_ledger,
    read_game_catalog,
    read_quality_report,
)


def cached_rotation_candidates(
    catalog: list[CatalogGame],
    failed_builds: list[GameBuildRecord],
    cache: NbaStatsRawCache,
) -> list[tuple[CatalogGame, GameBuildRecord, RotationEvidence]]:
    """Return latest failures with structurally valid cached rotations."""

    catalog_by_id = {game.game_id: game for game in catalog}
    candidates: list[tuple[CatalogGame, GameBuildRecord, RotationEvidence]] = []
    for record in failed_builds:
        game = catalog_by_id.get(record.game_id)
        if game is None:
            continue
        response = cache.read(NbaStatsEndpoint.GAME_ROTATION, game.game_id)
        if response is None:
            continue
        evidence = game_rotation_evidence(response.payload)
        if evidence.available:
            candidates.append((game, record, evidence))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reprocess latest failed regular-season games with cached Game Rotation data."
        )
    )
    parser.add_argument("--catalog", default="data/catalog/games.parquet")
    parser.add_argument("--builds", default="data/manifests/builds.parquet")
    parser.add_argument("--quality-games", default="data/quality/games.parquet")
    parser.add_argument("--quality-summary", default="data/quality/summary.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts/reports/game_rotation_recovery")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--checkpoint-size", type=int, default=25)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.max_workers < 1 or args.checkpoint_size < 1:
        raise SystemExit("--max-workers and --checkpoint-size must be positive")

    catalog = read_game_catalog(args.catalog).games
    before = latest_failed_builds(read_build_ledger(args.builds).records)
    cache = NbaStatsRawCache(Path(args.raw_dir) / "stats")
    candidates = cached_rotation_candidates(catalog, before, cache)
    if not candidates:
        raise SystemExit("No latest failed regular-season games have valid cached Game Rotation")

    started_at = datetime.now(UTC)
    run_id = args.run_id or f"game-rotation-recovery-{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    by_season: dict[str, list[str]] = {}
    for game, _record, _evidence in candidates:
        by_season.setdefault(game.season, []).append(game.game_id)

    summaries = []
    configured_flow = process_season_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )
    for season, game_ids in sorted(by_season.items()):
        summaries.append(
            configured_flow(
                season,
                catalog_path=args.catalog,
                raw_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                ledger_path=args.builds,
                quality_games_path=args.quality_games,
                quality_summary_path=args.quality_summary,
                season_types=["regular"],
                game_ids=sorted(game_ids),
                checkpoint_size=args.checkpoint_size,
                run_id=f"{run_id}-{season}",
            )
        )

    latest_builds = _latest_by_game(read_build_ledger(args.builds).records)
    quality_by_game = {
        record.game_id: record for record in read_quality_report(args.quality_games).records
    }
    rows = []
    for game, before_record, evidence in candidates:
        after = latest_builds[game.game_id]
        quality = quality_by_game.get(game.game_id)
        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "before_stage": before_record.terminal_stage,
                "before_error": before_record.error_message,
                "after_status": after.status,
                "after_stage": after.terminal_stage,
                "after_error": after.error_message,
                "after_quality_status": quality.status if quality is not None else None,
                "after_quality_issues": list(quality.issue_codes) if quality is not None else [],
                "game_rotation_sha256": after.game_rotation_sha256,
                **asdict(evidence),
            }
        )

    report_dir = Path(args.output_dir) / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_parquet(report_dir / "games.parquet", index=False)
    recovered_count = sum(row["after_status"] == "succeeded" for row in rows)
    summary = {
        "run_id": run_id,
        "candidate_count": len(rows),
        "recovered_count": recovered_count,
        "recovery_rate": recovered_count / len(rows),
        "season_summaries": [summary.model_dump(mode="json") for summary in summaries],
        "report_path": str(report_dir / "games.parquet"),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Game Rotation recovery {run_id}: candidates={len(rows)}, "
        f"recovered={recovered_count}, rate={recovered_count / len(rows):.1%}; "
        f"report={report_dir}"
    )


def _latest_by_game(records: list[GameBuildRecord]) -> dict[str, GameBuildRecord]:
    latest: dict[str, GameBuildRecord] = {}
    for record in sorted(records, key=lambda item: item.finished_at):
        latest[record.game_id] = record
    return latest


if __name__ == "__main__":
    main()
