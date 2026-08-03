"""Fetch Game Rotation only for unresolved period-lineup failures."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from prefect.task_runners import ThreadPoolTaskRunner

from nba_lineup_model.audit.game_rotation_probe import latest_failed_builds
from nba_lineup_model.flows.fetch_stats_history import fetch_stats_history_flow
from nba_lineup_model.ingest.nba_stats import NbaStatsEndpoint
from nba_lineup_model.season.schema import (
    CatalogGame,
    GameBuildRecord,
    validate_season,
)
from nba_lineup_model.season.storage import read_build_ledger, read_game_catalog

_PERIOD_LINEUP_ERROR_MARKERS = (
    "Period lineup remains ambiguous",
    "No legal period lineup can be inferred",
)


def unresolved_period_lineup_candidates(
    catalog: list[CatalogGame],
    failed_builds: list[GameBuildRecord],
) -> list[tuple[CatalogGame, GameBuildRecord]]:
    """Return current regular-season failures addressable by rotation evidence."""

    catalog_by_id = {game.game_id: game for game in catalog}
    candidates = []
    for record in failed_builds:
        message = record.error_message or ""
        if not any(marker in message for marker in _PERIOD_LINEUP_ERROR_MARKERS):
            continue
        game = catalog_by_id.get(record.game_id)
        if game is not None:
            candidates.append((game, record))
    return candidates


def ordered_candidate_seasons(
    candidates: list[tuple[CatalogGame, GameBuildRecord]],
    *,
    max_season: str | None = None,
    reverse: bool = False,
) -> list[str]:
    """Return selected candidate seasons in reproducible chronological order."""

    maximum = validate_season(max_season) if max_season is not None else None
    seasons = {
        game.season
        for game, _record in candidates
        if maximum is None or game.season <= maximum
    }
    return sorted(seasons, reverse=reverse)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Game Rotation for unresolved regular-season period-lineup failures."
    )
    parser.add_argument("--catalog", default="data/catalog/games.parquet")
    parser.add_argument("--builds", default="data/manifests/builds.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifests/stats_fetches.parquet")
    parser.add_argument("--output-dir", default="artifacts/reports/game_rotation_fetch")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-season",
        help="Latest season to include, such as 2018-19",
    )
    parser.add_argument(
        "--reverse-seasons",
        action="store_true",
        help="Fetch one season at a time from latest to earliest",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--min-request-interval", type=float, default=1.0)
    parser.add_argument("--request-interval-jitter", type=float, default=0.25)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    catalog = read_game_catalog(args.catalog).games
    failures = latest_failed_builds(read_build_ledger(args.builds).records)
    candidates = unresolved_period_lineup_candidates(catalog, failures)
    seasons = ordered_candidate_seasons(
        candidates,
        max_season=args.max_season,
        reverse=args.reverse_seasons,
    )
    selected_seasons = set(seasons)
    candidates = [
        (game, record)
        for game, record in candidates
        if game.season in selected_seasons
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("No unresolved regular-season period-lineup failures are available")

    started_at = datetime.now(UTC)
    run_id = args.run_id or f"game-rotation-fetch-{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    report_dir = Path(args.output_dir) / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        {
            "game_id": game.game_id,
            "season": game.season,
            "game_date": game.game_date,
            "failure_stage": record.terminal_stage,
            "failure_message": record.error_message,
        }
        for game, record in candidates
    ]
    pd.DataFrame(rows).to_parquet(report_dir / "selection.parquet", index=False)

    configured_flow = fetch_stats_history_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )
    candidate_groups = (
        [
            (season, [candidate for candidate in candidates if candidate[0].season == season])
            for season in seasons
        ]
        if args.reverse_seasons
        else [("all", candidates)]
    )
    summaries = []
    for season, group in candidate_groups:
        summaries.append(
            configured_flow(
                sorted({game.season for game, _record in group})
                if season == "all"
                else [season],
                catalog_path=args.catalog,
                raw_dir=args.raw_dir,
                manifest_path=args.manifest,
                season_types=["regular"],
                game_ids=[game.game_id for game, _record in group],
                endpoints=[NbaStatsEndpoint.GAME_ROTATION],
                run_id=f"{run_id}-{season}" if season != "all" else run_id,
                min_request_interval_seconds=args.min_request_interval,
                request_interval_jitter_seconds=args.request_interval_jitter,
                max_retries=0,
            )
        )
    succeeded_count = sum(summary.succeeded_count for summary in summaries)
    skipped_count = sum(summary.skipped_count for summary in summaries)
    failed_count = sum(summary.failed_count for summary in summaries)
    report = {
        "run_id": run_id,
        "candidate_count": len(rows),
        "selection_path": str(report_dir / "selection.parquet"),
        "fetch_summaries": [summary.model_dump(mode="json") for summary in summaries],
    }
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Game Rotation fetch {run_id}: candidates={len(rows)}, "
        f"succeeded={succeeded_count}, skipped={skipped_count}, "
        f"failed={failed_count}; report={report_dir}"
    )


if __name__ == "__main__":
    main()
