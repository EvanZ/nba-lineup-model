"""Recover catalog games absent from the historical RAPM stint inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from prefect.task_runners import ThreadPoolTaskRunner

from nba_lineup_model.flows.compact_season import compact_season_flow
from nba_lineup_model.flows.process_season import process_season_flow
from nba_lineup_model.modeling.stints import build_rapm_stint_dataset
from nba_lineup_model.season.fetch import select_catalog_games
from nba_lineup_model.season.storage import read_game_catalog


def unmodeled_regular_game_ids(
    season: str,
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    analytical_dir: Path | str = Path("data/analytical"),
) -> list[str]:
    """Return catalog games not represented in a season's RAPM stint partition."""

    catalog_games = select_catalog_games(
        read_game_catalog(catalog_path),
        season=season,
        season_types=["regular"],
    )
    expected = {game.game_id for game in catalog_games}
    stint_dir = Path(analytical_dir) / "rapm_stints" / season / "regular"
    part_paths = sorted(stint_dir.glob("part-*.parquet"))
    modeled: set[str] = set()
    for path in part_paths:
        modeled.update(pd.read_parquet(path, columns=["game_id"])["game_id"].astype(str))
    return sorted(expected - modeled)


def recover_season_modeling_coverage(
    season: str,
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    raw_dir: Path | str = Path("data/raw"),
    processed_dir: Path | str = Path("data/processed"),
    curated_dir: Path | str = Path("data/curated"),
    analytical_dir: Path | str = Path("data/analytical"),
    ledger_path: Path | str = Path("data/manifests/builds.parquet"),
    quality_games_path: Path | str = Path("data/quality/games.parquet"),
    quality_summary_path: Path | str = Path("data/quality/summary.parquet"),
    max_workers: int = 4,
    checkpoint_size: int = 25,
    run_id: str | None = None,
) -> dict[str, object]:
    """Reprocess only missing regular games, then publish the recovered subset."""

    missing_game_ids = unmodeled_regular_game_ids(
        season,
        catalog_path=catalog_path,
        analytical_dir=analytical_dir,
    )
    started_at = datetime.now(UTC)
    run_id = run_id or (
        f"recover-modeling-coverage-{season}-{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    result: dict[str, object] = {
        "run_id": run_id,
        "season": season,
        "started_at": started_at.isoformat(),
        "selected_missing_game_count": len(missing_game_ids),
    }
    if not missing_game_ids:
        result["status"] = "already_complete"
        result["finished_at"] = datetime.now(UTC).isoformat()
        return result

    process = process_season_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=max_workers)
    )(
        season,
        catalog_path=str(catalog_path),
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        ledger_path=str(ledger_path),
        quality_games_path=str(quality_games_path),
        quality_summary_path=str(quality_summary_path),
        season_types=["regular"],
        game_ids=missing_game_ids,
        force=True,
        checkpoint_size=checkpoint_size,
        run_id=run_id,
    )
    result["process"] = process.model_dump(mode="json")

    compact = compact_season_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=max_workers)
    )(
        season,
        catalog_path=str(catalog_path),
        processed_dir=str(processed_dir),
        curated_dir=str(curated_dir),
        ledger_path=str(ledger_path),
        quality_games_path=str(quality_games_path),
        season_types=["regular"],
        force=True,
        quality_eligible_only=True,
        run_id=run_id,
    )
    result["compact"] = compact.model_dump(mode="json")
    result["rapm_stints"] = build_rapm_stint_dataset(
        season,
        curated_dir=curated_dir,
        analytical_dir=analytical_dir,
    ).model_dump(mode="json")
    result["status"] = "completed"
    result["finished_at"] = datetime.now(UTC).isoformat()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover historical catalog games missing from regular-season RAPM stints."
        )
    )
    parser.add_argument("--season", action="append", dest="seasons", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--checkpoint-size", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default="data/audit/historical_modeling_coverage/recovery_runs",
    )
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1 or args.checkpoint_size < 1:
        raise SystemExit("--max-workers and --checkpoint-size must be positive")
    created_at = datetime.now(UTC)
    run_id = args.run_id or f"recovery-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output_path = Path(args.output_dir) / f"{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for season in args.seasons:
        record = recover_season_modeling_coverage(
            season,
            max_workers=args.max_workers,
            checkpoint_size=args.checkpoint_size,
            run_id=f"{run_id}-{season}",
        )
        records.append(record)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "created_at": created_at.isoformat(),
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"{season}: selected={record['selected_missing_game_count']}, "
            f"status={record['status']}"
        )
    print(f"Recovery manifest: {output_path}")


if __name__ == "__main__":
    main()
