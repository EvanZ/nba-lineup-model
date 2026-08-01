"""Reproducible audits over the locally retained historical raw cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_lineup_model.audit.runner import (
    _exception_result,
    _write_parquet,
    audit_reconstruction,
    audit_results_frame,
    audit_summary_frame,
)
from nba_lineup_model.audit.schema import AuditGameResult, AuditGameSpec
from nba_lineup_model.build_game import reconstruct_game_payloads
from nba_lineup_model.season.schema import CatalogGame
from nba_lineup_model.season.source import load_game_source_documents
from nba_lineup_model.season.storage import read_game_catalog

DEFAULT_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25")


def audit_cached_history(
    games: Iterable[CatalogGame],
    *,
    raw_dir: Path | str = Path("data/raw"),
    output_dir: Path | str,
) -> tuple[list[AuditGameResult], dict[str, Path]]:
    """Audit local source-selected raw data and persist complete provenance."""

    root = Path(output_dir)
    results: list[AuditGameResult] = []
    source_rows: list[dict[str, object]] = []
    for game in games:
        spec = AuditGameSpec(
            game_id=game.game_id,
            season=game.season,
            season_type=game.season_type,
            sample_group="historical_regular",
            expected_overtime=game.is_overtime,
        )
        try:
            documents = load_game_source_documents(game.game_id, raw_dir=raw_dir)
            source_rows.append(
                {
                    "game_id": game.game_id,
                    "play_by_play_source": documents.play_by_play.source,
                    "play_by_play_sha256": documents.play_by_play.sha256,
                    "play_by_play_bytes": documents.play_by_play.byte_count,
                    "boxscore_source": documents.boxscore.source,
                    "boxscore_sha256": documents.boxscore.sha256,
                    "boxscore_bytes": documents.boxscore.byte_count,
                }
            )
            reconstruction = reconstruct_game_payloads(
                documents.play_by_play.payload, documents.boxscore.payload
            )
            results.append(audit_reconstruction(spec, reconstruction, documents.boxscore.payload))
        except Exception as exc:
            results.append(_exception_result(spec, "reconstruct", exc))

    paths = {
        "games": root / "games.parquet",
        "summary": root / "summary.parquet",
        "sources": root / "sources.parquet",
        "manifest": root / "manifest.json",
    }
    _write_parquet(audit_results_frame(results), paths["games"])
    _write_parquet(audit_summary_frame(results), paths["summary"])
    _write_parquet(pd.DataFrame(source_rows), paths["sources"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "game_count": len(results),
        "status_counts": dict(sorted(Counter(result.status for result in results).items())),
        "raw_dir": str(Path(raw_dir)),
        "outputs": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return results, paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit locally cached historical NBA regular-season games."
    )
    parser.add_argument("--catalog-path", default="data/catalog/games.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--season", action="append", dest="seasons")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    seasons = tuple(args.seasons or DEFAULT_SEASONS)
    catalog = read_game_catalog(args.catalog_path)
    games = [
        game
        for game in catalog.games
        if game.season in seasons and game.season_type == "regular"
    ]
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--offset must be non-negative and --limit must be positive")
    end = None if args.limit is None else args.offset + args.limit
    games = games[args.offset:end]
    results, paths = audit_cached_history(
        games, raw_dir=args.raw_dir, output_dir=args.output_dir
    )
    statuses = Counter(result.status for result in results)
    print(
        f"{len(results)} games: {statuses['pass']} passed, {statuses['warning']} warnings, "
        f"{statuses['fail']} failed, {statuses['error']} errors"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
