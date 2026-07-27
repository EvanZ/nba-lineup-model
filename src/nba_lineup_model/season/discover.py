from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from nba_lineup_model.season.schedule import (
    NbaScheduleClient,
    SeasonScheduleCache,
    catalog_from_schedule,
    replace_catalog_season,
)
from nba_lineup_model.season.schema import GameCatalog
from nba_lineup_model.season.storage import read_game_catalog, write_game_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover one NBA season directly from the NBA Stats schedule endpoint."
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument(
        "--output",
        default="data/catalog/games.parquet",
        help="Canonical multi-season catalog output path",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Root directory for byte-preserved NBA responses",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore a cached schedule response and fetch it again",
    )
    parser.add_argument(
        "--replace-catalog",
        action="store_true",
        help="Write only the discovered season instead of retaining other seasons",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    client = NbaScheduleClient(cache=SeasonScheduleCache(Path(args.raw_dir)))
    response = client.fetch(args.season, use_cache=not args.refresh)
    discovered = catalog_from_schedule(response)

    catalog = discovered
    if output_path.exists() and not args.replace_catalog:
        catalog = replace_catalog_season(read_game_catalog(output_path), discovered)
    write_game_catalog(catalog, output_path)
    _print_summary(discovered, catalog, output_path)


def _print_summary(
    discovered: GameCatalog,
    catalog: GameCatalog,
    output_path: Path,
) -> None:
    season = discovered.games[0].season
    counts = Counter(game.season_type for game in discovered.games)
    breakdown = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    print(
        f"Discovered {len(discovered.games)} games for {season} "
        f"({breakdown}); catalog has {len(catalog.games)} games: {output_path}"
    )


if __name__ == "__main__":
    main()
