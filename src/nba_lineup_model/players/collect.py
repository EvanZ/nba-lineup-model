from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nba_lineup_model.players.normalize import (
    player_catalog_from_response,
    player_season_bios_from_response,
)
from nba_lineup_model.players.source import PlayerStatsCache, PlayerStatsClient
from nba_lineup_model.players.storage import (
    merge_player_catalogs,
    player_season_partition_dir,
    read_player_catalog,
    write_player_catalog,
    write_player_season_bios,
)


class PlayerBioCollectionSummary(BaseModel):
    """Terminal summary for one bulk player reference collection."""

    model_config = ConfigDict(strict=True, extra="forbid")

    season: str
    season_type: str
    historical_player_count: int = Field(ge=1)
    player_season_count: int = Field(ge=1)
    player_index_cache_hit: bool
    player_bios_cache_hit: bool
    player_catalog_path: str
    player_season_partition_path: str
    player_index_raw_path: str
    player_bios_raw_path: str


def collect_player_bios(
    season: str,
    *,
    season_type: str = "regular",
    raw_dir: Path | str = Path("data/raw"),
    player_catalog_path: Path | str = Path("data/catalog/players.parquet"),
    curated_dir: Path | str = Path("data/curated"),
    refresh: bool = False,
    client: PlayerStatsClient | None = None,
) -> PlayerBioCollectionSummary:
    """Fetch two bulk NBA endpoints and publish player reference datasets."""

    started_at = datetime.now(UTC)
    owns_client = client is None
    cache = PlayerStatsCache(raw_dir)
    active_client = client or PlayerStatsClient(cache=cache)
    try:
        player_index = active_client.fetch_player_index(
            season,
            use_cache=not refresh,
        )
        player_bios = active_client.fetch_player_season_bios(
            season,
            season_type=season_type,
            use_cache=not refresh,
        )
    finally:
        if owns_client:
            active_client.close()

    source_catalog = player_catalog_from_response(player_index)
    bios = player_season_bios_from_response(player_bios, source_catalog)
    catalog_path = Path(player_catalog_path)
    catalog = source_catalog
    if catalog_path.exists():
        catalog = merge_player_catalogs(
            read_player_catalog(catalog_path),
            source_catalog,
        )
    catalog_path = write_player_catalog(catalog, player_catalog_path)
    write_player_season_bios(bios, curated_dir)
    index_raw_path = active_client.cache.path_for(
        player_index.endpoint,
        player_index.season,
        player_index.season_type,
    )
    bios_raw_path = active_client.cache.path_for(
        player_bios.endpoint,
        player_bios.season,
        player_bios.season_type,
    )
    return PlayerBioCollectionSummary(
        season=season,
        season_type=season_type,
        historical_player_count=len(catalog.players),
        player_season_count=len(bios.players),
        player_index_cache_hit=player_index.fetched_at < started_at,
        player_bios_cache_hit=player_bios.fetched_at < started_at,
        player_catalog_path=str(catalog_path),
        player_season_partition_path=str(
            player_season_partition_dir(season, season_type, curated_dir)
        ),
        player_index_raw_path=str(index_raw_path),
        player_bios_raw_path=str(bios_raw_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect historical player identities and leakage-safe season bios "
            "directly from NBA Stats."
        )
    )
    parser.add_argument("season", help="NBA season in YYYY-YY format")
    parser.add_argument(
        "--season-type",
        default="regular",
        choices=("regular", "playoffs", "preseason", "all_star"),
        help="Competition type for the season bio table",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Byte-preserved NBA response cache root",
    )
    parser.add_argument(
        "--player-catalog",
        default="data/catalog/players.parquet",
        help="Historical player identity catalog path",
    )
    parser.add_argument(
        "--curated-dir",
        default="data/curated",
        help="Curated player-season dataset root",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass both validated raw caches",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = collect_player_bios(
        args.season,
        season_type=args.season_type,
        raw_dir=args.raw_dir,
        player_catalog_path=args.player_catalog,
        curated_dir=args.curated_dir,
        refresh=args.refresh,
    )
    print(
        f"{summary.season} {summary.season_type} player bios: "
        f"historical={summary.historical_player_count}, "
        f"player_seasons={summary.player_season_count}, "
        f"index_cache_hit={summary.player_index_cache_hit}, "
        f"bios_cache_hit={summary.player_bios_cache_hit}; "
        f"catalog={summary.player_catalog_path}, "
        f"partition={summary.player_season_partition_path}"
    )


if __name__ == "__main__":
    main()
