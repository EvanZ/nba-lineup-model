"""Season discovery, catalogs, build ledgers, and curated dataset layout."""

from nba_lineup_model.season.layout import (
    CURATED_TABLES,
    CuratedDatasetLayout,
    CuratedPartition,
)
from nba_lineup_model.season.schedule import (
    NbaScheduleClient,
    NbaScheduleError,
    ScheduleResponse,
    SeasonScheduleCache,
    catalog_from_schedule,
    replace_catalog_season,
)
from nba_lineup_model.season.schema import (
    BuildLedger,
    CatalogGame,
    GameBuildRecord,
    GameCatalog,
)
from nba_lineup_model.season.storage import (
    append_build_record,
    build_ledger_frame,
    build_ledger_from_frame,
    catalog_frame,
    catalog_from_frame,
    read_build_ledger,
    read_game_catalog,
    write_build_ledger,
    write_game_catalog,
)

__all__ = [
    "CURATED_TABLES",
    "BuildLedger",
    "CatalogGame",
    "CuratedDatasetLayout",
    "CuratedPartition",
    "GameBuildRecord",
    "GameCatalog",
    "NbaScheduleClient",
    "NbaScheduleError",
    "ScheduleResponse",
    "SeasonScheduleCache",
    "append_build_record",
    "build_ledger_frame",
    "build_ledger_from_frame",
    "catalog_frame",
    "catalog_from_frame",
    "catalog_from_schedule",
    "read_build_ledger",
    "read_game_catalog",
    "replace_catalog_season",
    "write_build_ledger",
    "write_game_catalog",
]
