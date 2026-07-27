"""Direct NBA player reference ingestion and normalized bio datasets."""

from nba_lineup_model.players.collect import (
    PlayerBioCollectionSummary,
    collect_player_bios,
)
from nba_lineup_model.players.normalize import (
    player_catalog_from_response,
    player_season_bios_from_response,
)
from nba_lineup_model.players.schema import (
    PlayerCatalog,
    PlayerIdentity,
    PlayerSeasonBio,
    PlayerSeasonBioDataset,
    PlayerSeasonBioManifest,
)
from nba_lineup_model.players.source import (
    PlayerStatsCache,
    PlayerStatsClient,
    PlayerStatsEndpoint,
    PlayerStatsError,
    PlayerStatsResponse,
)
from nba_lineup_model.players.storage import (
    player_catalog_frame,
    player_catalog_from_frame,
    player_season_bio_frame,
    player_season_bios_from_frame,
    player_season_partition_dir,
    read_player_catalog,
    read_player_season_bios,
    read_player_season_manifest,
    validate_player_season_partition,
    write_player_catalog,
    write_player_season_bios,
)

__all__ = [
    "PlayerBioCollectionSummary",
    "PlayerCatalog",
    "PlayerIdentity",
    "PlayerSeasonBio",
    "PlayerSeasonBioDataset",
    "PlayerSeasonBioManifest",
    "PlayerStatsCache",
    "PlayerStatsClient",
    "PlayerStatsEndpoint",
    "PlayerStatsError",
    "PlayerStatsResponse",
    "collect_player_bios",
    "player_catalog_frame",
    "player_catalog_from_frame",
    "player_catalog_from_response",
    "player_season_bio_frame",
    "player_season_bios_from_frame",
    "player_season_bios_from_response",
    "player_season_partition_dir",
    "read_player_catalog",
    "read_player_season_bios",
    "read_player_season_manifest",
    "validate_player_season_partition",
    "write_player_catalog",
    "write_player_season_bios",
]
