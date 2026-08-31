"""Materialize historical team travel features from the canonical game catalog."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nba_lineup_model.modeling.schema import ArtifactRecord

DEFAULT_CATALOG_PATH = Path("data/catalog/games.parquet")
DEFAULT_OUTPUT_DIR = Path("data/analytical/team_game_travel")
COMPETITIVE_SEASON_TYPES = frozenset({"regular", "play_in", "playoffs", "nba_cup_final"})
ACTIVE_GAME_STATUSES = frozenset({"final", "live", "scheduled"})
MAX_TRAVEL_MILES = 2_500.0
TRAVEL_WINDOW_HOURS = 48.0

# Arena-area coordinates, keyed by the historical tricode retained in the
# canonical schedule. The schedule does not identify neutral-site venues, so
# a nominal home game is necessarily represented by its listed home venue.
VENUE_COORDINATES: dict[str, tuple[float, float]] = {
    "ATL": (33.7573, -84.3963),
    "BKN": (40.6826, -73.9754),
    "BOS": (42.3662, -71.0621),
    "CHA": (35.2251, -80.8392),
    "CHH": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742),
    "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103),
    "DEN": (39.7487, -105.0077),
    "DET": (42.3314, -83.0458),
    "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621),
    "IND": (39.7639, -86.1555),
    "LAC": (34.0430, -118.2673),
    "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0506),
    "MIA": (25.7814, -80.1870),
    "MIL": (43.0451, -87.9173),
    "MIN": (44.9795, -93.2760),
    "NJN": (40.7330, -74.1710),
    "NOH": (29.9490, -90.0821),
    "NOK": (35.4634, -97.5151),
    "NOP": (29.9490, -90.0821),
    "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3840),
    "PHI": (39.9012, -75.1720),
    "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668),
    "SAC": (38.5802, -121.4997),
    "SAS": (29.4270, -98.4375),
    "SEA": (47.6221, -122.3540),
    "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011),
    "VAN": (49.2778, -123.1088),
    "WAS": (38.8981, -77.0209),
}
REQUIRED_CATALOG_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "game_time_utc",
    "game_status",
    "home_team_id",
    "home_team_tricode",
    "away_team_id",
    "away_team_tricode",
}
TRAVEL_COLUMNS = (
    "game_id",
    "season",
    "season_type",
    "team_id",
    "team_tricode",
    "opponent_team_id",
    "opponent_team_tricode",
    "is_home",
    "game_time_utc",
    "venue_tricode",
    "venue_latitude",
    "venue_longitude",
    "previous_game_id",
    "previous_game_time_utc",
    "previous_venue_tricode",
    "previous_venue_latitude",
    "previous_venue_longitude",
    "has_prior_competitive_game",
    "hours_since_previous_tipoff",
    "travel_miles",
    "travel_miles_capped",
    "travel_within_48_miles",
    "travel_within_48_thousand_miles",
)
TRAVEL_GAME_FEATURE_COLUMNS = (
    "game_id",
    "home_short_rest_travel_thousand_miles",
    "away_short_rest_travel_thousand_miles",
    "home_minus_away_short_rest_travel_thousand_miles",
    "has_complete_short_rest_travel",
)


class TeamGameTravelManifest(BaseModel):
    """Integrity and source contract for the team-game travel mart."""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    created_at: datetime
    builder_code_version: str
    source_catalog_path: str
    source_catalog_sha256: str
    travel_window_hours: float
    travel_miles_cap: float
    row_count: int
    game_count: int
    season_count: int
    venue_tricodes: tuple[str, ...]
    artifacts: tuple[ArtifactRecord, ...]


def build_team_game_travel_mart(
    *,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[TeamGameTravelManifest, Path]:
    """Build one historical travel row per competitive team-game."""

    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    catalog = pd.read_parquet(catalog_path)
    travel = build_team_game_travel_features(catalog)
    coverage = _season_coverage(travel)
    return _write_mart(
        travel=travel,
        coverage=coverage,
        catalog_path=catalog_path,
        output_dir=output_dir,
    )


def build_team_game_travel_features(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return audited travel features from prior scheduled game locations.

    The interval is measured between scheduled UTC tipoffs, which is the
    timestamp available uniformly in the historical canonical catalog. It is
    not an estimate of the time between a prior final buzzer and current tipoff.
    """

    missing = REQUIRED_CATALOG_COLUMNS - set(catalog)
    if missing:
        raise ValueError(f"Game catalog lacks travel columns: {sorted(missing)}")
    games = catalog.loc[:, sorted(REQUIRED_CATALOG_COLUMNS)].copy()
    games = games.loc[games["season_type"].isin(COMPETITIVE_SEASON_TYPES)].copy()
    games = games.loc[games["game_status"].isin(ACTIVE_GAME_STATUSES)].copy()
    if games.empty:
        raise ValueError("Game catalog has no competitive scheduled games")
    if games["game_id"].duplicated().any():
        raise ValueError("Game catalog must have one row per game")
    games["game_id"] = games["game_id"].astype(str)
    games["game_time_utc"] = pd.to_datetime(games["game_time_utc"], utc=True, errors="raise")
    if games["game_time_utc"].isna().any():
        raise ValueError("Travel mart requires a UTC tipoff timestamp for every game")

    game_columns = [
        "game_id",
        "season",
        "season_type",
        "game_time_utc",
        "home_team_id",
        "home_team_tricode",
        "away_team_id",
        "away_team_tricode",
    ]
    home = games.loc[:, game_columns].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_tricode": "team_tricode",
            "away_team_id": "opponent_team_id",
            "away_team_tricode": "opponent_team_tricode",
        }
    )
    home["is_home"] = True
    away = games.loc[
        :,
        [
            "game_id",
            "season",
            "season_type",
            "game_time_utc",
            "away_team_id",
            "away_team_tricode",
            "home_team_id",
            "home_team_tricode",
        ],
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_tricode": "team_tricode",
            "home_team_id": "opponent_team_id",
            "home_team_tricode": "opponent_team_tricode",
        }
    )
    away["is_home"] = False
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["venue_tricode"] = games.set_index("game_id").loc[
        team_games["game_id"], "home_team_tricode"
    ].to_numpy()
    _attach_venue_coordinates(team_games, prefix="venue", tricode_column="venue_tricode")
    team_games = team_games.sort_values(
        ["season", "team_id", "game_time_utc", "game_id"], kind="stable"
    ).reset_index(drop=True)

    grouped = team_games.groupby(["season", "team_id"], sort=False)
    team_games["previous_game_id"] = grouped["game_id"].shift()
    team_games["previous_game_time_utc"] = grouped["game_time_utc"].shift()
    team_games["previous_venue_tricode"] = grouped["venue_tricode"].shift()
    team_games["previous_venue_latitude"] = grouped["venue_latitude"].shift()
    team_games["previous_venue_longitude"] = grouped["venue_longitude"].shift()
    team_games["has_prior_competitive_game"] = team_games["previous_game_id"].notna()
    team_games["hours_since_previous_tipoff"] = (
        (team_games["game_time_utc"] - team_games["previous_game_time_utc"]).dt.total_seconds()
        / 3600.0
    )
    prior_intervals = team_games.loc[
        team_games["has_prior_competitive_game"], "hours_since_previous_tipoff"
    ]
    if prior_intervals.le(0).any():
        raise ValueError("Team schedule is not strictly chronological within a season")

    valid = team_games["has_prior_competitive_game"]
    team_games["travel_miles"] = pd.NA
    team_games.loc[valid, "travel_miles"] = _great_circle_miles(
        team_games.loc[valid, "previous_venue_latitude"],
        team_games.loc[valid, "previous_venue_longitude"],
        team_games.loc[valid, "venue_latitude"],
        team_games.loc[valid, "venue_longitude"],
    )
    team_games["travel_miles"] = team_games["travel_miles"].astype("Float64")
    team_games["travel_miles_capped"] = team_games["travel_miles"].clip(upper=MAX_TRAVEL_MILES)
    within_window = valid & team_games["hours_since_previous_tipoff"].le(TRAVEL_WINDOW_HOURS)
    team_games["travel_within_48_miles"] = team_games["travel_miles"].where(within_window, 0.0)
    team_games.loc[~valid, "travel_within_48_miles"] = pd.NA
    team_games["travel_within_48_thousand_miles"] = (
        team_games["travel_within_48_miles"].clip(upper=MAX_TRAVEL_MILES) / 1_000.0
    )
    output = team_games.loc[:, TRAVEL_COLUMNS].sort_values(
        ["season", "game_time_utc", "game_id", "is_home"], kind="stable"
    )
    _validate_travel_features(output)
    return output.reset_index(drop=True)


def validate_team_game_travel_mart(root: Path | str) -> TeamGameTravelManifest:
    """Validate an already materialized team-game travel mart."""

    root = Path(root)
    manifest = TeamGameTravelManifest.model_validate_json((root / "_manifest.json").read_text())
    travel = pd.read_parquet(root / "team_game_travel.parquet")
    coverage = pd.read_parquet(root / "season_travel_coverage.parquet")
    if len(travel) != manifest.row_count:
        raise ValueError("Travel mart row count does not match manifest")
    if travel["game_id"].nunique() != manifest.game_count:
        raise ValueError("Travel mart game count does not match manifest")
    if travel["season"].nunique() != manifest.season_count:
        raise ValueError("Travel mart season count does not match manifest")
    if tuple(sorted(travel["venue_tricode"].unique())) != manifest.venue_tricodes:
        raise ValueError("Travel mart venue coverage does not match manifest")
    _validate_travel_features(travel)
    if len(coverage) != manifest.season_count:
        raise ValueError("Travel coverage does not contain one row per season")
    for artifact in manifest.artifacts:
        path = root / artifact.filename
        if not path.is_file() or _sha256_file(path) != artifact.sha256:
            raise ValueError(f"Travel mart artifact checksum mismatch: {artifact.filename}")
    return manifest


def build_short_rest_travel_game_features(travel: pd.DataFrame) -> pd.DataFrame:
    """Pivot nullable team travel rows into a signed game-level feature.

    The signed travel value is deliberately null when either team is in its
    season opener. Downstream modeling must choose an explicit treatment for
    that unavailable preseason-to-opening-night journey.
    """

    required = {"game_id", "is_home", "travel_within_48_thousand_miles"}
    missing = required - set(travel)
    if missing:
        raise ValueError(f"Travel mart lacks game feature columns: {sorted(missing)}")
    if travel.duplicated(["game_id", "is_home"]).any():
        raise ValueError("Travel mart must be unique by game and side")
    home = travel.loc[
        travel["is_home"], ["game_id", "travel_within_48_thousand_miles"]
    ].rename(columns={"travel_within_48_thousand_miles": TRAVEL_GAME_FEATURE_COLUMNS[1]})
    away = travel.loc[
        ~travel["is_home"], ["game_id", "travel_within_48_thousand_miles"]
    ].rename(columns={"travel_within_48_thousand_miles": TRAVEL_GAME_FEATURE_COLUMNS[2]})
    output = home.merge(away, on="game_id", how="outer", validate="one_to_one")
    if len(output) != travel["game_id"].nunique():
        raise ValueError("Travel game feature adapter lost cataloged games")
    output[TRAVEL_GAME_FEATURE_COLUMNS[4]] = output[
        [TRAVEL_GAME_FEATURE_COLUMNS[1], TRAVEL_GAME_FEATURE_COLUMNS[2]]
    ].notna().all(axis=1)
    output[TRAVEL_GAME_FEATURE_COLUMNS[3]] = (
        output[TRAVEL_GAME_FEATURE_COLUMNS[1]] - output[TRAVEL_GAME_FEATURE_COLUMNS[2]]
    )
    return output.loc[:, TRAVEL_GAME_FEATURE_COLUMNS].sort_values("game_id").reset_index(drop=True)


def _attach_venue_coordinates(frame: pd.DataFrame, *, prefix: str, tricode_column: str) -> None:
    tricodes = frame[tricode_column].astype(str)
    unknown = sorted(set(tricodes) - set(VENUE_COORDINATES))
    if unknown:
        raise ValueError("Travel mart lacks venue coordinates for: " + ", ".join(unknown))
    frame[f"{prefix}_latitude"] = tricodes.map(lambda code: VENUE_COORDINATES[code][0])
    frame[f"{prefix}_longitude"] = tricodes.map(lambda code: VENUE_COORDINATES[code][1])


def _great_circle_miles(
    latitude_one: pd.Series,
    longitude_one: pd.Series,
    latitude_two: pd.Series,
    longitude_two: pd.Series,
) -> pd.Series:
    """Vectorized haversine distance using a mean Earth radius in miles."""

    lat_one = latitude_one.astype(float).map(math.radians)
    lon_one = longitude_one.astype(float).map(math.radians)
    lat_two = latitude_two.astype(float).map(math.radians)
    lon_two = longitude_two.astype(float).map(math.radians)
    angle = (
        ((lat_two - lat_one) / 2.0).map(math.sin) ** 2
        + lat_one.map(math.cos)
        * lat_two.map(math.cos)
        * ((lon_two - lon_one) / 2.0).map(math.sin) ** 2
    )
    return 3_958.7613 * 2.0 * angle.map(lambda value: math.asin(math.sqrt(value)))


def _season_coverage(travel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season, frame in travel.groupby("season", sort=True):
        prior = frame["has_prior_competitive_game"]
        within_48 = frame.loc[prior, "travel_within_48_miles"]
        short_rest = frame.loc[prior, "hours_since_previous_tipoff"].le(TRAVEL_WINDOW_HOURS)
        rows.append(
            {
                "season": season,
                "team_game_count": len(frame),
                "game_count": frame["game_id"].nunique(),
                "team_games_with_prior": int(prior.sum()),
                "within_48_team_games": int(short_rest.sum()),
                "within_48_nonzero_travel_team_games": int(within_48.gt(0).sum()),
                "within_48_travel_miles": float(frame["travel_within_48_miles"].sum(skipna=True)),
                "maximum_travel_miles": float(frame["travel_miles"].max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def _validate_travel_features(travel: pd.DataFrame) -> None:
    if tuple(travel.columns) != TRAVEL_COLUMNS:
        raise ValueError("Travel mart column contract changed")
    if travel.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Travel mart must be unique by game and team")
    if travel.groupby("game_id")["team_id"].size().ne(2).any():
        raise ValueError("Every travel mart game must contain exactly two team rows")
    if travel["venue_latitude"].isna().any() or travel["venue_longitude"].isna().any():
        raise ValueError("Travel mart venue coordinates cannot be missing")
    prior = travel["has_prior_competitive_game"]
    prior_required = travel.loc[
        prior, ["previous_game_id", "hours_since_previous_tipoff", "travel_miles"]
    ]
    if prior_required.isna().any().any():
        raise ValueError("Prior team-games must have a prior game, interval, and distance")
    opening_required_null = travel.loc[
        ~prior, ["previous_game_id", "travel_miles", "travel_within_48_miles"]
    ]
    if opening_required_null.notna().any().any():
        raise ValueError("Season-opening team-games must preserve unavailable travel as null")
    invalid_distance = travel.loc[prior, "travel_miles"].lt(0).any()
    invalid_cap = travel.loc[prior, "travel_miles_capped"].gt(MAX_TRAVEL_MILES).any()
    if invalid_distance or invalid_cap:
        raise ValueError("Travel miles violate non-negative capped contract")


def _write_mart(
    *,
    travel: pd.DataFrame,
    coverage: pd.DataFrame,
    catalog_path: Path,
    output_dir: Path,
) -> tuple[TeamGameTravelManifest, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    backup = output_dir.parent / f".{output_dir.name}.bak-{uuid4().hex}"
    temporary.mkdir()
    try:
        outputs = {"team_game_travel.parquet": travel, "season_travel_coverage.parquet": coverage}
        for filename, frame in outputs.items():
            frame.to_parquet(temporary / filename, index=False)
        artifacts = tuple(
            ArtifactRecord(
                filename=filename,
                row_count=len(frame),
                byte_count=(temporary / filename).stat().st_size,
                sha256=_sha256_file(temporary / filename),
            )
            for filename, frame in outputs.items()
        )
        manifest = TeamGameTravelManifest(
            created_at=datetime.now(UTC),
            builder_code_version=f"sha256:{_sha256_file(Path(__file__))}",
            source_catalog_path=str(catalog_path),
            source_catalog_sha256=_sha256_file(catalog_path),
            travel_window_hours=TRAVEL_WINDOW_HOURS,
            travel_miles_cap=MAX_TRAVEL_MILES,
            row_count=len(travel),
            game_count=travel["game_id"].nunique(),
            season_count=travel["season"].nunique(),
            venue_tricodes=tuple(sorted(travel["venue_tricode"].unique())),
            artifacts=artifacts,
        )
        (temporary / "_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        validate_team_game_travel_mart(temporary)
        if output_dir.exists():
            output_dir.replace(backup)
        temporary.replace(output_dir)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest, output_dir
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not output_dir.exists():
            backup.replace(output_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build the validated historical team-game travel mart."""

    parser = argparse.ArgumentParser(description="Build the historical team-game travel mart")
    parser.add_argument("--catalog-path", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest, output = build_team_game_travel_mart(
        catalog_path=args.catalog_path,
        output_dir=args.output_dir,
    )
    print(
        f"Team-game travel mart: games={manifest.game_count:,} rows={manifest.row_count:,} "
        f"seasons={manifest.season_count} output={output}"
    )


if __name__ == "__main__":
    main()
