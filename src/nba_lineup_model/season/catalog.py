from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.season.storage import catalog_from_frame, write_game_catalog


def read_catalog_source(path: Path) -> pd.DataFrame:
    """Read a canonical catalog source from Parquet or CSV."""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(
            path,
            dtype={
                "game_id": "string",
                "season": "string",
                "season_type": "string",
                "game_status": "string",
                "home_team_tricode": "string",
                "away_team_tricode": "string",
                "source_status_text": "string",
                "source_url": "string",
            },
        )
        for column in ("game_date", "game_time_utc", "source_fetched_at"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=column != "game_date")
        if "game_date" in frame:
            frame["game_date"] = frame["game_date"].dt.date
        for column in (
            "schema_version",
            "home_team_id",
            "away_team_id",
            "period_count",
            "source_status_code",
        ):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column]).astype("Int64")
        if "is_overtime" in frame:
            frame["is_overtime"] = frame["is_overtime"].astype("boolean")
        return frame
    raise ValueError("Catalog source must be a .parquet or .csv file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and normalize a canonical NBA game catalog."
    )
    parser.add_argument("source", help="Canonical catalog in Parquet or CSV format")
    parser.add_argument(
        "--output",
        default="data/catalog/games.parquet",
        help="Validated catalog Parquet output path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = catalog_from_frame(read_catalog_source(Path(args.source)))
    output_path = write_game_catalog(catalog, Path(args.output))
    seasons = sorted({game.season for game in catalog.games})
    print(f"{len(catalog.games)} games across {len(seasons)} seasons: {output_path}")


if __name__ == "__main__":
    main()
