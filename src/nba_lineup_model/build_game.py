from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.events import canonical_events, event_records_frame
from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache
from nba_lineup_model.lineups import (
    event_lineups_frame,
    lineup_stints_frame,
    reconstruct_lineups,
)
from nba_lineup_model.normalize.boxscore import boxscore_players_frame


@dataclass(frozen=True)
class ProcessedGame:
    game_id: str
    event_count: int
    stint_count: int
    issue_count: int
    output_paths: dict[str, Path]


def process_game_payloads(
    play_by_play_payload: Mapping[str, Any],
    boxscore_payload: Mapping[str, Any],
    *,
    output_root: Path | str = Path("data/processed"),
) -> ProcessedGame:
    """Build and persist canonical event and lineup tables for one game."""

    events = canonical_events(play_by_play_payload)
    reconstruction = reconstruct_lineups(events, boxscore_payload)
    game_id = events[0].game_id
    root = Path(output_root)

    frames = {
        "events": event_records_frame(events),
        "players": boxscore_players_frame(boxscore_payload),
        "event_lineups": event_lineups_frame(reconstruction),
        "lineup_stints": lineup_stints_frame(reconstruction),
    }
    output_paths: dict[str, Path] = {}
    for table_name, frame in frames.items():
        path = root / table_name / f"{game_id}.parquet"
        _write_parquet(frame, path)
        output_paths[table_name] = path

    return ProcessedGame(
        game_id=game_id,
        event_count=len(events),
        stint_count=len(reconstruction.stints),
        issue_count=len(reconstruction.issues),
        output_paths=output_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and process one NBA game into event and lineup Parquet tables."
    )
    parser.add_argument("game_id", help="10-digit NBA game ID, e.g. 0022000180")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw JSON cache directory")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Processed Parquet output directory",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached NBA responses")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = NbaCdnClient(cache=RawJsonCache(Path(args.raw_dir)))
    use_cache = not args.refresh
    play_by_play = client.fetch_play_by_play(args.game_id, use_cache=use_cache)
    boxscore = client.fetch_boxscore(args.game_id, use_cache=use_cache)
    result = process_game_payloads(
        play_by_play.payload,
        boxscore.payload,
        output_root=Path(args.processed_dir),
    )

    print(
        f"{result.game_id}: {result.event_count} events, "
        f"{result.stint_count} lineup stints, {result.issue_count} validation issues"
    )
    for table_name, path in result.output_paths.items():
        print(f"{table_name}: {path}")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
