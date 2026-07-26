from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.events import canonical_events, event_records_frame
from nba_lineup_model.events.schema import Event
from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache
from nba_lineup_model.lineups import (
    LineupReconstruction,
    event_lineups_frame,
    lineup_stints_frame,
    reconstruct_lineups,
)
from nba_lineup_model.normalize.boxscore import boxscore_players_frame
from nba_lineup_model.possessions import (
    PossessionReconstruction,
    PossessionSegmentation,
    build_possession_segments,
    lineup_segment_counts,
    possession_segments_frame,
    possessions_frame,
    reconstruct_possessions,
)


@dataclass(frozen=True)
class GameReconstruction:
    game_id: str
    events: list[Event]
    lineups: LineupReconstruction
    possessions: PossessionReconstruction
    possession_segments: PossessionSegmentation


@dataclass(frozen=True)
class ProcessedGame:
    game_id: str
    event_count: int
    stint_count: int
    possession_count: int
    possession_segment_count: int
    issue_count: int
    output_paths: dict[str, Path]


def reconstruct_game_payloads(
    play_by_play_payload: Mapping[str, Any],
    boxscore_payload: Mapping[str, Any],
) -> GameReconstruction:
    """Run the full in-memory reconstruction pipeline for one game."""

    events = canonical_events(play_by_play_payload)
    lineups = reconstruct_lineups(events, boxscore_payload)
    initial_lineup = lineups.event_lineups[0].lineup_before
    if initial_lineup.home_team_id is None or initial_lineup.away_team_id is None:
        raise ValueError("Lineup reconstruction did not identify both game teams")
    possessions = reconstruct_possessions(
        events,
        home_team_id=initial_lineup.home_team_id,
        away_team_id=initial_lineup.away_team_id,
    )
    possession_segments = build_possession_segments(
        events,
        possessions,
        lineups,
    )
    return GameReconstruction(
        game_id=events[0].game_id,
        events=events,
        lineups=lineups,
        possessions=possessions,
        possession_segments=possession_segments,
    )


def process_game_payloads(
    play_by_play_payload: Mapping[str, Any],
    boxscore_payload: Mapping[str, Any],
    *,
    output_root: Path | str = Path("data/processed"),
) -> ProcessedGame:
    """Build and persist canonical event, lineup, and possession tables for one game."""

    game = reconstruct_game_payloads(play_by_play_payload, boxscore_payload)
    root = Path(output_root)

    frames = {
        "events": event_records_frame(game.events),
        "players": boxscore_players_frame(boxscore_payload),
        "event_lineups": event_lineups_frame(game.lineups),
        "lineup_stints": lineup_stints_frame(game.lineups),
        "possessions": possessions_frame(
            game.possessions,
            lineup_segment_counts=lineup_segment_counts(game.possession_segments),
        ),
        "possession_segments": possession_segments_frame(game.possession_segments),
    }
    output_paths: dict[str, Path] = {}
    for table_name, frame in frames.items():
        path = root / table_name / f"{game.game_id}.parquet"
        _write_parquet(frame, path)
        output_paths[table_name] = path

    return ProcessedGame(
        game_id=game.game_id,
        event_count=len(game.events),
        stint_count=len(game.lineups.stints),
        possession_count=len(game.possessions.possessions),
        possession_segment_count=len(game.possession_segments.segments),
        issue_count=(
            len(game.lineups.issues)
            + len(game.possessions.issues)
            + len(game.possession_segments.issues)
        ),
        output_paths=output_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and process one NBA game into event, lineup, and possession "
            "Parquet tables."
        )
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
        f"{result.stint_count} lineup stints, "
        f"{result.possession_count} possessions, "
        f"{result.possession_segment_count} possession segments, "
        f"{result.issue_count} validation issues"
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
