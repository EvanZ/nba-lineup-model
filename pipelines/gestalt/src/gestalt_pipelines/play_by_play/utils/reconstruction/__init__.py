"""NBA event, lineup and possession reconstruction owned by this pipeline.

The reconstruction rules are ported without behavioral changes. This package
contains no source fetching, table exporters, model code, or legacy imports.
"""

from dataclasses import dataclass

from .event_schema import Event
from .events import canonical_events
from .lineups import LineupReconstruction, reconstruct_lineups
from .possession_schema import PossessionReconstruction, PossessionSegmentation
from .possessions import reconstruct_possessions
from .segments import build_possession_segments


@dataclass(frozen=True)
class GameReconstruction:
    game_id: str
    events: list[Event]
    lineups: LineupReconstruction
    possessions: PossessionReconstruction
    possession_segments: PossessionSegmentation


def reconstruct_game_payloads(play_by_play_payload: dict, boxscore_payload: dict) -> GameReconstruction:
    events = canonical_events(play_by_play_payload)
    lineups = reconstruct_lineups(events, boxscore_payload)
    initial = lineups.event_lineups[0].lineup_before
    if initial.home_team_id is None or initial.away_team_id is None:
        raise ValueError("Lineup reconstruction did not identify both game teams")
    possessions = reconstruct_possessions(
        events, home_team_id=initial.home_team_id, away_team_id=initial.away_team_id,
    )
    segments = build_possession_segments(events, possessions, lineups)
    return GameReconstruction(
        game_id=events[0].game_id, events=events, lineups=lineups,
        possessions=possessions, possession_segments=segments,
    )


__all__ = ["GameReconstruction", "reconstruct_game_payloads"]
