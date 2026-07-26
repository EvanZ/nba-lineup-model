from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from nba_lineup_model.events.schema import Event


class LineupReconstructionError(RuntimeError):
    """Raised when play-by-play substitutions produce an impossible lineup."""


class LineupState(BaseModel, frozen=True):
    home_team_id: int | None = None
    away_team_id: int | None = None
    home_player_ids: tuple[int, ...]
    away_player_ids: tuple[int, ...]

    @field_validator("home_player_ids", "away_player_ids")
    @classmethod
    def validate_players(cls, player_ids: tuple[int, ...]) -> tuple[int, ...]:
        normalized = normalize_lineup(player_ids)
        return normalized

    def validate(self) -> None:
        """Retained as an explicit assertion hook for callers and tests."""

        normalize_lineup(self.home_player_ids)
        normalize_lineup(self.away_player_ids)


class EventLineup(BaseModel):
    game_id: str
    event_id: str
    event_index: int
    source_order_number: int
    period: int
    lineup_before: LineupState
    lineup_after: LineupState
    substitution_batch_id: int | None = None


class LineupStint(BaseModel):
    game_id: str
    stint_index: int = Field(ge=0)
    period: int = Field(ge=1)
    period_type: str
    start_event_index: int
    end_event_index: int
    start_order_number: int
    end_order_number: int
    start_clock: str
    end_clock: str
    start_elapsed_game_seconds: float
    end_elapsed_game_seconds: float
    duration_seconds: float = Field(ge=0)
    start_score_home: int
    start_score_away: int
    end_score_home: int
    end_score_away: int
    points_home: int
    points_away: int
    home_player_ids: tuple[int, ...]
    away_player_ids: tuple[int, ...]
    start_reason: Literal["period_start", "substitution"]
    end_reason: Literal["period_end", "substitution", "feed_end"]


class LineupIssue(BaseModel):
    code: str
    severity: Literal["warning", "error"]
    detail: str
    event_id: str | None = None
    player_id: int | None = None


class LineupReconstruction(BaseModel):
    game_id: str
    event_lineups: list[EventLineup]
    stints: list[LineupStint]
    issues: list[LineupIssue]


@dataclass
class _OpenStint:
    period: int
    period_type: str
    start_event_index: int
    start_order_number: int
    start_clock: str
    start_elapsed_game_seconds: float
    start_score_home: int
    start_score_away: int
    state: LineupState
    start_reason: Literal["period_start", "substitution"]


def normalize_lineup(player_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Return a stable five-player lineup key independent of player order."""

    normalized = tuple(sorted(int(player_id) for player_id in player_ids))
    if len(normalized) != 5:
        raise ValueError("A lineup must contain exactly five players")
    if len(set(normalized)) != 5:
        raise ValueError("A lineup cannot contain duplicate players")
    if any(player_id <= 0 for player_id in normalized):
        raise ValueError("Player IDs must be positive")
    return normalized


def starting_lineup_from_boxscore(payload: Mapping[str, Any]) -> LineupState:
    game = payload.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("Expected boxscore payload['game'] to be an object")

    home_team_id, home_starters = _team_starters(game, "homeTeam")
    away_team_id, away_starters = _team_starters(game, "awayTeam")
    return LineupState(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_player_ids=home_starters,
        away_player_ids=away_starters,
    )


def reconstruct_lineups(
    events: Sequence[Event],
    boxscore_payload: Mapping[str, Any],
    *,
    validate_boxscore_minutes: bool = True,
    minute_tolerance_seconds: float = 2.0,
) -> LineupReconstruction:
    """Reconstruct event-level lineups and stable lineup stints."""

    if not events:
        raise ValueError("Cannot reconstruct lineups from an empty event stream")

    ordered_events = sorted(events, key=lambda event: event.source_order_number)
    if list(events) != ordered_events:
        raise ValueError("Events must be ordered by source_order_number")

    game_id = ordered_events[0].game_id
    if any(event.game_id != game_id for event in ordered_events):
        raise ValueError("All events must belong to the same game")
    boxscore_game = boxscore_payload.get("game")
    if not isinstance(boxscore_game, Mapping):
        raise ValueError("Expected boxscore payload['game'] to be an object")
    if boxscore_game.get("gameId") != game_id:
        raise ValueError(
            f"Play-by-play game {game_id} does not match "
            f"boxscore game {boxscore_game.get('gameId')!r}"
        )

    state = starting_lineup_from_boxscore(boxscore_payload)
    event_lineups: list[EventLineup] = []
    stints: list[LineupStint] = []
    issues: list[LineupIssue] = []
    open_stint: _OpenStint | None = None
    period_active = False
    substitution_batch_id = 0
    index = 0

    while index < len(ordered_events):
        event = ordered_events[index]
        if event.event_type == "substitution":
            batch_end = index + 1
            while (
                batch_end < len(ordered_events)
                and ordered_events[batch_end].event_type == "substitution"
            ):
                batch_end += 1
            batch = ordered_events[index:batch_end]
            if len({batch_event.period for batch_event in batch}) != 1:
                raise LineupReconstructionError(
                    f"Substitution batch {substitution_batch_id} crosses a period boundary"
                )

            before = state
            after = _apply_substitution_batch(before, batch, substitution_batch_id)
            boundary_start = batch[0]
            boundary_end = batch[-1]

            if open_stint is not None:
                stints.append(
                    _close_stint(
                        open_stint,
                        boundary_start,
                        len(stints),
                        end_reason="substitution",
                    )
                )
                open_stint = None

            for batch_event in batch:
                event_lineups.append(
                    _event_lineup(
                        batch_event,
                        before,
                        after,
                        substitution_batch_id=substitution_batch_id,
                    )
                )

            state = after
            if period_active:
                open_stint = _open_stint(
                    boundary_end,
                    state,
                    start_reason="substitution",
                )

            substitution_batch_id += 1
            index = batch_end
            continue

        before = state
        after = state

        if event.event_type == "period" and event.event_subtype == "start":
            if period_active:
                raise LineupReconstructionError(
                    f"Period {event.period} started while period state was already active"
                )
            period_active = True
            open_stint = _open_stint(event, state, start_reason="period_start")
        elif not period_active and event.event_type not in {
            "game",
            "instantreplay",
            "stoppage",
        }:
            raise LineupReconstructionError(
                f"Event {event.event_id} occurred outside an active period"
            )

        event_lineups.append(_event_lineup(event, before, after))
        _validate_primary_actor(event, before, issues)

        if event.event_type == "period" and event.event_subtype == "end":
            if open_stint is None:
                raise LineupReconstructionError(
                    f"Period {event.period} ended without an open lineup stint"
                )
            stints.append(
                _close_stint(open_stint, event, len(stints), end_reason="period_end")
            )
            open_stint = None
            period_active = False

        index += 1

    if open_stint is not None:
        final_event = ordered_events[-1]
        stints.append(_close_stint(open_stint, final_event, len(stints), end_reason="feed_end"))
        issues.append(
            LineupIssue(
                code="feed_ended_during_period",
                severity="warning",
                detail="The event feed ended without a period-end event.",
                event_id=final_event.event_id,
            )
        )

    if validate_boxscore_minutes:
        issues.extend(
            _minute_validation_issues(
                stints,
                boxscore_payload,
                tolerance_seconds=minute_tolerance_seconds,
            )
        )

    return LineupReconstruction(
        game_id=game_id,
        event_lineups=event_lineups,
        stints=stints,
        issues=issues,
    )


def event_lineups_frame(reconstruction: LineupReconstruction) -> pd.DataFrame:
    rows = []
    for assignment in reconstruction.event_lineups:
        row = assignment.model_dump(exclude={"lineup_before", "lineup_after"})
        row.update(
            {
                "home_player_ids_before": assignment.lineup_before.home_player_ids,
                "away_player_ids_before": assignment.lineup_before.away_player_ids,
                "home_player_ids_after": assignment.lineup_after.home_player_ids,
                "away_player_ids_after": assignment.lineup_after.away_player_ids,
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "substitution_batch_id" in frame:
        frame["substitution_batch_id"] = frame["substitution_batch_id"].astype("Int64")
    return frame


def lineup_stints_frame(reconstruction: LineupReconstruction) -> pd.DataFrame:
    return pd.DataFrame(stint.model_dump() for stint in reconstruction.stints)


def _team_starters(game: Mapping[str, Any], side: str) -> tuple[int, tuple[int, ...]]:
    team = game.get(side)
    if not isinstance(team, Mapping):
        raise ValueError(f"Expected boxscore game[{side!r}] to be an object")
    team_id = int(team.get("teamId"))
    players = team.get("players")
    if not isinstance(players, list):
        raise ValueError(f"Expected boxscore game[{side!r}]['players'] to be a list")
    starters = [
        int(player["personId"])
        for player in players
        if isinstance(player, Mapping) and str(player.get("starter")) == "1"
    ]
    try:
        return team_id, normalize_lineup(starters)
    except ValueError as exc:
        raise ValueError(f"Expected exactly five unique starters for team {team_id}") from exc


def _apply_substitution_batch(
    state: LineupState,
    batch: Sequence[Event],
    batch_id: int,
) -> LineupState:
    changes: dict[int, dict[str, set[int]]] = {}
    for event in batch:
        if event.team_id is None or event.player_id is None:
            raise LineupReconstructionError(
                f"Substitution event {event.event_id} has no team or player"
            )
        if event.event_subtype not in {"in", "out"}:
            raise LineupReconstructionError(
                f"Substitution event {event.event_id} has subtype {event.event_subtype!r}"
            )
        team_changes = changes.setdefault(event.team_id, {"in": set(), "out": set()})
        if event.player_id in team_changes[event.event_subtype]:
            raise LineupReconstructionError(
                f"Duplicate substitution {event.event_subtype} for player "
                f"{event.player_id} in batch {batch_id}"
            )
        team_changes[event.event_subtype].add(event.player_id)

    home = _apply_team_changes(
        state.home_player_ids,
        changes.get(state.home_team_id, {"in": set(), "out": set()}),
        state.home_team_id,
        batch_id,
    )
    away = _apply_team_changes(
        state.away_player_ids,
        changes.get(state.away_team_id, {"in": set(), "out": set()}),
        state.away_team_id,
        batch_id,
    )
    known_team_ids = {state.home_team_id, state.away_team_id}
    unknown_team_ids = set(changes) - known_team_ids
    if unknown_team_ids:
        raise LineupReconstructionError(
            f"Substitution batch {batch_id} contains unknown teams {sorted(unknown_team_ids)}"
        )

    return LineupState(
        home_team_id=state.home_team_id,
        away_team_id=state.away_team_id,
        home_player_ids=home,
        away_player_ids=away,
    )


def _apply_team_changes(
    lineup: tuple[int, ...],
    changes: dict[str, set[int]],
    team_id: int | None,
    batch_id: int,
) -> tuple[int, ...]:
    current = set(lineup)
    cancelled = changes["out"] & changes["in"]
    outgoing = changes["out"] - cancelled
    incoming = changes["in"] - cancelled
    missing = outgoing - current
    already_on_court = incoming & current
    if missing:
        raise LineupReconstructionError(
            f"Batch {batch_id} subs out players not on court for team {team_id}: "
            f"{sorted(missing)}"
        )
    if already_on_court:
        raise LineupReconstructionError(
            f"Batch {batch_id} subs in players already on court for team {team_id}: "
            f"{sorted(already_on_court)}"
        )
    updated = (current - outgoing) | incoming
    try:
        return normalize_lineup(tuple(updated))
    except ValueError as exc:
        raise LineupReconstructionError(
            f"Batch {batch_id} leaves team {team_id} with {len(updated)} players"
        ) from exc


def _event_lineup(
    event: Event,
    before: LineupState,
    after: LineupState,
    *,
    substitution_batch_id: int | None = None,
) -> EventLineup:
    return EventLineup(
        game_id=event.game_id,
        event_id=event.event_id,
        event_index=event.event_index,
        source_order_number=event.source_order_number,
        period=event.period,
        lineup_before=before,
        lineup_after=after,
        substitution_batch_id=substitution_batch_id,
    )


def _open_stint(
    event: Event,
    state: LineupState,
    *,
    start_reason: Literal["period_start", "substitution"],
) -> _OpenStint:
    return _OpenStint(
        period=event.period,
        period_type=event.period_type,
        start_event_index=event.event_index,
        start_order_number=event.source_order_number,
        start_clock=event.clock,
        start_elapsed_game_seconds=event.elapsed_game_seconds,
        start_score_home=event.score_home,
        start_score_away=event.score_away,
        state=state,
        start_reason=start_reason,
    )


def _close_stint(
    stint: _OpenStint,
    event: Event,
    stint_index: int,
    *,
    end_reason: Literal["period_end", "substitution", "feed_end"],
) -> LineupStint:
    duration = event.elapsed_game_seconds - stint.start_elapsed_game_seconds
    if duration < 0:
        raise LineupReconstructionError(
            f"Stint {stint_index} has negative duration at event {event.event_id}"
        )
    return LineupStint(
        game_id=event.game_id,
        stint_index=stint_index,
        period=stint.period,
        period_type=stint.period_type,
        start_event_index=stint.start_event_index,
        end_event_index=event.event_index,
        start_order_number=stint.start_order_number,
        end_order_number=event.source_order_number,
        start_clock=stint.start_clock,
        end_clock=event.clock,
        start_elapsed_game_seconds=stint.start_elapsed_game_seconds,
        end_elapsed_game_seconds=event.elapsed_game_seconds,
        duration_seconds=duration,
        start_score_home=stint.start_score_home,
        start_score_away=stint.start_score_away,
        end_score_home=event.score_home,
        end_score_away=event.score_away,
        points_home=event.score_home - stint.start_score_home,
        points_away=event.score_away - stint.start_score_away,
        home_player_ids=stint.state.home_player_ids,
        away_player_ids=stint.state.away_player_ids,
        start_reason=stint.start_reason,
        end_reason=end_reason,
    )


def _validate_primary_actor(
    event: Event,
    state: LineupState,
    issues: list[LineupIssue],
) -> None:
    if event.player_id is None or event.team_id is None:
        return
    if event.event_type in {
        "game",
        "instantreplay",
        "period",
        "stoppage",
        "substitution",
        "timeout",
    }:
        return
    if event.event_type == "foul" and event.event_subtype == "technical":
        return

    if event.team_id == state.home_team_id:
        lineup = state.home_player_ids
    elif event.team_id == state.away_team_id:
        lineup = state.away_player_ids
    else:
        issues.append(
            LineupIssue(
                code="unknown_event_team",
                severity="error",
                detail=f"Event team {event.team_id} is not a game participant.",
                event_id=event.event_id,
                player_id=event.player_id,
            )
        )
        return

    if event.player_id not in lineup:
        issues.append(
            LineupIssue(
                code="primary_actor_not_on_court",
                severity="warning",
                detail=(
                    f"Player {event.player_id} is not in the reconstructed lineup "
                    f"for team {event.team_id}."
                ),
                event_id=event.event_id,
                player_id=event.player_id,
            )
        )


def _minute_validation_issues(
    stints: Sequence[LineupStint],
    boxscore_payload: Mapping[str, Any],
    *,
    tolerance_seconds: float,
) -> list[LineupIssue]:
    game = boxscore_payload.get("game")
    if not isinstance(game, Mapping):
        return []

    reconstructed: dict[int, float] = {}
    for stint in stints:
        for player_id in (*stint.home_player_ids, *stint.away_player_ids):
            reconstructed[player_id] = reconstructed.get(player_id, 0.0) + stint.duration_seconds

    issues: list[LineupIssue] = []
    for side in ("homeTeam", "awayTeam"):
        team = game.get(side)
        if not isinstance(team, Mapping):
            continue
        players = team.get("players")
        if not isinstance(players, list):
            continue
        for player in players:
            if not isinstance(player, Mapping):
                continue
            statistics = player.get("statistics")
            if not isinstance(statistics, Mapping):
                continue
            minutes = statistics.get("minutes")
            if not isinstance(minutes, str) or not minutes:
                continue
            player_id = int(player["personId"])
            from_boxscore = _parse_minutes(minutes)
            from_stints = reconstructed.get(player_id, 0.0)
            difference = from_stints - from_boxscore
            if abs(difference) > tolerance_seconds:
                issues.append(
                    LineupIssue(
                        code="boxscore_minutes_mismatch",
                        severity="warning",
                        detail=(
                            f"Reconstructed {from_stints:.2f}s versus boxscore "
                            f"{from_boxscore:.2f}s; difference {difference:+.2f}s."
                        ),
                        player_id=player_id,
                    )
                )
    return issues


def _parse_minutes(value: str) -> float:
    from nba_lineup_model.events.normalize import parse_nba_clock

    return parse_nba_clock(value)
