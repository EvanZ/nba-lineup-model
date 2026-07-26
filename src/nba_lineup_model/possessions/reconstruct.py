from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from nba_lineup_model.events.schema import Event
from nba_lineup_model.possessions.schema import (
    EventPossession,
    Possession,
    PossessionIssue,
    PossessionReconstruction,
    PossessionStartReason,
    PossessionTerminalReason,
)

_FREE_THROW_SEQUENCE_RE = re.compile(
    r"^(?P<attempt>\d+)\s*of\s*(?P<total>\d+)$"
)
_ADMINISTRATIVE_EVENT_TYPES = {
    "game",
    "instantreplay",
    "period",
    "stoppage",
    "substitution",
    "timeout",
}


@dataclass
class _OpenPossession:
    offense_team_id: int
    defense_team_id: int
    start_event: Event
    start_reason: PossessionStartReason
    events: list[Event] = field(default_factory=list)
    pending_terminal: tuple[Event, PossessionTerminalReason] | None = None
    source_possession_mismatch_count: int = 0


def reconstruct_possessions(
    events: Sequence[Event],
    *,
    home_team_id: int,
    away_team_id: int,
) -> PossessionReconstruction:
    """Reconstruct team possessions from canonical NBA events."""

    if not events:
        raise ValueError("Cannot reconstruct possessions from an empty event stream")
    if home_team_id == away_team_id:
        raise ValueError("Home and away team IDs must differ")

    ordered_events = sorted(events, key=lambda event: event.source_order_number)
    if list(events) != ordered_events:
        raise ValueError("Events must be ordered by source_order_number")

    game_id = ordered_events[0].game_id
    if any(event.game_id != game_id for event in ordered_events):
        raise ValueError("All events must belong to the same game")

    team_ids = {home_team_id, away_team_id}
    possessions: list[Possession] = []
    event_possessions: list[EventPossession] = []
    issues: list[PossessionIssue] = []
    period_counts: dict[int, int] = {}
    open_possession: _OpenPossession | None = None
    period_active = False

    def other_team(team_id: int) -> int:
        return away_team_id if team_id == home_team_id else home_team_id

    def open_new(
        event: Event,
        offense_team_id: int,
        start_reason: PossessionStartReason,
    ) -> _OpenPossession:
        return _OpenPossession(
            offense_team_id=offense_team_id,
            defense_team_id=other_team(offense_team_id),
            start_event=event,
            start_reason=start_reason,
        )

    def close_current(
        current: _OpenPossession,
        end_event: Event,
        terminal_reason: PossessionTerminalReason,
    ) -> None:
        if not current.events:
            return
        period_possession_index = period_counts.get(current.start_event.period, 0)
        possession, assignments = _close_possession(
            current,
            end_event,
            terminal_reason,
            possession_index=len(possessions),
            period_possession_index=period_possession_index,
            home_team_id=home_team_id,
        )
        possessions.append(possession)
        event_possessions.extend(assignments)
        period_counts[current.start_event.period] = period_possession_index + 1

    for event in ordered_events:
        if event.event_type == "period" and event.event_subtype == "start":
            if open_possession is not None and open_possession.events:
                close_current(
                    open_possession,
                    event,
                    PossessionTerminalReason.PERIOD_END,
                )
            open_possession = None
            period_active = True
            continue

        if event.event_type == "period" and event.event_subtype == "end":
            if open_possession is not None and open_possession.events:
                if open_possession.pending_terminal is not None:
                    end_event, terminal_reason = open_possession.pending_terminal
                else:
                    end_event = event
                    terminal_reason = PossessionTerminalReason.PERIOD_END
                close_current(open_possession, end_event, terminal_reason)
            open_possession = None
            period_active = False
            continue

        if event.event_type in _ADMINISTRATIVE_EVENT_TYPES:
            continue

        if not period_active:
            issues.append(
                PossessionIssue(
                    code="event_outside_active_period",
                    severity="error",
                    detail="A substantive event occurred outside an active period.",
                    event_id=event.event_id,
                )
            )
            continue

        hint = _offense_hint(event, open_possession)
        if hint is not None and hint not in team_ids:
            issues.append(
                PossessionIssue(
                    code="unknown_possession_team",
                    severity="error",
                    detail=f"Possession hint {hint} is not a game participant.",
                    event_id=event.event_id,
                )
            )
            hint = None

        if open_possession is None:
            if hint is None:
                issues.append(
                    PossessionIssue(
                        code="unassigned_substantive_event",
                        severity="warning",
                        detail="No team signal was available to start a possession.",
                        event_id=event.event_id,
                    )
                )
                continue
            open_possession = open_new(event, hint, _start_reason(event))

        if _is_defensive_rebound(event, open_possession.offense_team_id):
            _append_event(open_possession, event)
            close_current(
                open_possession,
                event,
                PossessionTerminalReason.DEFENSIVE_REBOUND,
            )
            open_possession = open_new(
                event,
                event.team_id,
                PossessionStartReason.DEFENSIVE_REBOUND,
            )
            continue

        if _is_opponent_held_ball_recovery(event, open_possession.offense_team_id):
            _append_event(open_possession, event)
            open_possession.pending_terminal = (
                event,
                PossessionTerminalReason.HELD_BALL,
            )
            continue

        if hint is not None and hint != open_possession.offense_team_id:
            previous = open_possession
            if previous.pending_terminal is not None:
                boundary_event, terminal_reason = previous.pending_terminal
                next_start_reason = _start_reason_after_terminal(terminal_reason)
            else:
                boundary_event = event
                terminal_reason = PossessionTerminalReason.SOURCE_CHANGE
                next_start_reason = _start_reason(event)
                if previous.events:
                    issues.append(
                        PossessionIssue(
                            code="unexplained_possession_team_change",
                            severity="warning",
                            detail=(
                                f"Possession team changed from "
                                f"{previous.offense_team_id} to {hint} without an "
                                "explicit terminal event."
                            ),
                            event_id=event.event_id,
                        )
                    )
            close_current(previous, boundary_event, terminal_reason)
            open_possession = open_new(boundary_event, hint, next_start_reason)

        if (
            open_possession.pending_terminal is not None
            and open_possession.pending_terminal[1] is PossessionTerminalReason.HELD_BALL
            and hint == open_possession.offense_team_id
            and not _is_turnover(event)
        ):
            open_possession.pending_terminal = None

        _append_event(open_possession, event)

        if _is_turnover(event):
            offense_team_id = open_possession.offense_team_id
            close_current(open_possession, event, PossessionTerminalReason.TURNOVER)
            open_possession = open_new(
                event,
                other_team(offense_team_id),
                PossessionStartReason.TURNOVER,
            )
        elif _is_made_field_goal(event):
            open_possession.pending_terminal = (
                event,
                PossessionTerminalReason.MADE_FIELD_GOAL,
            )
        elif _is_made_final_nontechnical_free_throw(event):
            open_possession.pending_terminal = (
                event,
                PossessionTerminalReason.MADE_FINAL_FREE_THROW,
            )

    if open_possession is not None and open_possession.events:
        final_event = ordered_events[-1]
        if open_possession.pending_terminal is not None:
            end_event, terminal_reason = open_possession.pending_terminal
        else:
            end_event = final_event
            terminal_reason = PossessionTerminalReason.FEED_END
        close_current(open_possession, end_event, terminal_reason)
        if period_active:
            issues.append(
                PossessionIssue(
                    code="feed_ended_during_period",
                    severity="warning",
                    detail="The event feed ended without a period-end event.",
                    event_id=final_event.event_id,
                )
            )

    issues.extend(
        _validation_issues(
            ordered_events,
            possessions,
            event_possessions,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )
    )
    return PossessionReconstruction(
        game_id=game_id,
        possessions=possessions,
        event_possessions=event_possessions,
        issues=issues,
    )


def possessions_frame(
    reconstruction: PossessionReconstruction,
    *,
    lineup_segment_counts: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    rows = []
    for possession in reconstruction.possessions:
        row = possession.model_dump(mode="json")
        if lineup_segment_counts is not None:
            row["lineup_segment_count"] = lineup_segment_counts.get(
                possession.possession_index,
                0,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _append_event(possession: _OpenPossession, event: Event) -> None:
    possession.events.append(event)
    if (
        event.source_possession_team_id is not None
        and event.source_possession_team_id != possession.offense_team_id
    ):
        possession.source_possession_mismatch_count += 1


def _close_possession(
    possession: _OpenPossession,
    end_event: Event,
    terminal_reason: PossessionTerminalReason,
    *,
    possession_index: int,
    period_possession_index: int,
    home_team_id: int,
) -> tuple[Possession, list[EventPossession]]:
    duration = end_event.elapsed_game_seconds - possession.start_event.elapsed_game_seconds
    if duration < 0:
        raise ValueError(
            f"Possession {possession_index} has negative duration at {end_event.event_id}"
        )

    points_home = sum(event.score_home_delta for event in possession.events)
    points_away = sum(event.score_away_delta for event in possession.events)
    offense_points = (
        points_home if possession.offense_team_id == home_team_id else points_away
    )
    opponent_points = (
        points_away if possession.offense_team_id == home_team_id else points_home
    )
    flags: list[str] = []
    if opponent_points:
        opponent_scoring_events = [
            event
            for event in possession.events
            if (
                event.score_away_delta
                if possession.offense_team_id == home_team_id
                else event.score_home_delta
            )
        ]
        if opponent_scoring_events and all(
            _is_technical_free_throw(event) for event in opponent_scoring_events
        ):
            flags.append("opponent_technical_free_throw")
        else:
            flags.append("opponent_scored")
    if points_home < 0 or points_away < 0:
        flags.append("negative_score_delta")
    if possession.source_possession_mismatch_count:
        flags.append("source_possession_mismatch")

    possession_id = f"{possession.start_event.game_id}:{possession_index:04d}"
    record = Possession(
        game_id=possession.start_event.game_id,
        possession_id=possession_id,
        possession_index=possession_index,
        period_possession_index=period_possession_index,
        period=possession.start_event.period,
        period_type=possession.start_event.period_type,
        offense_team_id=possession.offense_team_id,
        defense_team_id=possession.defense_team_id,
        start_event_id=possession.start_event.event_id,
        end_event_id=end_event.event_id,
        start_event_index=possession.start_event.event_index,
        end_event_index=end_event.event_index,
        start_order_number=possession.start_event.source_order_number,
        end_order_number=end_event.source_order_number,
        start_clock=possession.start_event.clock,
        end_clock=end_event.clock,
        start_elapsed_game_seconds=possession.start_event.elapsed_game_seconds,
        end_elapsed_game_seconds=end_event.elapsed_game_seconds,
        duration_seconds=duration,
        points_home=points_home,
        points_away=points_away,
        offense_points=offense_points,
        event_count=len(possession.events),
        start_reason=possession.start_reason,
        terminal_reason=terminal_reason,
        source_possession_mismatch_count=possession.source_possession_mismatch_count,
        validation_flags=tuple(flags),
    )
    assignments = [
        EventPossession(
            game_id=event.game_id,
            event_id=event.event_id,
            event_index=event.event_index,
            possession_id=possession_id,
            possession_index=possession_index,
            offense_team_id=possession.offense_team_id,
        )
        for event in possession.events
    ]
    return record, assignments


def _offense_hint(
    event: Event,
    possession: _OpenPossession | None,
) -> int | None:
    if _is_post_score_loose_ball_foul(event, possession):
        return possession.offense_team_id
    if event.event_type in {"2pt", "3pt", "rebound", "steal", "turnover", "jumpball"}:
        return event.team_id or event.source_possession_team_id
    if event.event_type == "freethrow":
        if event.descriptor == "technical":
            return (
                possession.offense_team_id
                if possession is not None
                else event.source_possession_team_id
            )
        return event.team_id or event.source_possession_team_id
    if event.event_type == "foul" and event.event_subtype == "offensive":
        return event.team_id or event.source_possession_team_id
    return event.source_possession_team_id or (
        possession.offense_team_id if possession is not None else None
    )


def _start_reason(event: Event) -> PossessionStartReason:
    if event.event_type == "steal":
        return PossessionStartReason.TURNOVER
    if event.event_type == "jumpball":
        return PossessionStartReason.JUMP_BALL
    if event.event_type == "rebound" and event.event_subtype == "defensive":
        return PossessionStartReason.DEFENSIVE_REBOUND
    return PossessionStartReason.SOURCE_SIGNAL


def _start_reason_after_terminal(
    terminal_reason: PossessionTerminalReason,
) -> PossessionStartReason:
    if terminal_reason is PossessionTerminalReason.HELD_BALL:
        return PossessionStartReason.JUMP_BALL
    return PossessionStartReason.MADE_SCORE


def _is_turnover(event: Event) -> bool:
    return event.event_type == "turnover"


def _is_post_score_loose_ball_foul(
    event: Event,
    possession: _OpenPossession | None,
) -> bool:
    if (
        possession is None
        or possession.pending_terminal is None
        or possession.pending_terminal[1] is not PossessionTerminalReason.MADE_FIELD_GOAL
        or event.event_type != "foul"
        or (event.descriptor or "").replace(" ", "").replace("-", "") != "looseball"
    ):
        return False
    made_field_goal = possession.pending_terminal[0]
    return event.elapsed_game_seconds == made_field_goal.elapsed_game_seconds


def _is_defensive_rebound(event: Event, offense_team_id: int) -> bool:
    return (
        event.event_type == "rebound"
        and event.event_subtype == "defensive"
        and event.team_id is not None
        and event.team_id != offense_team_id
    )


def _is_opponent_held_ball_recovery(event: Event, offense_team_id: int) -> bool:
    return (
        event.event_type == "jumpball"
        and event.descriptor == "heldball"
        and event.team_id is not None
        and event.team_id != offense_team_id
    )


def _is_made_field_goal(event: Event) -> bool:
    return (
        event.event_type in {"2pt", "3pt"}
        and event.shot_result == "Made"
    )


def _is_made_final_nontechnical_free_throw(event: Event) -> bool:
    if (
        event.event_type != "freethrow"
        or event.shot_result != "Made"
        or event.descriptor == "technical"
        or event.event_subtype is None
    ):
        return False
    match = _FREE_THROW_SEQUENCE_RE.fullmatch(event.event_subtype)
    return match is not None and match.group("attempt") == match.group("total")


def _is_technical_free_throw(event: Event) -> bool:
    return event.event_type == "freethrow" and event.descriptor == "technical"


def _validation_issues(
    events: Sequence[Event],
    possessions: Sequence[Possession],
    assignments: Sequence[EventPossession],
    *,
    home_team_id: int,
    away_team_id: int,
) -> list[PossessionIssue]:
    issues: list[PossessionIssue] = []
    assignment_by_event = {assignment.event_index: assignment for assignment in assignments}
    if len(assignment_by_event) != len(assignments):
        issues.append(
            PossessionIssue(
                code="duplicate_event_assignment",
                severity="error",
                detail="At least one event was assigned to multiple possessions.",
            )
        )

    for event in events:
        if (
            event.score_home_delta or event.score_away_delta
        ) and event.event_type not in _ADMINISTRATIVE_EVENT_TYPES:
            if event.event_index not in assignment_by_event:
                issues.append(
                    PossessionIssue(
                        code="unassigned_scoring_event",
                        severity="error",
                        detail="A scoring event was not assigned to a possession.",
                        event_id=event.event_id,
                    )
                )

    first_substantive = next(
        (event for event in events if event.event_type not in _ADMINISTRATIVE_EVENT_TYPES),
        None,
    )
    if first_substantive is not None:
        initial_home = first_substantive.score_home - first_substantive.score_home_delta
        initial_away = first_substantive.score_away - first_substantive.score_away_delta
        expected_home = events[-1].score_home - initial_home
        expected_away = events[-1].score_away - initial_away
        actual_home = sum(possession.points_home for possession in possessions)
        actual_away = sum(possession.points_away for possession in possessions)
        if (actual_home, actual_away) != (expected_home, expected_away):
            issues.append(
                PossessionIssue(
                    code="score_conservation_failed",
                    severity="error",
                    detail=(
                        f"Possessions account for {actual_home}-{actual_away}; "
                        f"event stream requires {expected_home}-{expected_away}."
                    ),
                )
            )

    for possession in possessions:
        if "opponent_scored" in possession.validation_flags:
            issues.append(
                PossessionIssue(
                    code="opponent_scored_during_possession",
                    severity="error",
                    detail="Points were credited to the defensive team.",
                    possession_id=possession.possession_id,
                )
            )
        if possession.offense_team_id not in {home_team_id, away_team_id}:
            issues.append(
                PossessionIssue(
                    code="unknown_offense_team",
                    severity="error",
                    detail=f"Unknown offense team {possession.offense_team_id}.",
                    possession_id=possession.possession_id,
                )
            )

    for previous, current in zip(possessions, possessions[1:], strict=False):
        if (
            previous.period == current.period
            and previous.offense_team_id == current.offense_team_id
        ):
            issues.append(
                PossessionIssue(
                    code="same_offense_consecutive_possessions",
                    severity="warning",
                    detail="Consecutive possessions in one period have the same offense.",
                    possession_id=current.possession_id,
                )
            )
        if previous.end_order_number > current.start_order_number:
            issues.append(
                PossessionIssue(
                    code="overlapping_possessions",
                    severity="error",
                    detail="Possession time boundaries overlap.",
                    possession_id=current.possession_id,
                )
            )

    return issues
