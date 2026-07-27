from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from nba_lineup_model.events.schema import Event
from nba_lineup_model.lineups.reconstruct import LineupReconstruction, LineupState
from nba_lineup_model.possessions.schema import (
    Possession,
    PossessionIssue,
    PossessionReconstruction,
    PossessionSegment,
    PossessionSegmentation,
)


@dataclass
class _OpenSegment:
    start_event: Event
    state: LineupState
    start_reason: Literal["possession_start", "substitution"]
    points_home: int = 0
    points_away: int = 0
    event_count: int = 0


def build_possession_segments(
    events: Sequence[Event],
    possession_reconstruction: PossessionReconstruction,
    lineup_reconstruction: LineupReconstruction,
) -> PossessionSegmentation:
    """Intersect possessions with atomic lineup substitution boundaries."""

    if possession_reconstruction.game_id != lineup_reconstruction.game_id:
        raise ValueError("Possession and lineup reconstructions must be for the same game")
    if not events:
        raise ValueError("Cannot segment possessions from an empty event stream")

    event_by_index = {event.event_index: event for event in events}
    lineup_by_index = {
        assignment.event_index: assignment
        for assignment in lineup_reconstruction.event_lineups
    }
    assignment_by_index = {
        assignment.event_index: assignment
        for assignment in possession_reconstruction.event_possessions
    }
    segments: list[PossessionSegment] = []
    issues: list[PossessionIssue] = []

    for possession in possession_reconstruction.possessions:
        assigned_event_indexes = {
            assignment.event_index
            for assignment in possession_reconstruction.event_possessions
            if assignment.possession_index == possession.possession_index
        }
        nominal_end_event = event_by_index[possession.end_event_index]
        assigned_end_event = max(
            (event_by_index[event_index] for event_index in assigned_event_indexes),
            key=lambda event: event.source_order_number,
        )
        segment_end_event = max(
            nominal_end_event,
            assigned_end_event,
            key=lambda event: event.source_order_number,
        )
        possession_events = [
            event
            for event in events
            if possession.start_order_number
            <= event.source_order_number
            <= segment_end_event.source_order_number
        ]
        if not possession_events:
            issues.append(
                PossessionIssue(
                    code="possession_has_no_event_range",
                    severity="error",
                    detail="No events fall within the possession boundaries.",
                    possession_id=possession.possession_id,
                )
            )
            continue

        start_event = event_by_index[possession.start_event_index]
        start_lineup = lineup_by_index[possession.start_event_index].lineup_before
        open_segment = _OpenSegment(
            start_event=start_event,
            state=start_lineup,
            start_reason="possession_start",
        )
        possession_segments: list[PossessionSegment] = []
        index = 0

        while index < len(possession_events):
            event = possession_events[index]
            lineup_assignment = lineup_by_index[event.event_index]
            if event.event_type == "substitution":
                batch_id = lineup_assignment.substitution_batch_id
                if batch_id is None:
                    raise ValueError(
                        f"Substitution {event.event_id} has no lineup batch assignment"
                    )
                batch_end = index + 1
                while batch_end < len(possession_events):
                    next_event = possession_events[batch_end]
                    next_lineup = lineup_by_index[next_event.event_index]
                    if next_lineup.substitution_batch_id != batch_id:
                        break
                    batch_end += 1
                batch = possession_events[index:batch_end]
                possession_segments.append(
                    _close_segment(
                        possession,
                        open_segment,
                        batch[0],
                        segment_index=len(segments) + len(possession_segments),
                        possession_segment_index=len(possession_segments),
                        end_reason="substitution",
                        ends_possession=False,
                    )
                )
                open_segment = _OpenSegment(
                    start_event=batch[-1],
                    state=lineup_by_index[batch[-1].event_index].lineup_after,
                    start_reason="substitution",
                )
                index = batch_end
                continue

            event_possession = assignment_by_index.get(event.event_index)
            if (
                event_possession is not None
                and event_possession.possession_index == possession.possession_index
            ):
                open_segment.points_home += event.score_home_delta
                open_segment.points_away += event.score_away_delta
                open_segment.event_count += 1
            index += 1

        possession_segments.append(
            _close_segment(
                possession,
                open_segment,
                segment_end_event,
                segment_index=len(segments) + len(possession_segments),
                possession_segment_index=len(possession_segments),
                end_reason="possession_end",
                ends_possession=True,
            )
        )
        segments.extend(possession_segments)
        issues.extend(_segment_validation_issues(possession, possession_segments))

    return PossessionSegmentation(
        game_id=possession_reconstruction.game_id,
        segments=segments,
        issues=issues,
    )


def possession_segments_frame(segmentation: PossessionSegmentation) -> pd.DataFrame:
    return pd.DataFrame(segment.model_dump(mode="json") for segment in segmentation.segments)


def lineup_segment_counts(segmentation: PossessionSegmentation) -> dict[int, int]:
    return dict(Counter(segment.possession_index for segment in segmentation.segments))


def _close_segment(
    possession: Possession,
    segment: _OpenSegment,
    end_event: Event,
    *,
    segment_index: int,
    possession_segment_index: int,
    end_reason: Literal["substitution", "possession_end"],
    ends_possession: bool,
) -> PossessionSegment:
    duration = end_event.elapsed_game_seconds - segment.start_event.elapsed_game_seconds
    if duration < 0:
        raise ValueError(
            f"Segment {segment_index} has negative duration at {end_event.event_id}"
        )
    offense_points = (
        segment.points_home
        if possession.offense_team_id == segment.state.home_team_id
        else segment.points_away
    )
    return PossessionSegment(
        game_id=possession.game_id,
        segment_id=f"{possession.possession_id}:{possession_segment_index:02d}",
        segment_index=segment_index,
        possession_id=possession.possession_id,
        possession_index=possession.possession_index,
        possession_segment_index=possession_segment_index,
        period=possession.period,
        period_type=possession.period_type,
        offense_team_id=possession.offense_team_id,
        defense_team_id=possession.defense_team_id,
        home_player_ids=segment.state.home_player_ids,
        away_player_ids=segment.state.away_player_ids,
        start_event_id=segment.start_event.event_id,
        end_event_id=end_event.event_id,
        start_event_index=segment.start_event.event_index,
        end_event_index=end_event.event_index,
        start_order_number=segment.start_event.source_order_number,
        end_order_number=end_event.source_order_number,
        start_clock=segment.start_event.clock,
        end_clock=end_event.clock,
        start_elapsed_game_seconds=segment.start_event.elapsed_game_seconds,
        end_elapsed_game_seconds=end_event.elapsed_game_seconds,
        duration_seconds=duration,
        points_home=segment.points_home,
        points_away=segment.points_away,
        offense_points=offense_points,
        event_count=segment.event_count,
        start_reason=segment.start_reason,
        end_reason=end_reason,
        starts_possession=possession_segment_index == 0,
        ends_possession=ends_possession,
    )


def _segment_validation_issues(
    possession: Possession,
    segments: Sequence[PossessionSegment],
) -> list[PossessionIssue]:
    issues: list[PossessionIssue] = []
    points_home = sum(segment.points_home for segment in segments)
    points_away = sum(segment.points_away for segment in segments)
    if (points_home, points_away) != (possession.points_home, possession.points_away):
        issues.append(
            PossessionIssue(
                code="segment_score_conservation_failed",
                severity="error",
                detail=(
                    f"Segments account for {points_home}-{points_away}; possession "
                    f"requires {possession.points_home}-{possession.points_away}."
                ),
                possession_id=possession.possession_id,
            )
        )

    segment_duration = sum(segment.duration_seconds for segment in segments)
    if abs(segment_duration - possession.duration_seconds) > 0.01:
        issues.append(
            PossessionIssue(
                code="segment_duration_conservation_failed",
                severity="error",
                detail=(
                    f"Segments account for {segment_duration:.2f}s; possession "
                    f"requires {possession.duration_seconds:.2f}s."
                ),
                possession_id=possession.possession_id,
            )
        )

    if not segments[0].starts_possession or not segments[-1].ends_possession:
        issues.append(
            PossessionIssue(
                code="incomplete_possession_segmentation",
                severity="error",
                detail="Segment boundaries do not cover the full possession.",
                possession_id=possession.possession_id,
            )
        )
    return issues
