from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.events.schema import Event

NBA_CLOCK_RE = re.compile(r"^PT(?:(?P<minutes>\d+)M)?(?P<seconds>\d+(?:\.\d+)?)S$")
REGULATION_PERIOD_SECONDS = 12 * 60
OVERTIME_PERIOD_SECONDS = 5 * 60


def period_duration_seconds(period: int) -> int:
    if period < 1:
        raise ValueError(f"period must be positive, got {period}")
    return REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS


def completed_period_seconds(period: int) -> int:
    regulation_periods = min(period - 1, 4)
    overtime_periods = max(period - 5, 0)
    return (
        regulation_periods * REGULATION_PERIOD_SECONDS
        + overtime_periods * OVERTIME_PERIOD_SECONDS
    )


def parse_nba_clock(clock: str) -> float:
    match = NBA_CLOCK_RE.fullmatch(clock)
    if match is None:
        raise ValueError(f"Invalid NBA clock value: {clock!r}")
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds"))
    return minutes * 60 + seconds


def format_game_clock(seconds_remaining: float) -> str:
    if seconds_remaining < 0:
        raise ValueError("seconds_remaining must be non-negative")
    minutes, seconds = divmod(seconds_remaining, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def canonical_events(payload: Mapping[str, Any]) -> list[Event]:
    """Convert NBA ``game.actions`` records into a canonical ordered event stream."""

    game = payload.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("Expected payload['game'] to be an object")

    game_id = game.get("gameId")
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("Expected payload['game']['gameId'] to be a non-empty string")

    source_actions = game.get("actions")
    if not isinstance(source_actions, list):
        raise ValueError("Expected payload['game']['actions'] to be a list")

    actions: list[Mapping[str, Any]] = []
    for action in source_actions:
        if not isinstance(action, Mapping):
            raise ValueError("Expected every play-by-play action to be an object")
        actions.append(action)

    actions.sort(key=lambda action: _required_int(action, "orderNumber"))
    order_numbers = [_required_int(action, "orderNumber") for action in actions]
    if len(order_numbers) != len(set(order_numbers)):
        raise ValueError(f"Duplicate orderNumber values in game {game_id}")

    events: list[Event] = []
    previous_home_score = 0
    previous_away_score = 0
    previous_elapsed_game_seconds: float | None = None

    for event_index, action in enumerate(actions):
        period = _required_int(action, "period")
        source_clock = _required_str(action, "clock")
        seconds_remaining = parse_nba_clock(source_clock)
        duration = period_duration_seconds(period)
        flags: list[str] = []
        if seconds_remaining > duration:
            flags.append("clock_exceeds_period_duration")

        home_score = _score(action.get("scoreHome"), previous_home_score)
        away_score = _score(action.get("scoreAway"), previous_away_score)
        home_delta = home_score - previous_home_score
        away_delta = away_score - previous_away_score
        if home_delta < 0 or away_delta < 0:
            flags.append("negative_score_delta")

        elapsed_game_seconds = completed_period_seconds(period) + max(
            duration - seconds_remaining,
            0,
        )
        if (
            previous_elapsed_game_seconds is not None
            and elapsed_game_seconds < previous_elapsed_game_seconds
        ):
            flags.append("nonmonotonic_source_clock")
            elapsed_game_seconds = previous_elapsed_game_seconds

        order_number = _required_int(action, "orderNumber")
        event = Event(
            game_id=game_id,
            event_id=f"{game_id}:{order_number}",
            event_index=event_index,
            source_action_number=_required_int(action, "actionNumber"),
            source_order_number=order_number,
            period=period,
            period_type=str(action.get("periodType") or ""),
            source_clock=source_clock,
            clock=format_game_clock(seconds_remaining),
            seconds_remaining_period=seconds_remaining,
            elapsed_game_seconds=elapsed_game_seconds,
            event_type=_required_str(action, "actionType"),
            event_subtype=_optional_str(action.get("subType")),
            descriptor=_optional_str(action.get("descriptor")),
            team_id=_positive_int_or_none(action.get("teamId")),
            team_tricode=_optional_str(action.get("teamTricode")),
            player_id=_positive_int_or_none(action.get("personId")),
            related_player_ids=_int_tuple(action.get("personIdsFilter")),
            qualifiers=_str_tuple(action.get("qualifiers")),
            source_possession_team_id=_positive_int_or_none(action.get("possession")),
            score_home=home_score,
            score_away=away_score,
            score_home_delta=home_delta,
            score_away_delta=away_delta,
            is_field_goal=bool(action.get("isFieldGoal", 0)),
            shot_result=_optional_str(action.get("shotResult")),
            description=_optional_str(action.get("description")),
            validation_flags=tuple(flags),
        )
        events.append(event)
        previous_home_score = home_score
        previous_away_score = away_score
        previous_elapsed_game_seconds = elapsed_game_seconds

    return events


def events_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Return one row per canonical event."""

    return event_records_frame(canonical_events(payload))


def event_records_frame(events: Sequence[Event]) -> pd.DataFrame:
    """Return canonical events with stable pandas dtypes for persistence."""

    frame = pd.DataFrame(event.model_dump() for event in events)
    for column in ("team_id", "player_id", "source_possession_team_id"):
        if column in frame:
            frame[column] = frame[column].astype("Int64")
    return frame


def write_events_parquet(payload: Mapping[str, Any], path: Path | str) -> None:
    events_frame(payload).to_parquet(path, index=False)


def _required_int(value: Mapping[str, Any], field: str) -> int:
    raw_value = value.get(field)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"Expected integer {field}, got {raw_value!r}")
    return raw_value


def _required_str(value: Mapping[str, Any], field: str) -> str:
    raw_value = value.get(field)
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"Expected non-empty string {field}, got {raw_value!r}")
    return raw_value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected positive integer identifier, got {value!r}")
    if value == 0:
        return None
    return value if value > 0 else None


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"Expected integer identifier list, got {value!r}")
    return tuple(item for item in value if item > 0)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _score(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    score = int(value)
    if score < 0:
        raise ValueError(f"Scores must be non-negative, got {score}")
    return score
