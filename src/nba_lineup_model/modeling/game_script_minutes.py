"""Game-script-aware minute observations and pregame adjustments.

The updater learns a player's latent role from the competitive portion of a
completed game.  A separate pregame transform turns that role into expected
realized minutes using only the projected margin.  Realized score paths are
therefore outcomes, never forecast features.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

REGULATION_SECONDS = 48.0 * 60.0
FOURTH_QUARTER_START = 36.0 * 60.0


@dataclass(frozen=True)
class GameScriptConfig:
    """Frozen thresholds for competitive-minute reconstruction."""

    minimum_margin: int = 10
    margin_per_minute: int = 3
    extra_margin: int = 3
    persistence_margin: int = 8
    minimum_garbage_seconds: float = 60.0
    blowout_final_margin: float = 20.0
    margin_residual_sd: float = 12.0
    garbage_minutes_when_blowout: float = 7.34

    def validate(self) -> None:
        if self.minimum_margin <= 0 or self.margin_per_minute <= 0:
            raise ValueError("Garbage-time margin thresholds must be positive")
        if self.persistence_margin <= 0:
            raise ValueError("Persistence margin must be positive")
        if not 0.0 <= self.minimum_garbage_seconds <= 12.0 * 60.0:
            raise ValueError("Minimum garbage duration must lie in [0, 720] seconds")
        if self.blowout_final_margin <= 0 or self.margin_residual_sd <= 0:
            raise ValueError("Pregame blowout parameters must be positive")
        if not 0.0 <= self.garbage_minutes_when_blowout <= 12.0:
            raise ValueError("Expected garbage duration must lie in [0, 12]")


DEFAULT_GAME_SCRIPT_CONFIG = GameScriptConfig()


def garbage_margin_threshold(seconds_remaining: float, config: GameScriptConfig) -> int:
    """Return the lead required to call the remaining regulation noncompetitive."""

    config.validate()
    minutes = max(0, ceil(float(seconds_remaining) / 60.0))
    return max(
        config.minimum_margin,
        config.extra_margin + config.margin_per_minute * minutes,
    )


def competitive_boundary_seconds(
    events: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    config: GameScriptConfig = DEFAULT_GAME_SCRIPT_CONFIG,
) -> float:
    """Find the first persistent fourth-quarter garbage-time boundary.

    The eventual winner must already lead by a clock-dependent threshold and
    may never again lead by fewer than ``persistence_margin`` points.  Games
    reaching overtime are treated as competitive throughout.
    """

    config.validate()
    frame = _event_frame(events)
    if frame.empty:
        raise ValueError("Cannot derive game script from an empty event stream")
    required = {"period", "elapsed_game_seconds", "score_home", "score_away"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError("Game-script events lack: " + ", ".join(missing))
    frame = frame.sort_values(
        ["elapsed_game_seconds", *(column for column in ("event_index",) if column in frame)],
        kind="stable",
    ).reset_index(drop=True)
    numeric = ("period", "elapsed_game_seconds", "score_home", "score_away")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    game_end = float(frame["elapsed_game_seconds"].max())
    if int(frame["period"].max()) > 4:
        return game_end
    final = frame.iloc[-1]
    final_margin = float(final["score_home"] - final["score_away"])
    if final_margin == 0.0:
        return game_end
    winner_sign = 1.0 if final_margin > 0 else -1.0
    frame["winner_lead"] = winner_sign * (
        frame["score_home"].astype(float) - frame["score_away"].astype(float)
    )
    fourth = frame.loc[
        frame["period"].eq(4)
        & frame["elapsed_game_seconds"].between(FOURTH_QUARTER_START, REGULATION_SECONDS)
    ]
    for row in fourth.itertuples(index=True):
        elapsed = float(row.elapsed_game_seconds)
        remaining = max(0.0, REGULATION_SECONDS - elapsed)
        if remaining < config.minimum_garbage_seconds:
            continue
        if float(row.winner_lead) < garbage_margin_threshold(remaining, config):
            continue
        later = frame.loc[frame.index >= int(row.Index), "winner_lead"]
        if not later.empty and float(later.min()) >= config.persistence_margin:
            return elapsed
    return game_end


def competitive_role_minutes(
    stints: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    boundary_seconds: float,
) -> dict[int, float]:
    """Extrapolate pre-boundary lineup exposure to a 48-minute team game."""

    frame = pd.DataFrame(stints).copy()
    required = {
        "start_elapsed_game_seconds",
        "end_elapsed_game_seconds",
        "home_player_ids",
        "away_player_ids",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError("Lineup stints lack: " + ", ".join(missing))
    boundary = float(boundary_seconds)
    if not np.isfinite(boundary) or boundary < FOURTH_QUARTER_START:
        raise ValueError("Competitive boundary must be finite and no earlier than Q4")
    seconds: dict[int, float] = {}
    home_total = 0.0
    away_total = 0.0
    for row in frame.itertuples(index=False):
        start = max(0.0, float(row.start_elapsed_game_seconds))
        end = min(boundary, float(row.end_elapsed_game_seconds))
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue
        home = _player_ids(row.home_player_ids)
        away = _player_ids(row.away_player_ids)
        if len(home) != 5 or len(away) != 5:
            raise ValueError("Every competitive lineup stint must contain five players per team")
        home_total += duration
        away_total += duration
        for player_id in (*home, *away):
            seconds[player_id] = seconds.get(player_id, 0.0) + duration
    tolerance = 2.0
    if abs(home_total - boundary) > tolerance or abs(away_total - boundary) > tolerance:
        raise ValueError(
            "Competitive lineup coverage is incomplete: "
            f"home={home_total:.2f}s away={away_total:.2f}s boundary={boundary:.2f}s"
        )
    scale = REGULATION_SECONDS / boundary / 60.0
    output = {player_id: value * scale for player_id, value in seconds.items()}
    if abs(sum(output.values()) - 480.0) > 0.1:
        raise ValueError("Competitive role minutes do not reconcile to 240 per team")
    return output


def reconstruct_competitive_role_targets(
    events: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    config: GameScriptConfig = DEFAULT_GAME_SCRIPT_CONFIG,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Build player-game role targets from already-curated historical marts."""

    if "game_id" not in events or "game_id" not in stints:
        raise ValueError("Events and lineup stints must contain game_id")
    stint_groups = {str(key): value for key, value in stints.groupby("game_id", sort=False)}
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    for raw_game_id, game_events in events.groupby("game_id", sort=True):
        game_id = str(raw_game_id)
        game_stints = stint_groups.get(game_id)
        if game_stints is None:
            exclusions.append({"game_id": game_id, "reason": "missing lineup stints"})
            continue
        try:
            boundary = competitive_boundary_seconds(game_events, config=config)
            game_end = float(
                pd.to_numeric(game_events["elapsed_game_seconds"], errors="raise").max()
            )
            roles = competitive_role_minutes(
                game_stints,
                boundary_seconds=boundary,
            )
        except (TypeError, ValueError) as error:
            exclusions.append({"game_id": game_id, "reason": str(error)})
            continue
        for player_id, role_minutes in roles.items():
            rows.append(
                {
                    "game_id": game_id,
                    "player_id": player_id,
                    "competitive_role_minutes": role_minutes,
                    "competitive_boundary_seconds": boundary,
                    "game_duration_seconds": game_end,
                    "garbage_time_detected": boundary + 1e-6 < game_end,
                }
            )
    return pd.DataFrame(rows), exclusions


def expected_garbage_minutes(
    projected_margin: float,
    *,
    config: GameScriptConfig = DEFAULT_GAME_SCRIPT_CONFIG,
) -> float:
    """Estimate expected garbage duration from the pregame margin distribution."""

    config.validate()
    mean = float(projected_margin)
    if not np.isfinite(mean):
        raise ValueError("Projected margin must be finite")
    cutoff = config.blowout_final_margin
    sd = config.margin_residual_sd
    probability = float(ndtr((-cutoff - mean) / sd) + 1.0 - ndtr((cutoff - mean) / sd))
    return config.garbage_minutes_when_blowout * probability


def apply_expected_game_script(
    minutes: Mapping[int, float],
    *,
    projected_margin: float,
    config: GameScriptConfig = DEFAULT_GAME_SCRIPT_CONFIG,
) -> dict[int, float]:
    """Blend a competitive role allocation with an expected garbage rotation.

    Zero-minute players remain excluded, which preserves injury assumptions.
    The returned allocation always sums to the same team total as the input.
    """

    config.validate()
    values = {int(player_id): float(value) for player_id, value in minutes.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("Minute allocations must be finite and nonnegative")
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError("Minute allocation must have a positive total")
    eligible = {player_id: value for player_id, value in values.items() if value > 0.0}
    if len(eligible) <= 5:
        return values
    # An inverse-role distribution approximates a garbage unit while never
    # activating a player whose injury-adjusted projection is zero.
    weights = {player_id: 1.0 / (2.0 + value) for player_id, value in eligible.items()}
    garbage = _capped_allocation(weights, total=total, cap=48.0)
    blend = expected_garbage_minutes(projected_margin, config=config) / 48.0
    adjusted = {
        player_id: (1.0 - blend) * value + blend * garbage.get(player_id, 0.0)
        for player_id, value in values.items()
    }
    correction = total / sum(adjusted.values())
    return {player_id: value * correction for player_id, value in adjusted.items()}


def _capped_allocation(
    weights: Mapping[int, float], *, total: float, cap: float
) -> dict[int, float]:
    remaining = set(weights)
    output = {player_id: 0.0 for player_id in weights}
    unallocated = float(total)
    while remaining:
        weight_total = sum(float(weights[player_id]) for player_id in remaining)
        if weight_total <= 0.0:
            raise ValueError("Garbage-rotation weights must be positive")
        proposed = {
            player_id: unallocated * float(weights[player_id]) / weight_total
            for player_id in remaining
        }
        capped = {player_id for player_id, value in proposed.items() if value > cap}
        if not capped:
            output.update(proposed)
            break
        for player_id in capped:
            output[player_id] = cap
            unallocated -= cap
            remaining.remove(player_id)
    return output


def _event_frame(
    events: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events.copy()
    records = [
        event.model_dump() if hasattr(event, "model_dump") else dict(event)
        for event in events
    ]
    return pd.DataFrame(records)


def _player_ids(value: Iterable[int] | np.ndarray) -> tuple[int, ...]:
    return tuple(int(player_id) for player_id in value)
