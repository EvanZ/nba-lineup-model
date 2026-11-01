"""Immutable player-to-spread attribution for one game prediction."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


def enrich_projected_minutes(
    projected_minutes: Mapping[str, object],
    *,
    ratings: Mapping[int, float],
    game_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Attach the exact game rating and contribution to frozen minute rows."""

    output: dict[str, list[dict[str, Any]]] = {}
    for side in ("away", "home"):
        source_rows = projected_minutes.get(side)
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"Frozen projected {side} rotation is missing for {game_id}")
        rows: list[dict[str, Any]] = []
        for source in source_rows:
            if not isinstance(source, dict) or source.get("player_id") is None:
                raise ValueError(
                    f"Frozen projected {side} rotation has an invalid row for {game_id}"
                )
            player_id = int(source["player_id"])
            if player_id not in ratings:
                raise ValueError(
                    f"Frozen rotation player {player_id} lacks a game rating for {game_id}"
                )
            minutes = float(source.get("projected_minutes") or 0.0)
            rating = float(ratings[player_id])
            row = copy.deepcopy(source)
            row["projected_nail"] = rating
            row["expected_contribution"] = minutes / 48.0 * rating
            rows.append(row)
        output[side] = rows
    return output


def build_spread_ledger(
    *,
    game_id: str,
    away_team: str,
    home_team: str,
    projected_minutes: Mapping[str, object],
    home_court_adjustment: float,
    back_to_back_coefficient: float,
    home_back_to_back: bool,
    away_back_to_back: bool,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build and validate the exact ledger used by one game prediction."""

    def team_ledger(side: str, team: str) -> dict[str, object]:
        players = projected_minutes.get(side)
        if not isinstance(players, list) or not players:
            raise ValueError(f"Projection rotation has no players for {team}/{game_id}")
        rows: list[dict[str, object]] = []
        for player in players:
            if not isinstance(player, dict):
                raise ValueError(f"Projection rotation has an invalid row for {team}/{game_id}")
            minutes = float(player["projected_minutes"])
            rating = float(player["projected_nail"])
            contribution = float(player["expected_contribution"])
            expected = minutes / 48.0 * rating
            if not math.isclose(contribution, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Player contribution does not reconcile for {player['player_id']}/{game_id}"
                )
            rows.append(
                {
                    "player_id": int(player["player_id"]),
                    "player_name": str(player["player_name"]),
                    "projected_minutes": minutes,
                    "frozen_nail": rating,
                    "expected_contribution": contribution,
                    "status": player.get("status"),
                    "play_probability": player.get("play_probability"),
                    "reason": player.get("reason"),
                }
            )
        total_minutes = sum(float(row["projected_minutes"]) for row in rows)
        if not math.isclose(total_minutes, 240.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Projected minutes sum to {total_minutes:.9f}, not 240, for {team}/{game_id}"
            )
        team_strength = sum(float(row["expected_contribution"]) for row in rows)
        return {
            "team": team,
            "players": rows,
            "projected_minutes": total_minutes,
            "team_strength": team_strength,
        }

    away = team_ledger("away", away_team)
    home = team_ledger("home", home_team)
    team_strength_difference = float(home["team_strength"]) - float(
        away["team_strength"]
    )
    back_to_back_adjustment = float(back_to_back_coefficient) * (
        float(home_back_to_back) - float(away_back_to_back)
    )
    home_margin = (
        team_strength_difference
        + float(home_court_adjustment)
        + back_to_back_adjustment
    )
    return {
        "schema_version": 1,
        "artifact_kind": "immutable_game_prediction_attribution",
        "game_id": game_id,
        "contract": "minutes / 48 * frozen NAIL; home minus away plus HCA plus B2B",
        "away": away,
        "home": home,
        "team_strength_difference": team_strength_difference,
        "home_court_adjustment": float(home_court_adjustment),
        "back_to_back": {
            "coefficient": float(back_to_back_coefficient),
            "home_team_is_back_to_back": bool(home_back_to_back),
            "away_team_is_back_to_back": bool(away_back_to_back),
            "adjustment": back_to_back_adjustment,
        },
        "home_margin": home_margin,
        "fair_home_spread": -home_margin,
        "provenance": dict(provenance),
    }


def build_actual_minutes_ledger(
    *,
    game_id: str,
    away_team: str,
    home_team: str,
    actual_minutes: Mapping[str, object],
    ratings: Mapping[int, float],
    pregame_ledger: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the same game ledger with official final minutes.

    Actual team minutes can exceed 240 in overtime.  Normalizing each team's
    minute weights to five players preserves the per-100-possession NAIL scale
    while exposing every player's exact contribution.
    """

    def team_ledger(side: str, team: str) -> dict[str, object]:
        players = actual_minutes.get(side)
        if not isinstance(players, list) or not players:
            raise ValueError(f"Actual rotation has no players for {team}/{game_id}")
        total_minutes = sum(
            float(player.get("actual_minutes") or 0.0)
            for player in players
            if isinstance(player, dict)
        )
        if total_minutes <= 0.0:
            raise ValueError(f"Actual rotation has no minutes for {team}/{game_id}")
        rows: list[dict[str, object]] = []
        for player in players:
            if not isinstance(player, dict) or player.get("player_id") is None:
                raise ValueError(f"Actual rotation has an invalid row for {team}/{game_id}")
            player_id = int(player["player_id"])
            if player_id not in ratings:
                raise ValueError(
                    f"Actual rotation player {player_id} lacks a game rating for {game_id}"
                )
            minutes = float(player.get("actual_minutes") or 0.0)
            rating = float(ratings[player_id])
            contribution = 5.0 * minutes / total_minutes * rating
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": str(player["player_name"]),
                    "actual_minutes": minutes,
                    "frozen_nail": rating,
                    "actual_contribution": contribution,
                    "rating_matched": True,
                }
            )
        return {
            "team": team,
            "players": sorted(
                rows,
                key=lambda row: (-float(row["actual_minutes"]), str(row["player_name"])),
            ),
            "actual_minutes": total_minutes,
            "team_strength": sum(float(row["actual_contribution"]) for row in rows),
        }

    back_to_back = pregame_ledger.get("back_to_back")
    if not isinstance(back_to_back, Mapping):
        raise ValueError(f"Pregame context is missing for {game_id}")
    away = team_ledger("away", away_team)
    home = team_ledger("home", home_team)
    team_strength_difference = float(home["team_strength"]) - float(
        away["team_strength"]
    )
    home_court_adjustment = float(pregame_ledger["home_court_adjustment"])
    back_to_back_adjustment = float(back_to_back["adjustment"])
    home_margin = (
        team_strength_difference + home_court_adjustment + back_to_back_adjustment
    )
    return {
        "schema_version": 1,
        "artifact_kind": "retrospective_actual_minutes_attribution",
        "game_id": game_id,
        "contract": (
            "5 * actual minutes / team actual minutes * frozen NAIL; "
            "home minus away plus frozen HCA plus frozen B2B"
        ),
        "away": away,
        "home": home,
        "team_strength_difference": team_strength_difference,
        "home_court_adjustment": home_court_adjustment,
        "back_to_back": dict(back_to_back),
        "home_margin": home_margin,
        "fair_home_spread": -home_margin,
        "provenance": dict(provenance),
    }


def require_projection_reconciliation(
    projection: Mapping[str, object], *, game_id: str
) -> None:
    ledger = projection.get("ledger")
    if not isinstance(ledger, dict):
        raise ValueError(f"Game prediction attribution is missing for {game_id}")
    home_margin = float(projection["home_margin"])
    ledger_margin = float(ledger["home_margin"])
    if not math.isclose(home_margin, ledger_margin, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"Game prediction attribution does not reproduce the spread for {game_id}: "
            f"{ledger_margin} != {home_margin}"
        )
