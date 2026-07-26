from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def boxscore_players_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert NBA CDN boxscore JSON into one row per player."""

    game = payload.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("Expected payload['game'] to be an object")

    rows: list[dict[str, Any]] = []
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
            base = {
                "game_id": game.get("gameId"),
                "team_side": "home" if side == "homeTeam" else "away",
                "team_id": team.get("teamId"),
                "team_tricode": team.get("teamTricode"),
            }
            player_flat = pd.json_normalize(player, sep="_").iloc[0].to_dict()
            rows.append(base | player_flat)

    return pd.DataFrame(rows)


def write_boxscore_players_parquet(payload: Mapping[str, Any], path: str) -> None:
    boxscore_players_frame(payload).to_parquet(path, index=False)
