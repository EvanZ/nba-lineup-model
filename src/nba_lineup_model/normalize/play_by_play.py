from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def play_by_play_actions_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Convert NBA CDN play-by-play JSON into one row per action."""

    game = payload.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("Expected payload['game'] to be an object")

    game_id = game.get("gameId")
    actions = game.get("actions")
    if not isinstance(actions, list):
        raise ValueError("Expected payload['game']['actions'] to be a list")

    frame = pd.json_normalize(actions)
    frame.insert(0, "game_id", game_id)

    if "period" in frame:
        frame["period"] = frame["period"].astype("Int64")
    if "actionNumber" in frame:
        frame["actionNumber"] = frame["actionNumber"].astype("Int64")
    if "orderNumber" in frame:
        frame["orderNumber"] = frame["orderNumber"].astype("Int64")

    return frame


def write_actions_parquet(payload: Mapping[str, Any], path: str) -> None:
    play_by_play_actions_frame(payload).to_parquet(path, index=False)
