"""Leakage-safe prior-season teammate-continuity features."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd

PAIR_EXPOSURE_COLUMNS = ("player_id_1", "player_id_2", "shared_possessions")


def build_teammate_pair_exposure(stints: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one season of same-unit possession exposure for teammate pairs."""

    required = {"home_player_ids", "away_player_ids", "possessions"}
    missing = required - set(stints)
    if missing:
        raise ValueError(f"Pair exposure stints lack: {sorted(missing)}")
    possessions = stints["possessions"].to_numpy(dtype=float)
    if len(stints) == 0 or not np.isfinite(possessions).all() or (possessions <= 0).any():
        raise ValueError("Pair exposure requires positive, finite stint possessions")

    exposure: dict[tuple[int, int], float] = {}
    for lineup_column in ("home_player_ids", "away_player_ids"):
        for lineup, weight in zip(stints[lineup_column], possessions, strict=True):
            player_ids = _validated_lineup(lineup)
            for player_id_1, player_id_2 in combinations(player_ids, 2):
                pair = (player_id_1, player_id_2)
                exposure[pair] = exposure.get(pair, 0.0) + float(weight)
    return pd.DataFrame(
        [
            {
                "player_id_1": player_id_1,
                "player_id_2": player_id_2,
                "shared_possessions": shared_possessions,
            }
            for (player_id_1, player_id_2), shared_possessions in sorted(exposure.items())
        ],
        columns=list(PAIR_EXPOSURE_COLUMNS),
    )


def teammate_continuity_side_feature(
    lineups: Sequence[Sequence[int]], pair_exposure: pd.DataFrame
) -> np.ndarray:
    """Average ``log(1 + shared possessions)`` over a unit's ten pairs."""

    missing = set(PAIR_EXPOSURE_COLUMNS) - set(pair_exposure)
    if missing:
        raise ValueError(f"Pair exposure table lacks: {sorted(missing)}")
    shared = pair_exposure["shared_possessions"].to_numpy(dtype=float)
    if not np.isfinite(shared).all() or (shared < 0).any():
        raise ValueError("Pair exposure values must be finite and non-negative")
    if pair_exposure.duplicated(["player_id_1", "player_id_2"]).any():
        raise ValueError("Pair exposure rows must be unique by ordered player pair")
    pair_lookup = {
        (int(row.player_id_1), int(row.player_id_2)): float(row.shared_possessions)
        for row in pair_exposure.itertuples(index=False)
    }
    lineup_cache: dict[tuple[int, ...], float] = {}
    values: list[float] = []
    for lineup in lineups:
        player_ids = _validated_lineup(lineup)
        value = lineup_cache.get(player_ids)
        if value is None:
            value = float(
                np.mean(
                    [
                        np.log1p(pair_lookup.get((player_id_1, player_id_2), 0.0))
                        for player_id_1, player_id_2 in combinations(player_ids, 2)
                    ]
                )
            )
            lineup_cache[player_ids] = value
        values.append(value)
    return np.asarray(values, dtype=float)


def _validated_lineup(lineup: Sequence[int]) -> tuple[int, ...]:
    player_ids = tuple(sorted({int(player_id) for player_id in lineup}))
    if len(player_ids) != 5:
        raise ValueError("Teammate continuity requires five unique players per unit")
    return player_ids
