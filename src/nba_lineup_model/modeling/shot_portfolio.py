"""Forward-safe rim-pressure and spacing-capacity contextual profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.shot_taxonomy import DEFAULT_OUTPUT_DIR, validate_shot_taxonomy

RATE_PSEUDO_POSSESSIONS = 300.0


def add_shot_portfolio_profiles(
    profiles: pd.DataFrame,
    *,
    target_season: str,
    shot_taxonomy_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Add prior-season, era-standardized shot-capacity traits to target profiles."""

    root = Path(shot_taxonomy_dir)
    validate_shot_taxonomy(root)
    source_season = _previous_season(target_season)
    shots = pd.read_parquet(root / "player_season_shot_profiles.parquet")
    source = shots.loc[shots["season"].astype(str).eq(source_season)].copy()
    if source.empty:
        raise ValueError(f"Shot taxonomy has no source season {source_season}")
    source["rim_pressure_raw"] = _shrunk_rate(source, "rim_attempts")
    source["spacing_capacity_raw"] = _shrunk_rate(source, "three_attempts")
    reference = source.loc[source["shot_profile_available"].astype(bool)]
    if reference.empty:
        raise ValueError(f"Shot taxonomy has no available source profiles for {source_season}")
    for raw, output in (
        ("rim_pressure_raw", "rim_pressure"),
        ("spacing_capacity_raw", "spacing_capacity"),
    ):
        mean = float(reference[raw].mean())
        scale = float(reference[raw].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Shot taxonomy has invalid {raw} scale for {source_season}")
        source[output] = (source[raw] - mean) / scale
    values = source.loc[:, ["player_id", "rim_pressure", "spacing_capacity"]]
    output = profiles.merge(values, on="player_id", how="left", validate="one_to_one")
    output[["rim_pressure", "spacing_capacity"]] = output[
        ["rim_pressure", "spacing_capacity"]
    ].fillna(0.0)
    return output


def _shrunk_rate(frame: pd.DataFrame, count_column: str) -> pd.Series:
    possession = frame["on_court_possessions"].to_numpy(dtype=float)
    count = frame[count_column].to_numpy(dtype=float)
    raw = 100.0 * count / np.maximum(possession, 1.0)
    league_rate = float(np.average(raw, weights=np.maximum(possession, 1.0)))
    return pd.Series(
        100.0
        * (count + RATE_PSEUDO_POSSESSIONS * league_rate / 100.0)
        / (possession + RATE_PSEUDO_POSSESSIONS),
        index=frame.index,
    )


def _previous_season(season: str) -> str:
    start = int(str(season)[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"
