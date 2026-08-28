"""Canonical single-lineup possession frames derived from processed segments."""

from __future__ import annotations

import numpy as np
import pandas as pd


SINGLE_LINEUP_POSSESSION_COLUMNS = (
    "schema_version", "season", "season_type", "game_id", "game_date", "game_time_utc",
    "possession_id", "possession_index", "period", "offense_team_id", "defense_team_id",
    "offense_team_tricode", "defense_team_tricode", "offense_player_ids",
    "defense_player_ids", "home_offense", "home_offense_sign", "offense_points",
    "defense_points", "target_offense_margin", "target_home_margin", "quality_status",
    "source_build_run_id", "processing_code_version", "play_by_play_sha256", "boxscore_sha256",
)

_REQUIRED_SEGMENT_COLUMNS = {
    "season", "season_type", "game_id", "game_date", "game_time_utc", "possession_id",
    "possession_index", "period", "offense_team_id", "defense_team_id", "home_player_ids",
    "away_player_ids", "points_home", "points_away", "offense_points", "catalog_home_team_id",
    "catalog_away_team_id", "catalog_home_team_tricode", "catalog_away_team_tricode",
    "quality_status", "source_build_run_id", "processing_code_version", "play_by_play_sha256",
    "boxscore_sha256",
}


def single_lineup_possessions_frame(possession_segments: pd.DataFrame) -> pd.DataFrame:
    """Orient unambiguous processed possession segments by offense."""

    missing = _REQUIRED_SEGMENT_COLUMNS - set(possession_segments.columns)
    if missing:
        raise ValueError(f"Possession segments missing columns: {sorted(missing)}")
    if possession_segments.empty:
        raise ValueError("Possession segments cannot be empty")
    if possession_segments["season_type"].astype(str).nunique() != 1:
        raise ValueError("Possession segments must come from one season type")

    keys = ["game_id", "possession_id"]
    counts = possession_segments.groupby(keys, sort=False)["possession_id"].transform("size")
    output = possession_segments.loc[counts.eq(1)].copy()
    if output.empty:
        raise ValueError("No single-lineup possessions are available")
    if output.duplicated(keys).any():
        raise ValueError("Single-lineup possession keys must be unique")

    home_offense = output["offense_team_id"].eq(output["catalog_home_team_id"])
    away_offense = output["offense_team_id"].eq(output["catalog_away_team_id"])
    if not (home_offense | away_offense).all() or (home_offense & away_offense).any():
        raise ValueError("Possession offense team must match exactly one game team")
    expected_defense = output["catalog_away_team_id"].where(home_offense, output["catalog_home_team_id"])
    if not output["defense_team_id"].eq(expected_defense).all():
        raise ValueError("Possession defense team does not match its opponent")

    output["schema_version"] = 1
    output["home_offense"] = home_offense
    output["home_offense_sign"] = np.where(home_offense, 1.0, -1.0)
    output["offense_player_ids"] = [
        list(home if is_home else away)
        for home, away, is_home in zip(
            output["home_player_ids"], output["away_player_ids"], home_offense, strict=True
        )
    ]
    output["defense_player_ids"] = [
        list(away if is_home else home)
        for home, away, is_home in zip(
            output["home_player_ids"], output["away_player_ids"], home_offense, strict=True
        )
    ]
    _validate_lineups(output)
    output["offense_team_tricode"] = output["catalog_home_team_tricode"].where(
        home_offense, output["catalog_away_team_tricode"]
    )
    output["defense_team_tricode"] = output["catalog_away_team_tricode"].where(
        home_offense, output["catalog_home_team_tricode"]
    )
    offense_points = output["points_home"].where(home_offense, output["points_away"])
    defense_points = output["points_away"].where(home_offense, output["points_home"])
    if not offense_points.eq(output["offense_points"]).all():
        raise ValueError("Source offense points do not match game-relative points")
    output["offense_points"] = offense_points
    output["defense_points"] = defense_points
    output["target_offense_margin"] = offense_points - defense_points
    output["target_home_margin"] = output["target_offense_margin"] * output["home_offense_sign"]
    return _type_frame(
        output.loc[:, SINGLE_LINEUP_POSSESSION_COLUMNS]
        .sort_values(["game_time_utc", "game_id", "possession_index"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_lineups(frame: pd.DataFrame) -> None:
    for side in ("offense", "defense"):
        lineups = frame[f"{side}_player_ids"]
        if not lineups.map(len).eq(5).all():
            raise ValueError("Single-lineup possessions require exactly five players per team")
        if not lineups.map(lambda players: len(set(players)) == 5).all():
            raise ValueError("Single-lineup possession lineups cannot contain duplicates")
    if any(
        bool(set(offense) & set(defense))
        for offense, defense in zip(
            frame["offense_player_ids"], frame["defense_player_ids"], strict=True
        )
    ):
        raise ValueError("A player cannot appear on both sides of one possession")


def _type_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "season", "season_type", "game_id", "possession_id", "offense_team_tricode",
        "defense_team_tricode", "quality_status", "source_build_run_id",
        "processing_code_version", "play_by_play_sha256", "boxscore_sha256",
    ):
        frame[column] = frame[column].astype("string")
    for column in (
        "schema_version", "possession_index", "period", "offense_team_id", "defense_team_id",
        "offense_points", "defense_points", "target_offense_margin", "target_home_margin",
    ):
        frame[column] = frame[column].astype("int64")
    frame["home_offense"] = frame["home_offense"].astype(bool)
    frame["home_offense_sign"] = frame["home_offense_sign"].astype("float64")
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    return frame
