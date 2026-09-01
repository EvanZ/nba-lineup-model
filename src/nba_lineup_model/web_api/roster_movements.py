"""Derive current roster movement edges from an end-of-season team snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

DEFAULT_CURRENT_ROSTER_PATH = Path("data/curated/team_rosters/2026-27/part-00000.parquet")
DEFAULT_PRIOR_PANEL_PATH = Path(
    "data/analytical/player_season_panel/player_seasons.parquet"
)
DEFAULT_PRIOR_SNAPSHOT_PATH = Path(
    "data/curated/team_rosters/2025-26/final_regular_season_snapshot.parquet"
)
DEFAULT_PLAYER_CATALOG_PATH = Path("data/catalog/players.parquet")
DEFAULT_CURRENT_SEASON = "2026-27"
DEFAULT_PRIOR_SEASON = "2025-26"

RosterMoveType = Literal["trade", "signing", "waiver", "other"]


def build_roster_movement_payload(
    *,
    current_roster_path: Path | str = DEFAULT_CURRENT_ROSTER_PATH,
    prior_panel_path: Path | str = DEFAULT_PRIOR_PANEL_PATH,
    prior_snapshot_path: Path | str = DEFAULT_PRIOR_SNAPSHOT_PATH,
    player_catalog_path: Path | str = DEFAULT_PLAYER_CATALOG_PATH,
    preseason_rankings: pd.DataFrame | None = None,
    current_season: str = DEFAULT_CURRENT_SEASON,
    prior_season: str = DEFAULT_PRIOR_SEASON,
) -> dict[str, object]:
    """Return one directed player edge for every returning player who changed teams.

    Source teams come from the final observed regular-season game roster, not
    the season's primary team by minutes. This excludes players who changed
    teams during the prior regular season from the following offseason graph.
    Players without a prior roster record are returned as external arrivals so
    clients can show them entering from outside the NBA map.
    """

    current_roster = pd.read_parquet(current_roster_path)
    prior_panel = pd.read_parquet(prior_panel_path)
    prior_snapshot = pd.read_parquet(prior_snapshot_path)
    player_catalog = pd.read_parquet(player_catalog_path)
    _validate_current_roster(current_roster)
    _validate_prior_panel(prior_panel)
    _validate_prior_snapshot(prior_snapshot)
    _validate_player_catalog(player_catalog)
    projected_ratings = _projected_rating_by_player(preseason_rankings)

    current = current_roster.loc[
        :,
        [
            "player_id",
            "player_name",
            "team_abbreviation",
            "experience",
            "school",
            "how_acquired",
        ],
    ].copy()
    current["player_id"] = current["player_id"].astype(str)
    prior_minutes = prior_panel.loc[
        prior_panel["season"].astype(str).eq(prior_season),
        ["player_id", "minutes"],
    ].copy()
    prior_minutes["player_id"] = prior_minutes["player_id"].astype(str)
    prior_minutes = prior_minutes.rename(columns={"minutes": "prior_season_minutes"})
    prior = prior_snapshot.loc[:, ["player_id", "source_team"]].copy()
    prior["player_id"] = prior["player_id"].astype(str)
    prior = prior.merge(
        prior_minutes,
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    catalog = player_catalog.loc[:, ["player_id", "college", "country"]].copy()
    catalog["player_id"] = catalog["player_id"].astype(str)
    merged = current.merge(prior, on="player_id", how="left", validate="one_to_one")
    merged = merged.merge(catalog, on="player_id", how="left", validate="one_to_one")
    merged["target_team"] = merged["team_abbreviation"].astype(str)
    known_prior = merged["source_team"].notna()
    direct_moves = merged.loc[
        known_prior & merged["source_team"].ne(merged["target_team"])
    ].copy()
    direct_moves["move_type"] = direct_moves["how_acquired"].map(_move_type)
    direct_moves["prior_season_minutes"] = pd.to_numeric(
        direct_moves["prior_season_minutes"], errors="coerce"
    ).fillna(0.0)
    direct_moves = direct_moves.sort_values(
        ["prior_season_minutes", "player_name", "player_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    moves = [
        {
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
            "source_team": str(row.source_team),
            "target_team": str(row.target_team),
            "move_type": str(row.move_type),
            "how_acquired": _nullable_text(row.how_acquired),
            "prior_season_minutes": float(row.prior_season_minutes),
            "projected_rating": projected_ratings.get(str(row.player_id)),
        }
        for row in direct_moves.itertuples(index=False)
    ]
    external_arrivals = merged.loc[~known_prior].copy()
    external_arrivals["move_type"] = external_arrivals["how_acquired"].map(_move_type)
    external_arrivals = external_arrivals.sort_values(
        ["player_name", "player_id"],
        kind="stable",
    )
    arrivals = [
        {
            "player_id": int(row.player_id),
            "player_name": str(row.player_name),
            "target_team": str(row.target_team),
            "move_type": str(row.move_type),
            "how_acquired": _nullable_text(row.how_acquired),
            "school": _nullable_text(row.school) or _nullable_text(row.college),
            "country": _nullable_text(row.country),
            "is_rookie": str(row.experience).strip().upper() == "R",
            "projected_rating": projected_ratings.get(str(row.player_id)),
        }
        for row in external_arrivals.itertuples(index=False)
    ]
    teams = sorted(current["team_abbreviation"].dropna().astype(str).unique().tolist())
    return {
        "current_season": current_season,
        "prior_season": prior_season,
        "source_definition": f"{prior_season} final observed regular-season team",
        "teams": teams,
        "moves": moves,
        "external_arrivals": arrivals,
        "current_roster_count": int(len(current)),
        "returning_mover_count": int(len(moves)),
        "new_or_unmatched_current_player_count": int(len(arrivals)),
    }


def _move_type(value: object) -> RosterMoveType:
    text = _nullable_text(value)
    if text is None:
        return "other"
    normalized = text.lower()
    if "trade" in normalized:
        return "trade"
    if "waiver" in normalized:
        return "waiver"
    if "signed" in normalized:
        return "signing"
    return "other"


def _nullable_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _projected_rating_by_player(preseason_rankings: pd.DataFrame | None) -> dict[str, float | None]:
    """Return the published 2026-27 preseason NAIL rating keyed by player."""

    if preseason_rankings is None:
        return {}
    required = {"player_id", "rapm"}
    missing = sorted(required - set(preseason_rankings.columns))
    if missing:
        raise ValueError(f"Preseason rankings lack columns: {missing}")
    if preseason_rankings["player_id"].astype(str).duplicated().any():
        raise ValueError("Preseason rankings contain duplicate player IDs")
    ratings = pd.to_numeric(preseason_rankings["rapm"], errors="coerce")
    return {
        str(player_id): None if pd.isna(rating) else float(rating)
        for player_id, rating in zip(preseason_rankings["player_id"], ratings, strict=True)
    }


def _validate_current_roster(frame: pd.DataFrame) -> None:
    required = {
        "player_id",
        "player_name",
        "team_abbreviation",
        "experience",
        "school",
        "how_acquired",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Current roster table lacks columns: {missing}")
    if frame["player_id"].astype(str).duplicated().any():
        raise ValueError("Current roster table contains duplicate player IDs")


def _validate_prior_panel(frame: pd.DataFrame) -> None:
    required = {"season", "player_id", "primary_team_tricode", "minutes"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Player-season panel lacks columns: {missing}")


def _validate_prior_snapshot(frame: pd.DataFrame) -> None:
    required = {"player_id", "source_team"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prior roster snapshot lacks columns: {missing}")
    if frame["player_id"].astype(str).duplicated().any():
        raise ValueError("Prior roster snapshot contains duplicate player IDs")


def _validate_player_catalog(frame: pd.DataFrame) -> None:
    required = {"player_id", "college", "country"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Player catalog lacks columns: {missing}")
    if frame["player_id"].astype(str).duplicated().any():
        raise ValueError("Player catalog contains duplicate player IDs")
