"""Construct availability-loss episode and player-season actuarial targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_lineup_model.players.availability import (
    STATE_AVAILABLE_DNP_COACH,
    STATE_UNAVAILABLE_INJURY,
    _correct_isolated_coach_dnp_with_medical_neighbors,
)

DEFAULT_CURATED_DIR = Path("data/curated")
DEFAULT_PLAYER_PANEL_PATH = Path("data/analytical/player_season_panel/player_seasons.parquet")
DEFAULT_OUTPUT_DIR = Path("data/analytical/availability_loss_episodes")
DEFAULT_SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2026))


def build_availability_episode_frame(player_games: pd.DataFrame) -> pd.DataFrame:
    """Return binary-unavailable episodes from one or more player-game seasons.

    A one-game coach-DNP state directly bracketed by injury/illness rows is
    reconciled into the same availability-loss episode. The source mart applies this
    correction already, but repeating it here keeps episode construction robust
    to an older or independently rebuilt mart.
    """

    required = {
        "season",
        "game_id",
        "game_date",
        "team_id",
        "team",
        "player_id",
        "player_name",
        "availability_state",
        "available",
        "availability_state_known",
    }
    missing = sorted(required - set(player_games))
    if missing:
        raise ValueError(f"Player games missing episode columns: {missing}")

    corrected = _prepare_availability_episode_rows(player_games)
    if corrected.empty:
        return _empty_episode_frame()

    unavailable = corrected.loc[corrected["is_unavailable"]].copy()
    if unavailable.empty:
        return _empty_episode_frame()
    episodes = (
        unavailable.groupby(
            ["player_id", "availability_episode_number"],
            as_index=False,
            sort=False,
        )
        .agg(
            player_name=("player_name", "last"),
            episode_start_season=("season", "first"),
            episode_end_season=("season", "last"),
            episode_start_team_id=("team_id", "first"),
            episode_start_team=("team", "first"),
            episode_end_team_id=("team_id", "last"),
            episode_end_team=("team", "last"),
            episode_start_date=("game_date", "min"),
            episode_end_date=("game_date", "max"),
            unavailable_games=("game_id", "size"),
            reconciled_coach_dnp_games=("is_reconciled_coach_dnp", "sum"),
            final_unavailable_game_order=("game_order", "max"),
            final_known_game_order=("last_known_game_order", "max"),
        )
        .sort_values(
            ["episode_start_season", "player_id", "availability_episode_number"], kind="stable"
        )
        .reset_index(drop=True)
    )
    episodes["availability_episode_number"] = episodes["availability_episode_number"].astype(int)
    episodes["unavailable_games"] = episodes["unavailable_games"].astype(int)
    episodes["reconciled_coach_dnp_games"] = episodes["reconciled_coach_dnp_games"].astype(int)
    episodes["right_censored"] = episodes["final_unavailable_game_order"].eq(
        episodes["final_known_game_order"]
    )
    episodes["episode_id"] = episodes.apply(
        lambda row: (
            f"{row.episode_start_season}:{int(row.player_id)}:"
            f"{int(row.availability_episode_number)}"
        ),
        axis=1,
    )
    return episodes.loc[
        :,
        [
            "episode_id",
            "player_id",
            "player_name",
            "availability_episode_number",
            "episode_start_season",
            "episode_end_season",
            "episode_start_team_id",
            "episode_start_team",
            "episode_end_team_id",
            "episode_end_team",
            "episode_start_date",
            "episode_end_date",
            "unavailable_games",
            "reconciled_coach_dnp_games",
            "right_censored",
        ],
    ]


def build_availability_episode_season_summary(
    player_games: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    player_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate episode frequency, severity, and real workload by player-season.

    The episode count belongs to the season in which an episode began. Unavailable
    games remain allocated to the actual season/game in which they occur.
    """

    known = _prepare_availability_episode_rows(player_games)
    if known.empty:
        return pd.DataFrame()
    known["minutes"] = pd.to_numeric(known.get("minutes", 0.0), errors="coerce").fillna(0.0)
    summary = (
        known.groupby(["season", "player_id"], as_index=False, sort=False)
        .agg(
            player_name=("player_name", "last"),
            rostered_team_games=("game_id", "size"),
            available_rostered_games=("available", "sum"),
            realized_gp=("minutes", lambda values: int(values.gt(0.0).sum())),
            realized_minutes=("minutes", "sum"),
        )
        .sort_values(["season", "player_id"], kind="stable")
        .reset_index(drop=True)
    )
    for column in (
        "rostered_team_games",
        "available_rostered_games",
        "realized_gp",
    ):
        summary[column] = pd.to_numeric(summary[column], errors="raise").astype(int)
    summary["realized_minutes"] = pd.to_numeric(
        summary["realized_minutes"], errors="raise"
    ).astype(float)
    unavailable = known.loc[known["is_unavailable"]].copy()
    episode_summary = (
        unavailable.groupby(["season", "player_id"], as_index=False, sort=False)
        .agg(
            availability_episode_count=("is_availability_episode_start", "sum"),
            unavailable_games=("game_id", "size"),
            reconciled_coach_dnp_games=("is_reconciled_coach_dnp", "sum"),
        )
        if not unavailable.empty
        else pd.DataFrame(columns=["season", "player_id"])
    )
    episode_starts = (
        episodes.groupby(["episode_start_season", "player_id"], as_index=False, sort=False)
        .agg(
            mean_episode_missed_games=("unavailable_games", "mean"),
            max_episode_missed_games=("unavailable_games", "max"),
            censored_availability_episode_count=("right_censored", "sum"),
        )
        .rename(columns={"episode_start_season": "season"})
        if not episodes.empty
        else pd.DataFrame(columns=["season", "player_id"])
    )
    episode_summary = episode_summary.merge(
        episode_starts,
        on=["season", "player_id"],
        how="outer",
        validate="one_to_one",
    )
    if episode_summary.empty:
        episode_summary = pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "availability_episode_count",
                "unavailable_games",
                "reconciled_coach_dnp_games",
                "mean_episode_missed_games",
                "max_episode_missed_games",
                "censored_availability_episode_count",
            ]
        )
    else:
        episode_summary["availability_episode_count"] = episode_summary[
            "availability_episode_count"
        ].fillna(0)
        episode_summary["unavailable_games"] = episode_summary[
            "unavailable_games"
        ].fillna(0)
        episode_summary["reconciled_coach_dnp_games"] = episode_summary[
            "reconciled_coach_dnp_games"
        ].fillna(0)
    summary = summary.merge(
        episode_summary,
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    for column in (
        "availability_episode_count",
        "unavailable_games",
        "censored_availability_episode_count",
        "reconciled_coach_dnp_games",
    ):
        summary[column] = summary[column].fillna(0).astype(int)
    summary["mean_episode_missed_games"] = summary["mean_episode_missed_games"].fillna(0.0)
    summary["max_episode_missed_games"] = summary["max_episode_missed_games"].fillna(0).astype(int)
    summary["unavailability_loss_cost"] = (
        summary["unavailable_games"] / summary["rostered_team_games"]
    )
    summary["minutes_per_available_game"] = (
        summary["realized_minutes"] / summary["available_rostered_games"].clip(lower=1)
    )
    if player_panel is not None:
        required_panel = {"season", "player_id", "games", "games_started", "minutes"}
        missing_panel = sorted(required_panel - set(player_panel))
        if missing_panel:
            raise ValueError(f"Player panel missing workload columns: {missing_panel}")
        panel_columns = set(required_panel)
        if "age" in player_panel:
            panel_columns.add("age")
        panel = player_panel.loc[:, sorted(panel_columns)].copy()
        panel["player_id"] = pd.to_numeric(panel["player_id"], errors="raise").astype(int)
        if panel.duplicated(["season", "player_id"]).any():
            raise ValueError("Player panel has duplicate player-season workload rows")
        panel = panel.rename(
            columns={
                "games": "panel_gp",
                "games_started": "panel_gs",
                "minutes": "panel_minutes",
                "age": "panel_age",
            }
        )
        summary = summary.merge(panel, on=["season", "player_id"], how="left")
    return summary


def build_availability_episode_mart(
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    player_panel_path: Path | str = DEFAULT_PLAYER_PANEL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all requested episode and player-season actuarial tables."""

    root = Path(curated_dir) / "player_availability"
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = root / season / "part-00000.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing availability mart: {path}")
        frames.append(pd.read_parquet(path))
    player_games = pd.concat(frames, ignore_index=True)
    episodes = build_availability_episode_frame(player_games)
    panel_path = Path(player_panel_path)
    panel = pd.read_parquet(panel_path) if panel_path.exists() else None
    return episodes, build_availability_episode_season_summary(
        player_games,
        episodes,
        player_panel=panel,
    )


def _empty_episode_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "episode_id",
            "player_id",
            "player_name",
            "availability_episode_number",
            "episode_start_season",
            "episode_end_season",
            "episode_start_team_id",
            "episode_start_team",
            "episode_end_team_id",
            "episode_end_team",
            "episode_start_date",
            "episode_end_date",
            "unavailable_games",
            "reconciled_coach_dnp_games",
            "right_censored",
        ]
    )


def _prepare_availability_episode_rows(player_games: pd.DataFrame) -> pd.DataFrame:
    """Reconcile row-level artifacts and assign cross-season episode membership."""

    required = {
        "season",
        "game_id",
        "game_date",
        "team_id",
        "team",
        "player_id",
        "player_name",
        "availability_state",
        "available",
        "availability_state_known",
    }
    missing = sorted(required - set(player_games))
    if missing:
        raise ValueError(f"Player games missing episode columns: {missing}")
    known = player_games.loc[player_games["availability_state_known"].astype(bool)].copy()
    if known.empty:
        return known
    if known.duplicated(["player_id", "game_id"]).any():
        raise ValueError("Player games contain duplicate player/game rows")
    known["game_date"] = pd.to_datetime(known["game_date"], utc=True, errors="raise")
    ordered = known.sort_values(["player_id", "game_date", "game_id"], kind="stable").copy()
    groups = ordered.groupby("player_id", sort=False)
    prior_state = groups["availability_state"].shift(1)
    next_state = groups["availability_state"].shift(-1)
    ordered["is_reconciled_coach_dnp"] = (
        ordered["availability_state"].eq(STATE_AVAILABLE_DNP_COACH)
        & prior_state.eq(STATE_UNAVAILABLE_INJURY)
        & next_state.eq(STATE_UNAVAILABLE_INJURY)
    )
    corrected = _correct_isolated_coach_dnp_with_medical_neighbors(ordered)
    # The source-mart correction is intentionally team-scoped. Episode identity
    # is player-scoped so an injury can continue through an offseason or trade.
    bridged_index = ordered.index[ordered["is_reconciled_coach_dnp"]]
    corrected.loc[bridged_index, "availability_state"] = STATE_UNAVAILABLE_INJURY
    corrected.loc[bridged_index, "available"] = False
    corrected["is_injury_or_illness"] = corrected["availability_state"].eq(
        STATE_UNAVAILABLE_INJURY
    )
    # The established availability contract treats every known non-available
    # state as unavailable. Injury/illness remains an audit subtype, not the
    # actuarial event definition.
    corrected["is_unavailable"] = ~corrected["available"].astype(bool)
    groups = corrected.groupby("player_id", sort=False)
    previous_unavailable = groups["is_unavailable"].shift(1, fill_value=False).astype(bool)
    corrected["is_availability_episode_start"] = (
        corrected["is_unavailable"] & ~previous_unavailable
    )
    corrected["availability_episode_number"] = corrected["is_availability_episode_start"].astype(
        int
    ).groupby(corrected["player_id"], sort=False).cumsum()
    corrected["game_order"] = groups.cumcount()
    corrected["last_known_game_order"] = groups["game_order"].transform("max")
    return corrected


def main() -> int:
    """Materialize the episode mart used by actuarial availability experiments."""

    parser = argparse.ArgumentParser(description="Build availability-loss episode mart")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--player-panel", type=Path, default=DEFAULT_PLAYER_PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--first-season", default=DEFAULT_SEASONS[0])
    parser.add_argument("--last-season", default=DEFAULT_SEASONS[-1])
    args = parser.parse_args()
    seasons = tuple(
        season
        for season in DEFAULT_SEASONS
        if args.first_season <= season <= args.last_season
    )
    episodes, summary = build_availability_episode_mart(
        seasons=seasons,
        curated_dir=args.curated_dir,
        player_panel_path=args.player_panel,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(args.output_dir / "episodes.parquet", index=False)
    summary.to_parquet(args.output_dir / "player_seasons.parquet", index=False)
    print(f"Wrote {len(episodes):,} availability-loss episodes and {len(summary):,} player seasons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
