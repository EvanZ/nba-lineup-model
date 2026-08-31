"""Build an end-of-regular-season player-team snapshot from game rosters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_final_regular_season_roster_snapshot(
    season: str,
    *,
    schedule_path: Path | str,
    processed_players_dir: Path | str,
    output_path: Path | str,
) -> Path:
    """Publish each player's final observed team in a regular-season game roster.

    Box-score player tables include inactive players, making the final observed
    roster more faithful to the end-of-season team state than minutes-weighted
    season aggregates. This is a roster snapshot, not a transaction ledger.
    """

    schedule = _regular_season_games(Path(schedule_path))
    players_root = Path(processed_players_dir)
    rows: list[pd.DataFrame] = []
    for game in schedule.itertuples(index=False):
        player_path = players_root / f"{game.game_id}.parquet"
        if not player_path.exists():
            continue
        frame = pd.read_parquet(
            player_path,
            columns=["personId", "name", "team_tricode"],
        ).rename(
            columns={
                "personId": "player_id",
                "name": "player_name",
                "team_tricode": "team",
            }
        )
        frame["player_id"] = frame["player_id"].astype(str)
        frame["game_id"] = str(game.game_id)
        frame["game_date"] = game.game_date
        rows.append(frame)
    if not rows:
        raise ValueError(f"No regular-season player game rosters found for {season}")

    observed = pd.concat(rows, ignore_index=True)
    observed = observed.dropna(subset=["player_id", "team"]).copy()
    observed["team"] = observed["team"].astype(str)
    snapshot = (
        observed.sort_values(["player_id", "game_date", "game_id"], kind="stable")
        .groupby("player_id", as_index=False, sort=False)
        .tail(1)
        .loc[:, ["player_id", "player_name", "team", "game_id", "game_date"]]
        .rename(columns={"team": "source_team", "game_id": "last_regular_season_game_id"})
        .sort_values("player_id", kind="stable")
        .reset_index(drop=True)
    )
    if snapshot["player_id"].duplicated().any():
        raise ValueError("Final regular-season snapshot contains duplicate player IDs")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(target, index=False)
    return target


def _regular_season_games(schedule_path: Path) -> pd.DataFrame:
    payload = json.loads(schedule_path.read_text())
    dates = payload.get("leagueSchedule", {}).get("gameDates", [])
    rows: list[dict[str, str]] = []
    for game_date in dates:
        if not isinstance(game_date, dict):
            continue
        for game in game_date.get("games", []):
            if not isinstance(game, dict):
                continue
            game_id = str(game.get("gameId", ""))
            if not game_id.startswith("002"):
                continue
            game_date_est = game.get("gameDateEst")
            if not isinstance(game_date_est, str):
                continue
            rows.append({"game_id": game_id, "game_date": game_date_est})
    if not rows:
        raise ValueError(f"Schedule {schedule_path} has no completed regular-season games")
    games = pd.DataFrame(rows)
    games["game_date"] = pd.to_datetime(games["game_date"], utc=True)
    return games.drop_duplicates("game_id").sort_values(["game_date", "game_id"], kind="stable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a final observed regular-season player-team snapshot"
    )
    parser.add_argument("season", help="NBA season label, such as 2025-26")
    parser.add_argument(
        "--schedule-path",
        type=Path,
        help="Defaults to data/raw/scheduleleaguev2/<season>.json",
    )
    parser.add_argument(
        "--processed-players-dir",
        type=Path,
        default=Path("data/processed/players"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Defaults to "
            "data/curated/team_rosters/<season>/final_regular_season_snapshot.parquet"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    schedule_path = args.schedule_path or (
        Path("data/raw/scheduleleaguev2") / f"{args.season}.json"
    )
    output_path = args.output_path or Path(
        "data/curated/team_rosters"
        / args.season
        / "final_regular_season_snapshot.parquet"
    )
    print(
        build_final_regular_season_roster_snapshot(
            args.season,
            schedule_path=schedule_path,
            processed_players_dir=args.processed_players_dir,
            output_path=output_path,
        )
    )
