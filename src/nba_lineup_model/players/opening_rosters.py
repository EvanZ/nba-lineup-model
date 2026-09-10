"""Reconstruct and validate historical NBA opening-day roster snapshots.

The official NBA player-movement feed begins on 2015-07-01.  For each team,
the mart begins with the prior season's final observed game roster, applies
player movements strictly before that team's first regular-season game date,
and then reconciles missing membership against that opening game's box-score
roster.  Same-day movements are deliberately excluded because the source has
no transaction timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache
from nba_lineup_model.normalize.boxscore import boxscore_players_frame
from nba_lineup_model.normalize.stats_v3 import adapt_stats_v3_boxscore
from nba_lineup_model.players.rosters import TEAM_ABBREVIATIONS
from nba_lineup_model.players.season_roster_snapshot import (
    final_regular_season_roster_frame,
    player_game_roster_frame,
)
from nba_lineup_model.season.schema import validate_season

PLAYER_MOVEMENT_URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
SUPPORTED_TRANSACTION_TYPES = frozenset(
    {"AwardOnWaivers", "ContractConverted", "Signing", "Trade", "Waive"}
)
_ADD_TRANSACTION_TYPES = SUPPORTED_TRANSACTION_TYPES - {"Waive"}


class OpeningRosterError(RuntimeError):
    """Raised when an opening-roster source cannot be read or reconciled."""


def player_movement_cache_path(raw_dir: Path | str = Path("data/raw")) -> Path:
    """Return the byte-preserved official player-movement cache path."""

    return Path(raw_dir) / "player_movement" / "NBA_Player_Movement.json"


def fetch_player_movement(
    *,
    raw_dir: Path | str = Path("data/raw"),
    refresh: bool = False,
    http_client: httpx.Client | None = None,
) -> Path:
    """Fetch the official NBA movement feed once and preserve its provenance."""

    path = player_movement_cache_path(raw_dir)
    metadata_path = path.with_suffix(".meta.json")
    if path.exists() and not refresh:
        if not metadata_path.exists():
            raise OpeningRosterError(f"Cached movement feed lacks metadata: {path}")
        _validate_cached_movement(path, metadata_path)
        return path

    owns_client = http_client is None
    client = http_client or httpx.Client(
        timeout=60.0,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nba.com/",
            "User-Agent": "Mozilla/5.0",
        },
        follow_redirects=True,
    )
    try:
        response = client.get(PLAYER_MOVEMENT_URL)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OpeningRosterError("NBA player-movement request failed") from exc
    finally:
        if owns_client:
            client.close()

    raw_body = response.content
    _load_movement_payload(raw_body, source=PLAYER_MOVEMENT_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(raw_body, path)
    _atomic_write_text(
        json.dumps(
            {
                "url": str(response.url),
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(raw_body).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        metadata_path,
    )
    return path


def build_opening_roster_mart(
    season: str,
    *,
    transactions_path: Path | str,
    raw_dir: Path | str = Path("data/raw"),
    processed_players_dir: Path | str = Path("data/processed/players"),
    curated_dir: Path | str = Path("data/curated"),
) -> Path:
    """Publish one validated historical opening-roster snapshot.

    When the opening-game roster is available, it defines published membership.
    Transaction-only candidates remain in a separate audit artifact; this
    prevents incomplete waiver coverage from inflating the usable roster.
    """

    season = validate_season(season)
    prior_season = _previous_season(season)
    raw_root = Path(raw_dir)
    curated_root = Path(curated_dir)
    schedule_path = raw_root / "scheduleleaguev2" / f"{season}.json"
    prior_schedule_path = raw_root / "scheduleleaguev2" / f"{prior_season}.json"
    if not schedule_path.exists() or not prior_schedule_path.exists():
        raise OpeningRosterError(
            f"Opening roster reconstruction requires schedules for {prior_season} and {season}"
        )

    movements = _movement_frame(Path(transactions_path))
    prior_snapshot = final_regular_season_roster_frame(
        prior_season,
        schedule_path=prior_schedule_path,
        processed_players_dir=processed_players_dir,
    )
    opening_games = _opening_games(schedule_path)
    output, validation, reconciliation, transaction_only_candidates = _reconstruct_opening_rosters(
        season=season,
        prior_snapshot=prior_snapshot,
        movements=movements,
        opening_games=opening_games,
        processed_players_dir=Path(processed_players_dir),
    )

    target = curated_root / "opening_rosters" / season
    target.mkdir(parents=True, exist_ok=True)
    output_path = target / "part-00000.parquet"
    output.to_parquet(output_path, index=False)
    validation.to_parquet(target / "validation.parquet", index=False)
    reconciliation.to_parquet(target / "reconciliation.parquet", index=False)
    transaction_only_candidates.to_parquet(
        target / "transaction_only_candidates.parquet",
        index=False,
    )
    _atomic_write_text(
        json.dumps(
            {
                "season": season,
                "source": {
                    "player_movement_url": PLAYER_MOVEMENT_URL,
                    "player_movement_path": str(Path(transactions_path)),
                    "opening_cutoff": "strictly before each team's first regular-season game date",
                },
                "prior_season": prior_season,
                "team_count": int(len(opening_games)),
                "player_team_rows": int(len(output)),
                "reconciled_player_team_rows": int(
                    (output["membership_source"] == "opening_game_reconciliation").sum()
                ),
                "transaction_only_candidate_rows": int(len(transaction_only_candidates)),
                "opening_games_missing_player_roster": int(
                    (~validation["opening_game_roster_available"]).sum()
                ),
            },
            indent=2,
        )
        + "\n",
        target / "_manifest.json",
    )
    return output_path


def recover_missing_opening_game_rosters(
    seasons: list[str],
    *,
    raw_dir: Path | str = Path("data/raw"),
    processed_players_dir: Path | str = Path("data/processed/players"),
    client: NbaCdnClient | None = None,
) -> list[Path]:
    """Recover only missing opening-game player rosters from official box scores.

    This deliberately does not run lineup or possession reconstruction. Those
    components can fail for an otherwise valid historical game, while player
    roster membership is independently available in the official box score.
    """

    raw_root = Path(raw_dir)
    target_root = Path(processed_players_dir)
    missing_games: dict[str, Path] = {}
    for season in seasons:
        schedule_path = raw_root / "scheduleleaguev2" / f"{validate_season(season)}.json"
        for game in _opening_games(schedule_path).itertuples(index=False):
            target = target_root / f"{game.game_id}.parquet"
            if not target.exists():
                missing_games[str(game.game_id)] = target
    if not missing_games:
        return []

    owns_client = client is None
    active_client = client or NbaCdnClient(cache=RawJsonCache(raw_root))
    recovered: list[Path] = []
    try:
        for game_id, target in sorted(missing_games.items()):
            stats_path = (
                raw_root / "stats" / "boxscoretraditionalv3" / f"{game_id}.json"
            )
            if stats_path.exists():
                payload = json.loads(stats_path.read_text())
                frame = boxscore_players_frame(adapt_stats_v3_boxscore(payload))
            else:
                response = active_client.fetch_boxscore(game_id, use_cache=True)
                frame = boxscore_players_frame(response.payload)
            if frame.empty:
                raise OpeningRosterError(f"Official box score has no players: {game_id}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
            recovered.append(target)
    finally:
        if owns_client:
            active_client.close()
    return recovered


def build_opening_roster_coverage(
    *,
    curated_dir: Path | str = Path("data/curated"),
) -> Path:
    """Publish one cross-season coverage table for completed opening-roster marts."""

    root = Path(curated_dir) / "opening_rosters"
    rows: list[dict[str, Any]] = []
    for season_path in sorted(path for path in root.iterdir() if path.is_dir()):
        roster_path = season_path / "part-00000.parquet"
        validation_path = season_path / "validation.parquet"
        candidates_path = season_path / "transaction_only_candidates.parquet"
        if not (roster_path.exists() and validation_path.exists() and candidates_path.exists()):
            continue
        roster = pd.read_parquet(roster_path)
        validation = pd.read_parquet(validation_path)
        candidates = pd.read_parquet(candidates_path)
        rows.append(
            {
                "season": season_path.name,
                "team_count": int(len(validation)),
                "opening_game_roster_team_count": int(
                    validation["opening_game_roster_available"].sum()
                ),
                "published_player_team_rows": int(len(roster)),
                "transaction_reconstruction_rows": int(
                    (roster["membership_source"] == "transaction_reconstruction").sum()
                ),
                "opening_game_reconciliation_rows": int(
                    (roster["membership_source"] == "opening_game_reconciliation").sum()
                ),
                "transaction_state_unvalidated_rows": int(
                    (roster["membership_source"] == "transaction_state_unvalidated").sum()
                ),
                "transaction_only_candidate_rows": int(len(candidates)),
            }
        )
    if not rows:
        raise OpeningRosterError(f"No completed opening-roster marts found in {root}")
    coverage = pd.DataFrame(rows).sort_values("season", kind="stable").reset_index(drop=True)
    target = root / "coverage.parquet"
    coverage.to_parquet(target, index=False)
    return target


def _reconstruct_opening_rosters(
    *,
    season: str,
    prior_snapshot: pd.DataFrame,
    movements: pd.DataFrame,
    opening_games: pd.DataFrame,
    processed_players_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_state = {
        str(row.player_id): {
            "team": str(row.source_team),
            "player_name": str(row.player_name),
            "last_regular_season_game_id": str(row.last_regular_season_game_id),
            "last_regular_season_game_date": pd.Timestamp(row.game_date).date(),
        }
        for row in prior_snapshot.itertuples(index=False)
    }
    roster_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    transaction_only_rows: list[dict[str, Any]] = []

    for game in opening_games.itertuples(index=False):
        state = _apply_movements(
            base_state,
            movements,
            start_after=max(
                value["last_regular_season_game_date"] for value in base_state.values()
            ),
            cutoff_before=game.game_date,
        )
        team_state = {
            player_id: value
            for player_id, value in state.items()
            if value["team"] == game.team
        }
        opening_roster_path = processed_players_dir / f"{game.game_id}.parquet"
        observed = _opening_game_roster(
            opening_roster_path,
            team=game.team,
        )
        reconciliation = _reconcile_opening_game_roster(
            team_state,
            observed,
            team=game.team,
        )
        for row in reconciliation:
            reconciliation_rows.append(
                {
                    "season": season,
                    "team": game.team,
                    "opening_game_id": game.game_id,
                    "opening_game_date": game.game_date,
                    **row,
                }
            )
        if observed:
            included_player_ids = set(observed)
            for player_id, value in team_state.items():
                if player_id not in included_player_ids:
                    transaction_only_rows.append(
                        {
                            "season": season,
                            "team": game.team,
                            "team_id": game.team_id,
                            "opening_game_id": game.game_id,
                            "opening_game_date": game.game_date,
                            "player_id": player_id,
                            "player_name": value["player_name"],
                            "candidate_reason": "absent_from_opening_game_roster",
                        }
                    )
        else:
            included_player_ids = set(team_state)
            for value in team_state.values():
                value["membership_source"] = "transaction_state_unvalidated"
        for player_id in included_player_ids:
            value = team_state[player_id]
            roster_rows.append(
                {
                    "season": season,
                    "team": game.team,
                    "team_id": game.team_id,
                    "player_id": player_id,
                    "player_name": value["player_name"],
                    "opening_game_id": game.game_id,
                    "opening_game_date": game.game_date,
                    "membership_source": value.get(
                        "membership_source", "transaction_reconstruction"
                    ),
                    "last_regular_season_game_id": value[
                        "last_regular_season_game_id"
                    ],
                    "last_regular_season_game_date": value[
                        "last_regular_season_game_date"
                    ],
                }
            )
        observed_ids = set(observed)
        reconstructed_ids = set(team_state) - {
            row["player_id"] for row in reconciliation
        }
        validation_rows.append(
            {
                "season": season,
                "team": game.team,
                "team_id": game.team_id,
                "opening_game_id": game.game_id,
                "opening_game_date": game.game_date,
                "opening_game_roster_available": opening_roster_path.exists(),
                "opening_game_player_count": len(observed_ids),
                "reconstructed_player_count": len(reconstructed_ids),
                "matched_player_count": len(observed_ids & reconstructed_ids),
                "reconciled_player_count": len(reconciliation),
                "unobserved_transaction_state_count": len(reconstructed_ids - observed_ids),
            }
        )

    if not roster_rows:
        raise OpeningRosterError(f"No opening roster rows produced for {season}")
    output = (
        pd.DataFrame(roster_rows)
        .sort_values(["team", "player_name", "player_id"], kind="stable")
        .reset_index(drop=True)
    )
    if output.duplicated(["team", "player_id"]).any():
        raise OpeningRosterError("Opening roster output contains duplicate player-team rows")
    return (
        output,
        pd.DataFrame(validation_rows).sort_values("team", kind="stable").reset_index(drop=True),
        pd.DataFrame(
            reconciliation_rows,
            columns=[
                "season",
                "team",
                "opening_game_id",
                "opening_game_date",
                "player_id",
                "player_name",
                "reconciliation_reason",
            ],
        )
        .sort_values(["team", "player_id"], kind="stable")
        .reset_index(drop=True),
        pd.DataFrame(
            transaction_only_rows,
            columns=[
                "season",
                "team",
                "team_id",
                "opening_game_id",
                "opening_game_date",
                "player_id",
                "player_name",
                "candidate_reason",
            ],
        )
        .sort_values(["team", "player_id"], kind="stable")
        .reset_index(drop=True),
    )


def _apply_movements(
    base_state: dict[str, dict[str, Any]],
    movements: pd.DataFrame,
    *,
    start_after: date,
    cutoff_before: date,
) -> dict[str, dict[str, Any]]:
    state = {player_id: dict(value) for player_id, value in base_state.items()}
    eligible = movements.loc[
        (movements["transaction_date"] > start_after)
        & (movements["transaction_date"] < cutoff_before)
    ].sort_values(["transaction_date", "additional_sort"], kind="stable")
    for row in eligible.itertuples(index=False):
        if row.transaction_type == "Waive":
            if row.player_id in state and state[row.player_id]["team"] == row.team:
                del state[row.player_id]
            continue
        if row.transaction_type in _ADD_TRANSACTION_TYPES:
            previous = state.get(row.player_id, {})
            state[row.player_id] = {
                "team": row.team,
                "player_name": previous.get("player_name", _name_from_slug(row.player_slug)),
                "last_regular_season_game_id": previous.get(
                    "last_regular_season_game_id", None
                ),
                "last_regular_season_game_date": previous.get(
                    "last_regular_season_game_date", None
                ),
            }
    return state


def _reconcile_opening_game_roster(
    team_state: dict[str, dict[str, Any]],
    observed: dict[str, str],
    *,
    team: str,
) -> list[dict[str, str]]:
    reconciliation: list[dict[str, str]] = []
    for player_id, player_name in observed.items():
        existing = team_state.get(player_id)
        if existing is not None:
            if existing["player_name"].strip():
                continue
        team_state[player_id] = {
            "team": team,
            "player_name": player_name,
            "last_regular_season_game_id": (
                existing.get("last_regular_season_game_id") if existing else None
            ),
            "last_regular_season_game_date": (
                existing.get("last_regular_season_game_date") if existing else None
            ),
            "membership_source": "opening_game_reconciliation",
        }
        reconciliation.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "reconciliation_reason": (
                    "missing_from_transaction_state"
                    if existing is None
                    else "missing_player_name_in_transaction_state"
                ),
            }
        )
    return reconciliation


def _movement_frame(path: Path) -> pd.DataFrame:
    payload = _load_movement_payload(path.read_bytes(), source=str(path))
    rows = payload["NBA_Player_Movement"]["rows"]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        transaction_type = str(row.get("Transaction_Type", "")).strip()
        if transaction_type not in SUPPORTED_TRANSACTION_TYPES:
            continue
        player_id = _id_text(row.get("PLAYER_ID"))
        team_id = _id_text(row.get("TEAM_ID"))
        team = TEAM_ABBREVIATIONS.get(int(team_id)) if team_id is not None else None
        if player_id is None or team is None:
            continue
        transaction_date = _date_value(row.get("TRANSACTION_DATE"))
        if transaction_date is None:
            continue
        normalized.append(
            {
                "transaction_date": transaction_date,
                "transaction_type": transaction_type,
                "player_id": player_id,
                "team": team,
                "player_slug": _slug_text(row.get("PLAYER_SLUG")) or "",
                "additional_sort": int(row.get("Additional_Sort", 0) or 0),
            }
        )
    if not normalized:
        raise OpeningRosterError(f"Movement feed has no supported player rows: {path}")
    return pd.DataFrame(normalized)


def _opening_games(schedule_path: Path) -> pd.DataFrame:
    payload = json.loads(schedule_path.read_text())
    game_dates = payload.get("leagueSchedule", {}).get("gameDates", [])
    rows: list[dict[str, Any]] = []
    for game_date in game_dates:
        if not isinstance(game_date, dict):
            continue
        for game in game_date.get("games", []):
            if not isinstance(game, dict) or not str(game.get("gameId", "")).startswith("002"):
                continue
            game_date_value = _date_value(game.get("gameDateEst") or game.get("gameDate"))
            if game_date_value is None:
                continue
            for side in ("homeTeam", "awayTeam"):
                team = game.get(side)
                if not isinstance(team, dict):
                    continue
                tricode = _slug_text(team.get("teamTricode"))
                team_id = _id_text(team.get("teamId"))
                if tricode is None or team_id is None:
                    continue
                rows.append(
                    {
                        "team": tricode.upper(),
                        "team_id": team_id,
                        "game_id": str(game["gameId"]),
                        "game_date": game_date_value,
                    }
                )
    if not rows:
        raise OpeningRosterError(f"Schedule has no regular-season games: {schedule_path}")
    return (
        pd.DataFrame(rows)
        .sort_values(["team", "game_date", "game_id"], kind="stable")
        .drop_duplicates("team", keep="first")
        .sort_values("team", kind="stable")
        .reset_index(drop=True)
    )


def _opening_game_roster(path: Path, *, team: str) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = player_game_roster_frame(path)
    team_rows = frame.loc[frame["team"].astype(str).str.upper() == team]
    return {
        str(row.player_id): str(row.player_name)
        for row in team_rows.dropna(subset=["player_id", "player_name"]).itertuples(index=False)
    }


def _previous_season(season: str) -> str:
    start_year = int(season[:4]) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _load_movement_payload(raw_body: bytes, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpeningRosterError(f"Movement source is not valid JSON: {source}") from exc
    rows = payload.get("NBA_Player_Movement", {}).get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise OpeningRosterError(f"Movement source has no NBA_Player_Movement rows: {source}")
    return payload


def _validate_cached_movement(path: Path, metadata_path: Path) -> None:
    raw_body = path.read_bytes()
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sha256") != hashlib.sha256(raw_body).hexdigest():
        raise OpeningRosterError(f"Cached movement feed hash mismatch: {path}")
    _load_movement_payload(raw_body, source=str(path))


def _id_text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return None
    text = str(value).strip()
    return text or None


def _slug_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _date_value(value: Any) -> date | None:
    text = _slug_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _name_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.replace("-", " ").split())


def _atomic_write_bytes(content: bytes, target: Path) -> None:
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(target)


def _atomic_write_text(content: str, target: Path) -> None:
    _atomic_write_bytes(content.encode("utf-8"), target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build transaction-reconstructed, opening-game-validated NBA rosters"
    )
    parser.add_argument("season", nargs="+", help="Season labels, such as 2024-25 2025-26")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--processed-players-dir", type=Path, default=Path("data/processed/players")
    )
    parser.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--refresh-movements", action="store_true")
    parser.add_argument(
        "--recover-missing-opening-games",
        action="store_true",
        help="Fetch official box scores for missing opening-game player rosters",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    transactions_path = fetch_player_movement(
        raw_dir=args.raw_dir,
        refresh=args.refresh_movements,
    )
    if args.recover_missing_opening_games:
        for path in recover_missing_opening_game_rosters(
            args.season,
            raw_dir=args.raw_dir,
            processed_players_dir=args.processed_players_dir,
        ):
            print(f"recovered opening-game player roster: {path}")
    for season in args.season:
        print(
            build_opening_roster_mart(
                season,
                transactions_path=transactions_path,
                raw_dir=args.raw_dir,
                processed_players_dir=args.processed_players_dir,
                curated_dir=args.curated_dir,
            )
        )
    print(build_opening_roster_coverage(curated_dir=args.curated_dir))
