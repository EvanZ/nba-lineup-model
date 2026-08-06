"""Direct NBA active-roster ingestion for pre-season player profiles."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC
from pathlib import Path

import pandas as pd

from nba_lineup_model.players.source import PlayerStatsCache, PlayerStatsClient
from nba_lineup_model.season.schema import validate_season

# NBA franchise identifiers are stable. Keeping the active league set explicit
# prevents historical All-Star and legacy team IDs from entering roster pulls.
NBA_TEAM_IDS = tuple(range(1610612737, 1610612767))
TEAM_ABBREVIATIONS = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}


def collect_team_rosters(
    season: str,
    *,
    raw_dir: Path | str = Path("data/raw"),
    curated_dir: Path | str = Path("data/curated"),
    team_ids: tuple[int, ...] = NBA_TEAM_IDS,
    request_delay_seconds: float = 0.5,
    refresh: bool = False,
    client: PlayerStatsClient | None = None,
) -> Path:
    """Fetch all active NBA rosters and publish one normalized season table."""

    season = validate_season(season)
    if (
        not team_ids
        or len(set(team_ids)) != len(team_ids)
        or any(team_id <= 0 for team_id in team_ids)
    ):
        raise ValueError("Team roster ingestion requires unique positive team IDs")
    if request_delay_seconds < 0:
        raise ValueError("Request delay cannot be negative")
    owns_client = client is None
    active_client = client or PlayerStatsClient(cache=PlayerStatsCache(raw_dir))
    responses = []
    try:
        for index, team_id in enumerate(team_ids):
            response = active_client.fetch_team_roster(season, team_id, use_cache=not refresh)
            responses.append(response)
            if request_delay_seconds and index + 1 < len(team_ids):
                time.sleep(request_delay_seconds)
    finally:
        if owns_client:
            active_client.close()
    frame = (
        pd.concat(
            [
                team_roster_frame(response.payload, season=season, team_id=response.team_id)
                for response in responses
            ],
            ignore_index=True,
        )
        .sort_values(["team_id", "player_name"], kind="stable")
        .reset_index(drop=True)
    )
    target = Path(curated_dir) / "team_rosters" / season
    target.mkdir(parents=True, exist_ok=True)
    output = target / "part-00000.parquet"
    frame.to_parquet(output, index=False)
    (target / "_manifest.json").write_text(
        json.dumps(
            {
                "season": season,
                "endpoint": "commonteamroster",
                "team_count": len(team_ids),
                "player_team_rows": len(frame),
                "raw_paths": [
                    str(
                        active_client.cache.path_for(
                            response.endpoint, season, None, response.team_id
                        )
                    )
                    for response in responses
                ],
                "source_fetched_at": [
                    response.fetched_at.astimezone(UTC).isoformat() for response in responses
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return output


def team_roster_frame(
    payload: dict[str, object], *, season: str, team_id: int | None
) -> pd.DataFrame:
    """Normalize a CommonTeamRoster response without lossy identifier coercion."""

    if team_id is None:
        raise ValueError("Team roster response lacks its requested team ID")
    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        raise ValueError("Team roster response has no resultSets array")
    result = next(
        (
            item
            for item in result_sets
            if isinstance(item, dict) and str(item.get("name", "")).lower() == "commonteamroster"
        ),
        None,
    )
    if not isinstance(result, dict):
        raise ValueError("Team roster response has no CommonTeamRoster result set")
    headers = result.get("headers")
    rows = result.get("rowSet")
    if not isinstance(headers, list) or not isinstance(rows, list):
        raise ValueError("Team roster result set has invalid headers or rowSet")
    positions = {str(name): index for index, name in enumerate(headers)}
    required = {
        "TeamID",
        "PLAYER",
        "POSITION",
        "HEIGHT",
        "WEIGHT",
        "BIRTH_DATE",
        "AGE",
        "EXP",
        "SCHOOL",
        "PLAYER_ID",
        "HOW_ACQUIRED",
    }
    if required - set(positions):
        raise ValueError(
            f"Team roster response missing columns: {sorted(required - set(positions))}"
        )
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) < len(headers):
            raise ValueError("Team roster row does not match headers")
        height_raw = _optional_text(row, positions, "HEIGHT")
        records.append(
            {
                "season": season,
                "team_id": str(row[positions["TeamID"]]),
                "team_abbreviation": TEAM_ABBREVIATIONS.get(int(row[positions["TeamID"]])),
                "player_id": str(row[positions["PLAYER_ID"]]),
                "player_name": str(row[positions["PLAYER"]]),
                "listed_position": _optional_text(row, positions, "POSITION"),
                "height_raw": height_raw,
                "height_inches": _height_inches(height_raw),
                "weight_pounds": _optional_int(row, positions, "WEIGHT"),
                "birth_date": _optional_text(row, positions, "BIRTH_DATE"),
                "age": _optional_float(row, positions, "AGE"),
                "experience": _optional_text(row, positions, "EXP"),
                "school": _optional_text(row, positions, "SCHOOL"),
                "how_acquired": _optional_text(row, positions, "HOW_ACQUIRED"),
                "supplemental_status": _optional_text(row, positions, "SUPPLEMENTAL_STATUS"),
            }
        )
    frame = pd.DataFrame(records)
    if frame["player_id"].duplicated().any():
        raise ValueError(f"Team roster response duplicates player IDs for team {team_id}")
    return frame.astype(
        {
            "team_id": "string",
            "team_abbreviation": "string",
            "player_id": "string",
            "player_name": "string",
            "listed_position": "string",
            "height_raw": "string",
            "birth_date": "string",
            "experience": "string",
            "school": "string",
            "how_acquired": "string",
            "supplemental_status": "string",
        }
    )


def _optional_text(row: list[object], positions: dict[str, int], name: str) -> str | None:
    value = row[positions[name]] if name in positions else None
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_float(row: list[object], positions: dict[str, int], name: str) -> float | None:
    value = _optional_text(row, positions, name)
    return float(value) if value is not None else None


def _optional_int(row: list[object], positions: dict[str, int], name: str) -> int | None:
    value = _optional_text(row, positions, name)
    return int(value) if value is not None else None


def _height_inches(value: str | None) -> int | None:
    if value is None or "-" not in value:
        return None
    feet, inches = value.split("-", maxsplit=1)
    return int(feet) * 12 + int(inches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch active NBA team rosters")
    parser.add_argument("season", help="NBA season in YYYY-YY format, for example 2026-27")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--request-delay-seconds", type=float, default=0.5)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    output = collect_team_rosters(
        args.season,
        raw_dir=args.raw_dir,
        curated_dir=args.curated_dir,
        request_delay_seconds=args.request_delay_seconds,
        refresh=args.refresh,
    )
    print(f"Team rosters: {output}")


if __name__ == "__main__":
    main()
