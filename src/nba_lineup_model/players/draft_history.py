"""Direct NBA Draft History ingestion and normalized drafted-player profiles."""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path

import pandas as pd

from nba_lineup_model.players.source import PlayerStatsCache, PlayerStatsClient
from nba_lineup_model.season.schema import validate_season


def collect_draft_history(
    season: str,
    *,
    raw_dir: Path | str = Path("data/raw"),
    curated_dir: Path | str = Path("data/curated"),
    refresh: bool = False,
    client: PlayerStatsClient | None = None,
) -> Path:
    """Fetch and publish one drafted-player class without roster assumptions."""

    season = validate_season(season)
    owns_client = client is None
    active_client = client or PlayerStatsClient(cache=PlayerStatsCache(raw_dir))
    try:
        response = active_client.fetch_draft_history(season, use_cache=not refresh)
    finally:
        if owns_client:
            active_client.close()
    frame = draft_history_frame(response.payload, season=season)
    target = Path(curated_dir) / "draft_history" / season
    target.mkdir(parents=True, exist_ok=True)
    output = target / "part-00000.parquet"
    frame.to_parquet(output, index=False)
    (target / "_manifest.json").write_text(
        json.dumps(
            {
                "season": season,
                "endpoint": "drafthistory",
                "source_url": response.url,
                "source_fetched_at": response.fetched_at.astimezone(UTC).isoformat(),
                "player_count": len(frame),
                "raw_path": str(
                    active_client.cache.path_for(response.endpoint, season, None)
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return output


def draft_history_frame(payload: dict[str, object], *, season: str) -> pd.DataFrame:
    """Normalize NBA Stats `DraftHistory` result rows without float coercion."""

    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        raise ValueError("Draft History response has no resultSets array")
    result = next(
        (
            item
            for item in result_sets
            if isinstance(item, dict) and str(item.get("name", "")).lower() == "drafthistory"
        ),
        None,
    )
    if not isinstance(result, dict):
        raise ValueError("Draft History response has no DraftHistory result set")
    headers = result.get("headers")
    rows = result.get("rowSet")
    if not isinstance(headers, list) or not isinstance(rows, list):
        raise ValueError("Draft History result set has invalid headers or rowSet")
    positions = {str(name): index for index, name in enumerate(headers)}
    required = {"PERSON_ID", "PLAYER_NAME", "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK"}
    if required - set(positions):
        raise ValueError(
            f"Draft History response missing columns: {sorted(required - set(positions))}"
        )
    records = []
    for row in rows:
        if not isinstance(row, list) or len(row) < len(headers):
            raise ValueError("Draft History row does not match headers")
        records.append(
            {
                "season": season,
                "draft_year": int(season[:4]),
                "player_id": str(row[positions["PERSON_ID"]]),
                "player_name": str(row[positions["PLAYER_NAME"]]),
                "draft_round": int(row[positions["ROUND_NUMBER"]]),
                "draft_round_pick": int(row[positions["ROUND_PICK"]]),
                "draft_number": int(row[positions["OVERALL_PICK"]]),
                "draft_team_id": _optional_text(row, positions, "TEAM_ID"),
                "draft_team_abbreviation": _optional_text(row, positions, "TEAM_ABBREVIATION"),
                "affiliation": _optional_text(row, positions, "ORGANIZATION"),
            }
        )
    frame = pd.DataFrame(records).sort_values("draft_number", kind="stable").reset_index(drop=True)
    if frame["player_id"].duplicated().any() or frame["draft_number"].duplicated().any():
        raise ValueError("Draft History response has duplicate players or overall picks")
    return frame.astype(
        {
            "player_id": "string",
            "draft_team_id": "string",
            "draft_team_abbreviation": "string",
            "affiliation": "string",
        }
    )


def _optional_text(row: list[object], positions: dict[str, int], name: str) -> str | None:
    value = row[positions[name]] if name in positions else None
    return None if value in (None, "") else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch an NBA Draft History class")
    parser.add_argument("season", help="NBA season in YYYY-YY format, for example 2026-27")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    output = collect_draft_history(
        args.season, raw_dir=args.raw_dir, curated_dir=args.curated_dir, refresh=args.refresh
    )
    print(f"Draft History: {output}")


if __name__ == "__main__":
    main()
