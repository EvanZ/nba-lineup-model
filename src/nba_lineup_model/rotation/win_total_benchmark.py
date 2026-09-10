"""Historical preseason market win-total benchmark for rotation forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

import httpx
import pandas as pd

from nba_lineup_model.season.schedule import SeasonScheduleCache
from nba_lineup_model.season.schema import validate_season

BASKETBALL_REFERENCE_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_preseason_odds.html"
BR_TEAM_CODE_ALIASES = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
_TABLE_RE = re.compile(
    r'<table\b[^>]*\bid="NBA_preseason_odds"[^>]*>(?P<body>.*?)</table>', re.DOTALL
)
_ROW_RE = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.DOTALL)
_TEAM_RE = re.compile(r"/teams/(?P<team>[A-Z]{3})/")
_WIN_TOTAL_RE = re.compile(
    r'<td\b[^>]*\bdata-stat="wins_ou"[^>]*>(?P<value>.*?)</td>', re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


class WinTotalBenchmarkError(RuntimeError):
    """Raised when a market benchmark source cannot be fetched or parsed."""


def source_url_for(season: str) -> str:
    """Return Basketball-Reference's archived preseason-odds page for a season."""

    validate_season(season)
    return BASKETBALL_REFERENCE_URL.format(year=int(season[:4]) + 1)


def raw_path_for(season: str, *, raw_dir: Path | str = Path("data/raw")) -> Path:
    return Path(raw_dir) / "basketball_reference" / "preseason_odds" / f"{season}.html"


def fetch_preseason_win_totals(
    season: str,
    *,
    raw_dir: Path | str = Path("data/raw"),
    refresh: bool = False,
    http_client: httpx.Client | None = None,
) -> Path:
    """Fetch and byte-preserve a published preseason win-total table."""

    season = validate_season(season)
    path = raw_path_for(season, raw_dir=raw_dir)
    metadata_path = path.with_suffix(".meta.json")
    if path.exists() and not refresh:
        _validate_cached_source(path, metadata_path)
        return path

    owns_client = http_client is None
    client = http_client or httpx.Client(
        timeout=60.0,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml"},
        follow_redirects=True,
    )
    url = source_url_for(season)
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise WinTotalBenchmarkError(f"Preseason win-total request failed: {season}") from exc
    finally:
        if owns_client:
            client.close()

    parse_preseason_win_totals(response.text, season=season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    metadata_path.write_text(
        json.dumps(
            {
                "season": season,
                "source_url": str(response.url),
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return path


def parse_preseason_win_totals(html: str, *, season: str) -> pd.DataFrame:
    """Extract team tricodes and the opening regular-season win total."""

    validate_season(season)
    match = _TABLE_RE.search(html)
    if match is None:
        raise WinTotalBenchmarkError(f"Preseason odds table is missing: {season}")
    records: list[dict[str, object]] = []
    for row in _ROW_RE.finditer(match.group("body")):
        team_match = _TEAM_RE.search(row.group("body"))
        total_match = _WIN_TOTAL_RE.search(row.group("body"))
        if team_match is None or total_match is None:
            continue
        value = _plain_text(total_match.group("value"))
        try:
            opening_win_total = float(value)
        except ValueError as exc:
            raise WinTotalBenchmarkError(
                f"Invalid opening win total for {season}: {value!r}"
            ) from exc
        team = BR_TEAM_CODE_ALIASES.get(team_match.group("team"), team_match.group("team"))
        records.append({"season": season, "team": team, "opening_win_total": opening_win_total})
    output = pd.DataFrame.from_records(records)
    if len(output) != 30 or output["team"].duplicated().any():
        raise WinTotalBenchmarkError(
            f"Expected 30 unique preseason win totals for {season}; found {len(output)}"
        )
    return output.sort_values("team", kind="stable").reset_index(drop=True)


def actual_regular_season_wins(season: str) -> pd.DataFrame:
    """Derive regular-season wins directly from cached official NBA schedule scores."""

    response = SeasonScheduleCache().read(season)
    if response is None:
        raise FileNotFoundError(f"No cached official NBA schedule for {season}")
    records: list[dict[str, object]] = []
    for game_date in response.payload.get("leagueSchedule", {}).get("gameDates", []):
        for game in game_date.get("games", []):
            game_id = str(game.get("gameId", ""))
            if not game_id.startswith("002") or int(game.get("gameStatus", 0)) != 3:
                continue
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            if home.get("score") is None or away.get("score") is None:
                raise WinTotalBenchmarkError(f"Final schedule game lacks a score: {game_id}")
            records.extend(
                (
                    {
                        "team": home["teamTricode"],
                        "actual_wins": int(home["score"] > away["score"]),
                    },
                    {
                        "team": away["teamTricode"],
                        "actual_wins": int(away["score"] > home["score"]),
                    },
                )
            )
    wins = pd.DataFrame.from_records(records).groupby("team", as_index=False).agg(
        actual_wins=("actual_wins", "sum"), games=("actual_wins", "size")
    )
    if len(wins) != 30 or not wins["games"].eq(82).all():
        raise WinTotalBenchmarkError(f"Incomplete 82-game regular-season results: {season}")
    return wins.loc[:, ["team", "actual_wins"]]


def build_win_total_benchmark(
    seasons: list[str],
    *,
    raw_dir: Path | str = Path("data/raw"),
    output_path: Path | str = Path("data/analytical/win_total_benchmark/part-00000.parquet"),
    refresh: bool = False,
) -> pd.DataFrame:
    """Materialize a season-level market benchmark with official final wins."""

    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = fetch_preseason_win_totals(season, raw_dir=raw_dir, refresh=refresh)
        metadata = json.loads(path.with_suffix(".meta.json").read_text())
        market = parse_preseason_win_totals(path.read_text(), season=season)
        actual = actual_regular_season_wins(season)
        joined = market.merge(actual, on="team", how="inner", validate="one_to_one")
        if len(joined) != 30:
            raise WinTotalBenchmarkError(f"Could not join all market rows to final wins: {season}")
        joined["market_error"] = joined["opening_win_total"] - joined["actual_wins"]
        joined["source_url"] = metadata["source_url"]
        joined["source_fetched_at"] = metadata["fetched_at"]
        frames.append(joined)
    output = pd.concat(frames, ignore_index=True).sort_values(
        ["season", "team"], kind="stable"
    ).reset_index(drop=True)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(target, index=False)
    return output


def _plain_text(value: str) -> str:
    return unescape(_TAG_RE.sub("", value)).strip()


def _validate_cached_source(path: Path, metadata_path: Path) -> None:
    if not metadata_path.exists():
        raise WinTotalBenchmarkError(f"Cached source lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise WinTotalBenchmarkError(f"Cached source hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seasons", nargs="+", help="Season labels, e.g. 2024-25 2025-26")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/analytical/win_total_benchmark/part-00000.parquet"),
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    output = build_win_total_benchmark(
        args.seasons,
        raw_dir=args.raw_dir,
        output_path=args.output_path,
        refresh=args.refresh,
    )
    print(f"Published {len(output)} preseason market rows to {args.output_path}")


if __name__ == "__main__":
    main()
