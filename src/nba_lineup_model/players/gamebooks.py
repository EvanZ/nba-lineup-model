"""Official NBA gamebook retrieval and inactive-player extraction.

Historical traditional box scores identify active DNPs but omit much of the
inactive roster. Official scorer's reports restore that list and, in many
seasons, also retain an explicit reason such as injury or G League assignment.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep

import httpx
import pandas as pd
from pypdf import PdfReader

from nba_lineup_model.season.fetch import select_catalog_games
from nba_lineup_model.season.schema import CatalogGame, validate_season
from nba_lineup_model.season.storage import read_game_catalog

GAMEBOOK_BASE_URL = "https://statsdmz.nba.com/pdfs"
_INACTIVE_PREFIX = "Inactive:"
_PLAYER_WITH_REASON = re.compile(
    r"\s*(?P<name>[^,(]+?)\s*(?:\((?P<reason>[^)]*)\))?\s*(?:,|$)"
)
_WHITESPACE = re.compile(r"\s+")


class GamebookError(RuntimeError):
    """Raised when an official scorer's report cannot be fetched or parsed."""


@dataclass(frozen=True)
class GamebookInactivePlayer:
    """One inactive player as reported in an official gamebook."""

    team_label: str
    player_name: str
    reason: str | None


@dataclass(frozen=True)
class GamebookFetchSummary:
    """Terminal result for a resumable batch of official scorer's reports."""

    selected_games: int
    downloaded_games: int
    cached_games: int
    failed_games: int


def gamebook_url(*, game_date: date, away_team: str, home_team: str) -> str:
    """Return the stable official scorer's-report URL for one game."""

    stamp = game_date.strftime("%Y%m%d")
    away = away_team.upper()
    home = home_team.upper()
    return f"{GAMEBOOK_BASE_URL}/{stamp}/{stamp}_{away}{home}_book.pdf"


def gamebook_cache_path(*, raw_dir: Path | str, game_id: str) -> Path:
    """Return the immutable raw cache location for one gamebook."""

    return Path(raw_dir) / "gamebooks" / f"{game_id}.pdf"


def fetch_gamebook(
    *,
    game_id: str,
    game_date: date,
    away_team: str,
    home_team: str,
    raw_dir: Path | str = Path("data/raw"),
    refresh: bool = False,
    http_client: httpx.Client | None = None,
) -> Path:
    """Fetch and retain one official gamebook without replacing a valid cache."""

    path = gamebook_cache_path(raw_dir=raw_dir, game_id=game_id)
    if path.exists() and not refresh:
        return path
    url = gamebook_url(game_date=game_date, away_team=away_team, home_team=home_team)
    owns_client = http_client is None
    client = http_client or httpx.Client(
        timeout=60.0,
        headers={"Referer": "https://www.nba.com/", "User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
    )
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise GamebookError(f"NBA gamebook request failed for {game_id}: {url}") from error
    finally:
        if owns_client:
            client.close()
    if not response.content.startswith(b"%PDF"):
        raise GamebookError(f"NBA gamebook response is not a PDF for {game_id}: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pdf.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    return path


def fetch_season_gamebooks(
    seasons: list[str],
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    raw_dir: Path | str = Path("data/raw"),
    refresh: bool = False,
    min_request_interval_seconds: float = 0.25,
    limit: int | None = None,
) -> GamebookFetchSummary:
    """Fetch regular-season official gamebooks sequentially and resumably."""

    if min_request_interval_seconds < 0:
        raise ValueError("Minimum request interval cannot be negative")
    normalized_seasons = [validate_season(season) for season in seasons]
    if not normalized_seasons:
        raise ValueError("At least one season is required")
    catalog = read_game_catalog(catalog_path)
    games: list[CatalogGame] = []
    for season in normalized_seasons:
        games.extend(select_catalog_games(catalog, season=season, season_types=["regular"]))
    games.sort(key=lambda game: (game.game_date, game.game_id))
    if limit is not None:
        games = games[:limit]

    downloaded = 0
    cached = 0
    failed = 0
    client = httpx.Client(
        timeout=60.0,
        headers={"Referer": "https://www.nba.com/", "User-Agent": "Mozilla/5.0"},
        follow_redirects=True,
    )
    try:
        for index, game in enumerate(games, start=1):
            path = gamebook_cache_path(raw_dir=raw_dir, game_id=game.game_id)
            if path.exists() and not refresh:
                cached += 1
                continue
            try:
                fetch_gamebook(
                    game_id=game.game_id,
                    game_date=game.game_date,
                    away_team=game.away_team_tricode,
                    home_team=game.home_team_tricode,
                    raw_dir=raw_dir,
                    refresh=refresh,
                    http_client=client,
                )
            except GamebookError as error:
                failed += 1
                print(f"[{index}/{len(games)}] failed {game.game_id}: {error}", flush=True)
            else:
                downloaded += 1
                print(f"[{index}/{len(games)}] retained {game.game_id}", flush=True)
            if min_request_interval_seconds > 0 and index < len(games):
                sleep(min_request_interval_seconds)
    finally:
        client.close()
    return GamebookFetchSummary(
        selected_games=len(games),
        downloaded_games=downloaded,
        cached_games=cached,
        failed_games=failed,
    )


def audit_gamebook_reason_coverage(
    seasons: list[str],
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    raw_dir: Path | str = Path("data/raw"),
) -> pd.DataFrame:
    """Summarize inactive-reason coverage in cached official gamebooks."""

    catalog = read_game_catalog(catalog_path)
    rows: list[dict[str, object]] = []
    for season in [validate_season(value) for value in seasons]:
        games = select_catalog_games(catalog, season=season, season_types=["regular"])
        for game in games:
            path = gamebook_cache_path(raw_dir=raw_dir, game_id=game.game_id)
            if not path.exists():
                rows.append(
                    {
                        "season": season,
                        "game_id": game.game_id,
                        "cached": False,
                        "inactive_player_count": 0,
                        "reasoned_inactive_count": 0,
                        "unreasoned_inactive_count": 0,
                        "parse_error": None,
                    }
                )
                continue
            try:
                records = parse_inactive_players(
                    gamebook_text(path),
                )
            except GamebookError as error:
                rows.append(
                    {
                        "season": season,
                        "game_id": game.game_id,
                        "cached": True,
                        "inactive_player_count": 0,
                        "reasoned_inactive_count": 0,
                        "unreasoned_inactive_count": 0,
                        "parse_error": str(error),
                    }
                )
                continue
            reasoned = sum(record.reason is not None for record in records)
            rows.append(
                {
                    "season": season,
                    "game_id": game.game_id,
                    "cached": True,
                    "inactive_player_count": len(records),
                    "reasoned_inactive_count": reasoned,
                    "unreasoned_inactive_count": len(records) - reasoned,
                    "parse_error": None,
                }
            )
    return pd.DataFrame(rows)


def gamebook_text(path: Path | str) -> str:
    """Extract the final-box page from one retained gamebook PDF.

    The first page contains the final team boxes, active DNPs, and inactive
    list. Later pages repeat that information alongside play-by-play detail,
    so reading only page one makes a full archive audit practical.
    """

    try:
        reader = PdfReader(str(path))
        text = reader.pages[0].extract_text() or ""
    except Exception as error:  # pypdf exposes several parser-specific errors.
        raise GamebookError(f"Could not extract gamebook text: {path}") from error
    if not text.strip():
        raise GamebookError(f"Gamebook has no extractable text: {path}")
    return text


def parse_inactive_players(
    text: str,
    *,
    team_labels: tuple[str, ...] = (),
) -> list[GamebookInactivePlayer]:
    """Return unique inactive records from scorer's-report text.

    Gamebooks repeat the final-box page several times. Keeping the first
    normalized record makes output deterministic and preserves the exact
    reason, when the historical report supplies one.
    """

    records: list[GamebookInactivePlayer] = []
    seen: set[tuple[str, str, str | None]] = set()
    for line in text.splitlines():
        if _INACTIVE_PREFIX not in line:
            continue
        content = line.split(_INACTIVE_PREFIX, maxsplit=1)[1].strip()
        for team_label, players in _inactive_sections(content, team_labels=team_labels):
            position = 0
            while position < len(players):
                match = _PLAYER_WITH_REASON.match(players, position)
                if match is None:
                    break
                name = _normalize(match.group("name"))
                reason = _normalize_optional(match.group("reason"))
                if name:
                    key = (team_label, name, reason)
                    if key not in seen:
                        seen.add(key)
                        records.append(
                            GamebookInactivePlayer(
                                team_label=team_label,
                                player_name=name,
                                reason=reason,
                            )
                        )
                if match.end() <= position:
                    break
                position = match.end()
    return records


def _inactive_sections(
    content: str,
    *,
    team_labels: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Split one report line into team-specific inactive-player lists.

    Newer reports use one `Inactive:` line per team. Older reports can place
    both teams on a single line, so callers should provide the two official
    team nicknames to make that boundary unambiguous. Without those labels,
    retain the first section conservatively instead of mistaking a hyphen in
    an injury description for a new team.
    """

    normalized_labels = tuple(_normalize(label) for label in team_labels if _normalize(label))
    if normalized_labels:
        alternatives = "|".join(re.escape(label) for label in normalized_labels)
        pattern = re.compile(rf"(?P<team>{alternatives})\s*-\s*", flags=re.IGNORECASE)
        matches = list(pattern.finditer(content))
        sections: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else None
            players = content[match.end() : end].strip().strip(",")
            sections.append((_normalize(match.group("team")), players))
        return sections
    match = re.match(r"(?P<team>[^-]+?)\s*-\s*(?P<players>.*)$", content)
    if match is None:
        return []
    return [(_normalize(match.group("team")), match.group("players").strip())]


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _normalize_optional(value: str | None) -> str | None:
    return _normalize(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    """Build the gamebook acquisition CLI parser."""

    parser = argparse.ArgumentParser(description="Fetch official NBA scorer's-report PDFs")
    parser.add_argument("--season", action="append", dest="seasons", required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/games.parquet"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--min-request-interval", type=float, default=0.25)
    return parser


def main() -> None:
    """Fetch one or more historical gamebook seasons."""

    args = build_parser().parse_args()
    summary = fetch_season_gamebooks(
        args.seasons,
        catalog_path=args.catalog,
        raw_dir=args.raw_dir,
        refresh=args.refresh,
        min_request_interval_seconds=args.min_request_interval,
        limit=args.limit,
    )
    print(
        "Gamebooks: "
        f"selected={summary.selected_games}, downloaded={summary.downloaded_games}, "
        f"cached={summary.cached_games}, failed={summary.failed_games}"
    )


if __name__ == "__main__":
    main()
