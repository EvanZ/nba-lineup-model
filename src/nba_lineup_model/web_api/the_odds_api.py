"""Quota-bounded NBA market snapshots from The Odds API."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.web_api.inference import TEAM_DISPLAY_NAMES

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
DEFAULT_BOOKMAKERS = (
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "polymarket",
)
DEFAULT_MARKETS = ("spreads", "h2h")
DEFAULT_MINIMUM_REMAINING_CREDITS = 50
DEFAULT_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_S3_URI = "s3://nba-gestalt-data/nba-gestalt/schedule"
DEFAULT_SEASON = "2026-27"


class OddsApiQuotaGuard(RuntimeError):
    """Raised before a request that would consume the protected quota reserve."""


def estimated_request_cost(
    *, bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS, markets: tuple[str, ...] = DEFAULT_MARKETS
) -> int:
    """Return the documented market-by-bookmaker-group request cost."""

    if not bookmakers or not markets:
        raise ValueError("At least one bookmaker and market are required")
    return len(markets) * math.ceil(len(bookmakers) / 10)


def collect_nba_odds_snapshot(
    *,
    api_key: str,
    state_dir: Path | str = DEFAULT_STATE_DIR,
    s3_uri: str | None = DEFAULT_S3_URI,
    season: str = DEFAULT_SEASON,
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
    markets: tuple[str, ...] = DEFAULT_MARKETS,
    minimum_remaining_credits: int = DEFAULT_MINIMUM_REMAINING_CREDITS,
    request_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch, archive, and publish one league-wide NBA odds snapshot.

    The API key is used only in the outgoing request. It is never serialized,
    logged, or embedded in a returned URL. A staged response is reused when a
    Dagster retry repeats the same ``request_id``, avoiding another API charge
    after an S3 failure.
    """

    if not api_key.strip():
        raise ValueError("GESTALT_ODDS_API_KEY is not configured")
    if minimum_remaining_credits < 0:
        raise ValueError("minimum_remaining_credits must be non-negative")

    collected_at = (now or datetime.now(UTC)).astimezone(UTC)
    state_root = Path(state_dir)
    odds_root = state_root / "odds-api"
    request_token = request_id or collected_at.strftime("%Y%m%dT%H%M%SZ")
    staging_path = state_root / "pending" / "odds-api" / f"{request_token}.json"

    if staging_path.is_file():
        snapshot = _read_object(staging_path)
    else:
        _enforce_quota_reserve(
            quota_path=odds_root / "quota.json",
            collected_at=collected_at,
            expected_cost=estimated_request_cost(bookmakers=bookmakers, markets=markets),
            minimum_remaining_credits=minimum_remaining_credits,
        )
        events, headers = _fetch_odds(
            api_key=api_key,
            bookmakers=bookmakers,
            markets=markets,
        )
        snapshot = {
            "schema_version": 1,
            "provider": "The Odds API",
            "sport_key": "basketball_nba",
            "season": season,
            "collected_at": collected_at.isoformat().replace("+00:00", "Z"),
            "request": {
                "bookmakers": list(bookmakers),
                "markets": list(markets),
                "odds_format": "american",
            },
            "quota": {
                "requests_used": _header_int(headers, "x-requests-used"),
                "requests_remaining": _header_int(headers, "x-requests-remaining"),
                "request_cost": _header_int(headers, "x-requests-last"),
                "minimum_remaining_credits": minimum_remaining_credits,
            },
            "events": events,
        }
        _write_json(staging_path, snapshot)

    timestamp = _snapshot_timestamp(snapshot)
    if s3_uri:
        prefix = f"{s3_uri.rstrip('/')}/season={season}/odds-api"
        _aws_s3_cp_exclusive(staging_path, f"{prefix}/snapshots/{timestamp}.json")

    latest_path = odds_root / "latest.json"
    quota_path = odds_root / "quota.json"
    _write_json(latest_path, snapshot)
    _write_json(
        quota_path,
        {
            "collected_at": snapshot["collected_at"],
            **dict(snapshot.get("quota", {})),
        },
    )
    if s3_uri:
        _aws_s3_cp(
            latest_path,
            f"{s3_uri.rstrip('/')}/season={season}/odds-api/latest.json",
        )
    staging_path.unlink(missing_ok=True)
    return snapshot


def read_latest_odds_snapshot(
    *, state_dir: Path | str = DEFAULT_STATE_DIR
) -> dict[str, Any] | None:
    path = Path(state_dir) / "odds-api" / "latest.json"
    return _read_object(path) if path.is_file() else None


def normalize_bookmaker_lines(
    snapshot: dict[str, Any] | None,
    *,
    bookmaker_key: str = "draftkings",
) -> dict[str, dict[str, Any]]:
    """Index one bookmaker's spread and moneyline by ``AWAY@HOME``."""

    if not isinstance(snapshot, dict):
        return {}
    inverse_team_names = {
        display_name: tricode
        for tricode, display_name in TEAM_DISPLAY_NAMES.items()
        if tricode not in {"NJN", "NOH", "NOK", "SEA", "VAN"}
    }
    result: dict[str, dict[str, Any]] = {}
    for event in snapshot.get("events", []):
        if not isinstance(event, dict):
            continue
        home_name = str(event.get("home_team", ""))
        away_name = str(event.get("away_team", ""))
        home_team = inverse_team_names.get(home_name)
        away_team = inverse_team_names.get(away_name)
        if not home_team or not away_team:
            continue
        bookmaker = next(
            (
                row
                for row in event.get("bookmakers", [])
                if isinstance(row, dict) and row.get("key") == bookmaker_key
            ),
            None,
        )
        if not isinstance(bookmaker, dict):
            continue
        markets = {
            str(row.get("key")): row
            for row in bookmaker.get("markets", [])
            if isinstance(row, dict)
        }
        spread = _outcomes_by_name(markets.get("spreads"))
        moneyline = _outcomes_by_name(markets.get("h2h"))
        home_spread = _number(spread.get(home_name, {}).get("point"))
        away_spread = _number(spread.get(away_name, {}).get("point"))
        home_moneyline = _integer(moneyline.get(home_name, {}).get("price"))
        away_moneyline = _integer(moneyline.get(away_name, {}).get("price"))
        if all(
            value is None for value in (home_spread, away_spread, home_moneyline, away_moneyline)
        ):
            continue
        market_updates = [
            str(row.get("last_update")) for row in markets.values() if row.get("last_update")
        ]
        commence_time = str(event.get("commence_time", ""))
        try:
            game_date = (
                datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                .astimezone(PACIFIC)
                .date()
                .isoformat()
            )
        except ValueError:
            continue
        result[f"{game_date}|{away_team}@{home_team}"] = {
            "provider": str(bookmaker.get("title") or bookmaker_key),
            "provider_key": bookmaker_key,
            "source": "The Odds API",
            "source_event_id": str(event.get("id", "")),
            "captured_at": snapshot.get("collected_at"),
            "last_update": max(market_updates, default=bookmaker.get("last_update")),
            "home_spread": home_spread,
            "away_spread": away_spread,
            "home_spread_open": None,
            "away_spread_open": None,
            "home_moneyline": home_moneyline,
            "away_moneyline": away_moneyline,
            "home_moneyline_open": None,
            "away_moneyline_open": None,
            "total": None,
            "total_open": None,
        }
    return result


def _fetch_odds(
    *, api_key: str, bookmakers: tuple[str, ...], markets: tuple[str, ...]
) -> tuple[list[dict[str, Any]], Any]:
    query = urllib.parse.urlencode(
        {
            "apiKey": api_key,
            "bookmakers": ",".join(bookmakers),
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
    )
    request = urllib.request.Request(
        f"{ODDS_API_URL}?{query}",
        headers={"User-Agent": "nba-gestalt/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
            headers = response.headers
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"The Odds API returned HTTP {error.code}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"The Odds API request failed: {error.reason}") from None
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("The Odds API did not return an event list")
    return payload, headers


def _enforce_quota_reserve(
    *,
    quota_path: Path,
    collected_at: datetime,
    expected_cost: int,
    minimum_remaining_credits: int,
) -> None:
    if not quota_path.is_file():
        return
    quota = _read_object(quota_path)
    previous_at = str(quota.get("collected_at", ""))
    if previous_at[:7] != collected_at.strftime("%Y-%m"):
        return
    remaining = _integer(quota.get("requests_remaining"))
    if remaining is not None and remaining - expected_cost < minimum_remaining_credits:
        raise OddsApiQuotaGuard(
            "The Odds API request was skipped to preserve the configured quota reserve: "
            f"remaining={remaining}, expected_cost={expected_cost}, "
            f"reserve={minimum_remaining_credits}"
        )


def _outcomes_by_name(market: object) -> dict[str, dict[str, Any]]:
    if not isinstance(market, dict):
        return {}
    return {
        str(row.get("name")): row
        for row in market.get("outcomes", [])
        if isinstance(row, dict) and row.get("name")
    }


def _number(value: object) -> float | None:
    try:
        return float(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    try:
        return int(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _header_int(headers: Any, name: str) -> int | None:
    return _integer(headers.get(name))


def _snapshot_timestamp(snapshot: dict[str, Any]) -> str:
    value = datetime.fromisoformat(str(snapshot["collected_at"]).replace("Z", "+00:00"))
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _aws_s3_cp(source: Path, destination: str) -> None:
    subprocess.run(
        ["aws", "s3", "cp", str(source), destination, "--only-show-errors"],
        check=True,
    )


def _aws_s3_cp_exclusive(source: Path, destination: str) -> None:
    if not destination.startswith("s3://"):
        raise ValueError(f"S3 destination is invalid: {destination}")
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(source),
            destination,
            "--no-overwrite",
            "--only-show-errors",
        ],
        check=True,
    )
    existing = subprocess.run(
        ["aws", "s3", "cp", destination, "-", "--only-show-errors"],
        capture_output=True,
        check=True,
    ).stdout
    if existing != source.read_bytes():
        raise RuntimeError(
            f"Refusing to overwrite immutable S3 object with different bytes: {destination}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive one league-wide NBA odds snapshot")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--s3-uri", default=DEFAULT_S3_URI)
    parser.add_argument("--no-s3", action="store_true")
    parser.add_argument(
        "--minimum-remaining-credits",
        type=int,
        default=DEFAULT_MINIMUM_REMAINING_CREDITS,
    )
    args = parser.parse_args()
    snapshot = collect_nba_odds_snapshot(
        api_key=os.environ.get("GESTALT_ODDS_API_KEY", ""),
        state_dir=args.state_dir,
        s3_uri=None if args.no_s3 else args.s3_uri,
        minimum_remaining_credits=args.minimum_remaining_credits,
    )
    quota = snapshot.get("quota", {})
    print(
        f"Archived {len(snapshot['events'])} NBA odds events; "
        f"credits remaining: {quota.get('requests_remaining', 'unknown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
