"""Fetch and publish the official NBA season schedule."""

import json
import re
from datetime import UTC, date, datetime, timedelta

import requests
from dagster import MaterializeResult, asset

from ...resources.nba_schedule import SOURCE_URL, NBAScheduleResource

CDN_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.9"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:158.0) "
        "Gecko/20100101 Firefox/158.0"
    ),
}


def _fetch_schedule() -> bytes:
    with requests.get(SOURCE_URL, headers=CDN_HEADERS, timeout=30, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f"Unexpected NBA schedule response status {response.status_code}.")
        return response.content


def _schedule_fields(pairs: list[tuple]) -> dict:
    if any("espn" in key.casefold() for key, _ in pairs):
        raise ValueError("Prohibited field in the official NBA schedule.")
    return dict(pairs)


def _validate_schedule(payload: bytes, season: str) -> int:
    schedule = json.loads(payload, object_pairs_hook=_schedule_fields)["leagueSchedule"]
    if str(schedule["seasonYear"]) != season or str(schedule["leagueId"]) != "00":
        raise ValueError("The official NBA schedule belongs to a different season or league.")
    season_year = int(season[:4])
    years = {season_year, season_year + 1}
    game_ids = set()
    assigned_count = 0
    for group in schedule["gameDates"]:
        for game in group["games"]:
            game_id = game["gameId"]
            if not re.fullmatch(r"[0-9]{10}", game_id) or game_id in game_ids:
                raise ValueError(f"Invalid or duplicate NBA game ID: {game_id}.")
            game_ids.add(game_id)

            home, away = game["homeTeam"], game["awayTeam"]
            home_id, away_id = int(home["teamId"]), int(away["teamId"])
            # The NBA publishes Cup slots before it has assigned teams.
            if home_id == away_id == 0 and str(game["gameStatus"]) == "1":
                continue
            if (
                min(home_id, away_id) <= 0 or home_id == away_id
                or home["teamTricode"] == away["teamTricode"]
                or any(not re.fullmatch(r"[A-Z0-9]{3}", team["teamTricode"]) for team in (home, away))
            ):
                raise ValueError(f"Invalid team identity for NBA game {game_id}.")

            raw_date = (game.get("gameDateEst") or group["gameDate"])[:10]
            game_date = (
                datetime.strptime(raw_date, "%m/%d/%Y").date()
                if "/" in raw_date else date.fromisoformat(raw_date)
            )
            if game_date.year not in years:
                raise ValueError(f"NBA game {game_id} falls outside season {season}.")
            timestamp = (game.get("gameDateTimeUTC") or "").strip()
            if timestamp.upper() not in {"", "TBD", "TBA"} or str(game["gameStatus"]) == "3":
                start = datetime.fromisoformat(timestamp)
                if start.year not in years or start.utcoffset() != timedelta(0):
                    raise ValueError(f"Invalid UTC start time for NBA game {game_id}.")
            assigned_count += 1

    if assigned_count == 0:
        raise ValueError(f"The official NBA schedule for {season} contains no assigned games.")
    return assigned_count


@asset(group_name="season_preparation")
def nba_season_schedule(nba_schedule: NBAScheduleResource) -> MaterializeResult:
    payload = _fetch_schedule()
    fetched_at = datetime.now(UTC)
    assigned_count = _validate_schedule(payload, nba_schedule.season)
    reference = nba_schedule.publish(payload, fetched_at=fetched_at)
    return MaterializeResult(metadata={
        "season": nba_schedule.season,
        "schedule_uri": reference["uri"],
        "schedule_sha256": reference["sha256"],
        "assigned_game_count": assigned_count,
    })
