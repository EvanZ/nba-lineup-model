"""Read existing official schedule, roster, and injury evidence for reconciliation."""

import json
import re
from datetime import date, datetime
from hashlib import sha256
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .archive import BUCKET, EXTRACTION_PREFIX, PREFIX, InjuryReportArchive, _report_prefix, _utc_text
from .official_nba import publication_time

PACIFIC = ZoneInfo("America/Los_Angeles")
SCHEDULE_PREFIX = "lake/raw/scheduleleaguev2/"
ROSTER_PREFIX = "lake/raw/commonteamroster/"


def _official_json(payload: bytes) -> dict:
    if b"espn" in payload.lower():
        raise ValueError("Prohibited source in the selected NBA evidence.")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError("Expected an NBA evidence object.")
    return record


def _check_schedule_fields(pairs: list[tuple]) -> dict:
    if any("espn" in key.casefold() for key, _ in pairs):
        raise ValueError("Prohibited field in the official schedule archive.")
    return dict(pairs)


def read_extraction(payload: bytes, uri: str, season: str) -> dict:
    report = _official_json(payload)
    published_at = publication_time(report["source_url"])
    if (
        report["schema_version"] != 1 or report["season"] != season
        or report["published_at"] != _utc_text(published_at)
    ):
        raise ValueError("Invalid extracted injury report schema or publication time.")
    if not re.fullmatch(r"[0-9a-f]{64}", report["sha256"]):
        raise ValueError("Invalid extracted injury report PDF checksum.")
    raw_prefix = _report_prefix(season, published_at, report["sha256"])
    revision = raw_prefix.removeprefix(f"{PREFIX}/")
    version = report["parser_version"]
    key = f"{EXTRACTION_PREFIX}/{revision}/parser_version={version}/report.json"
    if (
        not re.fullmatch(r"[a-zA-Z0-9_.-]+", version)
        or uri != f"s3://{BUCKET}/{key}"
        or report["pdf_uri"] != f"s3://{BUCKET}/{raw_prefix}/report.pdf"
        or report["capture_uri"] != f"s3://{BUCKET}/{raw_prefix}/capture.json"
    ):
        raise ValueError("Extraction reference differs from its official report revision.")
    return report

def read_game_catalog(payload: bytes, metadata_payload: bytes, season: str, day: date) -> list[dict]:
    metadata = _official_json(metadata_payload)
    source = urlsplit(metadata["url"])
    if (
        metadata["season"] != season
        or metadata["sha256"] != sha256(payload).hexdigest()
        or (source.scheme, source.netloc, source.path) != (
            "https", "stats.nba.com", "/stats/scheduleleaguev2"
        )
    ):
        raise ValueError("Schedule bytes differ from their official NBA capture metadata.")
    # The NBA response includes broadcast listings. Read only game identities and times.
    schedule = json.loads(payload, object_pairs_hook=_check_schedule_fields)["leagueSchedule"]
    if str(schedule["seasonYear"]) != season or str(schedule["leagueId"]) != "00":
        raise ValueError("The official schedule belongs to a different season or league.")

    games = []
    for group in schedule["gameDates"]:
        for game in group["games"]:
            timestamp = game.get("gameDateTimeUTC")
            raw_date = (game.get("gameDateEst") or group["gameDate"])[:10]
            reported_date = (
                datetime.strptime(raw_date, "%m/%d/%Y").date()
                if "/" in raw_date else date.fromisoformat(raw_date)
            )
            start = datetime.fromisoformat(timestamp) if timestamp else None
            if start is None or start.tzinfo is None:
                if reported_date == day:
                    raise ValueError(f"No verified UTC schedule time for game {game['gameId']}.")
                continue
            if start.astimezone(PACIFIC).date() != day:
                continue
            home, away = game["homeTeam"], game["awayTeam"]
            home_id, away_id = int(home["teamId"]), int(away["teamId"])
            if (
                not game["gameId"] or not home["teamTricode"] or not away["teamTricode"]
                or min(home_id, away_id) <= 0 or home_id == away_id
            ):
                raise ValueError("Invalid game or team identity in the selected NBA schedule.")
            games.append({
                "game_id": str(game["gameId"]),
                "game_date": reported_date,
                "game_time_utc": start,
                "away_team_id": away_id,
                "away_team_tricode": away["teamTricode"].upper(),
                "home_team_id": home_id,
                "home_team_tricode": home["teamTricode"].upper(),
            })
    if "espn" in json.dumps(games, default=str).casefold():
        raise ValueError("Prohibited source in the selected NBA game identities.")
    if not games or len({row["game_id"] for row in games}) != len(games):
        raise ValueError(f"Missing or duplicate official games for {day}.")
    return sorted(games, key=lambda row: row["game_id"])


def read_roster(
    archive: InjuryReportArchive, metadata_ref: dict, season: str,
) -> tuple[int, dict[int, str], dict]:
    prefix = f"{ROSTER_PREFIX}{season}/"
    metadata_uri = metadata_ref["uri"]
    metadata = _official_json(archive.read_pinned(metadata_uri, metadata_ref["sha256"], prefix=prefix))
    team_id = int(metadata["team_id"])
    source = urlsplit(metadata["url"])
    observed = datetime.fromisoformat(metadata["fetched_at"])
    if (
        metadata["season"] != season or metadata["endpoint"] != "commonteamroster"
        or metadata_uri != f"s3://{BUCKET}/{prefix}{team_id}.meta.json"
        or observed.tzinfo is None
        or (source.scheme, source.netloc, source.path) != (
            "https", "stats.nba.com", "/stats/commonteamroster"
        )
    ):
        raise ValueError(f"Invalid official roster metadata for team {team_id}.")
    raw_ref = {"uri": f"s3://{BUCKET}/{prefix}{team_id}.json", "sha256": metadata["sha256"]}
    raw = archive.read_pinned(raw_ref["uri"], raw_ref["sha256"], prefix=prefix)
    roster = next(
        (item for item in _official_json(raw)["resultSets"] if item["name"].casefold() == "commonteamroster"),
        None,
    )
    if roster is None:
        raise ValueError(f"Missing official CommonTeamRoster result set for team {team_id}.")
    columns = roster["headers"]
    team_col, id_col, name_col = (columns.index(name) for name in ("TeamID", "PLAYER_ID", "PLAYER"))
    rows = roster["rowSet"]
    players = {int(row[id_col]): row[name_col] for row in rows}
    if (
        {int(row[team_col]) for row in rows} != {team_id} or len(players) != len(rows)
        or any(pid <= 0 or not isinstance(name, str) or not name.strip() for pid, name in players.items())
    ):
        raise ValueError(f"Invalid official roster evidence for team {team_id}.")
    return team_id, players, {
        "metadata": metadata_ref,
        "roster": raw_ref,
        "fetched_at": metadata["fetched_at"],
        "roster_date_pacific": observed.astimezone(PACIFIC).date().isoformat(),
    }
