"""Read existing official schedule, roster, and injury evidence for reconciliation."""

import json
import re
from datetime import date, datetime, timedelta
from itertools import groupby
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from botocore.exceptions import ClientError

from ..resources import BUCKET, EXTRACTION_PREFIX, InjuryReportArchive, official_nba
from ..resources import archive as archive_module
from .parse_pdf import PARSER_VERSION
from .reconcile import report_game_key

PACIFIC = ZoneInfo("America/Los_Angeles")
ROSTER_PREFIX = "lake/raw/commonteamroster/"


def _official_json(payload: bytes) -> dict:
    if b"espn" in payload.lower():
        raise ValueError("Prohibited source in the selected NBA evidence.")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError("Expected an NBA evidence object.")
    return record


def read_extraction(payload: bytes, uri: str, season: str) -> dict:
    report = _official_json(payload)
    published_at = datetime.fromisoformat(report["published_at"])
    if (
        report["schema_version"] != 1 or report["season"] != season
        or published_at not in official_nba.publication_instants(report["source_url"])
        or report["published_at"] != archive_module._utc_text(published_at)
    ):
        raise ValueError("Invalid extracted injury report schema or publication time.")
    if not re.fullmatch(r"[0-9a-f]{64}", report["sha256"]):
        raise ValueError("Invalid extracted injury report PDF checksum.")
    raw_prefix = archive_module._report_prefix(season, published_at, report["sha256"])
    revision = raw_prefix.removeprefix(f"{archive_module.PREFIX}/")
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


def select_extractions(
    archive: InjuryReportArchive, games: list[dict], *, first_date: date, now: datetime,
) -> dict[str, tuple[dict, dict]]:
    """Select the newest saved report covering each game before its scheduled start."""
    if not games:
        return {}
    remaining = {game["game_id"]: game for game in games}
    last_cutoff = min(max(game["game_time_utc"] for game in games), now)
    day = last_cutoff.astimezone(official_nba.EASTERN).date()
    selected = {}
    prefix = f"{EXTRACTION_PREFIX}/season={archive.season}/"

    while remaining and day >= first_date:
        captures = archive.captures_for_date(day)
        for (published_at, _), revisions in groupby(captures, key=lambda item: item[:2]):
            eligible = [
                game for game in remaining.values()
                if published_at <= now and published_at < game["game_time_utc"]
            ]
            if not eligible:
                continue
            group = list(revisions)
            if len(group) != 1:
                raise ValueError(f"Multiple report revisions share a publication and archive time: {published_at}.")
            uri = archive.extraction_uri(group[0][2], PARSER_VERSION)
            try:
                raw, ref = archive.read_input(uri, prefix=prefix)
            except ClientError as error:
                if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                    raise FileNotFoundError(f"Extract Eastern publication date {day} first. Missing {uri}.") from error
                raise
            report = read_extraction(raw, uri, archive.season)
            for game in eligible:
                if report_game_key(game) in report["game_team_reports"]:
                    selected[game["game_id"]] = (report, ref)
                    del remaining[game["game_id"]]
            if not remaining:
                break
        day -= timedelta(days=1)
    return selected


def read_roster(
    archive: InjuryReportArchive, metadata_ref: dict | int, season: str,
) -> tuple[int, dict[int, str], dict]:
    prefix = f"{ROSTER_PREFIX}{season}/"
    if isinstance(metadata_ref, int):
        metadata_uri = f"s3://{BUCKET}/{prefix}{metadata_ref}.meta.json"
        payload, metadata_ref = archive.read_input(metadata_uri, prefix=prefix)
    else:
        metadata_uri = metadata_ref["uri"]
        payload = archive.read_pinned(metadata_uri, metadata_ref["sha256"], prefix=prefix)
    metadata = _official_json(payload)
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
