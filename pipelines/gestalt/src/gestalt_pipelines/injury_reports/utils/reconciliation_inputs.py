"""Read existing official schedule, roster, and injury evidence for reconciliation."""

import json
import re
from datetime import datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from ..resources import BUCKET, EXTRACTION_PREFIX, InjuryReportArchive, official_nba
from ..resources import archive as archive_module

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
    published_at = official_nba.publication_time(report["source_url"])
    if (
        report["schema_version"] != 1 or report["season"] != season
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
