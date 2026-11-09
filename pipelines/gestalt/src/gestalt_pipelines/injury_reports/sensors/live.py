"""Request injury processing from committed objects in the recent S3 folders."""

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo

from dagster import DefaultSensorStatus, RunRequest, SensorEvaluationContext, SensorResult, sensor

from ...resources import NBAScheduleResource
from ..jobs import extract_injury_pdf, reconcile_injury_date
from ..partitions import injury_game_dates, injury_publication_dates
from ..resources import EXTRACTION_PREFIX, InjuryReportArchive
from ..utils import PARSER_VERSION, read_extraction, read_roster, report_game_key, select_extractions

EASTERN = ZoneInfo(injury_publication_dates.timezone)
PACIFIC = ZoneInfo(injury_game_dates.timezone)


def _scan_window(context: SensorEvaluationContext, now: datetime):
    # S3 commit times have second precision. Leave the current second for the next scan.
    end = now.replace(microsecond=0)
    start = datetime.fromisoformat(context.cursor) if context.cursor else end - timedelta(hours=1)
    today = now.astimezone(EASTERN).date()
    days = [day for day in (today - timedelta(days=1), today) if day >= injury_publication_dates.start.date()]
    return start, end, days


@sensor(job=extract_injury_pdf, minimum_interval_seconds=60, default_status=DefaultSensorStatus.STOPPED)
def new_injury_pdf(
    context: SensorEvaluationContext, injury_report_archive: InjuryReportArchive,
) -> SensorResult:
    """Extract each newly committed PDF revision at the current parser version."""
    start, end, days = _scan_window(context, datetime.now(UTC))
    requests = []
    for day in days:
        captures = [
            uri for _, committed_at, uri in injury_report_archive.captures_for_date(day)
            if start <= committed_at < end
        ]
        if not captures:
            continue
        extracted = injury_report_archive.extraction_uris(day, PARSER_VERSION)
        for uri in captures:
            extraction_uri = injury_report_archive.extraction_uri(uri, PARSER_VERSION)
            if extraction_uri in extracted:
                continue
            requests.append(RunRequest(
                run_key=f"extract:{sha256(extraction_uri.encode()).hexdigest()}",
                partition_key=day.isoformat(),
                tags={"dagster/priority": "5", "injury_reports/parser_version": PARSER_VERSION},
                run_config={"ops": {"injury_report_extractions": {"config": {"capture_uri": uri}}}},
            ))
    return SensorResult(run_requests=requests, cursor=end.isoformat())


@sensor(job=reconcile_injury_date, minimum_interval_seconds=60, default_status=DefaultSensorStatus.STOPPED)
def injury_report_dates_ready(
    context: SensorEvaluationContext, injury_report_archive: InjuryReportArchive, nba_schedule: NBAScheduleResource,
) -> SensorResult:
    """Reconcile open games affected by newly committed extractions."""
    now = datetime.now(UTC)
    start, end, days = _scan_window(context, now)
    season = injury_report_archive.season
    prefix = f"{EXTRACTION_PREFIX}/season={season}/"
    changed = {}
    for day in days:
        for committed_at, uri in injury_report_archive.extraction_records(day, PARSER_VERSION):
            if start <= committed_at < end:
                payload, _ = injury_report_archive.read_input(uri, prefix=prefix)
                changed[uri] = read_extraction(payload, uri, season)
    if not changed:
        return SensorResult(cursor=end.isoformat())

    covered = {key for report in changed.values() for key in report["game_team_reports"]}
    _, schedule_ref = nba_schedule.load()
    today = now.astimezone(PACIFIC).date()
    roster_refs = {}
    requests = []
    for day in (today, today + timedelta(days=1)):
        if day < injury_game_dates.start.date():
            continue
        games, _ = nba_schedule.games_for_date(day, schedule_ref, allow_empty=True)
        affected = [
            game["game_id"] for game in games
            if game["game_time_utc"] > now and report_game_key(game) in covered
        ]
        if not affected:
            continue
        selected = select_extractions(
            injury_report_archive, games, first_date=min(days), now=now, committed_only=True,
        )
        if not any(game_id in selected and selected[game_id][1]["uri"] in changed for game_id in affected):
            continue

        teams = {game[field] for game in games for field in ("away_team_id", "home_team_id")}
        for team in sorted(teams - roster_refs.keys()):
            _, _, evidence = read_roster(injury_report_archive, team, season)
            roster_refs[team] = evidence["metadata"]
        config = {
            "live": True,
            "schedule": schedule_ref,
            "extractions": {game_id: ref for game_id, (_, ref) in selected.items()},
            "roster_metadata": [roster_refs[team] for team in sorted(teams)],
        }
        identity = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        requests.append(RunRequest(
            run_key=f"reconcile:{day}:{identity}",
            partition_key=day.isoformat(),
            tags={"dagster/priority": "5"},
            run_config={"ops": {"injury_status_by_game_date": {"config": config}}},
        ))
    return SensorResult(run_requests=requests, cursor=end.isoformat())
