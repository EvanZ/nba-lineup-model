"""Request injury processing from committed objects in the recent S3 folders."""

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from dagster import DefaultSensorStatus, RunRequest, SensorEvaluationContext, SensorResult, sensor

from ...resources import NBAPlayerNamesResource, NBAScheduleResource
from ...rosters.resources import NBARosterArchive
from ..jobs import extract_injury_pdf, reconcile_injury_date
from ..partitions import injury_game_dates, injury_publication_dates
from ..resources import InjuryReportArchive
from ..utils import PARSER_VERSION, RESOLVER_VERSION, select_extractions

from ...utils import EASTERN, PACIFIC



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
    nba_roster_archive: NBARosterArchive, nba_player_names: NBAPlayerNamesResource,
) -> SensorResult:
    """Reconcile open dates when selected reports, rosters, names, or repairs change."""
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    season = injury_report_archive.season
    _, schedule_ref = nba_schedule.load()
    _, names_ref = nba_player_names.load()
    repairs = injury_report_archive.read_player_repairs()
    requests = []
    for day in (today, today + timedelta(days=1)):
        if day < injury_game_dates.start.date():
            continue
        games, _ = nba_schedule.games_for_date(day, schedule_ref, allow_empty=True)
        open_ids = {game["game_id"] for game in games if game["game_time_utc"] > now}
        if not open_ids:
            continue
        selected = select_extractions(
            injury_report_archive, games, first_date=injury_publication_dates.start.date(),
            now=now, committed_only=True,
        )
        if not open_ids.intersection(selected):
            continue
        # Select the newest accepted roster no later than the game date.
        # The asset selects again at execution; missing evidence fails visibly.
        try:
            _, roster_ref = nba_roster_archive.latest_selection(season, day)
        except FileNotFoundError:
            roster_ref = None
        config = {
            "live": True,
            "schedule": schedule_ref,
            "extractions": {game_id: ref for game_id, (_, ref) in selected.items()},
        }
        sources = {**config, "resolver_version": RESOLVER_VERSION, "rosters": roster_ref, "player_names": names_ref,
                   "player_repairs_sha256": repairs["sha256"]}
        identity = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        requests.append(RunRequest(
            run_key=f"reconcile:{day}:{identity}",
            partition_key=day.isoformat(),
            tags={"dagster/priority": "5"},
            run_config={"ops": {"injury_status_by_game_date": {"config": config}}},
        ))
    return SensorResult(run_requests=requests)
