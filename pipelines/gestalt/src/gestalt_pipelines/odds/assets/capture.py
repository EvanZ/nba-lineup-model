"""Archive one live league-wide spreads and moneylines snapshot."""

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from fcntl import LOCK_EX, flock
from pathlib import Path

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

from ...resources import NBAScheduleResource
from ..resources import OddsApi
from ..partitions import odds_capture_dates

from ...utils import PACIFIC



def _has_upcoming_game(nba_schedule: NBAScheduleResource, now: datetime) -> bool:
    today = now.astimezone(PACIFIC).date()
    reference = None
    for offset in range(8):
        games, reference = nba_schedule.games_for_date(
            today + timedelta(days=offset), reference, allow_empty=True,
        )
        if any(game["game_time_utc"] > now for game in games):
            return True
    return False


@asset(group_name="odds", output_required=False, partitions_def=odds_capture_dates)
def nba_market_odds(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource, odds_api: OddsApi,
) -> Iterator[MaterializeResult]:
    season = nba_schedule.season
    day = date.fromisoformat(context.partition_key)
    today = datetime.now(PACIFIC).date()
    if day > today:
        raise ValueError("Cannot capture odds for a future Pacific date.")
    state_dir = Path(context.instance.storage_directory()) / "market_odds" / season
    run_id = context.run.root_run_id or context.run.run_id
    staged = state_dir / f"{run_id}.json"
    if day < today and not staged.is_file():
        snapshots = odds_api.archived_snapshots(season, day)
        yield MaterializeResult(metadata={
            "capture_date_pacific": day.isoformat(), "archive_only": True,
            "snapshot_count": len(snapshots), "snapshots": MetadataValue.json(snapshots),
        })
        return

    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "capture.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        if staged.is_file():
            payload = staged.read_bytes()
            snapshot = json.loads(payload)
        else:
            if not _has_upcoming_game(nba_schedule, datetime.now(UTC)):
                context.log.info("No upcoming NBA games through the next seven Pacific dates.")
                return
            snapshot = odds_api.fetch(season)
            if snapshot is None:
                context.log.info("Skipped the odds poll to preserve the 50-credit reserve.")
                return
            payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
            temporary = staged.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(staged)

        collected_day = datetime.fromisoformat(snapshot["collected_at"]).astimezone(PACIFIC).date()
        if collected_day != day:
            raise ValueError("The saved odds capture belongs to another date; materialize that archived date separately.")
        uri = odds_api.publish(season, payload)
        staged.unlink()

    yield MaterializeResult(metadata={
        "capture_date_pacific": day.isoformat(), "archive_only": False,
        "snapshot_uri": uri,
        "collected_at": snapshot["collected_at"],
        "events": len(snapshot["events"]),
        **snapshot["quota"],
    })
