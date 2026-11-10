"""Archive one live league-wide spreads and moneylines snapshot."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from fcntl import LOCK_EX, flock
from pathlib import Path
from zoneinfo import ZoneInfo

from dagster import AssetExecutionContext, MaterializeResult, asset

from ...resources import NBAScheduleResource
from ..resources import OddsApi

PACIFIC = ZoneInfo("America/Los_Angeles")


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


@asset(group_name="odds", output_required=False)
def nba_market_odds(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource, odds_api: OddsApi,
) -> Iterator[MaterializeResult]:
    season = nba_schedule.season
    state_dir = Path(context.instance.storage_directory()) / "market_odds" / season
    state_dir.mkdir(parents=True, exist_ok=True)
    run_id = context.run.root_run_id or context.run.run_id
    staged = state_dir / f"{run_id}.json"

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

        uri = odds_api.publish(season, payload)
        staged.unlink()

    yield MaterializeResult(metadata={
        "snapshot_uri": uri,
        "collected_at": snapshot["collected_at"],
        "events": len(snapshot["events"]),
        **snapshot["quota"],
    })
