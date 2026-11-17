"""Refresh per-game pointers after saved captures and finalize them at cutoff."""
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

from dagster import RunRequest, SensorEvaluationContext, SkipReason, sensor

from ...resources import NBAScheduleResource
from ...utils import PACIFIC
from ..jobs import publish_game_odds
from ..resources import GameOddsArchive
from ..utils import instant


@sensor(job=publish_game_odds, minimum_interval_seconds=60)
def publish_game_odds_sensor(context: SensorEvaluationContext, nba_schedule: NBAScheduleResource,
                             game_odds_archive: GameOddsArchive):
    now, archive = datetime.now(UTC), game_odds_archive
    today = now.astimezone(PACIFIC).date()
    source = archive.source_identity()
    # The cutoff still needs evaluation during a collector outage, even if
    # no season latest.json exists yet. Only saved captures are ever selected.
    _, schedule = nba_schedule.load()
    source_identity = {"raw_latest_etag": source, "schedule": schedule}
    cursor = json.loads(context.cursor) if context.cursor else {}
    if cursor.get("source") == source_identity and now.timestamp() < cursor.get("next_check", 0):
        yield SkipReason("Saved odds are unchanged and no game has reached its cutoff.")
        return
    days = {today+timedelta(days=offset): [] for offset in range(-1, 8)}
    games, _ = nba_schedule.games_between(today-timedelta(days=1), today+timedelta(days=7), schedule, allow_empty=True)
    for game in games:
        days[game["game_time_utc"].astimezone(PACIFIC).date()].append((game["game_id"], game["game_time_utc"]))
    next_check = now + timedelta(days=1)
    for day, games in days.items():
        open_games, due = [], []
        for gid, start in games:
            pointer, _ = archive.selection(str(day), gid)
            # Only finish existing overnight pointers for yesterday. Missed
            # historical days remain an explicit manual backfill.
            if day < today and pointer is None:
                continue
            if pointer is not None and pointer["frozen"]:
                continue
            cutoff = instant(pointer["cutoff_utc"]) if pointer else start
            open_games.append(gid)
            if now >= cutoff:
                due.append(gid)
                next_check = min(next_check, now+timedelta(seconds=60))
            else:
                next_check = min(next_check, cutoff)
        if open_games and (cursor.get("source") != source_identity or due):
            identity = json.dumps([str(day), source_identity, sorted(due)], sort_keys=True)
            yield RunRequest(run_key="game-odds-"+sha256(identity.encode()).hexdigest(),
                partition_key=str(day), tags={"dagster/priority": "-1"})
    context.update_cursor(json.dumps({"source": source_identity, "next_check": next_check.timestamp()}, sort_keys=True))


__all__ = ["publish_game_odds_sensor"]
