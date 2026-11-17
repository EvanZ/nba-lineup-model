"""Materialize per-game references without collecting or copying odds."""
import json
from datetime import UTC, date, datetime

from dagster import AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy, MaterializeResult, MetadataValue, asset

from ...resources import NBAScheduleResource
from ..partitions import game_odds_dates
from ..resources import GameOddsArchive
from ..utils import eligible_event, instant


@asset(
    group_name="odds", partitions_def=game_odds_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[AssetDep(AssetKey("nba_season_schedule")),
          AssetDep(AssetKey("nba_market_odds"), partition_mapping=AllPartitionMapping())],
    description="Select and freeze one saved pregame Odds API event per official game.",
)
def game_odds(context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
              game_odds_archive: GameOddsArchive) -> MaterializeResult:
    day, now = date.fromisoformat(context.partition_key), datetime.now(UTC)
    archive = game_odds_archive
    if nba_schedule.season != archive.season or nba_schedule.bucket != archive.bucket:
        raise ValueError("Schedule and game odds must use the same season and bucket.")
    games, schedule = nba_schedule.games_for_date(day, allow_empty=True)
    raw, _ = nba_schedule.load(schedule)
    raw_games = {g["gameId"]: g for group in json.loads(raw)["leagueSchedule"]["gameDates"] for g in group["games"]}
    names = {g["game_id"]: {side: raw_games[g["game_id"]][side+"Team"]["teamName"] for side in ("home", "away")} for g in games}
    previous, selected, frozen = {}, {}, []
    for game in games:
        gid = game["game_id"]
        pointer, etag = archive.selection(str(day), gid)
        previous[gid] = (pointer, etag)
        if pointer is not None:
            archive.validate(pointer, game, names[gid])
            if pointer["frozen"]:
                frozen.append(gid)
    pending = [g for g in games if g["game_id"] not in frozen]
    if pending:
        before = min(now, max(g["game_time_utc"] for g in pending))
        for snapshot, reference in archive.snapshots(day, before):
            for game in pending:
                gid = game["game_id"]
                if gid in selected:
                    continue
                match = eligible_event(snapshot, game, names[gid])
                if match is not None:
                    selected[gid] = (reference, *match)
            if len(selected) == len(pending):
                break
    missing, written, unchanged, warnings = [], [], list(frozen), []
    # Check the official selection once before publishing, rather than silently
    # mixing schedule versions if it changes during this run.
    if nba_schedule.load()[1] != schedule:
        raise ValueError("The official schedule changed; rerun this date.")
    for game in pending:
        gid = game["game_id"]
        old, etag = previous[gid]
        candidate = selected.get(gid)
        if old is not None and (candidate is None or instant(old["snapshot"]["collected_at"]) >= instant(candidate[0]["collected_at"])):
            pointer = dict(old)
            pointer["frozen"] = now >= instant(old["cutoff_utc"])
            if pointer == old:
                unchanged.append(gid)
                continue
            pointer["frozen_at"] = now.isoformat()
        elif candidate is not None:
            reference, event, cutoff = candidate
            pointer = {"schema_version": 1, "selection_rule": "latest_saved_pregame_event_v1",
                "season": archive.season, "game_date_pacific": str(day), "game_id": gid,
                "home_team_id": game["home_team_id"], "away_team_id": game["away_team_id"],
                "home_team": event["home_team"], "away_team": event["away_team"],
                "scheduled_start_utc": game["game_time_utc"].isoformat(), "cutoff_utc": cutoff.isoformat(),
                "selected_at": now.isoformat(), "frozen": now >= cutoff,
                "purpose": "historical_replay" if now >= cutoff else "pregame_selection",
                "snapshot": reference, "provider_event_id": event["id"], "inputs": {"schedule": schedule}}
            if pointer["frozen"]:
                pointer["frozen_at"] = now.isoformat()
        else:
            missing.append(gid)
            if now >= game["game_time_utc"]:
                warning = archive.save_missing_odds_warning(game=game, names=names[gid], schedule=schedule,
                    run_id=context.run.run_id, occurred_at=now)
                warnings.append({"game_id": gid, **warning})
            continue
        archive.validate(pointer, game, names[gid])
        written.append(archive.save(pointer, etag))
    if missing:
        context.log.warning(f"No saved eligible pregame odds for games {missing}; no odds were invented.")
    return MaterializeResult(metadata={"game_date_pacific": str(day), "scheduled_game_count": len(games),
        "selected_game_count": len(games)-len(missing), "written_count": len(written),
        "unchanged_count": len(unchanged), "missing_game_ids": MetadataValue.json(missing),
        "pointers": MetadataValue.json(written), "schedule": MetadataValue.json(schedule),
        "warnings": MetadataValue.json(warnings)})
