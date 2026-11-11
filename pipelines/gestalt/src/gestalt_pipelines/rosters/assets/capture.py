"""Build one complete NBA roster snapshot for a selected Pacific date."""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, Failure,
    MaterializeResult, MetadataValue, asset,
)

from ...resources import NBAScheduleResource
from ..partitions import roster_dates
from ..resources import NBARosterArchive
from ..utils import RosterError, build_rosters
PACIFIC = ZoneInfo(roster_dates.timezone)


@asset(
    group_name="rosters", partitions_def=roster_dates,
    deps=[
        AssetDep(AssetKey("transactions_by_date"), partition_mapping=AllPartitionMapping()),
        AssetDep(AssetKey("nba_box_scores"), partition_mapping=AllPartitionMapping()),
    ],
)
def rosters_by_date(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    nba_roster_archive: NBARosterArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    if day > today:
        raise Failure("Cannot reconstruct a future Pacific roster date.", allow_retries=False)
    inputs = {}
    try:
        opening = nba_roster_archive.opening_rosters(nba_schedule.season)
        if opening["date"] >= day:
            raise ValueError("Opening roster observations must precede the selected date.")
        games, schedule = nba_schedule.games_between(
            opening["date"], day, allow_empty=True,
        )
        games = [
            game for game in games
            if game["game_time_utc"] <= now
            and game["game_time_utc"].astimezone(PACIFIC).date() > opening["date"]
        ]
        evidence = nba_roster_archive.daily_evidence(
            nba_schedule.season, day, opening, games, today=today,
        )
        inputs = {
            "opening_rosters": opening["references"], "schedule": schedule,
            **evidence["references"],
        }
        record = build_rosters(
            season=nba_schedule.season, day=day, opening=opening,
            transactions=evidence["transactions"], boxes=evidence["boxes"], inputs=inputs,
        )
        published = nba_roster_archive.publish(record)
    except RosterError as error:
        raise Failure(str(error), metadata={
            "game_date_pacific": day.isoformat(), "details": MetadataValue.json(error.details),
            "inputs": MetadataValue.json(inputs),
        }, allow_retries=False) from error
    except (ValueError, FileNotFoundError) as error:
        raise Failure(str(error), metadata={"game_date_pacific": day.isoformat(), "inputs": MetadataValue.json(inputs)}, allow_retries=False) from error
    return MaterializeResult(metadata={
        "season": nba_schedule.season, "game_date_pacific": day.isoformat(),
        "team_count": len(record["teams"]),
        "player_count": sum(len(ids) for ids in record["teams"].values()),
        "rosters_uri": published["uri"], "sha256": published["sha256"],
        "artifact_reused": published["reused"],
        "materialized_at": now.isoformat(),
        "opening_roster_date_pacific": opening["date"].isoformat(),
        "box_score_bodies_read": evidence["box_bodies_read"],
        "inputs": MetadataValue.json(inputs),
    })
