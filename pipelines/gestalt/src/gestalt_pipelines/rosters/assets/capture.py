"""Build one complete NBA roster snapshot for a selected Pacific date."""

import os
from datetime import UTC, date, datetime, timedelta

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, Failure,
    MaterializeResult, MetadataValue, asset,
)

from ..partitions import roster_dates
from ..resources import NBARosterArchive
from ..utils import RosterError, build_rosters, ten_day_signings
from ...resources import NBAScheduleResource

from ...utils import PACIFIC


@asset(
    group_name="rosters", partitions_def=roster_dates,
    deps=[
        AssetDep(AssetKey("transactions_by_date"), partition_mapping=AllPartitionMapping()),
    ],
)
def rosters_by_date(
    context: AssetExecutionContext,
    nba_roster_archive: NBARosterArchive,
    nba_schedule: NBAScheduleResource,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    season = os.environ["GESTALT_NBA_SEASON"]
    now = datetime.now(UTC)
    today = now.astimezone(PACIFIC).date()
    if day > today:
        raise Failure("Cannot reconstruct a future Pacific roster date.", allow_retries=False)
    inputs = {}
    warning_artifact = None
    try:
        _, expected_etag = nba_roster_archive.selection(season, day.isoformat())
        opening = nba_roster_archive.opening_rosters(season)
        if opening["date"] >= day:
            raise ValueError("Opening roster observations must precede the selected date.")
        transactions, references = nba_roster_archive.transactions(
            season, opening["date"] + timedelta(days=1), day,
        )
        repairs, repair_reference = nba_roster_archive.repair_transactions(season)
        inputs = {
            "opening_rosters": opening["references"], "transactions": references,
            "repair_transactions": repair_reference,
        }
        games = None
        signings = ten_day_signings(transactions)
        if signings:
            first_signing = min(date.fromisoformat(row["TRANSACTION_DATE"][:10]) for row in signings)
            games, schedule_reference = nba_schedule.games_between(first_signing, day, allow_empty=True)
            inputs["schedule"] = schedule_reference
        record = build_rosters(
            season=season, day=day, opening=opening,
            transactions=transactions, inputs=inputs, repair_transactions=repairs, games=games,
        )
        roster_size_warnings = {
            team: len(players) for team, players in record["teams"].items() if len(players) > 18
        }
        for team, count in roster_size_warnings.items():
            context.log.warning(
                f"NBA team {team} has {count} rostered players, above the 18-player warning threshold."
            )
        published = nba_roster_archive.publish(record, expected_etag=expected_etag)
        if roster_size_warnings:
            warning_artifact = nba_roster_archive.save_roster_size_warning(
                record=record, artifact=published, counts=roster_size_warnings,
                run_id=context.run.run_id, occurred_at=now,
            )
    except RosterError as error:
        raise Failure(str(error), metadata={
            "game_date_pacific": day.isoformat(), "details": MetadataValue.json(error.details),
            "inputs": MetadataValue.json(inputs),
        }, allow_retries=False) from error
    except (ValueError, FileNotFoundError) as error:
        raise Failure(str(error), metadata={"game_date_pacific": day.isoformat(), "inputs": MetadataValue.json(inputs)}, allow_retries=False) from error
    return MaterializeResult(metadata={
        "season": season, "game_date_pacific": day.isoformat(),
        "team_count": len(record["teams"]),
        "ten_day_contracts": MetadataValue.json(record["ten_day_contracts"]),
        "player_count": sum(len(ids) for ids in record["teams"].values()),
        "roster_size_warnings": MetadataValue.json(roster_size_warnings),
        "warning_artifact": MetadataValue.json(warning_artifact),
        "rosters_uri": published["uri"], "sha256": published["sha256"],
        "artifact_reused": published["reused"], "latest_uri": published["latest_uri"],
        "materialized_at": now.isoformat(),
        "opening_roster_date_pacific": opening["date"].isoformat(),
        "inputs": MetadataValue.json(inputs),
    })
