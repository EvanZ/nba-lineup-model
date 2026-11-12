"""Capture official transactions for one selected Pacific game date."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from dagster import AssetExecutionContext, AssetMaterialization, MaterializeResult, asset

from ..partitions import transaction_dates
from ..resources import NBATransactions

PACIFIC = ZoneInfo(transaction_dates.timezone)


@asset(group_name="transactions", partitions_def=transaction_dates)
def transactions_by_date(
    context: AssetExecutionContext, nba_transactions: NBATransactions,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    today = datetime.now(PACIFIC).date()
    if day > today:
        raise ValueError("Cannot capture transactions for a future Pacific date.")
    payload = nba_transactions.fetch()
    observed_at = datetime.now(UTC)
    if day == today:
        results = nba_transactions.poll(
            max(transaction_dates.start.date(), day - timedelta(days=1)), day,
            payload, observed_at=observed_at,
        )
    else:
        results = {day: nba_transactions.publish(day, payload, observed_at=observed_at)}
    for affected_day, result in results.items():
        if affected_day != day and result["new_transaction_count"]:
            context.log_event(AssetMaterialization(
                asset_key=context.asset_key, partition=affected_day.isoformat(),
                metadata=_metadata(nba_transactions.season, affected_day, observed_at, result),
            ))
    return MaterializeResult(metadata=_metadata(nba_transactions.season, day, observed_at, results[day]))


def _metadata(season: str, day: date, observed_at: datetime, result: dict) -> dict:
    return {
        "season": season,
        "game_date_pacific": day.isoformat(),
        "observed_at": observed_at.isoformat(),
        "transactions_uri": result["uri"], "sha256": result["sha256"],
        "transaction_count": result["transaction_count"],
        "new_transaction_count": result["new_transaction_count"],
        "source_date_row_count": result["source_date_row_count"],
    }
