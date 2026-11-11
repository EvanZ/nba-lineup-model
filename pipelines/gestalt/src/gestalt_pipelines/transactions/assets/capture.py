"""Capture official transactions for one selected Pacific game date."""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from dagster import AssetExecutionContext, MaterializeResult, asset

from ..partitions import transaction_dates
from ..resources import NBATransactions

PACIFIC = ZoneInfo(transaction_dates.timezone)


@asset(group_name="transactions", partitions_def=transaction_dates)
def transactions_by_date(
    context: AssetExecutionContext, nba_transactions: NBATransactions,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    if day > datetime.now(PACIFIC).date():
        raise ValueError("Cannot capture transactions for a future Pacific date.")
    payload = nba_transactions.fetch()
    observed_at = datetime.now(UTC)
    result = nba_transactions.publish(day, payload, observed_at=observed_at)
    return MaterializeResult(metadata={
        "season": nba_transactions.season,
        "game_date_pacific": day.isoformat(),
        "observed_at": observed_at.isoformat(),
        "transactions_uri": result["uri"], "sha256": result["sha256"],
        "transaction_count": result["transaction_count"],
        "new_transaction_count": result["new_transaction_count"],
        "source_date_row_count": result["source_date_row_count"],
    })
