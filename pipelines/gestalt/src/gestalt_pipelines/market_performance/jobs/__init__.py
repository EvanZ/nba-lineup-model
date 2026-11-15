from dagster import AssetSelection, define_asset_job, in_process_executor
from ..assets import market_performance_by_date

evaluate_market_performance = define_asset_job(
    "evaluate_market_performance", selection=AssetSelection.assets(market_performance_by_date),
    executor_def=in_process_executor,
    description="Evaluate a completed Pacific slate against saved pre-tipoff markets without changing its sources.",
)

__all__ = ["evaluate_market_performance"]
