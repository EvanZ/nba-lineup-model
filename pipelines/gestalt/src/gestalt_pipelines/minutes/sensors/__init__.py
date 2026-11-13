"""Wait for successful morning inputs, then update the latest completed date."""

from dagster import (
    AssetSelection, AutomationCondition, AutomationConditionSensorDefinition,
    DefaultSensorStatus,
)

from ...utils import PACIFIC

# Yesterday's rosters may already exist from an evening transaction. Require the
# morning refresh, as well as fresh stints, instead of launching against that copy.
morning_inputs = AssetSelection.assets("nba_stints", "rosters_by_date")
minutes_ready_condition = (
    AutomationCondition.eager()
    & AutomationCondition.all_deps_updated_since_cron(
        "0 0 * * *", cron_timezone=PACIFIC.key,
    ).allow(morning_inputs)
    & ~AutomationCondition.any_deps_match(AutomationCondition.execution_failed())
)

minutes_ready = AutomationConditionSensorDefinition(
    "minutes_ready",
    target=AssetSelection.assets("player_minutes"),
    default_status=DefaultSensorStatus.STOPPED,
    minimum_interval_seconds=60,
    run_tags={"dagster/priority": "-1"},
    description=(
        "Update the latest completed Pacific date once morning stints and rosters "
        "succeed; wait for missing, running, or failed inputs and the previous minutes date."
    ),
)

__all__ = ["minutes_ready", "minutes_ready_condition"]

from .projections import game_minutes_ready
__all__ += ["game_minutes_ready"]
