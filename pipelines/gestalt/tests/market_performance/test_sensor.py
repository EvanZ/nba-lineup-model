from dagster import (
    AssetMaterialization, DagsterEventType, DagsterInstance, MetadataValue,
    RunRequest, SkipReason, build_run_status_sensor_context, job, op,
)
from gestalt_pipelines.market_performance.sensors import evaluate_market_performance_sensor
from .conftest import DAY


@op(config_schema={"digest": str, "reused": bool, "materialize": bool})
def counterfactual(context):
    if context.op_config["materialize"]:
        context.log_event(AssetMaterialization(asset_key="game_predictions_actual_minutes", partition=DAY, metadata={
            "counterfactuals": MetadataValue.json({"0022600001": {"uri": "s3://nba-gestalt-data/test/counter.json",
                "sha256": context.op_config["digest"]*64, "reused": context.op_config["reused"]}})}))


@job
def producer():
    counterfactual()


def request(instance, digest="a", reused=False, materialize=True, tagged=True):
    result = producer.execute_in_process(instance=instance, tags={"dagster/partition": DAY} if tagged else {},
        run_config={"ops": {"counterfactual": {"config": {"digest": digest, "reused": reused, "materialize": materialize}}}})
    event = next(e for e in result.all_events if e.event_type == DagsterEventType.RUN_SUCCESS)
    with build_run_status_sensor_context(sensor_name="evaluate_market_performance_sensor", dagster_event=event,
        dagster_instance=instance, dagster_run=instance.get_run_by_id(result.run_id)) as context:
        return evaluate_market_performance_sensor(context)


def test_trigger_uses_same_partition_and_suppresses_unchanged_source_successes():
    with DagsterInstance.ephemeral() as instance:
        first = request(instance)
        reused = request(instance, reused=True)
        changed = request(instance, digest="b")
    assert isinstance(first, RunRequest) and first.partition_key == DAY
    assert first.run_key == reused.run_key and changed.run_key != first.run_key
    assert first.tags["dagster/priority"] == "-1"


def test_trigger_waits_for_actual_materialization_and_single_date():
    with DagsterInstance.ephemeral() as instance:
        assert isinstance(request(instance, materialize=False), SkipReason)
        assert isinstance(request(instance, tagged=False), SkipReason)
