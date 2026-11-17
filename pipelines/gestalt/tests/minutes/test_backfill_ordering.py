"""Real asset-backfill planning and minutes execution with isolated storage."""
import json
import logging
import socket
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import boto3
from dagster import AssetKey, DagsterInstance, RunsFilter
from dagster._core.asset_graph_view.asset_graph_view import AssetGraphView, TemporalContext
from dagster._core.execution.asset_backfill import AssetBackfillData, execute_asset_backfill_iteration_inner
from dagster._core.execution.backfill import BulkActionStatus, PartitionBackfill
from dagster._core.storage.tags import ASSET_PARTITION_RANGE_START_TAG, BACKFILL_ID_TAG
from dagster._core.test_utils import in_process_test_workspace
from dagster._core.types.loadable_target_origin import LoadableTargetOrigin

from gestalt_pipelines.minutes.utils import json_bytes
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.rosters.resources import NBARosterArchive

from .helpers import FIRST, SEASON, reference, slate
from .test_asset import minutes_job


def test_failed_date_blocks_asset_backfill_then_retry_resumes(minutes_job, monkeypatch, tmp_path):
    job, archive, objects, _, _ = minutes_job
    client = boto3.client("s3")
    assert client.__class__.__name__ == "SimpleNamespace"
    def deny_network(*args, **kwargs):
        raise AssertionError("Isolated ordering test must never connect to a network service")
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setenv("DAGSTER_HOME", str(tmp_path / "instance"))
    monkeypatch.setenv("DAGSTER_DISABLE_TELEMETRY", "1")
    days = [str(FIRST+timedelta(days=i)) for i in range(3)]
    data = {day: slate(FIRST+timedelta(days=i), gid=f"002260000{i+1}") for i, day in enumerate(days)}
    monkeypatch.setattr(NBAScheduleResource, "games_for_date",
        lambda self, day, **kwargs: (data[str(day)]["games"], reference("schedule")))
    def put(key, doc):
        client.put_object(Key=key, Body=json_bytes(doc))
    for day in days[1:]:
        for kind, doc in [("stints", data[day]["stints"]), ("injury_status_by_game_date", data[day]["injuries"])]:
            digest = sha256(json_bytes(doc)).hexdigest()
            prefix = f"lake/artifacts/{kind}/season={SEASON}/game_date_pacific={day}/"
            suffix = digest+".json" if kind == "stints" else f"selection_sha256={digest}/statuses.json"
            key = prefix+suffix
            put(key, doc)
            put(prefix+"latest.json", {"uri": "s3://nba-gestalt-data/"+key, "sha256": digest})
        data[day]["rosters"]["schema_version"] = 1
        NBARosterArchive().publish(data[day]["rosters"], expected_etag=None)
    missing_key = f"lake/artifacts/stints/season={SEASON}/game_date_pacific={days[1]}/latest.json"
    removed_input = objects.pop(missing_key)
    origin = LoadableTargetOrigin(executable_path=sys.executable,
        module_name="gestalt_pipelines.definitions", attribute="defs")
    events = []
    logger = logging.getLogger("isolated_minutes_backfill")
    key = AssetKey("player_minutes")
    (tmp_path / "instance").mkdir()
    with DagsterInstance.local_temp(str(tmp_path / "instance"), overrides={"telemetry": {"enabled": False}}) as instance:
        assert instance.root_directory.startswith(str(tmp_path))
        with in_process_test_workspace(instance, origin) as workspace:
            graph = workspace.asset_graph
            def view():
                return AssetGraphView(temporal_context=TemporalContext(effective_dt=datetime.now(UTC),
                    last_event_id=instance.event_log_storage.get_maximum_record_id()), instance=instance, asset_graph=graph)
            start = datetime.now(UTC).timestamp()
            state = AssetBackfillData.from_asset_partitions(asset_graph=graph, partition_names=days,
                asset_selection=[key], dynamic_partitions_store=instance, backfill_start_timestamp=start, all_partitions=False)
            def register(bid, state):
                instance.add_backfill(PartitionBackfill(backfill_id=bid, status=BulkActionStatus.REQUESTED,
                    from_failure=False, tags={}, backfill_timestamp=state.backfill_start_timestamp,
                    asset_selection=[key], asset_backfill_data=state))
            def tick(bid, state):
                v = view()
                result = execute_asset_backfill_iteration_inner(backfill_id=bid, asset_backfill_data=state,
                    asset_graph_view=v, backfill_start_timestamp=state.backfill_start_timestamp, logger=logger, run_config=None)
                state = result.backfill_data.with_run_requests_submitted(result.run_requests, v)
                instance.update_backfill(instance.get_backfill(bid).with_asset_backfill_data(state, dynamic_partitions_store=instance, asset_graph=graph))
                return state, result.run_requests
            def run_requested(bid, requests):
                assert len(requests) == 1, "Ordering must come from the planner, not a worker-count limit"
                request = requests[0]
                day = request.partition_key or request.tags[ASSET_PARTITION_RANGE_START_TAG]
                result = job.execute_in_process(instance=instance,
                    tags={**request.tags, BACKFILL_ID_TAG: bid}, raise_on_error=False)
                events.append({"backfill": bid, "date": day, "run_id": result.run_id,
                    "status": instance.get_run_by_id(result.run_id).status.value})
                return day, result
            register("isolated-first", state)
            state, requested = tick("isolated-first", state)
            day, first = run_requested("isolated-first", requested)
            assert day == days[0] and first.success
            _, first_ref, _ = archive.state(days[0])
            first_pointer = objects[archive.prefix+f"game_date_pacific={days[0]}/latest.json"]
            state, requested = tick("isolated-first", state)
            day, failed = run_requested("isolated-first", requested)
            assert day == days[1] and not failed.success
            state, requested = tick("isolated-first", state)
            assert not requested
            assert archive.state(days[1]) == archive.state(days[2]) == (None, None, None)
            assert not instance.get_run_records(filters=RunsFilter(tags={ASSET_PARTITION_RANGE_START_TAG: days[2]}))
            failed_dates = sorted(p.partition_key for p in state.failed_and_downstream_subset.iterate_asset_partitions())
            assert failed_dates == days[1:]
            # Native failed-and-downstream scope retries only the failed suffix.
            objects[missing_key] = removed_input
            retry = AssetBackfillData.from_asset_graph_subset(state.failed_and_downstream_subset,
                dynamic_partitions_store=instance, backfill_start_timestamp=datetime.now(UTC).timestamp())
            register("isolated-retry", retry)
            retry, requested = tick("isolated-retry", retry)
            day, second = run_requested("isolated-retry", requested)
            assert day == days[1] and second.success
            retry, requested = tick("isolated-retry", retry)
            day, third = run_requested("isolated-retry", requested)
            assert day == days[2] and third.success
            retry, requested = tick("isolated-retry", retry)
            assert not requested and retry.failed_and_downstream_subset.is_empty
            assert sorted(p.partition_key for p in retry.materialized_subset.iterate_asset_partitions()) == days[1:]
            second_state, second_ref, _ = archive.state(days[1])
            third_state, third_ref, _ = archive.state(days[2])
            assert second_state["inputs"]["parent"] == first_ref
            assert third_state["inputs"]["parent"] == second_ref
            assert objects[archive.prefix+f"game_date_pacific={days[0]}/latest.json"] == first_pointer
            assert len(instance.get_run_records(filters=RunsFilter(tags={ASSET_PARTITION_RANGE_START_TAG: days[0]}))) == 1
            report = {"test_storage": str(tmp_path), "network_connections_allowed": False,
                "events": events, "failed_and_blocked_dates": failed_dates,
                "third_date_started_before_repair": False, "first_date_rerun": False,
                "second_parent": second_state["inputs"]["parent"], "third_parent": third_state["inputs"]["parent"],
                "second_reference": second_ref, "third_reference": third_ref,
                "object_count": len(objects), "production_s3_calls": 0}
            (tmp_path / "ordering-evidence.json").write_text(json.dumps(report, indent=2))
            print("ORDERING_EVIDENCE "+json.dumps(report))
