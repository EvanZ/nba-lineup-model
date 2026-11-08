from test_daily_instance_wiring import pipeline
import json
import sqlite3
from pathlib import Path

import pytest
from dagster import DagsterInstance

from nba_lineup_model.orchestration.daily_instance import execute_manual, initialize
from nba_lineup_model.orchestration.daily_history_migration import migrate, rows, record_digest
from nba_lineup_model.orchestration.daily_run_status import read_daily_runs
from nba_lineup_model.orchestration.daily_work import atomic_json, digest
from test_daily_instance_wiring import environment, request

pytest_plugins = ("test_daily_instance_wiring",)

TEMPLATE = Path(__file__).resolve().parents[1] / "deploy/dagster-daily.yaml"


def source_history(binding, name):
    source = binding.root / name / "dagster"
    source.parent.mkdir()
    binding.home.rename(source)
    return source


def test_preserves_failures_successes_events_and_ids_and_is_repeatable(tmp_path, pipeline):
    binding = environment(tmp_path, "dev")
    req = request(binding)
    failed = execute_manual(binding.home, "prepare_daily_partition", req)
    source1 = source_history(binding, "first")
    initialize("dev", binding.root, TEMPLATE)
    pipeline["fail"] = False
    other = execute_manual(binding.home, "prepare_daily_partition", request(binding, "2026-10-31"))
    source2 = source_history(binding, "second")
    originals = {r["run_id"]:record_digest(r) for src in [source1, source2]
                 for r in rows(src / "history/runs.db")}
    result = migrate("dev", binding.root, [source1,source2], TEMPLATE, "test approval")
    assert result["verified"] and result["migrated_runs"] == 2
    assert migrate("dev", binding.root, [source2,source1], TEMPLATE, "test approval") == result
    with DagsterInstance.from_config(str(binding.home)) as instance:
        assert instance.get_run_by_id(failed.run_id).status.value == "FAILURE"
        assert instance.get_run_by_id(other.run_id).status.value == "SUCCESS"
        assert instance.all_logs(failed.run_id) and instance.all_logs(other.run_id)
    migrated = rows(binding.home / "history/runs.db")
    assert {r["run_id"]:record_digest(r) for r in migrated} == originals
    state = read_daily_runs(binding.home, "dev", binding.root)
    assert [r["run_id"] for r in state["unresolved_failures"]] == [failed.run_id]
    assert execute_manual(binding.home, "prepare_daily_partition", req).success
    assert not read_daily_runs(binding.home, "dev", binding.root)["unresolved_failures"]
    # Original logs remain unchanged even after the new retry.
    assert {r["run_id"]:record_digest(r) for src in [source1,source2]
            for r in rows(src / "history/runs.db")} == originals
    assert migrate("dev", binding.root, [source1,source2], TEMPLATE, "test approval")["verified"]


def test_migrated_status_tampering_is_rejected(tmp_path, pipeline):
    binding = environment(tmp_path, "dev")
    failed = execute_manual(binding.home, "prepare_daily_partition", request(binding))
    source = source_history(binding, "first")
    migrate("dev", binding.root, [source], TEMPLATE, "test approval")
    with sqlite3.connect(binding.home / "history/runs.db") as c:
        c.execute("update runs set status='SUCCESS' where run_id=?", (failed.run_id,))
    with pytest.raises(ValueError, match="Migrated run record changed"):
        read_daily_runs(binding.home, "dev", binding.root)


@pytest.mark.parametrize("change_edition", [False, True])
def test_code_fix_can_recover_same_output_but_other_edition_cannot(tmp_path, pipeline, change_edition):
    binding = environment(tmp_path, "dev")
    req = request(binding)
    failed = execute_manual(binding.home, "prepare_daily_partition", req)
    old = json.loads(Path(req["spec_path"]).read_text())
    old["identity"]["code_identity"] = "fixed-code"
    if change_edition:
        old["identity"]["edition_id"] = "different-model"
    old["work_id"] = digest(old["identity"])
    fixed = binding.root / "fixed-spec.json"
    atomic_json(fixed, old)
    pipeline["fail"] = False
    assert execute_manual(binding.home, "prepare_daily_partition", {**req,"spec_path":str(fixed)}).success
    unresolved = read_daily_runs(binding.home, "dev", binding.root)["unresolved_failures"]
    assert bool(unresolved) == change_edition
    if change_edition:
        assert unresolved[0]["run_id"] == failed.run_id
