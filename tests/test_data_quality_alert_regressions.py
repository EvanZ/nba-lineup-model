import io
import json
import sqlite3

import pytest

from nba_lineup_model.orchestration import data_quality_probe as probe
from nba_lineup_model.season.data_quality_report import (
    publish_data_quality_report,
    reconcile_alert_state,
)
from nba_lineup_model.web_api.data_quality_reports import (
    read_data_quality_status,
    compose_data_quality_report,
)
from test_data_quality_report import collector, finding_codes, report


@pytest.mark.parametrize("quiet", [True, False])
def test_failed_job_sets_urgent_status_even_when_current_data_needs_no_refresh(tmp_path, quiet):
    row = collector("daily_publication", required_now=not quiet, latest_status="FAILURE")
    payload = report(collectors=[row])
    assert "collector_run_failed" in finding_codes(payload)
    bundle = publish_data_quality_report(payload, tmp_path)
    state, events = reconcile_alert_state(payload, None)
    (tmp_path / "alert-state.json").write_text(json.dumps(state))
    status = read_data_quality_status(tmp_path)
    assert status["urgent"] and status["open_incident_count"] == 1
    rendered = compose_data_quality_report(
        (bundle / "report.html").read_text(),
        "<html><body><main><h1>Model audit</h1>Current data</main></body></html>",
        quality_status=status,
    )
    assert 'role="alert"' in rendered and "Urgent attention required" in rendered
    assert len(events) == 1


@pytest.mark.parametrize("status", ["QUEUED", "STARTING", "STARTED", "CANCELED"])
def test_retry_does_not_clear_failure_before_verified_success(status):
    row = collector("daily_publication", required_now=False, latest_status=status)
    row.update(last_failure_run_id="failed-run", last_failure_at="2026-11-04T15:45:00Z")
    payload = report(collectors=[row])
    assert "collector_run_failed" in finding_codes(payload)
    state, opened = reconcile_alert_state(payload, None)
    _, unchanged = reconcile_alert_state(payload, state)
    assert len(opened) == 1 and not unchanged
    row.update(latest_run_status="SUCCESS", last_success_at="2026-11-04T15:55:00Z")
    recovered = report(collectors=[row])
    assert "collector_run_failed" not in finding_codes(recovered)
    new_state, events = reconcile_alert_state(recovered, state)
    assert not new_state["active"] and events[0]["event"] == "recovered"


def test_stalled_job_alerts_during_quiet_period():
    row = collector(
        "daily_publication",
        required_now=False,
        latest_status="STARTED",
        created_at="2026-11-04T14:00:00Z",
    )
    assert "collector_run_stalled" in finding_codes(report(collectors=[row]))


def states_database(tmp_path, status):
    path = tmp_path / "schedules.db"
    c = sqlite3.connect(path)
    c.execute("create table instigators (status text, instigator_body text)")
    c.execute(
        "insert into instigators values (?, ?)",
        (status, json.dumps({"origin": {"job_name": "collector"}})),
    )
    c.commit()
    c.close()
    return path


@pytest.mark.parametrize(
    "stored,effective,source",
    [
        ("DECLARED_IN_CODE", "RUNNING", "code_default"),
        ("DECLARED_IN_CODE", "STOPPED", "code_default"),
        ("STOPPED", "STOPPED", "operator_override"),
        ("RUNNING", "RUNNING", "operator_override"),
    ],
)
def test_resolves_loaded_default_and_override(tmp_path, monkeypatch, stored, effective, source):
    db = states_database(tmp_path, stored)
    response = {
        "data": {
            "repositoriesOrError": {
                "__typename": "RepositoryConnection",
                "nodes": [
                    {
                        "schedules": [
                            {"name": "collector", "scheduleState": {"status": effective}}
                        ],
                        "sensors": [],
                    }
                ],
            }
        }
    }
    monkeypatch.setattr(
        probe.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(response).encode())
    )
    state = probe.effective_automation_states(db, "http://localhost")["collector"]
    assert state["automation_status"] == effective
    assert state["automation_status_source"] == source
    row = collector("schedule")
    row.update(state)
    payload = report(collectors=[row])
    assert ("collector_automation_inactive" in finding_codes(payload)) == (effective == "STOPPED")
    bundle = publish_data_quality_report(payload, tmp_path / "reports")
    html = (bundle / "report.html").read_text()
    assert "DECLARED_IN_CODE" not in html
    assert ("Enabled" if effective == "RUNNING" else "Disabled") in html


def test_unavailable_workspace_cannot_be_reported_as_enabled(tmp_path, monkeypatch):
    db = states_database(tmp_path, "DECLARED_IN_CODE")

    def unavailable(*args, **kwargs):
        raise OSError("Dagster unavailable")

    monkeypatch.setattr(probe.urllib.request, "urlopen", unavailable)
    state = probe.effective_automation_states(db, "http://localhost")["collector"]
    assert state["automation_status"] == "UNKNOWN"
    row = collector("schedule", required_now=False)
    row.update(state)
    assert "collector_automation_inactive" in finding_codes(report(collectors=[row]))


def test_probe_retains_failure_while_retry_is_queued(tmp_path):
    db = tmp_path / "runs.db"
    c = sqlite3.connect(db)
    c.execute("""create table runs (id integer primary key, run_id text, status text,
      pipeline_name text, create_timestamp text, start_time real, end_time real, update_timestamp text)""")
    c.executemany(
        "insert into runs values (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                1,
                "failure",
                "FAILURE",
                "job",
                "2026-11-04 12:00:00",
                1793793600,
                1793793610,
                "2026-11-04 12:00:10",
            ),
            (2, "retry", "QUEUED", "job", "2026-11-04 12:05:00", None, None, "2026-11-04 12:05:00"),
        ],
    )
    c.commit()
    c.close()
    state = probe.dagster_run_states(db, ["job"])["job"]
    assert state["status"] == "QUEUED" and state["last_failure_run_id"] == "failure"
    assert state["last_failure_at"] and not state["last_success_at"]
