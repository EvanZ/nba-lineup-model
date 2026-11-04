from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nba_lineup_model.season.data_quality_report import (
    build_data_quality_report,
    publish_data_quality_report,
    publish_failed_data_quality_report,
    reconcile_alert_state,
    verify_data_quality_report,
)

NOW = datetime(2026, 11, 4, 16, 0, tzinfo=UTC)


def artifact(
    name: str,
    *,
    evidence_date: str = "2026-11-02",
    edition: str = "edition-v2",
    revision: str = "data-r2",
    expected_games: int = 107,
    observed_games: int = 107,
    states: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "layer": "served_surface",
        "artifact_id": f"{name}-id",
        "checksum": f"{name}-sha",
        "evidence_through_date": evidence_date,
        "observed_data_revision": revision,
        "model_edition": edition,
        "coverage": {
            "completed_games": {
                "expected": expected_games,
                "observed": observed_games,
            }
        },
        "states": states
        or {
            "materialized": True,
            "validated": True,
            "activated": True,
            "served": True,
        },
        "required_states": ["materialized", "validated", "activated", "served"],
    }


def collector(
    name: str,
    *,
    required_now: bool = True,
    latest_status: str = "SUCCESS",
    automation_status: str = "RUNNING",
    last_success_at: str = "2026-11-04T15:30:00Z",
    artifact_observed_at: str = "2026-11-04T15:30:00Z",
    created_at: str = "2026-11-04T15:29:00Z",
    required_since_at: str = "",
    expected_by: str = "",
) -> dict[str, object]:
    return {
        "name": name,
        "job_name": f"collect_{name}",
        "automation_name": f"{name}_schedule",
        "automation_status": automation_status,
        "latest_run_id": f"{name}-run",
        "latest_run_status": latest_status,
        "latest_run_created_at": created_at,
        "latest_run_started_at": created_at,
        "latest_run_finished_at": last_success_at,
        "last_success_at": last_success_at,
        "artifact_observed_at": artifact_observed_at,
        "required_since_at": required_since_at,
        "expected_by": expected_by,
        "required_now": required_now,
        "quiet_reason": "" if required_now else "No NBA games are scheduled today.",
        "stale_after_minutes": 90,
        "stall_after_minutes": 45,
    }


def report(
    *,
    artifacts: list[dict[str, object]] | None = None,
    collectors: list[dict[str, object]] | None = None,
    activation: dict[str, object] | None = None,
    failures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_data_quality_report(
        report_date="2026-11-04",
        environment="dev",
        expected={
            "season": "2026-27",
            "latest_eligible_completed_date": "2026-11-02",
            "completed_games": 107,
            "in_progress_games": 0,
            "scheduled_games_today": 0,
        },
        activation=activation
        or {
            "state": "active",
            "generation": 6,
            "release_id": "release-6",
            "edition_id": "edition-v2",
            "observed_data_revision": "data-r2",
        },
        artifacts=artifacts
        if artifacts is not None
        else [
            artifact("player_pages"),
            artifact("rankings"),
            artifact("teams"),
            artifact("lineups"),
            artifact("wins"),
            artifact("schedule_ledgers"),
            artifact("grades"),
            artifact("performance"),
        ],
        collectors=collectors
        if collectors is not None
        else [
            collector("schedule_injuries"),
            collector("odds"),
            collector("official_scores", required_now=False),
        ],
        pipeline_failures=failures or [],
        generated_at=NOW,
    )


def finding_codes(payload: dict[str, object]) -> set[str]:
    return {str(item["code"]) for item in payload["findings"]}


def test_clean_report_has_all_four_quality_dimensions_available():
    payload = report()
    assert payload["status"] == "clean"
    assert payload["finding_counts"] == {"error": 0, "warning": 0, "info": 0}
    assert len(payload["artifacts"]) == 8
    assert len(payload["collectors"]) == 3


def test_stale_player_surface_is_visible():
    payload = report(artifacts=[artifact("player_pages", evidence_date="2026-11-01")])
    assert payload["status"] == "failed"
    assert "artifact_evidence_stale" in finding_codes(payload)
    finding = payload["findings"][0]
    assert finding["category"] == "freshness"
    assert finding["scope"] == "artifact:player_pages"


def test_missing_games_are_visible_without_dropping_other_artifacts():
    artifacts = [
        artifact("player_pages", observed_games=106),
        artifact("rankings"),
    ]
    payload = report(artifacts=artifacts)
    assert "coverage_mismatch" in finding_codes(payload)
    assert len(payload["artifacts"]) == 2
    finding = next(item for item in payload["findings"] if item["code"] == "coverage_mismatch")
    assert finding["expected"] == 107
    assert finding["observed"] == 106


def test_mixed_model_edition_and_data_revision_are_visible():
    payload = report(
        artifacts=[
            artifact("rankings", edition="old-edition"),
            artifact("wins", revision="old-data"),
        ]
    )
    assert finding_codes(payload) == {
        "model_edition_mismatch",
        "observed_data_revision_mismatch",
    }
    assert {item["category"] for item in payload["findings"]} == {"consistency"}


@pytest.mark.parametrize("status", ["FAILURE", "FAILED"])
def test_collector_failure_is_visible(status: str):
    payload = report(collectors=[collector("official_scores", latest_status=status)])
    assert "collector_run_failed" in finding_codes(payload)
    assert payload["status"] == "failed"


@pytest.mark.parametrize("status", ["QUEUED", "STARTED"])
def test_orphaned_or_stalled_run_is_visible(status: str):
    payload = report(
        collectors=[
            collector(
                "schedule_injuries",
                latest_status=status,
                created_at="2026-11-04T14:00:00Z",
            )
        ]
    )
    assert "collector_run_stalled" in finding_codes(payload)


def test_stalled_activation_is_visible():
    payload = report(
        activation={
            "state": "staging",
            "generation": 7,
            "edition_id": "edition-v2",
            "observed_data_revision": "data-r2",
            "state_changed_at": "2026-11-04T14:00:00Z",
            "stall_after_minutes": 30,
        },
        artifacts=[],
    )
    assert "activation_stalled" in finding_codes(payload)


def test_zero_game_day_and_in_progress_window_do_not_false_alarm():
    payload = report(
        collectors=[
            collector(
                "official_scores",
                required_now=False,
                latest_status="missing",
                automation_status="RUNNING",
                last_success_at="",
                artifact_observed_at="",
            )
        ]
    )
    assert payload["status"] == "clean"
    assert payload["collectors"][0]["quiet_reason"]


def test_required_collector_checks_both_run_and_artifact_freshness():
    payload = report(
        collectors=[
            collector(
                "odds",
                last_success_at="2026-11-04T12:00:00Z",
                artifact_observed_at="2026-11-04T11:00:00Z",
            )
        ]
    )
    assert finding_codes(payload) == {
        "collector_success_stale",
        "collector_artifact_stale",
    }


def test_required_simulated_incidents_are_persisted_in_browsable_html(
    tmp_path: Path,
):
    scenarios = [
        (
            report(artifacts=[artifact("player_pages", evidence_date="2026-11-01")]),
            "artifact_evidence_stale",
        ),
        (
            report(artifacts=[artifact("rankings", observed_games=106)]),
            "coverage_mismatch",
        ),
        (
            report(artifacts=[artifact("rankings", edition="old-edition")]),
            "model_edition_mismatch",
        ),
        (
            report(collectors=[collector("schedule_injuries", latest_status="FAILURE")]),
            "collector_run_failed",
        ),
        (
            report(
                activation={
                    "state": "staging",
                    "generation": 7,
                    "edition_id": "edition-v2",
                    "observed_data_revision": "data-r2",
                    "state_changed_at": "2026-11-04T14:00:00Z",
                    "stall_after_minutes": 30,
                },
                artifacts=[],
            ),
            "activation_stalled",
        ),
    ]
    for payload, expected_code in scenarios:
        bundle = publish_data_quality_report(payload, tmp_path)
        assert expected_code in (bundle / "report.html").read_text()
        assert verify_data_quality_report(bundle)["status"] == "failed"


def test_failed_report_is_still_browsable_and_immutable(tmp_path: Path):
    bundle = publish_failed_data_quality_report(
        report_date="2026-11-04",
        environment="dev",
        output_root=tmp_path,
        stage="collect_api",
        error=RuntimeError("API unavailable"),
        generated_at=NOW,
    )
    manifest = verify_data_quality_report(bundle)
    payload = json.loads((bundle / "report.json").read_text())
    assert manifest["status"] == "failed"
    assert "API unavailable" in (bundle / "report.html").read_text()
    assert payload["pipeline_failures"][0]["stage"] == "collect_api"
    pointer = json.loads((tmp_path / "latest.json").read_text())
    assert Path(pointer["report_path"]).parts[0] == "reports"

    with pytest.raises(ValueError, match="file changed"):
        (bundle / "report.html").write_text("changed")
        verify_data_quality_report(bundle)


def test_successful_report_publishes_date_and_global_pointers(tmp_path: Path):
    payload = report()
    bundle = publish_data_quality_report(payload, tmp_path)
    first = verify_data_quality_report(bundle)
    second_bundle = publish_data_quality_report(payload, tmp_path)
    assert second_bundle == bundle
    assert verify_data_quality_report(second_bundle) == first
    global_pointer = json.loads((tmp_path / "latest.json").read_text())
    date_pointer = json.loads((tmp_path / "reports" / "2026-11-04" / "latest.json").read_text())
    assert global_pointer == date_pointer
    assert global_pointer["status"] == "clean"
    assert (tmp_path / "current").resolve() == bundle
    (bundle / "unexpected.txt").write_text("not sealed")
    with pytest.raises(ValueError, match="unsealed"):
        verify_data_quality_report(bundle)


def test_alerts_open_once_remain_quiet_and_recover_once():
    failed = report(artifacts=[artifact("player_pages", observed_games=106)])
    state1, events1 = reconcile_alert_state(failed, None, observed_at=NOW)
    assert [event["event"] for event in events1] == ["opened"]

    later = datetime(2026, 11, 4, 16, 30, tzinfo=UTC)
    state2, events2 = reconcile_alert_state(failed, state1, observed_at=later)
    assert events2 == []
    active = next(iter(state2["active"].values()))
    assert active["occurrences"] == 2

    clean = report()
    state3, events3 = reconcile_alert_state(clean, state2, observed_at=later)
    assert [event["event"] for event in events3] == ["recovered"]
    assert state3["active"] == {}

    state4, events4 = reconcile_alert_state(clean, state3, observed_at=later)
    assert events4 == []
    assert state4["active"] == {}


def test_lifecycle_state_failure_is_reported():
    states = {
        "materialized": True,
        "validated": True,
        "activated": True,
        "served": False,
    }
    payload = report(artifacts=[artifact("player_pages", states=states)])
    assert "artifact_not_served" in finding_codes(payload)


def test_scheduled_poll_detects_missed_due_run_and_artifact():
    payload = report(
        collectors=[
            collector(
                "odds",
                last_success_at="2026-11-04T14:00:00Z",
                artifact_observed_at="2026-11-04T14:00:00Z",
                required_since_at="2026-11-04T15:20:00Z",
                expected_by="2026-11-04T15:50:00Z",
            )
        ]
    )
    assert finding_codes(payload) == {
        "collector_scheduled_success_missing",
        "collector_scheduled_artifact_missing",
    }


def test_quiet_collector_still_reports_disabled_automation():
    payload = report(
        collectors=[
            collector(
                "official_scores",
                required_now=False,
                automation_status="STOPPED",
            )
        ]
    )
    assert "collector_automation_inactive" in finding_codes(payload)


def test_nightly_publication_is_late_only_after_five_am_deadline():
    common = {
        "season": "2026-27",
        "latest_eligible_completed_date": "2026-11-03",
        "required_evidence_through_date": "2026-11-03",
        "active_evidence_through_date": "2026-11-02",
        "completed_games": 110,
        "in_progress_games": 0,
        "scheduled_games_today": 0,
    }
    before = build_data_quality_report(
        report_date="2026-11-04",
        environment="dev",
        expected={**common, "nightly_publication_due": False},
        activation={
            "state": "active",
            "edition_id": "edition-v2",
            "observed_data_revision": "data-r2",
        },
        artifacts=[],
        collectors=[],
        generated_at=datetime(2026, 11, 4, 12, 59, tzinfo=UTC),
    )
    after = build_data_quality_report(
        report_date="2026-11-04",
        environment="dev",
        expected={
            **common,
            "nightly_publication_due": True,
            "nightly_publication_deadline": "2026-11-04T13:00:00Z",
        },
        activation={
            "state": "active",
            "edition_id": "edition-v2",
            "observed_data_revision": "data-r2",
        },
        artifacts=[],
        collectors=[],
        generated_at=datetime(2026, 11, 4, 13, 1, tzinfo=UTC),
    )
    assert "nightly_publication_late" not in finding_codes(before)
    assert "nightly_publication_late" in finding_codes(after)
