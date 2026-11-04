from __future__ import annotations

import pytest

from nba_lineup_model.web_api.data_quality_reports import (
    compose_data_quality_report,
    read_data_quality_status,
)

OPERATIONAL = """<!doctype html><html><head><style>main{max-width:100%}</style></head>
<body><main><h1>Daily Data Quality — 2026-11-04</h1><p>No findings.</p></main></body></html>"""

CURRENT_SEASON = """<!doctype html><html><head><style>.cards{display:grid}</style></head>
<body><main><h1>Current-season serving-state audit</h1><div class="cards">
<div class="card"><strong>107</strong>final games</div></div>
<h2>Daily pipeline quality</h2><p class="clean">Clean run.</p></main></body></html>"""


def test_compose_data_quality_report_preserves_both_audits() -> None:
    report = compose_data_quality_report(OPERATIONAL, CURRENT_SEASON)

    assert "Daily Data Quality — 2026-11-04" in report
    assert "Current-season model audit" in report
    assert "107</strong>final games" in report
    assert "Clean run." in report
    assert report.count("<main>") == 1
    assert report.count("</main>") == 1
    assert 'class="dq-model-audit"' in report


@pytest.mark.parametrize(
    ("operational", "current", "message"),
    [
        ("<html></html>", CURRENT_SEASON, "Operational report"),
        (OPERATIONAL, "<html></html>", "Current-season report"),
        (OPERATIONAL, CURRENT_SEASON.replace("h1", "div"), "contain a title"),
    ],
)
def test_compose_data_quality_report_rejects_malformed_inputs(
    operational: str,
    current: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compose_data_quality_report(operational, current)


def test_open_incident_adds_urgent_banner():
    report = compose_data_quality_report(
        OPERATIONAL,
        CURRENT_SEASON,
        quality_status={
            "urgent": True,
            "open_incident_count": 1,
            "incidents": [
                {"finding": {"scope": "collector:odds", "message": "Scheduled poll is late."}}
            ],
        },
    )
    assert "Urgent attention required · 1 open incident" in report
    assert "Scheduled poll is late." in report
    assert 'role="alert"' in report


def test_status_reader_exposes_open_incidents(tmp_path):
    (tmp_path / "latest.json").write_text(
        '{"status":"failed","report_date":"2026-11-04","generated_at":"now"}'
    )
    (tmp_path / "alert-state.json").write_text(
        '{"updated_at":"now","active":{"abc":{"finding":{"code":"late"}}}}'
    )
    status = read_data_quality_status(tmp_path)
    assert status["urgent"] is True
    assert status["open_incident_count"] == 1
