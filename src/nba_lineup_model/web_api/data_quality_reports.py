"""Compose the operational and current-season model audits into one report."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

_MAIN = re.compile(r"<main(?:\s[^>]*)?>(?P<body>.*)</main>", re.IGNORECASE | re.DOTALL)
_FIRST_H1 = re.compile(r"<h1(?:\s[^>]*)?>(?P<title>.*?)</h1>", re.IGNORECASE | re.DOTALL)


def _main_body(document: str, *, label: str) -> str:
    match = _MAIN.search(document)
    if match is None:
        raise ValueError(f"{label} report does not contain a main element")
    return match.group("body").strip()


def read_data_quality_status(root: Path, *, files=None) -> dict[str, object]:
    """Return the current incident state used by the report and site indicator."""

    pointer_path = root / "latest.json"
    state_path = root / "alert-state.json"
    if files is None:
        from nba_lineup_model.storage import website_data
        uri = os.environ.get('GESTALT_OPERATIONAL_REPORTS_S3_URI')
        if uri:
            from nba_lineup_model.storage.operational_reports import read
            files = read(uri, 'data-quality')
        elif website_data.active():
            raise ValueError('Durable website reports require an S3 source')
    if files is not None:
        pointer = json.loads(files['latest.json'])
        state = json.loads(files['alert-state.json'])
    else:
        pointer = json.loads(pointer_path.read_text()) if pointer_path.is_file() else {}
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    active = [
        dict(value) for value in dict(state.get("active") or {}).values() if isinstance(value, dict)
    ]
    status = str(pointer.get("status") or "unavailable")
    return {
        "status": status,
        "urgent": bool(active) or status == "failed",
        "open_incident_count": len(active),
        "updated_at": str(state.get("updated_at") or pointer.get("generated_at") or ""),
        "report_date": str(pointer.get("report_date") or ""),
        "incidents": active,
    }


def _alert_banner(status: dict[str, object] | None) -> str:
    if not status or not status.get("urgent"):
        return ""
    incidents = status.get("incidents") or []
    items = []
    for incident in incidents:
        finding = incident.get("finding") if isinstance(incident, dict) else {}
        finding = finding if isinstance(finding, dict) else {}
        items.append(
            "<li><strong>"
            + html.escape(str(finding.get("scope") or "Data quality"))
            + ":</strong> "
            + html.escape(str(finding.get("message") or "Urgent review required."))
            + "</li>"
        )
    if not items:
        items.append("<li>The latest data-quality report failed and requires review.</li>")
    count = int(status.get("open_incident_count") or len(items))
    return (
        '<section class="dq-alert-banner" role="alert">'
        f"<h2>Urgent attention required · {count} open incident{'s' if count != 1 else ''}</h2>"
        + "<ul>"
        + "".join(items)
        + "</ul></section>"
    )


def compose_data_quality_report(
    operational_report: str,
    current_season_report: str,
    *,
    quality_status: dict[str, object] | None = None,
) -> str:
    """Return one browser report while preserving both immutable inputs.

    The operational report remains the outer document because it is generated
    more frequently than a model release. The immutable current-season audit
    is appended as a clearly labelled section; neither source artifact is
    rewritten.
    """

    operational_body = _main_body(operational_report, label="Operational")
    model_body = _main_body(current_season_report, label="Current-season")
    model_body, replacements = _FIRST_H1.subn(
        r"<h2>Current-season model audit</h2>", model_body, count=1
    )
    if replacements != 1:
        raise ValueError("Current-season report does not contain a title")

    combined_body = (
        _alert_banner(quality_status)
        + operational_body
        + '<section class="dq-model-audit" aria-labelledby="current-season-model-audit">'
        + model_body.replace(
            "<h2>Current-season model audit</h2>",
            '<h2 id="current-season-model-audit">Current-season model audit</h2>',
            1,
        )
        + "</section>"
    )
    document = _MAIN.sub(f"<main>{combined_body}</main>", operational_report, count=1)
    extra_style = (
        ".dq-model-audit{margin-top:3rem;padding-top:2rem;border-top:3px solid var(--ink)}"
        ".dq-model-audit .cards{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(180px,1fr));gap:12px}"
        ".dq-model-audit .card strong{display:block;font-size:24px}"
        ".dq-model-audit .clean,.dq-model-audit .warning{padding:12px 16px;background:#fff}"
        ".dq-model-audit .clean{border-left:4px solid var(--good)}"
        ".dq-model-audit .warning{border-left:4px solid var(--warn)}"
        ".dq-model-audit table{margin-top:22px}"
        ".dq-alert-banner{padding:14px 18px;margin-bottom:20px;border:2px solid var(--bad);"
        "background:#fbe9e4;color:var(--bad)}"
        ".dq-alert-banner h2{margin:0 0 8px}.dq-alert-banner ul{margin:0;padding-left:20px}"
    )
    if "</style>" not in document:
        raise ValueError("Operational report does not contain a style block")
    return document.replace("</style>", extra_style + "</style>", 1)
