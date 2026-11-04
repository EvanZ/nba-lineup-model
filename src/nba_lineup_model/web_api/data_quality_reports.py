"""Compose the operational and current-season model audits into one report."""

from __future__ import annotations

import re

_MAIN = re.compile(r"<main(?:\s[^>]*)?>(?P<body>.*)</main>", re.IGNORECASE | re.DOTALL)
_FIRST_H1 = re.compile(
    r"<h1(?:\s[^>]*)?>(?P<title>.*?)</h1>", re.IGNORECASE | re.DOTALL
)


def _main_body(document: str, *, label: str) -> str:
    match = _MAIN.search(document)
    if match is None:
        raise ValueError(f"{label} report does not contain a main element")
    return match.group("body").strip()


def compose_data_quality_report(
    operational_report: str,
    current_season_report: str,
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
        operational_body
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
    )
    if "</style>" not in document:
        raise ValueError("Operational report does not contain a style block")
    return document.replace("</style>", extra_style + "</style>", 1)
