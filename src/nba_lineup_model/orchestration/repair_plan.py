"""Reviewable correction plans and exact-scope execution approvals.

Planning never activates data or runs a model. An approval is an audit record of
an external user decision, not something a worker is allowed to invent.
"""
from __future__ import annotations

import html
from datetime import date
from pathlib import Path

from nba_lineup_model.orchestration.daily_work import (
    atomic_json, digest, verify_reference,
)

ACTIONS = {"select_historical_data", "causal_replay", "replace_forecast_edition"}
CLASSIFICATIONS = {"presentation_only", "grade_only", "model_input", "missing_daily_update"}


def make_impact_plan(*, season: str, classification: str, changed_inputs: list[dict],
                     affected_games: dict[str, list[str]], affected_artifacts: list[str],
                     model_suffix: list[str], forecast_replacement: bool,
                     base_release_id: str, edition_id: str, explanation: str,
                     source_coverage: dict | None = None) -> dict:
    if classification not in CLASSIFICATIONS or not explanation.strip():
        raise ValueError("A supported classification and explanation are required")
    if not base_release_id or not edition_id or not season:
        raise ValueError("The plan must identify its base release and edition")
    for day, games in affected_games.items():
        date.fromisoformat(day)
        if len(games) != len(set(games)):
            raise ValueError("Duplicate games in impact plan")
    parsed = [date.fromisoformat(day) for day in model_suffix]
    if parsed != sorted(set(parsed)):
        raise ValueError("Model suffix must be ordered without duplicate dates")
    if any((b - a).days != 1 for a, b in zip(parsed, parsed[1:])):
        raise ValueError("Model suffix cannot skip a Pacific date")
    if classification in {"presentation_only", "grade_only"} and model_suffix:
        raise ValueError("An independent correction cannot request a causal replay")
    if classification == "model_input" and not model_suffix:
        raise ValueError("A model-input correction must identify its dependent suffix")
    for change in changed_inputs:
        if not change.get("reason") or not (change.get("before") or change.get("after")):
            raise ValueError("Input changes require explicit references and a reason")
        for key in ("before", "after"):
            if change.get(key):
                verify_reference(change[key])
    identity = {
        "schema_version": 1, "season": season, "classification": classification,
        "changed_inputs": changed_inputs, "affected_games": affected_games,
        "affected_artifacts": sorted(set(affected_artifacts)), "model_suffix": model_suffix,
        "forecast_replacement": bool(forecast_replacement),
        "base_release_id": base_release_id, "edition_id": edition_id,
        "explanation": explanation,
        "source_coverage": source_coverage or {},
    }
    return {"plan_id": digest(identity), "identity": identity}


def verify_plan(plan: dict) -> None:
    if plan.get("plan_id") != digest(plan["identity"]):
        raise ValueError("Impact plan checksum mismatch")
    identity = plan["identity"]
    # Reconstruct to check scope invariants as well as the checksum.
    rebuilt = make_impact_plan(**{k: v for k, v in identity.items() if k != "schema_version"})
    if rebuilt != plan:
        raise ValueError("Invalid impact plan schema")


def require_approval(plan: dict, approval: dict | None, action: str) -> None:
    verify_plan(plan)
    if action not in ACTIONS:
        raise ValueError("Unknown gated action")
    if not approval or approval.get("plan_id") != plan["plan_id"]:
        raise PermissionError("Explicit approval of this exact impact plan is required")
    if not approval.get("approval_reference") or not approval.get("approved_by"):
        raise PermissionError("Approval must cite the user decision and approver")
    if action not in approval.get("actions", []):
        raise PermissionError(f"Approval does not authorize {action}")
    if action == "causal_replay" and not plan["identity"]["model_suffix"]:
        raise ValueError("Plan does not request a model suffix")
    if action == "replace_forecast_edition" and not plan["identity"]["forecast_replacement"]:
        raise ValueError("Plan preserves historical forecasts")


def write_impact_plan(root: Path, plan: dict) -> Path:
    verify_plan(plan)
    target = root / plan["plan_id"]
    atomic_json(target / "plan.json", plan, immutable=True)
    item = plan["identity"]
    rows = "".join(
        f"<tr><td>{html.escape(day)}</td><td>{len(games)}</td>"
        "<td>" + (" ".join(f"<span class='game-id'>{html.escape(game)}</span>" for game in games) or "Zero-game day") + "</td></tr>"
        for day, games in sorted(item["affected_games"].items())
    )
    coverage = "".join(
        f"<li>{html.escape(str(label))}: <strong>{html.escape(str(value))}</strong></li>"
        for label, value in item["source_coverage"].items()
    )
    text = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Daily repair impact plan</title><style>"
            "body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}"
            "td,th{text-align:left;padding:.6rem;border-bottom:1px solid #ccc}"
            "td{vertical-align:top}.game-id{display:inline-block;white-space:nowrap;margin-right:.4rem}"
            "th:first-child{width:27%}th:nth-child(2){width:15%}"
            "td,code{overflow-wrap:anywhere}table{width:100%;table-layout:fixed}"
            "</style></head><body><h1>Repair impact plan</h1>"
            "<p><strong>Proposed only — not executed or activated.</strong></p>"
            f"<p>{html.escape(item['explanation'])}</p>"
            f"<p>Classification: {html.escape(item['classification'])}</p>"
            f"<p>Historical forecasts: {'replacement requested' if item['forecast_replacement'] else 'preserved unchanged'}</p>"
            f"<p>Model dates: {html.escape(', '.join(item['model_suffix']) or 'None')}</p>"
            f"<p>Artifacts: {html.escape(', '.join(item['affected_artifacts']))}</p>"
            f"<p>Plan: <code>{plan['plan_id']}</code></p>"
            "<h2>Verified input coverage</h2><ul>" + coverage + "</ul>"
            "<table><thead><tr><th>Pacific date</th><th>Games</th><th>NBA game IDs</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table></body></html>")
    # Presentation revisions never change the impact plan or model inputs.
    path = target / f"report-{digest(text)}.html"
    if path.exists() and path.read_text() != text:
        raise ValueError("Impact report changed for an immutable plan")
    if not path.exists():
        path.write_text(text)
    return path
