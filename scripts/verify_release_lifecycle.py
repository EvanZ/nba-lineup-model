"""Run the current #57 test tranche and produce an explicit, partial-scope report.

This does not deploy, ingest, train a real model, or activate a Dagster sensor.
Run on EC2 with its installed project runtime. Output must be a new directory.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

TESTS = [
    "tests/test_release_lifecycle_integration.py",
    "tests/test_release_artifact_contracts.py",
    "tests/test_release_publisher.py",
    "tests/test_forecast_publication.py",
    "tests/test_release_bound_serving.py",
    "tests/test_daily_work.py",
    "tests/test_resumable_pipeline.py",
    "tests/test_daily_artifacts.py",
    "tests/test_daily_model_replay.py",
    "tests/test_current_season_serving_materialization.py",
    "tests/test_player_game_values.py",
]
PENDING = [
    "One integrated nightly Dagster -> data -> coupled model -> all serving surfaces run.",
    "Publisher crash/reload/concurrent activation exercised with real assembled runtime/API processes.",
    "Multi-request browser coherence during an isolated activation.",
    "Full materialized dev API and rendered coverage rerun, including missed games and viewport resizing.",
    "Tablet header overflow and NBA-logo browser loading verification.",
]
PROVEN = [
    "Complete recorded slate: pregame -> cutoff -> final scores -> official boxes -> ledgers and metrics.",
    "Published injury revision before cutoff; late observations do not replace the locked forecast.",
    "Missing original publication remains visible and cannot be backdated.",
    "Missing, mismatched, or malformed box for one game does not hide other games.",
    "Late box arrival fills retrospective without changing projected minutes or ratings.",
    "Ordinary late-start rejection, official rescheduling, and suspended-game retention.",
    "API rejects stale release pins and changed selected forecast files.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_release.resolve(strict=True)
    if not root.is_relative_to(Path("/var/lib/nba-gestalt-dev")):
        parser.error("An explicitly selected EC2 dev release is required")
    manifest = json.loads((root / "runtime-manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    env = dict(os.environ, GESTALT_LIFECYCLE_SOURCE_RELEASE=str(root), OPENBLAS_NUM_THREADS="1")
    cmd = [sys.executable, "-m", "pytest", *TESTS, "-q", "--tb=short",
           "--basetemp", str(output / "pytest"),
           "--junitxml", str(output / "tests.xml")]
    result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "tests.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    cases = []
    if (output / "tests.xml").is_file():
        for case in ET.parse(output / "tests.xml").iter("testcase"):
            failure = case.find("failure")
            if failure is None:
                failure = case.find("error")
            skipped = case.find("skipped")
            cases.append({
                "name": case.get("classname", "") + "::" + case.get("name", ""),
                "seconds": case.get("time"),
                "status": "failed" if failure is not None else "skipped" if skipped is not None else "passed",
                "details": failure.text if failure is not None else skipped.get("message") if skipped is not None else "",
            })
    counts = {key: sum(c["status"] == key for c in cases) for key in ("passed", "failed", "skipped")}
    evidence = [
        json.loads(p.read_text()) for p in (output / "pytest").glob("*/lifecycle-evidence.json")
        if not p.parent.is_symlink()
    ]
    integration = [c for c in cases if c["name"].split("::")[0].endswith("test_release_lifecycle_integration")]
    clean = result.returncode == 0 and bool(integration) and all(c["status"] == "passed" for c in integration)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "issue-57 first integrated tranche; not full ticket acceptance",
        "status": "partial_scope_passed" if clean else "failed",
        "source_runtime_sha256": manifest["runtime_sha256"],
        "source_release": str(root), "tests": cases, "counts": counts, "evidence": evidence,
        "verified": PROVEN if clean else [], "pending": PENDING,
        "boundaries": {
            "production_changed": False, "dev_activation_performed": False,
            "nightly_sensor_enabled": False, "real_season_replay": False,
            "permissions_changed": False, "deferred_minutes_bug": 60,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    esc = lambda v: html.escape(str(v))
    rows = "".join(
        f'<tr><td>{esc(c["name"])}</td><td>{esc(c["status"])}</td><td>{esc(c["seconds"])}</td>'
        f'<td><pre>{esc(c["details"])}</pre></td></tr>' for c in cases
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Issue 57 lifecycle verification</title><style>'
        'body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;background:#faf9f5;color:#173c31}'
        'table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;vertical-align:top;'
        'border-bottom:1px solid #ddd;padding:.6rem;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'code{overflow-wrap:anywhere}th:first-child{width:50%}li{margin:.5rem 0}</style></head><body>'
        '<h1>Lifecycle verification — first tranche</h1>'
        '<p><strong>Not a complete #57 sign-off. No production change or real season replay.</strong></p>'
        f'<p>Status: {esc(report["status"])}. {counts["passed"]} passed; {counts["failed"]} failed; '
        f'{counts["skipped"]} skipped.</p><p>Runtime: <code>{esc(manifest["runtime_sha256"])}</code></p>'
        '<h2>Verified integration paths</h2><ul>' +
        "".join(f"<li>{esc(x)}</li>" for x in report["verified"]) + '</ul>'
        '<h2>Still required</h2><ul>' + "".join(f"<li>{esc(x)}</li>" for x in PENDING) + '</ul>'
        '<h2>Recorded-source fixture coverage</h2><pre>' + esc(json.dumps(evidence, indent=2)) + '</pre>'
        '<h2>Every test result</h2><table><thead><tr><th>Test</th><th>Result</th><th>Seconds</th>'
        '<th>Failure / skip detail</th></tr></thead><tbody>' + rows + '</tbody></table></body></html>'
    )
    (output / "report.html").write_text(document)
    print(json.dumps({"status": report["status"], "counts": counts, "report": str(output/"report.html")}))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
