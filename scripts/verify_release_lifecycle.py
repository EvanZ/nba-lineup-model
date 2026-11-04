"""Run issue #57's complete isolated lifecycle verification on EC2.

The verifier consumes explicit immutable dev inputs, builds a private release,
starts a private API process, and runs rendered browser checks. It never
activates dev/prod, enables a sensor, ingests live data, or retrains the model.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

TESTS = [
    "tests/test_release_lifecycle_integration.py",
    "tests/test_nightly_release_lifecycle_integration.py",
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
PROVEN = [
    "Complete recorded slate: pregame -> cutoff -> final scores -> official boxes -> ledgers and metrics.",
    "Published injury revision before cutoff; late observations do not replace the locked forecast.",
    "Missing original publication remains visible and cannot be backdated.",
    "Missing, mismatched, or malformed box for one game does not hide other games.",
    "Late box arrival fills retrospective without changing projected minutes or ratings.",
    "Ordinary late-start rejection, official rescheduling, and suspended-game retention.",
    "API rejects stale release pins and changed selected forecast files.",
    "Actual isolated Dagster job builds data, coupled model, six serving surfaces, and one runtime.",
    "Actual assembled API serves schedule, players, rankings, teams, lineups, wins, and metrics coherently.",
    "Publisher activation rejects a stale writer, rolls back failed verification, and recovers interruption.",
    "Rendered browser recovers from an old-to-new release switch without accepting mixed-release data.",
]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _health(api_root: str) -> dict:
    with urllib.request.urlopen(api_root.rstrip("/") + "/api/health", timeout=5) as response:
        return json.load(response)


def _release_id_from_health(payload: dict) -> str | None:
    loaded = payload.get("loaded_release")
    return str(loaded.get("release_id")) if isinstance(loaded, dict) and loaded.get("release_id") else None


@contextmanager
def _private_api(
    *,
    runtime: Path,
    observation_root: Path,
    output: Path,
):
    bindings = _read_json(runtime / "serving-bindings.json")
    pointer = output / "browser-new-live"
    pointer.symlink_to(runtime, target_is_directory=True)
    port = _free_port()
    api_root = "http://127.0.0.1:" + str(port)
    log_path = output / "browser-new-api.log"
    log_handle = log_path.open("w")
    env = dict(
        os.environ,
        GESTALT_ACTIVE_RELEASE_ROOT=str(pointer),
        GESTALT_LIVE_SCORE_STATE_DIR=str(observation_root),
        PYTHONPATH=str(Path.cwd() / "src"),
        OPENBLAS_NUM_THREADS="1",
    )
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "nba_lineup_model.web_api.app:app",
            "--host", "127.0.0.1", "--port", str(port), "--workers", "1",
        ],
        cwd=bindings["historical_runtime"],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    expected = _read_json(runtime / "runtime-manifest.json")["runtime_sha256"]
    deadline = time.monotonic() + 180
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Private API exited before readiness; see " + str(log_path))
            try:
                if _release_id_from_health(_health(api_root)) == expected:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Private API did not load assembled release; see " + str(log_path))
        yield api_root
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()


def _test_cases(path: Path) -> list[dict]:
    cases = []
    if not path.is_file():
        return cases
    for case in ET.parse(path).iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        skipped = case.find("skipped")
        cases.append({
            "name": case.get("classname", "") + "::" + case.get("name", ""),
            "seconds": case.get("time"),
            "status": "failed" if failure is not None else "skipped" if skipped is not None else "passed",
            "details": (
                failure.text if failure is not None
                else skipped.get("message") if skipped is not None
                else ""
            ),
        })
    return cases


def _evidence(root: Path, filename: str) -> list[dict]:
    return [
        _read_json(path)
        for path in root.glob("**/" + filename)
        if not path.parent.is_symlink()
    ]


def _run_browser(args, output: Path, nightly: dict, source_manifest: dict) -> tuple[int, dict]:
    old_health = _health(args.old_api)
    old_release = _release_id_from_health(old_health)
    expected_old = str(source_manifest["runtime_sha256"])
    if old_release != expected_old:
        raise RuntimeError(
            "Running old API does not match pinned source release: "
            + str(old_release) + " != " + expected_old
        )
    runtime = Path(nightly["runtime_release"]).resolve(strict=True)
    observation_root = Path(nightly["observation_root"]).resolve(strict=True)
    browser_output = output / "browser"
    browser_output.mkdir()
    with _private_api(runtime=runtime, observation_root=observation_root, output=output) as new_api:
        env = dict(
            os.environ,
            GESTALT_BROWSER_WEB=str(runtime / "web"),
            GESTALT_BROWSER_OUTPUT=str(browser_output),
            GESTALT_BROWSER_OLD_API=args.old_api,
            GESTALT_BROWSER_NEW_API=new_api,
            GESTALT_BROWSER_OLD_RELEASE=expected_old,
            GESTALT_BROWSER_NEW_RELEASE=str(nightly["runtime_sha256"]),
            GESTALT_PLAYWRIGHT_MODULE=str(args.playwright_tools / "node_modules/playwright/index.mjs"),
            PLAYWRIGHT_BROWSERS_PATH=str(args.playwright_tools / "browsers"),
            LD_LIBRARY_PATH=str(args.playwright_tools / "sysroot/usr/lib/aarch64-linux-gnu"),
        )
        result = subprocess.run(
            [str(args.node), "scripts/verify_release_browser.mjs"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    (output / "browser.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    evidence_path = browser_output / "browser-evidence.json"
    return result.returncode, _read_json(evidence_path) if evidence_path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--source-work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-api", default="http://127.0.0.1:8011")
    parser.add_argument(
        "--playwright-tools", type=Path,
        default=Path("/var/lib/nba-gestalt-dev/release-architecture-55/browser-tools"),
    )
    parser.add_argument(
        "--node", type=Path,
        default=Path("/home/ubuntu/.nvm/versions/node/v20.19.6/bin/node"),
    )
    args = parser.parse_args()

    source_release = args.source_release.resolve(strict=True)
    source_work = args.source_work_root.resolve(strict=True)
    dev_root = Path("/var/lib/nba-gestalt-dev")
    if not source_release.is_relative_to(dev_root) or not source_work.is_relative_to(dev_root):
        parser.error("Explicitly selected EC2 dev release and work root are required")
    source_manifest = _read_json(source_release / "runtime-manifest.json")
    if not args.node.is_file():
        parser.error("EC2 Node runtime is missing: " + str(args.node))
    if not (args.playwright_tools / "node_modules/playwright/index.mjs").is_file():
        parser.error("EC2 Playwright runtime is missing")

    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    env = dict(
        os.environ,
        GESTALT_LIFECYCLE_SOURCE_RELEASE=str(source_release),
        GESTALT_LIFECYCLE_SOURCE_WORK_ROOT=str(source_work),
        OPENBLAS_NUM_THREADS="1",
    )
    command = [
        sys.executable, "-m", "pytest", *TESTS, "-q", "--tb=short",
        "--basetemp", str(output / "pytest"),
        "--junitxml", str(output / "tests.xml"),
    ]
    result = subprocess.run(
        command, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (output / "tests.log").write_text(result.stdout)
    print(result.stdout, flush=True)

    cases = _test_cases(output / "tests.xml")
    counts = {
        key: sum(case["status"] == key for case in cases)
        for key in ("passed", "failed", "skipped")
    }
    recorded = _evidence(output / "pytest", "lifecycle-evidence.json")
    nightly = _evidence(output / "pytest", "nightly-lifecycle-evidence.json")
    activation = _evidence(output / "pytest", "activation-lifecycle-evidence.json")
    integrated_cases = [
        case for case in cases
        if case["name"].split("::")[0].endswith((
            "test_release_lifecycle_integration",
            "test_nightly_release_lifecycle_integration",
        ))
    ]
    test_clean = (
        result.returncode == 0
        and counts["skipped"] == 0
        and len(nightly) == 1
        and len(activation) == 1
        and bool(integrated_cases)
        and all(case["status"] == "passed" for case in integrated_cases)
    )

    browser_returncode = 1
    browser = {}
    browser_error = None
    if test_clean:
        try:
            browser_returncode, browser = _run_browser(
                args, output, nightly[0], source_manifest
            )
        except Exception as error:
            browser_error = repr(error)
            (output / "browser-error.txt").write_text(browser_error + "\n")
            print(browser_error, flush=True)

    clean = test_clean and browser_returncode == 0 and not browser.get("functionalFailures")
    layout_findings = browser.get("layoutFindings", [])
    status = (
        "passed_with_layout_findings" if clean and layout_findings
        else "passed" if clean
        else "failed"
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "complete issue-57 isolated acceptance; no live environment activation",
        "status": status,
        "source_runtime_sha256": source_manifest["runtime_sha256"],
        "source_release": str(source_release),
        "source_work_root": str(source_work),
        "tests": cases,
        "counts": counts,
        "recorded_source_evidence": recorded,
        "nightly_evidence": nightly,
        "activation_evidence": activation,
        "browser_evidence": browser,
        "browser_error": browser_error,
        "verified": PROVEN if clean else [],
        "known_layout_findings": layout_findings,
        "boundaries": {
            "production_changed": False,
            "dev_activation_performed": False,
            "nightly_sensor_enabled": False,
            "real_season_replay": False,
            "permissions_changed": False,
            "deferred_minutes_bug": 60,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    escape = lambda value: html.escape(str(value))
    rows = "".join(
        "<tr><td>" + escape(case["name"]) + "</td><td>" + escape(case["status"])
        + "</td><td>" + escape(case["seconds"]) + "</td><td><pre>"
        + escape(case["details"]) + "</pre></td></tr>"
        for case in cases
    )
    browser_summary = {
        "views": len(browser.get("views", [])),
        "functional_failures": browser.get("functionalFailures", []),
        "layout_findings": layout_findings,
        "transition": browser.get("transition"),
        "console_errors": browser.get("consoleErrors", []),
        "page_errors": browser.get("pageErrors", []),
        "espn_requests": browser.get("espnRequests", []),
    }
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>Issue 57 lifecycle verification</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;background:#faf9f5;color:#173c31}"
        "table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{text-align:left;vertical-align:top;"
        "border-bottom:1px solid #ddd;padding:.6rem;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "code{overflow-wrap:anywhere}th:first-child{width:50%}li{margin:.5rem 0}</style></head><body>"
        "<h1>Integrated lifecycle verification</h1>"
        "<p><strong>No production change, live dev activation, sensor enablement, or real season replay.</strong></p>"
        "<p>Status: " + escape(status) + ". " + str(counts["passed"]) + " passed; "
        + str(counts["failed"]) + " failed; " + str(counts["skipped"]) + " skipped.</p>"
        "<p>Source runtime: <code>" + escape(source_manifest["runtime_sha256"]) + "</code></p>"
        "<h2>Verified lifecycle</h2><ul>"
        + "".join("<li>" + escape(item) + "</li>" for item in report["verified"]) + "</ul>"
        "<h2>Rendered browser</h2><pre>" + escape(json.dumps(browser_summary, indent=2)) + "</pre>"
        "<h2>Known layout findings</h2><ul>"
        + ("".join("<li>" + escape(item) + "</li>" for item in layout_findings) or "<li>None</li>")
        + "</ul><h2>Nightly evidence</h2><pre>"
        + escape(json.dumps(nightly, indent=2)) + "</pre>"
        "<h2>Activation evidence</h2><pre>"
        + escape(json.dumps(activation, indent=2)) + "</pre>"
        "<h2>Every test result</h2><table><thead><tr><th>Test</th><th>Result</th>"
        "<th>Seconds</th><th>Failure / skip detail</th></tr></thead><tbody>"
        + rows + "</tbody></table></body></html>"
    )
    (output / "report.html").write_text(document)
    print(json.dumps({
        "status": status,
        "counts": counts,
        "report": str(output / "report.html"),
        "browser_views": len(browser.get("views", [])),
        "layout_findings": layout_findings,
    }))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
