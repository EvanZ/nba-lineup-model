"""Deploy API code to dev without running a data or forecasting job.

Retain all approved data and selections, exercise the candidate API, then use
the generation-fenced publisher. No direct pointer changes or daily receipts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from nba_lineup_model.orchestration.daily_activation import (
    reload_environment_api, verify_environment_api,
)
from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
from nba_lineup_model.season.dev_site_release import collect_api_validation, _normalize_api_validation
from nba_lineup_model.season.production_site_release import stage_code_release, verify_production_site_release
from nba_lineup_model.season.release_publisher import activate_release, read_active_activation


def fetch(api, path):
    with urllib.request.urlopen(api + path, timeout=90) as response:
        return json.load(response)


@contextmanager
def candidate_api(release, report_root):
    bindings = json.loads((release / "serving-bindings.json").read_text())
    # Isolated preflight link, never an environment's live pointer. There is no
    # activation record: only the real publisher can create one after validation.
    pointer = report_root / "preflight-release"
    if not pointer.is_symlink():
        pointer.symlink_to(release, target_is_directory=True)
    if pointer.resolve() != release.resolve():
        raise ValueError("Preflight link points at another release")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(release / "code/src"),
           "GESTALT_ACTIVE_RELEASE_ROOT": str(pointer),
           "GESTALT_LIVE_SCORE_STATE_DIR": "/var/lib/nba-gestalt/schedule-serving/current",
           "GESTALT_DATA_QUALITY_ROOT": "/var/lib/nba-gestalt-dev/data-quality"}
    for variable, relative in {
        "CURRENT_SEASON_SERVING_ROOT": "current-season-serving",
        "CURRENT_SEASON_LINEUPS_ROOT": "current-season-lineups",
        "SCHEDULE_STATE_DIR": "schedule", "GAME_CATALOG_PATH": "catalog/games.parquet",
        "SITE_RELEASE_ROOT": "site-releases", "PRESEASON_RANKINGS_PATH": "preseason/rankings.parquet",
        "PRESEASON_PROFILES_PATH": "preseason/profiles.parquet",
    }.items():
        env["GESTALT_" + variable] = str(release / relative)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    api = f"http://127.0.0.1:{port}"
    with (report_root / "api.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn",
            "nba_lineup_model.web_api.app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=bindings["historical_runtime"], env=env, stdout=log, stderr=subprocess.STDOUT,
            user="ubuntu" if os.geteuid() == 0 else None,
            group="ubuntu" if os.geteuid() == 0 else None)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Candidate API exited; see {report_root / 'api.log'}")
                try:
                    fetch(api, "/api/health")
                    break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("Candidate API did not start")
            yield api
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = DailyPipelineConfig.read(args.config)
    config.validate()
    if config.environment_root != Path("/var/lib/nba-gestalt-dev"):
        raise ValueError("This command is dev-only; production requires separate approval")
    if config.live_pointer != config.environment_root / "live-release":
        raise ValueError("Unexpected dev live pointer")
    active = read_active_activation(config.live_pointer)
    if active is None:
        raise ValueError("Code deployment requires an existing validated release")
    base = Path(active["release_path"])
    candidate = stage_code_release(base_release=base, code_root=args.code,
                                  output_root=config.environment_root / "code-releases")
    manifest = verify_production_site_release(candidate)
    identity = manifest["identity"]
    report_root = config.environment_root / "code-release-validation" / manifest["runtime_sha256"]
    report_root.mkdir(parents=True, exist_ok=True)
    report = {"status": "validating", "base_generation": active["generation"],
              "base_runtime": active["runtime_sha256"], "runtime": manifest["runtime_sha256"],
              "code_revision": identity["code_revision"], "data_changed": False}
    try:
        with candidate_api(candidate, report_root) as api:
            print("Candidate API started; validating unchanged data and all schedule dates", flush=True)
            evidence = collect_api_validation(api, evidence_date=identity["evidence_through_date"],
                                              daily_model_root=config.model_root)
            _normalize_api_validation(evidence, evidence_date=identity["evidence_through_date"],
                expected_game_count=int(identity["game_count"]),
                expected_final_game_count=int(manifest["approved_manifest"]["identity"]["schedule"]["final_game_count"]),
                serving_sha=identity["serving_bundle_sha256"], lineup_sha=identity["lineup_bundle_sha256"])
            report["api_validation"] = evidence
            days = sorted((candidate / "selected-schedule/latest").glob("*.json"))
            if not days:
                raise ValueError("Selected schedule contains no dates")
            coverage = {}
            for day in days:
                expected = {str(g["game_id"]) for g in json.loads(day.read_text())["games"]}
                result = fetch(api, f"/api/schedule?date={day.stem}")
                actual = {str(g["game_id"]) for g in result["games"]}
                if expected != actual:
                    raise ValueError(f"Schedule coverage differs on {day.stem}")
                coverage[day.stem] = len(actual)
            report["schedule_coverage"] = coverage
        report["status"] = "validated"
        if args.activate:
            result = activate_release(release=candidate, live_pointer=config.live_pointer,
                publisher_root=config.environment_root / "release-publisher", environment="dev",
                operation_id=f"code-{manifest['runtime_sha256']}-from-{active['generation']}",
                expected_previous_generation=active["generation"], reason="code-only dev deployment; data unchanged",
                reload_api=lambda record: reload_environment_api(config, record),
                verify_api=lambda record: verify_environment_api(config, Path(record["release_path"]), record))
            report.update(status="activated", generation=result.generation)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (report_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "api_validation"}), flush=True)


if __name__ == "__main__":
    main()
