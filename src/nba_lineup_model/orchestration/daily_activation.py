"""Activate one validated daily candidate; roll back the API as well as files.

No Dagster service is started here. Automatic activation is opt-in in the
environment configuration, and deployment approval is still required for prod.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from nba_lineup_model.orchestration.daily_pipeline import (
    DailyPipelineConfig, _report, schedule_lock,
)
from nba_lineup_model.season.dev_site_release import _tree_sha256, collect_api_validation
from nba_lineup_model.season.production_site_release import (
    switch_production_release, verify_production_site_release,
)
from nba_lineup_model.season.source_policy import require_clean_artifact_sources


def reload_environment_api(config: DailyPipelineConfig) -> None:
    service = "gestalt-web-dev" if config.environment_root.name == "nba-gestalt-dev" else "gestalt-web"
    subprocess.run(["sudo", "-n", "systemctl", "restart", service], check=True, timeout=60)


def verify_environment_api(config: DailyPipelineConfig, release: Path) -> None:
    manifest = verify_production_site_release(release)
    expected = manifest["identity"]
    port = 8011 if config.environment_root.name == "nba-gestalt-dev" else 8010
    api = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(api + "/api/health", timeout=5) as response:
                health = json.load(response)
            actual = health.get("site_release") or {}
            if actual.get("release_sha256") != expected["approved_release_sha256"]:
                raise ValueError("Running API is not using the activated release")
            validation = collect_api_validation(api, evidence_date=expected["evidence_through_date"])
            # Same full endpoint contracts as candidate preflight, not HTTP 200 alone.
            from nba_lineup_model.season.dev_site_release import _normalize_api_validation
            _normalize_api_validation(
                validation,
                evidence_date=expected["evidence_through_date"],
                expected_game_count=int(expected["game_count"]),
                expected_final_game_count=int(manifest["approved_manifest"]["identity"]["schedule"]["final_game_count"]),
                serving_sha=expected["serving_bundle_sha256"],
                lineup_sha=expected["lineup_bundle_sha256"],
            )
            return
        except (OSError, ValueError) as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"Activated API verification failed: {last_error}")


def activate_daily_release(config: DailyPipelineConfig, report: dict, *, reload_api=None, verify_api=None) -> None:
    config.validate()
    if report.get("status") != "validated_candidate":
        raise ValueError("Only a fully validated candidate can be activated")
    release = Path(report["runtime_release"]).resolve()
    if not release.is_relative_to(config.pipeline_root.resolve() / "releases"):
        raise ValueError("Candidate is outside this environment's pipeline")
    manifest = verify_production_site_release(release)
    if Path(manifest["identity"]["live_pointer"]) != config.live_pointer.absolute():
        raise ValueError("Candidate was staged for a different live pointer")
    if manifest["identity"]["evidence_through_date"] != report["game_date"]:
        raise ValueError("Candidate evidence date differs from its pipeline report")
    reload_api = reload_api or reload_environment_api
    verify_api = verify_api or verify_environment_api
    report_root = config.pipeline_root / "reports" / report["game_date"]
    with schedule_lock(config.schedule_root):
        require_clean_artifact_sources(config.schedule_root)
        if _tree_sha256(config.schedule_root) != report.get("source_schedule_sha256"):
            raise ValueError("Schedule changed while the candidate was built; rebuild before activation")
        require_clean_artifact_sources(release / "schedule")
        previous = switch_production_release(release, live_pointer=config.live_pointer)
        try:
            reload_api(config)
            verify_api(config, release)
        except Exception as error:
            if previous is None:
                config.live_pointer.unlink(missing_ok=True)
            else:
                switch_production_release(previous, live_pointer=config.live_pointer)
            report.update(status="activation_failed", stage="rollback", error=f"{type(error).__name__}: {error}")
            _report(report_root, report)
            # Restoring only the symlink leaves a process serving cached new data.
            # First-time host bootstrap must restore its old service configuration.
            if previous is not None:
                try:
                    reload_api(config)
                    verify_api(config, previous)
                except Exception as rollback_error:
                    report.update(status="rollback_failed", rollback_error=str(rollback_error))
                    _report(report_root, report)
                    raise ExceptionGroup("Activation and API rollback failed", [error, rollback_error])
            raise
        report.update(status="activated", stage="complete", previous_release=str(previous) if previous else None)
        _report(report_root, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = DailyPipelineConfig.read(args.config)
    report = json.loads(args.report.read_text())
    activate_daily_release(config, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
