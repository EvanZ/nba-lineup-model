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
    DailyPipelineConfig,
    _report,
    causal_schedule_sha256,
    schedule_lock,
)
from nba_lineup_model.season.dev_site_release import collect_api_validation
from nba_lineup_model.season.production_site_release import (
    verify_production_site_release,
)
from nba_lineup_model.season.release_publisher import (
    activate_release,
    read_active_activation,
    verify_loaded_identity,
)
from nba_lineup_model.season.source_policy import require_clean_artifact_sources


def _environment(config: DailyPipelineConfig) -> str:
    return "dev" if config.environment_root.name == "nba-gestalt-dev" else "prod"


def reload_environment_api(
    config: DailyPipelineConfig,
    _activation_record: dict,
) -> None:
    service = "gestalt-web-dev" if _environment(config) == "dev" else "gestalt-web"
    subprocess.run(
        ["sudo", "-n", "systemctl", "restart", service],
        check=True,
        timeout=60,
    )


def verify_environment_api(
    config: DailyPipelineConfig,
    release: Path,
    activation_record: dict,
) -> None:
    manifest = verify_production_site_release(release)
    expected = manifest["identity"]
    port = 8011 if _environment(config) == "dev" else 8010
    api = f"http://127.0.0.1:{port}"
    verify_loaded_identity(api, activation_record, timeout_seconds=60)
    deadline = time.monotonic() + 60
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(api + "/api/health", timeout=5) as response:
                health = json.load(response)
            actual = health.get("site_release") or {}
            if actual.get("release_sha256") != expected["approved_release_sha256"]:
                raise ValueError("Running API is not using the activated release")
            validation = collect_api_validation(
                api,
                evidence_date=expected["evidence_through_date"],
                daily_model_root=config.model_root,
            )
            # Same full endpoint contracts as candidate preflight, not HTTP 200 alone.
            from nba_lineup_model.season.dev_site_release import (
                _normalize_api_validation,
            )

            _normalize_api_validation(
                validation,
                evidence_date=expected["evidence_through_date"],
                expected_game_count=int(expected["game_count"]),
                expected_final_game_count=int(
                    manifest["approved_manifest"]["identity"]["schedule"][
                        "final_game_count"
                    ]
                ),
                serving_sha=expected["serving_bundle_sha256"],
                lineup_sha=expected["lineup_bundle_sha256"],
            )
            return
        except (OSError, ValueError) as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"Activated API verification failed: {last_error}")


def activate_daily_release(
    config: DailyPipelineConfig,
    report: dict,
    *,
    reload_api=None,
    verify_api=None,
) -> None:
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
    operation_id = str(report["activation_operation_id"])
    expected_generation = int(report["expected_previous_generation"])
    reload_api = reload_api or reload_environment_api
    verify_api = verify_api or verify_environment_api
    report_root = config.pipeline_root / "reports" / report["game_date"]

    # Validate the mutable observation overlay under its own short lock. The
    # restart and verification happen after release so collectors keep writing.
    with schedule_lock(config.schedule_root):
        require_clean_artifact_sources(config.schedule_root)
        if causal_schedule_sha256(config.schedule_root) != report.get(
            "source_causal_schedule_sha256"
        ):
            raise ValueError(
                "Causal schedule inputs changed while the candidate was built; "
                "rebuild before activation"
            )
        require_clean_artifact_sources(release / "schedule")

    previous = read_active_activation(config.live_pointer)
    try:
        result = activate_release(
            release=release,
            live_pointer=config.live_pointer,
            publisher_root=config.environment_root / "release-publisher",
            environment=_environment(config),
            operation_id=operation_id,
            expected_previous_generation=expected_generation,
            reason=f"daily release through {report['game_date']}",
            approval_reference=report.get("approval_reference"),
            reload_api=lambda record: reload_api(config, record),
            verify_api=lambda record: verify_api(
                config,
                Path(str(record["release_path"])).resolve(),
                record,
            ),
        )
    except Exception as error:
        status = (
            "rollback_failed"
            if isinstance(error, ExceptionGroup)
            else "activation_failed"
        )
        active = read_active_activation(config.live_pointer)
        report.update(
            status=status,
            stage="rollback",
            error=f"{type(error).__name__}: {error}",
            active_generation=int(active["generation"]) if active else 0,
        )
        _report(report_root, report)
        raise

    report.update(
        status="activated",
        stage="complete",
        activation_generation=result.generation,
        activation_release_id=result.release_id,
        activation_result=result.status,
        previous_release=previous.get("release_path") if previous else None,
    )
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
