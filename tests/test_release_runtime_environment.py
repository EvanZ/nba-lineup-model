from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_gestalt_runtime_environment.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("gestalt_runtime_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_project_dependencies_exclude_torch_and_lightning() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = "\n".join(project["project"]["dependencies"]).lower()
    assert "torch" not in dependencies
    assert "lightning" not in dependencies


def test_web_release_requirements_are_serving_only() -> None:
    requirements = (ROOT / "deploy" / "web-requirements.txt").read_text().lower()
    for forbidden in (
        "catboost",
        "dagster",
        "duckdb",
        "lightning",
        "matplotlib",
        "mlflow",
        "nvidia-",
        "prefect",
        "pypdf",
        "torch",
    ):
        assert forbidden not in requirements


def test_orchestration_requirements_extend_web_without_training_stack() -> None:
    requirements = (ROOT / "deploy" / "orchestration-requirements.txt").read_text().lower()
    assert "-r web-requirements.txt" in requirements
    for required in ("boto3", "dagster", "dagster-webserver", "duckdb", "pypdf"):
        assert required in requirements
    for forbidden in ("catboost", "lightning", "matplotlib", "mlflow", "prefect", "torch"):
        assert forbidden not in requirements


def test_all_dagster_units_use_shared_versioned_environment() -> None:
    units = [
        *ROOT.glob("deploy/gestalt-collection-*.service"),
        *ROOT.glob("deploy/gestalt-daily-*.service"),
        *ROOT.glob("deploy/gestalt-dagster-*.service"),
    ]
    assert units
    for unit in units:
        text = unit.read_text()
        assert "/opt/nba-gestalt/current/.venv/bin/" not in text
        assert "/opt/nba-gestalt-orchestration/current/bin/" in text


def test_web_deploy_does_not_install_orchestration_into_web_environment() -> None:
    script = (ROOT / "scripts" / "deploy_gestalt_beta.sh").read_text()
    assert "uv pip install --python .venv/bin/python -r deploy/web-requirements.txt" in script
    assert (
        "uv pip install --python .venv/bin/python -r deploy/orchestration-requirements.txt"
        not in script
    )
    assert "ensure_gestalt_orchestration_environment.sh" in script
    assert "--profile web" in script


def test_dev_web_uses_an_isolated_content_addressed_runtime() -> None:
    unit = (ROOT / "deploy" / "gestalt-web-dev.service").read_text()
    assert "/opt/nba-gestalt-web-dev/current/bin/uvicorn" in unit
    assert "/opt/nba-gestalt/current/.venv/bin/uvicorn" not in unit
    assert "WorkingDirectory=/opt/nba-gestalt/current" in unit
    assert (
        "PYTHONPATH=/var/lib/nba-gestalt-dev/live-release/code/src"
        in unit
    )
    assert (
        "GESTALT_ACTIVE_RELEASE_ROOT=/var/lib/nba-gestalt-dev/live-release"
        in unit
    )

    installer = (ROOT / "scripts" / "ensure_gestalt_web_environment.sh").read_text()
    assert "requirements_sha256=" in installer
    assert "--profile web" in installer
    assert "mv -Tf" in installer
    assert '$HOME/.local/bin/uv' in installer
    assert "venv --relocatable" in installer
    assert '"$target/bin/uvicorn" --version' in installer


def test_validator_rejects_training_and_cuda_packages() -> None:
    validator = _validator_module()
    packages = {name: "1" for name in validator.REQUIRED["web"]}
    packages["torch"] = "2.0"
    packages["nvidia-cublas-cu12"] = "12"
    errors = validator.validate_inventory("web", packages)
    assert errors == [
        "forbidden packages installed: nvidia-cublas-cu12, torch",
    ]


def test_validator_rejects_orchestration_packages_in_web_runtime() -> None:
    validator = _validator_module()
    packages = {name: "1" for name in validator.REQUIRED["web"]}
    packages["dagster"] = "1"
    packages["pypdf"] = "1"
    errors = validator.validate_inventory("web", packages)
    assert errors == ["forbidden packages installed: dagster, pypdf"]


def test_production_web_reports_the_active_release_root() -> None:
    unit = (ROOT / "deploy" / "gestalt-web.service").read_text()
    assert "GESTALT_ACTIVE_RELEASE_ROOT=/var/lib/nba-gestalt/live-release" in unit


def test_production_promotion_cannot_bypass_release_publisher() -> None:
    script = (ROOT / "scripts" / "promote_gestalt_dev_to_prod.sh").read_text()
    assert "GESTALT_PROD_APPROVAL_REFERENCE" in script
    assert "nba_lineup_model.season.release_publisher activate" in script
    assert "nba_lineup_model.season.release_publisher status" in script
    assert "production_site_release switch" not in script


def test_dev_activation_runs_as_the_authorized_root_publisher() -> None:
    script = (ROOT / "scripts" / "deploy_gestalt_dev.sh").read_text()
    assert "release-publisher" in script
    assert "sudo chown root:root /var/lib/nba-gestalt-dev" in script
    assert 'sudo env PYTHONPATH="$SOURCE_REPO/src"' in script
