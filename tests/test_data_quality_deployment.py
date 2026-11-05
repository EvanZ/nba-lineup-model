"""Regression guards for the independent watchdog's installed service contract."""

from pathlib import Path
import sqlite3

import pytest

from nba_lineup_model.orchestration.data_quality_probe import dagster_automation_states

ROOT = Path(__file__).resolve().parents[1]


def test_watchdog_allows_wal_sidecars_but_keeps_databases_read_only():
    unit = (ROOT / "deploy/gestalt-data-quality-dev.service").read_text()
    for root, home in (("nba-gestalt", "collection-dagster"), ("nba-gestalt-dev", "daily-dagster")):
        for folder, name in (("schedules", "schedules.db"), ("history", "runs.db")):
            directory = f"/var/lib/{root}/{home}/{folder}"
            assert f"ReadWritePaths={directory}\n" in unit
            assert any(
                f"{directory}/{name}" in line.split("=", 1)[1].split()
                for line in unit.splitlines()
                if line.startswith("ReadOnlyPaths=")
            )
    assert "@ORCHESTRATION_PYTHON@" in unit
    assert "/opt/nba-gestalt-orchestration/current" not in unit


def test_watchdog_reads_current_wal_commits_without_mutating_database(tmp_path):
    database = tmp_path / "schedules.db"
    writer = sqlite3.connect(database)
    try:
        writer.execute("pragma journal_mode=wal")
        writer.execute("pragma wal_autocheckpoint=0")
        writer.execute("create table instigators (status text, instigator_body text)")
        writer.execute(
            "insert into instigators values (?, ?)",
            ("RUNNING", '{"origin":{"job_name":"test_collector"}}'),
        )
        writer.commit()
        assert dagster_automation_states(database) == {"test_collector": "RUNNING"}
        reader = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.execute("delete from instigators")
        finally:
            reader.close()
    finally:
        writer.close()


def test_watchdog_release_is_sealed_and_independent_of_site(tmp_path):
    import subprocess
    import sys
    from nba_lineup_model.orchestration.watchdog_deployment import seal_watchdog
    from nba_lineup_model.season.sealed_code_package import verify_sealed_code_package

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "uv.lock").write_text("version = 1\n")
    (source / "watchdog.py").write_text("print('first')\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "watchdog",
        ],
        check=True,
    )
    package, checksum = seal_watchdog(source, tmp_path / "releases")
    assert verify_sealed_code_package(package)["identity"]["package_sha256"] == checksum
    subprocess.run([sys.executable, "-B", "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); import watchdog",
                    str(package / "code")], check=True, capture_output=True)
    assert not (package / "code/__pycache__").exists()
    assert verify_sealed_code_package(package)["identity"]["package_sha256"] == checksum
    assert '"$runtime/bin/python" -B -c' in (ROOT / "scripts/migrate_gestalt_daily_history.sh").read_text()
    assert "export PYTHONDONTWRITEBYTECODE=1" in (ROOT / "scripts/deploy_gestalt_daily_dev.sh").read_text()
    (source / "watchdog.py").write_text("print('changed')\n")
    assert (package / "code/watchdog.py").read_text() == "print('first')\n"
    assert verify_sealed_code_package(package)["identity"]["package_sha256"] == checksum
    path = package / "code/watchdog.py"
    path.chmod(0o644)
    path.write_text("print('tampered')\n")
    with pytest.raises(ValueError):
        verify_sealed_code_package(package)
    unit = (ROOT / "deploy/gestalt-data-quality-dev.service").read_text()
    assert "WorkingDirectory=@WATCHDOG_PACKAGE@/code" in unit
    assert "--watchdog-package-sha256 @WATCHDOG_SHA256@" in unit


def test_verified_sealed_package_can_be_copied_without_git(tmp_path):
    import subprocess
    from nba_lineup_model.season.sealed_code_package import (
        materialize_sealed_code_package,
        verify_sealed_code_package,
    )

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "uv.lock").write_text("version = 1\n")
    (source / "module.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "sealed",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    first = tmp_path / "first"
    first.mkdir()
    first_manifest = materialize_sealed_code_package(
        source, first, expected_revision=revision
    )
    second = tmp_path / "second"
    second.mkdir()
    second_manifest = materialize_sealed_code_package(
        first / "code", second, expected_revision=revision
    )
    assert second_manifest["identity"] == first_manifest["identity"]
    assert verify_sealed_code_package(second)["files"] == first_manifest["files"]


def test_wrong_watchdog_checksum_publishes_urgent_failure(tmp_path, monkeypatch):
    from nba_lineup_model.orchestration import data_quality_probe as probe
    from nba_lineup_model.season import sealed_code_package
    from nba_lineup_model.web_api.data_quality_reports import read_data_quality_status

    monkeypatch.setattr(
        sealed_code_package,
        "verify_sealed_code_package",
        lambda p: {"identity": {"package_sha256": "actual"}},
    )

    def must_not_run(**kwargs):
        pytest.fail("Invalid package must not run the environment probe")

    monkeypatch.setattr(probe, "collect_environment_report", must_not_run)
    result = probe.main(
        [
            "--environment",
            "dev",
            "--live-release",
            str(tmp_path),
            "--schedule-root",
            str(tmp_path),
            "--dagster-home",
            str(tmp_path),
            "--daily-dagster-home",
            str(tmp_path),
            "--api-root",
            "http://unused",
            "--output-root",
            str(tmp_path / "reports"),
            "--watchdog-package-root",
            str(tmp_path),
            "--watchdog-package-sha256",
            "wrong",
        ]
    )
    assert result == 1
    assert read_data_quality_status(tmp_path / "reports")["urgent"] is True
