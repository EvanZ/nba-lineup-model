"""Regression guards for the independent watchdog's installed service contract."""

from pathlib import Path
import sqlite3

import pytest

from nba_lineup_model.orchestration.data_quality_probe import dagster_automation_states

ROOT = Path(__file__).resolve().parents[1]


def test_watchdog_allows_wal_sidecars_but_keeps_databases_read_only():
    unit = (ROOT / "deploy/gestalt-data-quality-dev.service").read_text()
    for home in ("collection-dagster", "daily-dagster"):
        for folder, name in (("schedules", "schedules.db"), ("history", "runs.db")):
            directory = f"/var/lib/nba-gestalt/{home}/{folder}"
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
