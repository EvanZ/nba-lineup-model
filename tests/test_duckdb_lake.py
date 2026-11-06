from __future__ import annotations

import pytest

from nba_lineup_model.storage.duckdb_lake import (
    DEFAULT_LAKE_URI,
    s3_parquet_row_count,
    validate_lake_uri,
)


class FakeDuckDb:
    def __init__(self, count: int) -> None:
        self.count = count
        self.query: str | None = None
        self.parameters: list[str] | None = None

    def execute(self, query: str, parameters: list[str]):
        self.query = query
        self.parameters = parameters
        return self

    def fetchone(self) -> tuple[int]:
        return (self.count,)


def test_validate_lake_uri_rejects_objects_outside_configured_lake() -> None:
    assert validate_lake_uri(f"{DEFAULT_LAKE_URI}/curated/example.parquet").endswith(
        "example.parquet"
    )
    with pytest.raises(ValueError, match="beneath"):
        validate_lake_uri("s3://other-bucket/example.parquet")
    with pytest.raises(ValueError, match="beneath"):
        validate_lake_uri(DEFAULT_LAKE_URI)


def test_s3_parquet_count_uses_parameterized_duckdb_query() -> None:
    connection = FakeDuckDb(31)
    uri = f"{DEFAULT_LAKE_URI}/analytical/example.parquet"

    assert s3_parquet_row_count(uri, connection=connection) == 31
    assert connection.query == "SELECT count(*) FROM read_parquet(?)"
    assert connection.parameters == [uri]


def test_artifact_recovery_does_not_require_duckdb():
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = """
import builtins
original = builtins.__import__
def without_duckdb(name, *args, **kwargs):
    if name == 'duckdb' or name.startswith('duckdb.'):
        raise ModuleNotFoundError('DuckDB intentionally absent from web environment')
    return original(name, *args, **kwargs)
builtins.__import__ = without_duckdb
from nba_lineup_model.storage.artifact_store import ArtifactReference
from nba_lineup_model.season.serving_recovery import historical_path
from nba_lineup_model.storage.duckdb_lake import validate_lake_uri
assert validate_lake_uri('s3://nba-gestalt-data/lake/example')
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
