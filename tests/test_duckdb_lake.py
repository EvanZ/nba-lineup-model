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
