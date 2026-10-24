"""Bounded DuckDB access to immutable Parquet datasets in the S3 lake."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import duckdb

DEFAULT_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_S3_REGION = "us-west-1"
DEFAULT_MEMORY_LIMIT = "512MB"
DEFAULT_THREADS = 2
DEFAULT_HEALTHCHECK_PARQUET = (
    f"{DEFAULT_LAKE_URI}/analytical/box_score_prior_panel/season_cohort_summary.parquet"
)

_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*(?:KB|MB|GB)$", re.IGNORECASE)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def validate_lake_uri(uri: str, *, lake_uri: str = DEFAULT_LAKE_URI) -> str:
    """Require a concrete object path beneath the configured S3 lake."""

    root = lake_uri.rstrip("/")
    if not root.startswith("s3://") or uri == root or not uri.startswith(f"{root}/"):
        raise ValueError(f"S3 object must be beneath {root}/: {uri}")
    if any(part in {"", ".", ".."} for part in uri.removeprefix("s3://").split("/")):
        raise ValueError(f"S3 object path is invalid: {uri}")
    return uri


def connect_s3_lake(
    *,
    database: Path | str = ":memory:",
    lake_uri: str = DEFAULT_LAKE_URI,
    region: str = DEFAULT_S3_REGION,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
    install_extensions: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a resource-bounded connection authenticated by the AWS credential chain."""

    if not _REGION_PATTERN.fullmatch(region):
        raise ValueError(f"AWS region is invalid: {region}")
    if not _MEMORY_PATTERN.fullmatch(memory_limit):
        raise ValueError(f"DuckDB memory limit is invalid: {memory_limit}")
    if threads < 1 or threads > 8:
        raise ValueError("DuckDB threads must be between 1 and 8")
    root = lake_uri.rstrip("/")
    if not root.startswith("s3://"):
        raise ValueError(f"S3 lake URI is invalid: {lake_uri}")

    connection = duckdb.connect(str(database))
    try:
        connection.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
        connection.execute(f"SET threads = {threads}")
        if install_extensions:
            connection.execute("INSTALL aws")
            connection.execute("INSTALL httpfs")
        connection.execute("LOAD aws")
        connection.execute("LOAD httpfs")
        connection.execute(
            "CREATE OR REPLACE SECRET nba_gestalt_lake ("
            "TYPE s3, PROVIDER credential_chain, "
            f"REGION {_sql_string(region)}, SCOPE {_sql_string(root)}"
            ")"
        )
    except Exception:
        connection.close()
        raise
    return connection


def s3_parquet_row_count(
    uri: str,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
    install_extensions: bool = False,
) -> int:
    """Count one lake Parquet object without downloading the dataset to local disk."""

    object_uri = validate_lake_uri(uri)
    active = connection or connect_s3_lake(install_extensions=install_extensions)
    try:
        row = active.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [object_uri],
        ).fetchone()
        if row is None:
            raise RuntimeError(f"DuckDB returned no count for {object_uri}")
        return int(row[0])
    finally:
        if connection is None:
            active.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query an immutable Parquet object in S3")
    parser.add_argument("--uri", default=DEFAULT_HEALTHCHECK_PARQUET)
    parser.add_argument(
        "--install-extensions",
        action="store_true",
        help="Install DuckDB aws/httpfs extensions before loading them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    count = s3_parquet_row_count(args.uri, install_extensions=args.install_extensions)
    print(f"DuckDB S3 query succeeded: rows={count}; uri={args.uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
