from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.season.schema import (
    BuildLedger,
    CatalogGame,
    GameBuildRecord,
    GameCatalog,
)

CATALOG_COLUMNS = (
    "schema_version",
    "game_id",
    "season",
    "season_type",
    "game_date",
    "game_time_utc",
    "game_status",
    "home_team_id",
    "home_team_tricode",
    "away_team_id",
    "away_team_tricode",
    "period_count",
    "is_overtime",
    "source_status_code",
    "source_status_text",
    "source_url",
    "source_fetched_at",
)
BUILD_LEDGER_COLUMNS = (
    "schema_version",
    "run_id",
    "attempt_id",
    "attempt_number",
    "game_id",
    "season",
    "season_type",
    "started_at",
    "finished_at",
    "duration_seconds",
    "status",
    "terminal_stage",
    "use_cache",
    "code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
    "event_count",
    "lineup_stint_count",
    "possession_count",
    "possession_segment_count",
    "validation_issue_count",
    "output_table_count",
    "error_type",
    "error_message",
    "skip_reason",
)

_CATALOG_STRING_COLUMNS = (
    "game_id",
    "season",
    "season_type",
    "game_status",
    "home_team_tricode",
    "away_team_tricode",
    "source_status_text",
    "source_url",
)
_CATALOG_INTEGER_COLUMNS = (
    "schema_version",
    "home_team_id",
    "away_team_id",
    "period_count",
    "source_status_code",
)
_LEDGER_STRING_COLUMNS = (
    "run_id",
    "attempt_id",
    "game_id",
    "season",
    "season_type",
    "status",
    "terminal_stage",
    "code_version",
    "play_by_play_sha256",
    "boxscore_sha256",
    "error_type",
    "error_message",
    "skip_reason",
)
_LEDGER_INTEGER_COLUMNS = (
    "schema_version",
    "attempt_number",
    "event_count",
    "lineup_stint_count",
    "possession_count",
    "possession_segment_count",
    "validation_issue_count",
    "output_table_count",
)


def catalog_frame(catalog: GameCatalog | Sequence[CatalogGame]) -> pd.DataFrame:
    """Return a deterministically ordered, typed catalog frame."""

    games = catalog.games if isinstance(catalog, GameCatalog) else list(catalog)
    validated = GameCatalog(games=games)
    rows = [game.model_dump(mode="python") for game in validated.games]
    frame = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    frame = frame.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    for column in _CATALOG_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _CATALOG_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    frame["is_overtime"] = frame["is_overtime"].astype("boolean")
    frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True)
    frame["source_fetched_at"] = pd.to_datetime(frame["source_fetched_at"], utc=True)
    return frame


def catalog_from_frame(frame: pd.DataFrame) -> GameCatalog:
    """Validate a canonical catalog frame without lossy identifier coercion."""

    records = [_python_record(row) for row in frame.to_dict(orient="records")]
    return GameCatalog(games=[CatalogGame.model_validate(record) for record in records])


def write_game_catalog(catalog: GameCatalog | Sequence[CatalogGame], path: Path | str) -> Path:
    """Atomically write a canonical game catalog to Parquet."""

    output_path = Path(path)
    _atomic_write_parquet(catalog_frame(catalog), output_path)
    return output_path


def read_game_catalog(path: Path | str) -> GameCatalog:
    """Read and validate a canonical game catalog from Parquet."""

    return catalog_from_frame(pd.read_parquet(Path(path)))


def build_ledger_frame(
    ledger: BuildLedger | Sequence[GameBuildRecord],
) -> pd.DataFrame:
    """Return a typed build ledger frame ordered by attempt start time."""

    records = ledger.records if isinstance(ledger, BuildLedger) else list(ledger)
    validated = BuildLedger(records=records)
    rows = [record.model_dump(mode="python") for record in validated.records]
    frame = pd.DataFrame(rows, columns=BUILD_LEDGER_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["started_at", "run_id", "game_id", "attempt_number"],
            kind="stable",
        ).reset_index(drop=True)
    for column in _LEDGER_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _LEDGER_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    frame["duration_seconds"] = frame["duration_seconds"].astype("Float64")
    frame["use_cache"] = frame["use_cache"].astype("boolean")
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
    frame["finished_at"] = pd.to_datetime(frame["finished_at"], utc=True)
    return frame


def build_ledger_from_frame(frame: pd.DataFrame) -> BuildLedger:
    """Validate a build ledger frame."""

    records = [_python_record(row) for row in frame.to_dict(orient="records")]
    return BuildLedger(
        records=[GameBuildRecord.model_validate(record) for record in records]
    )


def write_build_ledger(
    ledger: BuildLedger | Sequence[GameBuildRecord],
    path: Path | str,
) -> Path:
    """Atomically write a build ledger to Parquet."""

    output_path = Path(path)
    _atomic_write_parquet(build_ledger_frame(ledger), output_path)
    return output_path


def read_build_ledger(path: Path | str) -> BuildLedger:
    """Read a build ledger, returning an empty ledger when it does not exist."""

    input_path = Path(path)
    if not input_path.exists():
        return BuildLedger()
    return build_ledger_from_frame(pd.read_parquet(input_path))


def append_build_record(
    record: GameBuildRecord,
    path: Path | str,
) -> BuildLedger:
    """Append one terminal record using an atomic single-writer rewrite."""

    ledger = read_build_ledger(path)
    updated = BuildLedger(records=[*ledger.records, record])
    write_build_ledger(updated, path)
    return updated


def _python_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _python_value(value) for key, value in record.items()}


def _python_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if pd.api.types.is_scalar(value) and bool(pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime | date | str | int | float | bool):
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
