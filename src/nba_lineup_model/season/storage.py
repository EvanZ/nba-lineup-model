from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.audit.runner import audit_summary_frame
from nba_lineup_model.season.schema import (
    BuildLedger,
    CatalogGame,
    FetchManifest,
    GameBuildRecord,
    GameCatalog,
    GameFetchRecord,
    GameQualityRecord,
    QualityReport,
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
    "prefect_flow_run_id",
    "prefect_task_run_id",
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
    "play_by_play_source",
    "boxscore_source",
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
FETCH_MANIFEST_COLUMNS = (
    "schema_version",
    "run_id",
    "prefect_flow_run_id",
    "prefect_task_run_id",
    "attempt_number",
    "game_id",
    "season",
    "season_type",
    "started_at",
    "finished_at",
    "duration_seconds",
    "status",
    "refresh",
    "play_by_play_cache_hit",
    "boxscore_cache_hit",
    "play_by_play_sha256",
    "boxscore_sha256",
    "play_by_play_bytes",
    "boxscore_bytes",
    "error_type",
    "error_message",
    "skip_reason",
)
QUALITY_RECORD_COLUMNS = tuple(GameQualityRecord.model_fields)

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
    "prefect_flow_run_id",
    "prefect_task_run_id",
    "game_id",
    "season",
    "season_type",
    "status",
    "terminal_stage",
    "code_version",
    "play_by_play_source",
    "boxscore_source",
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
_FETCH_STRING_COLUMNS = (
    "run_id",
    "prefect_flow_run_id",
    "prefect_task_run_id",
    "game_id",
    "season",
    "season_type",
    "status",
    "play_by_play_sha256",
    "boxscore_sha256",
    "error_type",
    "error_message",
    "skip_reason",
)
_FETCH_INTEGER_COLUMNS = (
    "schema_version",
    "attempt_number",
    "play_by_play_bytes",
    "boxscore_bytes",
)
_QUALITY_STRING_COLUMNS = (
    "run_id",
    "prefect_flow_run_id",
    "prefect_task_run_id",
    "code_version",
    "play_by_play_source",
    "boxscore_source",
    "play_by_play_sha256",
    "boxscore_sha256",
    "game_id",
    "season",
    "season_type",
    "sample_group",
    "status",
    "game_time_utc",
    "home_tricode",
    "away_tricode",
    "error_stage",
    "error_type",
    "error_message",
)
_QUALITY_INTEGER_COLUMNS = (
    "schema_version",
    "attempt_number",
    "home_team_id",
    "away_team_id",
    "score_home",
    "score_away",
    "period_count",
    "event_count",
    "lineup_stint_count",
    "possession_count",
    "home_possession_count",
    "away_possession_count",
    "possession_count_difference",
    "possession_segment_count",
    "multi_segment_possession_count",
    "source_possession_override_count",
    "source_change_terminal_count",
    "opponent_technical_free_throw_possession_count",
    "lineup_warning_count",
    "lineup_error_count",
    "possession_warning_count",
    "possession_error_count",
    "segment_warning_count",
    "segment_error_count",
    "event_warning_count",
)
_QUALITY_FLOAT_COLUMNS = (
    "estimated_home_possessions",
    "estimated_away_possessions",
    "home_possession_estimate_difference",
    "away_possession_estimate_difference",
)
_QUALITY_BOOLEAN_COLUMNS = (
    "is_overtime",
    "score_matches_boxscore",
    "possession_score_conserved",
    "segment_score_conserved",
    "segment_duration_conserved",
    "balanced_possession_counts",
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


def fetch_manifest_frame(
    manifest: FetchManifest | Sequence[GameFetchRecord],
) -> pd.DataFrame:
    """Return a typed fetch manifest ordered by run and game."""

    records = manifest.records if isinstance(manifest, FetchManifest) else list(manifest)
    validated = FetchManifest(records=records)
    rows = [record.model_dump(mode="python") for record in validated.records]
    frame = pd.DataFrame(rows, columns=FETCH_MANIFEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["started_at", "run_id", "game_id"],
            kind="stable",
        ).reset_index(drop=True)
    for column in _FETCH_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _FETCH_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    for column in ("refresh", "play_by_play_cache_hit", "boxscore_cache_hit"):
        frame[column] = frame[column].astype("boolean")
    frame["duration_seconds"] = frame["duration_seconds"].astype("Float64")
    frame["started_at"] = pd.to_datetime(frame["started_at"], utc=True)
    frame["finished_at"] = pd.to_datetime(frame["finished_at"], utc=True)
    return frame


def fetch_manifest_from_frame(frame: pd.DataFrame) -> FetchManifest:
    """Validate a raw fetch manifest frame."""

    records = [_python_record(row) for row in frame.to_dict(orient="records")]
    return FetchManifest(
        records=[GameFetchRecord.model_validate(record) for record in records]
    )


def write_fetch_manifest(
    manifest: FetchManifest | Sequence[GameFetchRecord],
    path: Path | str,
) -> Path:
    """Atomically write the append-oriented raw fetch manifest."""

    output_path = Path(path)
    _atomic_write_parquet(fetch_manifest_frame(manifest), output_path)
    return output_path


def read_fetch_manifest(path: Path | str) -> FetchManifest:
    """Read a raw fetch manifest, returning an empty manifest when absent."""

    input_path = Path(path)
    if not input_path.exists():
        return FetchManifest()
    return fetch_manifest_from_frame(pd.read_parquet(input_path))


def append_fetch_records(
    records: Sequence[GameFetchRecord],
    path: Path | str,
) -> FetchManifest:
    """Append a batch of terminal fetch records through one atomic writer."""

    manifest = read_fetch_manifest(path)
    updated = FetchManifest(records=[*manifest.records, *records])
    write_fetch_manifest(updated, path)
    return updated


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

    return append_build_records([record], path)


def append_build_records(
    records: Sequence[GameBuildRecord],
    path: Path | str,
) -> BuildLedger:
    """Append terminal build records using one atomic single-writer rewrite."""

    ledger = read_build_ledger(path)
    updated = BuildLedger(records=[*ledger.records, *records])
    write_build_ledger(updated, path)
    return updated


def quality_report_frame(
    report: QualityReport | Sequence[GameQualityRecord],
) -> pd.DataFrame:
    """Return a typed canonical game-quality frame."""

    records = report.records if isinstance(report, QualityReport) else list(report)
    validated = QualityReport(records=records)
    rows = [record.model_dump(mode="python") for record in validated.records]
    frame = pd.DataFrame(rows, columns=QUALITY_RECORD_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["season", "season_type", "game_id"],
            kind="stable",
        ).reset_index(drop=True)
    for column in _QUALITY_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _QUALITY_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("Int64")
    for column in _QUALITY_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("Float64")
    for column in _QUALITY_BOOLEAN_COLUMNS:
        frame[column] = frame[column].astype("boolean")
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at"], utc=True)
    frame["issue_codes"] = frame["issue_codes"].map(list)
    return frame


def quality_report_from_frame(frame: pd.DataFrame) -> QualityReport:
    """Validate a canonical game-quality frame."""

    records = []
    for row in frame.to_dict(orient="records"):
        record = _python_record(row)
        issue_codes = record.get("issue_codes")
        if issue_codes is None:
            record["issue_codes"] = ()
        else:
            tolist = getattr(issue_codes, "tolist", None)
            if callable(tolist):
                issue_codes = tolist()
            record["issue_codes"] = tuple(issue_codes)
        records.append(GameQualityRecord.model_validate(record))
    return QualityReport(records=records)


def write_quality_report(
    report: QualityReport | Sequence[GameQualityRecord],
    games_path: Path | str,
    *,
    summary_path: Path | str | None = None,
) -> QualityReport:
    """Atomically write canonical game quality and its aggregate summary."""

    validated = report if isinstance(report, QualityReport) else QualityReport(records=report)
    _atomic_write_parquet(quality_report_frame(validated), Path(games_path))
    if summary_path is not None:
        _atomic_write_parquet(
            audit_summary_frame(validated.records),
            Path(summary_path),
        )
    return validated


def read_quality_report(path: Path | str) -> QualityReport:
    """Read canonical game quality, returning an empty report when absent."""

    input_path = Path(path)
    if not input_path.exists():
        return QualityReport()
    return quality_report_from_frame(pd.read_parquet(input_path))


def merge_quality_records(
    records: Sequence[GameQualityRecord],
    games_path: Path | str,
    *,
    summary_path: Path | str | None = None,
) -> QualityReport:
    """Replace canonical quality rows by game ID and write the merged report."""

    incoming = QualityReport(records=list(records))
    by_game_id = {
        record.game_id: record for record in read_quality_report(games_path).records
    }
    by_game_id.update({record.game_id: record for record in incoming.records})
    merged = QualityReport(
        records=sorted(
            by_game_id.values(),
            key=lambda record: (record.season, record.season_type, record.game_id),
        )
    )
    return write_quality_report(
        merged,
        games_path,
        summary_path=summary_path,
    )


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
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
