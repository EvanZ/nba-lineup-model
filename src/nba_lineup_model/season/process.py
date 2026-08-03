from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.audit import (
    AuditGameResult,
    AuditGameSpec,
    audit_reconstruction,
)
from nba_lineup_model.build_game import (
    GameReconstruction,
    persist_game_reconstruction,
    reconstruct_game_payloads,
)
from nba_lineup_model.season.layout import CURATED_TABLES
from nba_lineup_model.season.schema import (
    BuildLedger,
    BuildStage,
    CatalogGame,
    GameBuildRecord,
    GameQualityRecord,
)
from nba_lineup_model.season.source import (
    SelectedRawArtifact,
    load_game_source_documents,
)

_PROCESSING_SOURCE_ENTRIES = (
    "audit/runner.py",
    "audit/schema.py",
    "build_game.py",
    "events",
    "ingest/nba_cdn.py",
    "ingest/nba_stats.py",
    "lineups",
    "normalize",
    "possessions",
    "season/fetch.py",
    "season/layout.py",
    "season/process.py",
    "season/schema.py",
    "season/source.py",
)


@dataclass(frozen=True)
class GameProcessOutcome:
    """Terminal build and quality evidence returned by one game processor."""

    record: GameBuildRecord
    quality: AuditGameResult | None = None
    retryable: bool = False


def process_catalog_game(
    game: CatalogGame,
    *,
    run_id: str,
    code_version: str,
    raw_dir: Path | str = Path("data/raw"),
    processed_dir: Path | str = Path("data/processed"),
    prior_success: GameBuildRecord | None = None,
    prior_quality: GameQualityRecord | None = None,
    force: bool = False,
    attempt_number: int = 1,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
    started_at: datetime | None = None,
) -> GameProcessOutcome:
    """Build one catalog game from validated local raw documents only."""

    started_at = started_at or datetime.now(UTC)
    attempt_id = f"{run_id}:{game.game_id}:{attempt_number}:{uuid4().hex[:8]}"
    play_by_play = None
    boxscore = None
    game_rotation = None
    try:
        documents = load_game_source_documents(game.game_id, raw_dir=raw_dir)
        play_by_play = documents.play_by_play
        boxscore = documents.boxscore
        game_rotation = documents.game_rotation
    except Exception as error:
        return _failure_outcome(
            game,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            code_version=code_version,
            started_at=started_at,
            stage="preflight",
            error=error,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            quality=_error_quality(game, "preflight", error),
        )

    if (
        not force
        and prior_success is not None
        and prior_quality is not None
        and _can_resume(
            game,
            prior_success=prior_success,
            prior_quality=prior_quality,
            code_version=code_version,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            processed_dir=processed_dir,
        )
    ):
        finished_at = datetime.now(UTC)
        return GameProcessOutcome(
            record=GameBuildRecord(
                run_id=run_id,
                attempt_id=attempt_id,
                prefect_flow_run_id=prefect_flow_run_id,
                prefect_task_run_id=prefect_task_run_id,
                attempt_number=attempt_number,
                game_id=game.game_id,
                season=game.season,
                season_type=game.season_type,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=(finished_at - started_at).total_seconds(),
                status="skipped",
                terminal_stage="preflight",
                use_cache=True,
                code_version=code_version,
                play_by_play_source=play_by_play.source,
                boxscore_source=boxscore.source,
                play_by_play_sha256=play_by_play.sha256,
                boxscore_sha256=boxscore.sha256,
                game_rotation_sha256=(
                    game_rotation.sha256 if game_rotation is not None else None
                ),
                event_count=prior_success.event_count,
                lineup_stint_count=prior_success.lineup_stint_count,
                possession_count=prior_success.possession_count,
                possession_segment_count=prior_success.possession_segment_count,
                validation_issue_count=prior_success.validation_issue_count,
                output_table_count=len(CURATED_TABLES),
                skip_reason="matching_source_code_quality_and_outputs",
            )
        )

    try:
        reconstruction = reconstruct_game_payloads(
            play_by_play.payload,
            boxscore.payload,
        )
    except Exception as error:
        return _failure_outcome(
            game,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            code_version=code_version,
            started_at=started_at,
            stage="reconstruct",
            error=error,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            quality=_error_quality(game, "reconstruct", error),
        )

    try:
        quality = audit_reconstruction(
            _audit_spec(game),
            reconstruction,
            boxscore.payload,
        )
        quality = _apply_catalog_invariants(game, quality)
    except Exception as error:
        return _failure_outcome(
            game,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            code_version=code_version,
            started_at=started_at,
            stage="validate",
            error=error,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            reconstruction=reconstruction,
            quality=_error_quality(game, "validate", error),
        )

    if quality.status in {"fail", "error"}:
        error = ValueError(
            "Quality gate failed: "
            + (", ".join(quality.issue_codes) or quality.status)
        )
        return _failure_outcome(
            game,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            code_version=code_version,
            started_at=started_at,
            stage="validate",
            error=error,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            reconstruction=reconstruction,
            quality=quality,
        )

    try:
        processed = persist_game_reconstruction(
            reconstruction,
            boxscore.payload,
            output_root=processed_dir,
        )
    except Exception as error:
        return _failure_outcome(
            game,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            code_version=code_version,
            started_at=started_at,
            stage="persist",
            error=error,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            reconstruction=reconstruction,
            quality=_error_quality(game, "persist", error),
            output_table_count=_valid_output_count(game.game_id, processed_dir),
        )

    try:
        require_processed_outputs(game.game_id, processed_dir)
    except Exception as error:
        return _failure_outcome(
            game,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            code_version=code_version,
            started_at=started_at,
            stage="validate",
            error=error,
            play_by_play=play_by_play,
            boxscore=boxscore,
            game_rotation=game_rotation,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            reconstruction=reconstruction,
            quality=_error_quality(game, "validate", error),
            output_table_count=_valid_output_count(game.game_id, processed_dir),
        )

    finished_at = datetime.now(UTC)
    return GameProcessOutcome(
        record=GameBuildRecord(
            run_id=run_id,
            attempt_id=attempt_id,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            attempt_number=attempt_number,
            game_id=game.game_id,
            season=game.season,
            season_type=game.season_type,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            status="succeeded",
            terminal_stage="complete",
            use_cache=True,
            code_version=code_version,
            play_by_play_source=play_by_play.source,
            boxscore_source=boxscore.source,
            play_by_play_sha256=play_by_play.sha256,
            boxscore_sha256=boxscore.sha256,
            game_rotation_sha256=(
                game_rotation.sha256 if game_rotation is not None else None
            ),
            event_count=processed.event_count,
            lineup_stint_count=processed.stint_count,
            possession_count=processed.possession_count,
            possession_segment_count=processed.possession_segment_count,
            validation_issue_count=len(quality.issue_codes),
            output_table_count=len(processed.output_paths),
        ),
        quality=quality,
    )


def quality_record_for_outcome(
    outcome: GameProcessOutcome,
) -> GameQualityRecord | None:
    """Attach build provenance to a game's audit result."""

    record = outcome.record
    if (
        outcome.quality is None
        or record.code_version is None
    ):
        return None
    return GameQualityRecord(
        **outcome.quality.model_dump(mode="python"),
        run_id=record.run_id,
        prefect_flow_run_id=record.prefect_flow_run_id,
        prefect_task_run_id=record.prefect_task_run_id,
        attempt_number=record.attempt_number,
        code_version=record.code_version,
        play_by_play_source=record.play_by_play_source,
        boxscore_source=record.boxscore_source,
        play_by_play_sha256=record.play_by_play_sha256,
        boxscore_sha256=record.boxscore_sha256,
        game_rotation_sha256=record.game_rotation_sha256,
        recorded_at=record.finished_at,
    )


def failed_process_outcome(
    game: CatalogGame,
    *,
    run_id: str,
    code_version: str,
    started_at: datetime,
    error: Exception,
    attempt_number: int = 1,
    prefect_flow_run_id: str | None = None,
    prefect_task_run_id: str | None = None,
) -> GameProcessOutcome:
    """Build a terminal record for an unexpected orchestration failure."""

    return _failure_outcome(
        game,
        run_id=run_id,
        attempt_id=f"{run_id}:{game.game_id}:{attempt_number}:{uuid4().hex[:8]}",
        attempt_number=attempt_number,
        code_version=code_version,
        started_at=started_at,
        stage="reconstruct",
        error=error,
        prefect_flow_run_id=prefect_flow_run_id,
        prefect_task_run_id=prefect_task_run_id,
    )


def processing_code_fingerprint(
    package_root: Path | str | None = None,
    *,
    source_entries: Sequence[Path | str] | None = None,
) -> str:
    """Hash processing-owned Python sources so algorithm changes invalidate resume."""

    root = (
        Path(package_root)
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    entries = (
        tuple(Path(entry) for entry in source_entries)
        if source_entries is not None
        else (
            tuple(Path(entry) for entry in _PROCESSING_SOURCE_ENTRIES)
            if package_root is None
            else ()
        )
    )
    if entries:
        paths = sorted(
            {
                path
                for entry in entries
                for path in _python_paths_for_entry(root, entry)
            }
        )
    else:
        paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise ValueError(f"No Python source files found under {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _python_paths_for_entry(root: Path, entry: Path) -> list[Path]:
    path = root / entry
    if path.is_file():
        return [path] if path.suffix == ".py" else []
    if path.is_dir():
        return [candidate for candidate in path.rglob("*.py") if candidate.is_file()]
    raise ValueError(f"Processing source entry does not exist: {path}")


def latest_successful_builds(ledger: BuildLedger) -> dict[str, GameBuildRecord]:
    """Return the latest successful build for each game."""

    latest: dict[str, GameBuildRecord] = {}
    for record in sorted(ledger.records, key=lambda item: item.finished_at):
        if record.status == "succeeded":
            latest[record.game_id] = record
    return latest


def sample_processing_games(
    games: list[CatalogGame],
    *,
    games_per_stratum: int,
    random_seed: int = 0,
) -> list[CatalogGame]:
    """Sample deterministically by season type and overtime status."""

    if games_per_stratum < 1:
        raise ValueError("games_per_stratum must be positive")
    grouped: dict[tuple[str, str], list[CatalogGame]] = {}
    for game in games:
        overtime_group = (
            "overtime"
            if game.is_overtime is True
            else "regulation"
            if game.is_overtime is False
            else "unknown"
        )
        grouped.setdefault((game.season_type, overtime_group), []).append(game)

    sampled: list[CatalogGame] = []
    for group_number, key in enumerate(sorted(grouped)):
        candidates = sorted(
            grouped[key],
            key=lambda game: (game.game_date, game.game_id),
        )
        sample_size = min(games_per_stratum, len(candidates))
        sampled.extend(
            random.Random(random_seed + group_number).sample(
                candidates,
                sample_size,
            )
        )
    return sorted(sampled, key=lambda game: (game.game_date, game.game_id))


def processed_output_paths(
    game_id: str,
    processed_dir: Path | str = Path("data/processed"),
) -> dict[str, Path]:
    """Return all conventional per-game processed output paths."""

    root = Path(processed_dir)
    return {
        table: root / table / f"{game_id}.parquet" for table in CURATED_TABLES
    }


def require_processed_outputs(
    game_id: str,
    processed_dir: Path | str = Path("data/processed"),
) -> dict[str, Path]:
    """Require six readable, non-empty Parquet tables for the expected game."""

    paths = processed_output_paths(game_id, processed_dir)
    for table, path in paths.items():
        if not path.exists():
            raise ValueError(f"Missing processed {table} output: {path}")
        identifiers = pd.read_parquet(path, columns=["game_id"])
        if identifiers.empty:
            raise ValueError(f"Processed {table} output is empty: {path}")
        if set(identifiers["game_id"].astype(str)) != {game_id}:
            raise ValueError(f"Processed {table} output has the wrong game_id: {path}")
    return paths


def processed_outputs_valid(
    game_id: str,
    processed_dir: Path | str = Path("data/processed"),
) -> bool:
    """Return whether all expected processed outputs pass preflight checks."""

    try:
        require_processed_outputs(game_id, processed_dir)
    except Exception:
        return False
    return True


def _can_resume(
    game: CatalogGame,
    *,
    prior_success: GameBuildRecord,
    prior_quality: GameQualityRecord,
    code_version: str,
    play_by_play: SelectedRawArtifact,
    boxscore: SelectedRawArtifact,
    game_rotation: SelectedRawArtifact | None,
    processed_dir: Path | str,
) -> bool:
    return (
        prior_success.code_version == code_version
        and prior_success.play_by_play_source == play_by_play.source
        and prior_success.boxscore_source == boxscore.source
        and prior_success.play_by_play_sha256 == play_by_play.sha256
        and prior_success.boxscore_sha256 == boxscore.sha256
        and prior_success.game_rotation_sha256
        == (game_rotation.sha256 if game_rotation is not None else None)
        and prior_quality.status in {"pass", "warning"}
        and prior_quality.code_version == code_version
        and prior_quality.play_by_play_source == play_by_play.source
        and prior_quality.boxscore_source == boxscore.source
        and prior_quality.play_by_play_sha256 == play_by_play.sha256
        and prior_quality.boxscore_sha256 == boxscore.sha256
        and prior_quality.game_rotation_sha256
        == (game_rotation.sha256 if game_rotation is not None else None)
        and processed_outputs_valid(game.game_id, processed_dir)
    )


def _audit_spec(game: CatalogGame) -> AuditGameSpec:
    return AuditGameSpec(
        game_id=game.game_id,
        season=game.season,
        season_type=game.season_type,
        sample_group="season_process",
        expected_overtime=(
            None if game.season_type == "all_star" else game.is_overtime
        ),
    )


def _apply_catalog_invariants(
    game: CatalogGame,
    quality: AuditGameResult,
) -> AuditGameResult:
    failure_codes = set(quality.issue_codes)
    if quality.home_team_id != game.home_team_id:
        failure_codes.add("catalog:home_team_mismatch")
    if quality.away_team_id != game.away_team_id:
        failure_codes.add("catalog:away_team_mismatch")
    if (
        game.season_type != "all_star"
        and game.period_count is not None
        and quality.period_count != game.period_count
    ):
        failure_codes.add("catalog:period_count_mismatch")
    if failure_codes == set(quality.issue_codes):
        return quality
    return quality.model_copy(
        update={
            "status": "fail",
            "issue_codes": tuple(sorted(failure_codes)),
        }
    )


def _error_quality(
    game: CatalogGame,
    stage: BuildStage,
    error: Exception,
) -> AuditGameResult:
    return AuditGameResult(
        game_id=game.game_id,
        season=game.season,
        season_type=game.season_type,
        sample_group="season_process",
        status="error",
        issue_codes=(f"process:{stage}:{type(error).__name__}",),
        error_stage=stage,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _failure_outcome(
    game: CatalogGame,
    *,
    run_id: str,
    attempt_id: str,
    attempt_number: int,
    code_version: str,
    started_at: datetime,
    stage: BuildStage,
    error: Exception,
    prefect_flow_run_id: str | None,
    prefect_task_run_id: str | None,
    play_by_play: SelectedRawArtifact | None = None,
    boxscore: SelectedRawArtifact | None = None,
    game_rotation: SelectedRawArtifact | None = None,
    reconstruction: GameReconstruction | None = None,
    quality: AuditGameResult | None = None,
    output_table_count: int = 0,
) -> GameProcessOutcome:
    finished_at = datetime.now(UTC)
    return GameProcessOutcome(
        record=GameBuildRecord(
            run_id=run_id,
            attempt_id=attempt_id,
            prefect_flow_run_id=prefect_flow_run_id,
            prefect_task_run_id=prefect_task_run_id,
            attempt_number=attempt_number,
            game_id=game.game_id,
            season=game.season,
            season_type=game.season_type,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=(finished_at - started_at).total_seconds(),
            status="failed",
            terminal_stage=stage,
            use_cache=True,
            code_version=code_version,
            play_by_play_source=(
                play_by_play.source if play_by_play is not None else None
            ),
            boxscore_source=boxscore.source if boxscore is not None else None,
            play_by_play_sha256=(
                play_by_play.sha256 if play_by_play is not None else None
            ),
            boxscore_sha256=boxscore.sha256 if boxscore is not None else None,
            game_rotation_sha256=(
                game_rotation.sha256 if game_rotation is not None else None
            ),
            event_count=(
                len(reconstruction.events) if reconstruction is not None else None
            ),
            lineup_stint_count=(
                len(reconstruction.lineups.stints)
                if reconstruction is not None
                else None
            ),
            possession_count=(
                len(reconstruction.possessions.possessions)
                if reconstruction is not None
                else None
            ),
            possession_segment_count=(
                len(reconstruction.possession_segments.segments)
                if reconstruction is not None
                else None
            ),
            validation_issue_count=(
                len(quality.issue_codes) if quality is not None else None
            ),
            output_table_count=output_table_count,
            error_type=type(error).__name__,
            error_message=str(error),
        ),
        quality=quality,
        retryable=isinstance(error, OSError),
    )


def _valid_output_count(
    game_id: str,
    processed_dir: Path | str,
) -> int:
    return sum(
        path.exists() for path in processed_output_paths(game_id, processed_dir).values()
    )
