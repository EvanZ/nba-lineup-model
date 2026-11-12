"""Build a complete Pacific date's stints from accepted raw NBA captures."""

from datetime import date

from dagster import AssetDep, AssetExecutionContext, AssetKey, Failure, MaterializeResult, MetadataValue, asset

from ...box_scores.resources import NBABoxScoreArchive
from ...resources import NBAScheduleResource
from ..partitions import play_by_play_dates
from ..resources import NBAPlayByPlayArchive
from ..utils import PROCESSING_VERSION, StintsError, parse_stints
from .capture import _parent_failure


@asset(
    group_name="play_by_play", partitions_def=play_by_play_dates,
    deps=[AssetDep(AssetKey("nba_play_by_play")), AssetDep(AssetKey("nba_box_scores"))],
    code_version=PROCESSING_VERSION,
)
def nba_stints(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    nba_play_by_play_archive: NBAPlayByPlayArchive, nba_box_score_archive: NBABoxScoreArchive,
) -> MaterializeResult:
    day = date.fromisoformat(context.partition_key)
    season = nba_schedule.season
    archive = nba_play_by_play_archive
    inputs = {"schedule": None, "games": {}}
    failures, rows, diagnostics = {}, [], {}
    metadata = {"season": season, "game_date_pacific": day.isoformat(), "processing_version": PROCESSING_VERSION}
    try:
        prior, etag = archive.stints_pointer(season, day.isoformat())
        games, schedule = nba_schedule.games_for_date(day, allow_empty=True)
        inputs["schedule"] = schedule
        for game in sorted(games, key=lambda g:g["game_id"]):
            game_id = game["game_id"]
            inputs["games"][game_id] = {}
            try:
                pbp, ref = archive.read_game(season, game)
                inputs["games"][game_id]["play_by_play"] = ref
                box, ref = nba_box_score_archive.read_saved(season, game)
                inputs["games"][game_id]["box_score"] = ref
                game_rows, quality = parse_stints(pbp, box, game)
                rows.extend(game_rows)
                diagnostics[game_id] = quality
            except Exception as error:
                failures[game_id] = {
                    "message": str(error), "error_type": type(error).__name__,
                    "details": error.details if isinstance(error, StintsError) else {},
                }
        if failures:
            raise ValueError(f"{len(failures)} of {len(games)} NBA games could not be parsed for {day}.")
        expected = sorted(game["game_id"] for game in games)
        record = {
            "schema_version": 1, "season": season, "game_date_pacific": day.isoformat(),
            "processing_version": PROCESSING_VERSION, "inputs": inputs,
            "validation": {
                "complete": True, "expected_game_ids": expected,
                "processed_game_ids": sorted(diagnostics), "games": diagnostics,
            },
            "rows": rows,
        }
        saved = archive.save_stints(record, prior=prior, expected_etag=etag)
    except Exception as error:
        message = str(error)
        metadata.update({"inputs": MetadataValue.json(inputs), "failed_games": MetadataValue.json(failures)})
        try:
            ref = archive.save_error_event(
                season=season, day=day.isoformat(), run_id=context.run.run_id,
                message=message, schedule=inputs["schedule"], failed_games=failures,
                stage="nba_stints", inputs=inputs,
            )
            metadata["stints_failure"] = MetadataValue.json(ref)
        except Exception as write_error:
            message += f" Could not save the S3 error record. {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    metadata.update({
        "stints_uri": saved["uri"], "sha256": saved["sha256"], "reused": saved["reused"],
        "game_count": len(games), "stint_count": len(rows),
        "zero_possession_stint_count": sum(not row["possessions"] for row in rows),
        "inputs": MetadataValue.json(inputs),
    })
    original = None
    try:
        original = _parent_failure(
            context, archive, season, day.isoformat(),
            job_name="parse_nba_games", stage="nba_stints", metadata_key="stints_failure",
        )
        if original:
            recovery = archive.save_error_event(
                season=season, day=day.isoformat(), run_id=context.run.run_id,
                message="All selected NBA games were parsed and validated.",
                schedule=inputs["schedule"], failed_games={}, original_failure=original,
                stage="nba_stints", inputs=inputs,
            )
            metadata["recoveries"] = MetadataValue.json([recovery])
    except Exception as error:
        if original:
            metadata["stints_failure"] = MetadataValue.json(original)
        raise Failure(f"Stints are saved; could not verify or record recovery. {error}", metadata=metadata) from error
    return MaterializeResult(metadata=metadata)
