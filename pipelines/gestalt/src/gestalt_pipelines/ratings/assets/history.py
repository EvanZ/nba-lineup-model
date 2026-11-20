"""Publish saved scalar ratings histories without recalculating the ratings."""
from collections import Counter
from datetime import date

from dagster import AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy, Failure, MaterializeResult, MetadataValue, asset

from ...predictions.resources import GamePredictionArchive
from ...resources import NBAScheduleResource
from ..partitions import ratings_dates
from ..resources import RatingsArchive
from ..utils import history_index, history_records


@asset(
    group_name="ratings", partitions_def=ratings_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[AssetDep(AssetKey("player_ratings")), AssetDep(AssetKey("player_rating_priors")), AssetDep(AssetKey("game_predictions_actual_minutes"))],
    description="Publish game-day ratings, inactive players and pinned final-box stats without recalculating ratings.",
)
def player_game_history(context: AssetExecutionContext, ratings_archive: RatingsArchive,
                        nba_schedule: NBAScheduleResource, game_prediction_archive: GamePredictionArchive) -> MaterializeResult:
    archive, day = ratings_archive, context.partition_key
    inputs = {}
    try:
        chain = archive.history_chain(day, first_date=str(ratings_dates.start.date()))
        inputs = {"head": chain[-1][1], "priors": chain[0][0]["inputs"]["priors"],
                  "ratings_chain": {row["game_date_pacific"]: ref for row, ref in chain}}
        if (len({archive.season, nba_schedule.season, game_prediction_archive.season}) != 1
                or len({archive.bucket, nba_schedule.bucket, game_prediction_archive.bucket}) != 1
                or archive.model_version != game_prediction_archive.model_version):
            raise ValueError("History resources have different season/model/bucket identities.")
        game_details, game_inputs = {}, {}
        for state, _ in chain:
            games, _ = nba_schedule.games_for_date(date.fromisoformat(state["game_date_pacific"]),
                state["inputs"]["schedule"], allow_empty=True)
            details, references = archive.history_games(state, games, game_prediction_archive)
            game_details[state["game_date_pacific"]] = details
            game_inputs.update(references)
        inputs["postgame"] = {f"{d}/{g}": ref for (d, g), ref in game_inputs.items()}
        histories = history_records(chain, game_details)
        # Validate every existing selection before any new publication. Each
        # player index is an independent coherent selection, not a global table.
        existing = {pid: archive.history_selection(pid) for pid in histories}
        for prior, _ in existing.values():
            if prior is not None and any(prior[k] != chain[-1][0][k] for k in ("model_id", "model_run_id")):
                raise ValueError("Accepted player history belongs to another model identity.")
        known_keys = archive.history_inventory()
        indexes, reused, verified = {}, 0, 0
        for pid, records in histories.items():
            prior, etag = existing[pid]
            accepted = {p["game_date_pacific"]: p for p in prior["points"]} if prior else {}
            references = []
            for record in records:
                reference, was_reused = archive.save_history_record(record, accepted=accepted.get(record["game_date_pacific"]), known_keys=known_keys)
                references.append(reference)
                reused += int(was_reused)
                verified += int(not was_reused)
            indexes[pid] = history_index(records, references,
                first_date=str(ratings_dates.start.date()), chain=chain, player_id=pid)
        archive.assert_history_sources_current(chain)
        archive.assert_history_games_current(game_inputs, game_prediction_archive)
        publication = Counter()
        for pid, index in indexes.items():
            # Recheck the head immediately before each conditional index write.
            if archive.state_selection(day)[0] != inputs["head"]:
                raise ValueError("Ratings head changed during history publication; rerun this date.")
            prior, etag = existing[pid]
            result = archive.save_history_index(index, prior=prior, expected_etag=etag)
            publication[result["status"]] += 1
        archive.assert_history_games_current(game_inputs, game_prediction_archive)
    except Exception as error:
        message = str(error)
        metadata = {"game_date_pacific": day}
        try:
            metadata["ratings_failure"] = MetadataValue.json(archive.save_event(run_id=context.run.run_id,
                stage="player_game_history", day=day, message=message, inputs=inputs))
        except Exception as reporting_error:
            message += f" Could not save the failure record: {reporting_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    recovery = None
    try:
        original = archive.parent_failure(context, stage="player_game_history", day=day)
        if original:
            recovery = archive.save_event(run_id=context.run.run_id, stage="player_game_history", day=day,
                message="Player rating histories validated and published.", inputs=inputs, original=original)
    except Exception as error:
        raise Failure(f"History published, but recovery reporting failed: {error}", allow_retries=False) from error
    return MaterializeResult(metadata={"game_date_pacific": day, "player_count": len(histories),
        "date_count": len(chain), "game_count": len(game_inputs), "record_count": sum(len(rows) for rows in histories.values()),
        "reused_record_count": reused, "verified_record_count": verified,
        "index_publication": MetadataValue.json(dict(publication)),
        "player_ids": MetadataValue.json(sorted(histories)), "history_root": f"s3://{archive.bucket}/{archive.history_root}",
        "source_head": MetadataValue.json(inputs["head"]), "initialization": MetadataValue.json(inputs["priors"]),
        "recovery": MetadataValue.json(recovery)})
