"""Evaluate saved model predictions against final scores and pre-tipoff markets."""
import json
from datetime import UTC, date, datetime

from dagster import (
    AllPartitionMapping, AssetDep, AssetExecutionContext, AssetKey, BackfillPolicy,
    Failure, MaterializeResult, MetadataValue, asset,
)

from ...box_scores.resources import NBABoxScoreArchive
from ...predictions.resources import GamePredictionArchive
from ...resources import NBAScheduleResource
from ...utils import PACIFIC
from ..partitions import market_performance_dates
from ..resources import MarketPerformanceArchive
from ..utils import (
    CONTRACT, aggregate_totals, daily_totals, fingerprint, game_result,
    implementation_identity, select_markets,
)


def _ref(pointer):
    return {key: pointer[key] for key in ("uri", "sha256")}


@asset(
    group_name="market_performance", partitions_def=market_performance_dates,
    backfill_policy=BackfillPolicy.multi_run(max_partitions_per_run=1),
    deps=[AssetDep(AssetKey(key)) for key in ("game_predictions", "game_predictions_actual_minutes", "nba_box_scores", "nba_season_schedule")]
         + [AssetDep(AssetKey("nba_market_odds"), partition_mapping=AllPartitionMapping())],
    code_version=implementation_identity()["source_sha256"],
    description="Saved postgame market evaluation and daily totals for the Schedule summary and calendar.",
)
def market_performance_by_date(
    context: AssetExecutionContext, nba_schedule: NBAScheduleResource,
    game_prediction_archive: GamePredictionArchive, nba_box_score_archive: NBABoxScoreArchive,
    market_performance_archive: MarketPerformanceArchive,
) -> MaterializeResult:
    day, now = context.partition_key, datetime.now(UTC)
    archive, predictions = market_performance_archive, game_prediction_archive
    inputs, sources = {}, {}
    try:
        if date.fromisoformat(day) >= now.astimezone(PACIFIC).date():
            raise ValueError("Select a completed Pacific game date for market performance.")
        if (len({archive.season, nba_schedule.season, predictions.season}) != 1
                or archive.model_version != predictions.model_version
                or len({archive.bucket, nba_schedule.bucket, predictions.bucket, nba_box_score_archive.bucket}) != 1):
            raise ValueError("Market-performance resources have different seasons, models or buckets.")
        _, expected_etag = archive.selection(day)
        games, schedule = nba_schedule.games_for_date(date.fromisoformat(day), allow_empty=True)
        raw, _ = nba_schedule.load(schedule)
        raw_games = {g["gameId"]: g for group in json.loads(raw)["leagueSchedule"]["gameDates"] for g in group["games"]}
        names = {g["game_id"]: {side: raw_games[g["game_id"]][side+"Team"]["teamName"] for side in ("home", "away")} for g in games}
        inputs = {"schedule": schedule, "games": {}, "odds_snapshots": []}
        for game in games:
            gid = game["game_id"]
            pointer, _ = predictions.selection(day, gid)
            counter_pointer, _ = predictions.selection(day, gid, actual_minutes=True)
            if pointer is None or counter_pointer is None:
                raise FileNotFoundError(f"Missing saved prediction or actual-minutes counterfactual for {gid}.")
            prediction = predictions.load(pointer)
            counter = predictions.load(counter_pointer, actual_minutes=True)
            box_raw, box_ref = nba_box_score_archive.read_saved(archive.season, game)
            box = json.loads(box_raw)["game"]
            refs = {"prediction": _ref(pointer), "actual_minutes_counterfactual": _ref(counter_pointer), "final_box_score": box_ref}
            inputs["games"][gid] = refs
            if (counter["inputs"]["prediction"] != refs["prediction"] or counter["inputs"]["box_score"] != box_ref
                    or counter["actual_home_margin"] != int(box["homeTeam"]["score"])-int(box["awayTeam"]["score"])):
                raise ValueError(f"The counterfactual for {gid} does not use the currently selected prediction/final box. Refresh that job first.")
            if (box["gameStatus"] != 3 or datetime.fromisoformat(prediction["scheduled_start_utc"]) != game["game_time_utc"]
                    or any(prediction[side]["team_id"] != game[side+"_team_id"] for side in ("home", "away"))):
                raise ValueError(f"Non-final game or differing official matchup for {gid}.")
            sources[gid] = (prediction, counter, box)
        before = max((game["game_time_utc"] for game in games), default=None)
        inventory = archive.snapshot_inventory(before) if before else []
        selected = select_markets(games, names, archive.snapshots(inventory)) if games else {}
        snapshot_refs = {q["snapshot"]["uri"]: q["snapshot"] for markets in selected.values() for kind in markets.values()
                         for q in [kind["polymarket"], *kind["books"]]}
        inputs["odds_snapshots"] = [snapshot_refs[key] for key in sorted(snapshot_refs)]
        rows = [game_result(game, *sources[game["game_id"]], selected[game["game_id"]], inputs["games"][game["game_id"]]) for game in games]
        totals = daily_totals(rows)
        record = {"schema_version": 1, "artifact_kind": "market_performance", "season": archive.season,
                  "model_version": archive.model_version, "game_date_pacific": day, "generated_at": now.isoformat(),
                  "calculation_contract": CONTRACT, "implementation": implementation_identity(), "inputs": inputs,
                  "input_fingerprint": fingerprint(inputs, archive.season, archive.model_version, day),
                  "games": rows, "daily_totals": totals, "summary": aggregate_totals([totals]),
                  "coverage": {"scheduled_game_count": len(games), "completed_game_count": len(rows),
                      "prediction_game_count": len(rows), "actual_minutes_game_count": len(rows),
                      "polymarket_game_count": sum(all(r["comparisons"]["polymarket"][k] is not None for k in ("home_win_probability", "predicted_home_margin")) for r in rows),
                      "bookmaker_consensus_game_count": sum(all(r["comparisons"]["bookmaker_consensus"][k] is not None for k in ("home_win_probability", "predicted_home_margin")) for r in rows),
                      "shared_comparison_game_count": totals["game_count"],
                      "missing_market_fields": [{"game_id": r["game_id"], "fields": r["missing_market_fields"]} for r in rows if r["missing_market_fields"]]}}
        if nba_schedule.load()[1] != schedule:
            raise ValueError("The official schedule changed during evaluation.")
        for game in games:
            gid = game["game_id"]
            current, _ = predictions.selection(day, gid)
            counter, _ = predictions.selection(day, gid, actual_minutes=True)
            if current is None or counter is None or _ref(current) != inputs["games"][gid]["prediction"] or _ref(counter) != inputs["games"][gid]["actual_minutes_counterfactual"]:
                raise ValueError("Accepted prediction inputs changed during evaluation.")
            if nba_box_score_archive.read_saved(archive.season, game)[1] != inputs["games"][gid]["final_box_score"]:
                raise ValueError("The selected final box changed during evaluation.")
        if before and archive.snapshot_inventory(before) != inventory:
            raise ValueError("The eligible odds archive changed during evaluation; rerun this date.")
        output = archive.save(record, expected_etag=expected_etag)
    except Exception as error:
        metadata = {"game_date_pacific": day}
        message = str(error)
        try:
            metadata["market_performance_failure"] = MetadataValue.json(archive.save_event(
                day=day, run_id=context.run.run_id, message=message, inputs=inputs))
        except Exception as write_error:
            message += f" Could not save the S3 failure record: {write_error}"
        raise Failure(message, metadata=metadata, allow_retries=False) from error
    try:
        parent = archive.parent_failure(context, day)
        if parent:
            recovery = archive.save_event(day=day, run_id=context.run.run_id, message="Daily market performance recovered successfully.",
                                          inputs=inputs, status="recovered", original=parent)
        else:
            recovery = None
    except Exception as error:
        raise Failure(f"Performance saved, but recovery reporting failed: {error}", allow_retries=False) from error
    return MaterializeResult(metadata={"game_date_pacific": day, "game_count": len(rows),
        "shared_comparison_game_count": totals["game_count"], "performance": MetadataValue.json(output),
        "missing_market_fields": MetadataValue.json(record["coverage"]["missing_market_fields"]),
        "summary": MetadataValue.json(record["summary"]), "recovery": MetadataValue.json(recovery)})
