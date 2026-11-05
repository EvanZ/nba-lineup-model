"""Automatic forecast-data handoff, retaining the already deployed application."""
from pathlib import Path

from nba_lineup_model.orchestration.daily_activation import reload_environment_api, verify_environment_api
from nba_lineup_model.orchestration.daily_work import atomic_json
from nba_lineup_model.orchestration.release_preflight import candidate_api, fetch
from nba_lineup_model.season.dev_site_release import collect_api_validation, _normalize_api_validation, _resolved_daily_data_tip
from nba_lineup_model.season.production_site_release import stage_schedule_release, verify_production_site_release
from nba_lineup_model.season.release_publisher import activate_release, read_active_activation
from nba_lineup_model.season.serving_selection import (
    FORECAST_FIELDS, MARKET_FIELDS, INDEX_FILENAME, forecast_inputs, read_object, verify_schedule_selection,
)


def validate_forecast_handoff(base: Path, schedule: Path, *, model_release: Path | None = None) -> list[str]:
    old = verify_schedule_selection(base / "selected-schedule")
    new = verify_schedule_selection(schedule)
    manifest = verify_production_site_release(model_release or base)
    model_sha = manifest["approved_manifest"]["identity"]["model"]["bundle_sha256"]
    from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
    from nba_lineup_model.season.game_prediction_artifact import require_projection_reconciliation
    model_path = Path(manifest["approved_manifest"]["identity"]["model"]["bundle_path"])
    model = verify_daily_model_artifact(model_path)
    hca_ref = model["identity"]["components"]["home_court_state"]
    hca = read_object(model_path / hca_ref["path"])["home_court_estimate"]
    if set(new["games"]) != set(old["games"]) or set(new["dates"]) != set(old["dates"]):
        raise ValueError("Forecast handoff changed schedule coverage")
    changed = []
    for day, ref in new["dates"].items():
        originals = {str(g["game_id"]): g for g in read_object(base / "selected-schedule" / old["dates"][day]["path"])["games"]}
        for game in read_object(schedule / ref["path"])["games"]:
            gid = str(game["game_id"])
            before = originals[gid].get("pregame", originals[gid])
            after = game.get("pregame", game)
            if before.get("market") != after.get("market"):
                repair = new.get("market_repair", {}).get("request", {})
                if new["games"][gid].get("provenance") != "original_live":
                    if (not repair or repair.get("source_selection") != old["selection_map_sha256"]
                        or repair.get("game_date") != day or not repair.get("approval_reference")
                        or repair.get("market_captures", {}).get(gid, {}).get("market") != after.get("market")
                        or forecast_inputs(before) != forecast_inputs(after)):
                        raise ValueError(f"Market-only correction violates its approval: {gid}")
                if not game.get("market_selection"):
                    raise ValueError(f"Market selection has no capture provenance: {gid}")
                if forecast_inputs(before) == forecast_inputs(after):
                    changed.append(gid)
                    continue
            if forecast_inputs(before) == forecast_inputs(after):
                continue
            selection = new["games"][gid]
            if selection.get("provenance") == "retrospective_replay":
                replay = new.get("injury_replay", {})
                if (not selection.get("approval_reference")
                    or replay.get("request", {}).get("approval_reference") != selection["approval_reference"]
                    or replay["request"].get("source_selection") != old["selection_map_sha256"]
                    or replay["request"].get("game_date") != day):
                    raise ValueError(f"Historical correction has no matching approval: {gid}")
            elif selection.get("provenance") != "original_live":
                raise ValueError(f"Forecast changed without a publication contract: {gid}")
            provenance = after["projection"]["input_provenance"]
            if provenance.get("rating_source_uri") != f"artifact://daily-model/{model_sha}/player_ratings":
                raise ValueError(f"Forecast publication changed accepted model: {gid}")
            if not after["injury_report"].get("capture_reference"):
                raise ValueError(f"Forecast lacks captured injury provenance: {gid}")
            require_projection_reconciliation(after["projection"], game_id=gid)
            if abs(after["projection"]["ledger"]["home_court_adjustment"] - hca) > 1e-9:
                raise ValueError(f"Forecast HCA differs from accepted model: {gid}")
            changed.append(gid)
    return sorted(changed)


def retained_market_warnings(before: dict, after: dict, existing: list, errors: list) -> list:
    """Retain existing retrospective warnings only for independent market changes."""
    if not errors:
        return []
    if (forecast_inputs(before) != forecast_inputs(after)
        or before.get("market") == after.get("market")
        or before.get("prediction_market") != after.get("prediction_market")
        or any(error not in existing or not isinstance(error, str)
               or not error.startswith("retrospective_unavailable:Official minutes player ")
               or not error.endswith(" has no rating-before state") for error in errors)):
        raise ValueError("API reported a new or publication-dependent game error")
    return list(errors)


def publish_forecast_schedule(config, *, schedule: Path, base: Path, work_id: str,
                              activate: bool) -> dict:
    """Called by the successful forecast job; failure propagates to Dagster."""
    base = base.resolve(strict=True)
    if config.observation_root is None:
        raise ValueError("Forecast publication requires the canonical observation source")
    active = read_active_activation(config.live_pointer)
    base_identity = verify_production_site_release(base)
    if active is None or active["runtime_sha256"] != base_identity["runtime_sha256"]:
        raise ValueError("Active release changed before forecast publication")
    changed = validate_forecast_handoff(base, schedule)
    candidate = stage_schedule_release(base_release=base, schedule_root=schedule,
        output_root=config.pipeline_root / "forecast-releases")
    manifest = verify_production_site_release(candidate)
    identity = manifest["identity"]
    report_root = config.pipeline_root / "forecast-publication" / work_id
    report_root.mkdir(parents=True, exist_ok=True)
    report = {"status": "validating", "work_id": work_id, "runtime_release": str(candidate),
        "runtime_sha256": manifest["runtime_sha256"],
        "base_runtime": active["runtime_sha256"], "base_generation": active["generation"],
        "code_changed": False, "model_changed": False, "changed_game_ids": changed}
    environment = "dev" if config.environment_root.name == "nba-gestalt-dev" else "prod"
    try:
        with candidate_api(candidate, report_root, observation_root=config.observation_root,
                           quality_root=config.environment_root / "data-quality") as api:
            evidence = collect_api_validation(api, evidence_date=identity["evidence_through_date"],
                                              daily_model_root=config.model_root)
            _normalize_api_validation(evidence, evidence_date=identity["evidence_through_date"],
                expected_schedule_game_ids=_resolved_daily_data_tip(config.data_root)[1]["identity"]["game_ids"],
                expected_game_count=identity["game_count"],
                expected_final_game_count=manifest["approved_manifest"]["identity"]["schedule"]["final_game_count"],
                serving_sha=identity["serving_bundle_sha256"], lineup_sha=identity["lineup_bundle_sha256"])
            index = verify_schedule_selection(candidate / "selected-schedule")
            coverage = {}
            retained_warnings = {}
            for day, ref in index["dates"].items():
                board = read_object(candidate / "selected-schedule" / ref["path"])
                actual = fetch(api, f"/api/schedule?date={day}")
                expected_games = {str(g["game_id"]): g for g in board["games"]}
                actual_games = {str(g["game_id"]): g for g in actual["games"]}
                if set(expected_games) != set(actual_games):
                    raise ValueError(f"Published schedule coverage differs on {day}")
                for gid in set(changed) & set(expected_games):
                    expected_game = expected_games[gid].get("pregame", expected_games[gid])
                    actual_game = actual_games[gid].get("pregame", actual_games[gid])
                    for field in (*FORECAST_FIELDS, *MARKET_FIELDS):
                        if expected_game.get(field) != actual_game.get(field):
                            raise ValueError(f"API does not serve selected {field}: {gid}")
                    if actual_games[gid].get("data_quality_errors"):
                        base_index = verify_schedule_selection(base / "selected-schedule")
                        before_board = read_object(base / "selected-schedule" / base_index["dates"][day]["path"])
                        before = next(g for g in before_board["games"] if str(g["game_id"]) == gid)
                        port = 8011 if environment == "dev" else 8010
                        existing_board = fetch(f"http://127.0.0.1:{port}", f"/api/schedule?date={day}")
                        existing = next(g for g in existing_board["games"] if str(g["game_id"]) == gid)
                        retained_warnings[gid] = retained_market_warnings(
                            before.get("pregame", before), expected_game,
                            existing.get("data_quality_errors", []), actual_games[gid]["data_quality_errors"])
                coverage[day] = len(actual_games)
            report.update(status="validated", api_validation=evidence, schedule_coverage=coverage,
                          retained_game_warnings=retained_warnings)
        if activate:
            result = activate_release(release=candidate, live_pointer=config.live_pointer,
                publisher_root=config.environment_root / "release-publisher", environment=environment,
                operation_id=f"forecast-{work_id}-from-{active['generation']}",
                expected_previous_generation=active["generation"], reason="validated forecast-data update",
                reload_api=lambda record: reload_environment_api(config, record),
                verify_api=lambda record: verify_environment_api(config, Path(record["release_path"]), record))
            report.update(status="activated", generation=result.generation)
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(report_root / "report.json", report)
