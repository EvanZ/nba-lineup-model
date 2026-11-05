"""Dated forecast requests resolve the accepted model automatically, without training."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from nba_lineup_model.orchestration.daily_pipeline import (
    DailyPipelineConfig,
    require_clean_code_revision,
)
from nba_lineup_model.orchestration.daily_work import (
    atomic_json,
    digest,
    file_reference,
    read_object,
    work_spec,
)
from nba_lineup_model.orchestration.resumable_pipeline import (
    PartitionSteps,
    artifact_files,
    bind_execution,
    config_identity,
    prepare_partition,
    validate_execution,
)
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact


class ForecastPartitionSteps(PartitionSteps):
    """Forecast-data publication; application code releases are out of scope."""

    def validate_and_stage(self, target: date) -> Path:
        from nba_lineup_model.season.sealed_code_package import (
            inspect_source_code,
            verify_sealed_code_package,
        )

        code_root = self.config.code_root.resolve()
        package_root = code_root.parent
        code_identity = (
            verify_sealed_code_package(package_root)["identity"]
            if code_root.name == "code" and (package_root / "code-package.json").is_file()
            else inspect_source_code(code_root)["identity"]
        )
        if code_identity["git_revision"] != self.revision:
            raise ValueError("Forecast code identity differs from the pinned revision")
        return self.refresh_forecast_schedule(target, code_identity=code_identity)

    def stage_receipt(self, stage, result):
        if stage == "validate_and_stage":
            from nba_lineup_model.season.serving_selection import INDEX_FILENAME

            schedule = Path(result)
            evidence = self.candidate / "forecast-refresh.json"
            return {
                "artifacts": [
                    file_reference(schedule / INDEX_FILENAME),
                    file_reference(evidence),
                ],
                "result": {"schedule_artifact": str(schedule)},
            }
        return super().stage_receipt(stage, result)


def latest_completed_model(live: Path, forecast_date: date) -> Path:
    """Walk the accepted chain, not directory mtime or a caller-selected model.

    A no-game day is not a new game-based update. Neither it nor same-day/future
    evidence can take precedence over the last completed game-based update.
    """
    approved = read_object(live / "runtime-manifest.json")["approved_manifest"]["identity"]
    tip = Path(approved["model"]["bundle_path"])
    model_root = tip.parents[2]
    current, expected = tip, approved["model"]["bundle_sha256"]
    while True:
        manifest = verify_daily_model_artifact(current)
        if manifest["bundle_sha256"] != expected:
            raise ValueError("Accepted model reference differs from verified artifact")
        identity = manifest["identity"]
        if (
            date.fromisoformat(identity["game_date_pacific"]) < forecast_date
            and identity["daily_game_ids"]
        ):
            return current
        expected = identity["parent_model_sha256"]
        if not expected:
            raise ValueError("No completed model update is available before forecast date")
        current = (
            model_root
            / f"season={identity['season']}"
            / f"game_date_pacific={identity['parent_game_date_pacific']}"
            / f"model_sha256={expected}"
        )


def prepare_forecast_date(template: DailyPipelineConfig, forecast_date: date, *,
                          prepared_schedule: Path | None = None,
                          approval_reference: str | None = None) -> dict:
    """The Dagster boundary: date in, verified selected model and release out."""
    from nba_lineup_model.orchestration.forecast_publication_workflow import publish_forecast_schedule
    live = template.live_pointer.resolve(strict=True)
    if prepared_schedule is not None:
        from nba_lineup_model.season.serving_selection import verify_schedule_selection
        prepared_schedule = prepared_schedule.resolve(strict=True)
        if not prepared_schedule.is_relative_to(template.environment_root.resolve()):
            raise ValueError("Prepared forecasts must belong to the job's environment")
        index = verify_schedule_selection(prepared_schedule)
        replay = index.get("market_repair") or index.get("injury_replay", {})
        request = replay.get("request", {})
        if not approval_reference or approval_reference != request.get("approval_reference"):
            raise ValueError("Prepared historical correction requires its explicit approval")
        if request.get("game_date") != str(forecast_date):
            raise ValueError("Prepared correction belongs to another forecast date")
        model_path = latest_completed_model(live, forecast_date)
        model = verify_daily_model_artifact(model_path)
        if model["bundle_sha256"] != request.get("model"):
            raise ValueError("Prepared forecasts no longer use the accepted model")
        cfg = replace(template, model_root=model_path.parents[2])
        prior_path = cfg.pipeline_root / "forecast-publication" / replay["request_id"] / "report.json"
        prior = read_object(prior_path) if prior_path.is_file() else None
        from nba_lineup_model.season.production_site_release import verify_production_site_release
        if (prior and prior.get("status") == "activated"
            and prior.get("runtime_sha256") == verify_production_site_release(live)["runtime_sha256"]):
            publication = prior
        else:
            publication = publish_forecast_schedule(cfg, schedule=prepared_schedule, base=live,
                work_id=replay["request_id"], activate=template.activate_after_validation)
        return {"forecast_date": str(forecast_date), "model_date": model["identity"]["game_date_pacific"],
                "model_sha256": model["bundle_sha256"], "work_id": replay["request_id"],
                "schedule_artifact": str(prepared_schedule), "publication": publication}
    base_ref = file_reference(live / "runtime-manifest.json")
    base = read_object(live / "runtime-manifest.json")["approved_manifest"]["identity"]
    bindings = read_object(live / "serving-bindings.json")
    if template.observation_root is None:
        raise ValueError("Forecast job requires the canonical collector observation root")
    observation_root = template.observation_root.resolve(strict=True)
    schedule = live / "selected-schedule"
    from nba_lineup_model.season.official_schedule_runtime import select_captured_injuries
    captures = select_captured_injuries(schedule_root=schedule, observation_root=observation_root,
        start_date=forecast_date.isoformat(), days_ahead=0, now=datetime.now(UTC))
    from nba_lineup_model.season.market_selection import select_captured_markets
    markets = select_captured_markets(schedule_root=schedule, observation_root=observation_root,
        start_date=forecast_date.isoformat(), days_ahead=0, now=datetime.now(UTC))
    selected = latest_completed_model(live, forecast_date)
    model = verify_daily_model_artifact(selected)
    evidence_date = date.fromisoformat(model["identity"]["game_date_pacific"])
    model_root = selected.parents[2]
    data_root = Path(base["data"]["bundle_path"]).parents[2]
    models, data = [], []
    current = selected
    while True:
        m = verify_daily_model_artifact(current)
        info = m["identity"]
        data_path = (
            data_root
            / f"season={info['season']}"
            / f"game_date_pacific={info['game_date_pacific']}"
            / f"data_sha256={info['input_data_sha256']}"
        )
        verify_daily_model_artifact(current, data_bundle=data_path)
        data.append(data_path)
        models.append(current)
        if not info["parent_model_sha256"]:
            break
        current = (
            model_root
            / f"season={info['season']}"
            / f"game_date_pacific={info['parent_game_date_pacific']}"
            / f"model_sha256={info['parent_model_sha256']}"
        )
    models.reverse()
    data.reverse()
    observed = verify_daily_data_artifact(data[-1])
    revision = require_clean_code_revision(template.code_root)
    run_identity = digest(
        {
            "base": base_ref["sha256"],
            "forecast_date": str(forecast_date),
            "code": revision,
            "model": model["bundle_sha256"],
            "injury_captures": captures,
            "market_captures": markets,
        }
    )
    output = template.pipeline_root / "forecasts" / str(forecast_date) / run_identity
    injury_inputs = output / "inputs" / "injury-captures.json"
    atomic_json(injury_inputs, captures, immutable=True)
    market_inputs = output / "inputs" / "market-captures.json"
    atomic_json(market_inputs, markets, immutable=True)
    config = replace(
        template,
        pipeline_root=output / "pipeline",
        data_root=output / "data",
        model_root=output / "model",
        schedule_root=schedule,
        observation_root=observation_root,
        forecast_injury_inputs_path=injury_inputs,
        catalog_path=live / "catalog/games.parquet",
        frontend_root=live / "web",
        preseason_rankings_path=live / "preseason/rankings.parquet",
        preseason_profiles_path=live / "preseason/profiles.parquet",
        forecast_date=str(forecast_date),
        forecast_market_inputs_path=market_inputs,
        activate_after_validation=False,
    )
    files = {
        "coverage.json": file_reference(config.coverage_path),
        "catalog.parquet": file_reference(config.catalog_path),
        "rankings.parquet": file_reference(config.preseason_rankings_path),
        "profiles.parquet": file_reference(config.preseason_profiles_path),
        "minutes.json": file_reference(config.preseason_minutes_path),
        "model-configuration.json": file_reference(config.model_configuration_path),
    }
    schedule_files = {
        str(p.relative_to(schedule)): file_reference(p)
        for p in sorted(schedule.rglob("*"))
        if p.is_file()
    }
    refs = [base_ref, *files.values(), *schedule_files.values()]
    refs.append(file_reference(injury_inputs))
    for bundle in [*models, *data]:
        refs.extend(artifact_files(bundle))
    spec = work_spec(
        season=config.season,
        game_date=evidence_date,
        source_revisions=observed["identity"]["game_sources"],
        model_configuration=model["identity"]["model_provenance"],
        parent_model=model["identity"]["parent_model_sha256"],
        edition_id=bindings["edition_id"],
        code_identity=revision,
        inputs={ref["path"]: ref for ref in refs},
        expected_game_ids=observed["identity"]["game_ids"],
    )
    spec = bind_execution(
        spec,
        config,
        {
            "files": files,
            "schedule_files": schedule_files,
            "base_release_manifest": base_ref,
            "data_manifests": [file_reference(p / "manifest.json") for p in data],
            "model_manifests": [file_reference(p / "manifest.json") for p in models[:-1]],
            "target_model_manifest": file_reference(selected / "manifest.json"),
            "reuse_accepted_model": True,
        },
    )
    validate_execution(config, spec, None)
    atomic_json(output / "config.json", config_identity(config), immutable=True)
    atomic_json(output / "spec.json", spec, immutable=True)
    result = prepare_partition(
        config,
        spec,
        steps_class=ForecastPartitionSteps,
        result_key="schedule_artifact",
    )
    publication = publish_forecast_schedule(
        replace(template, model_root=config.model_root), schedule=Path(result["schedule_artifact"]),
        base=live, work_id=spec["work_id"], activate=template.activate_after_validation)
    return {
        **result,
        "publication": publication,
        "forecast_date": str(forecast_date),
        "model_date": str(evidence_date),
        "model_sha256": model["bundle_sha256"],
        "config_path": str(output / "config.json"),
        "work_root": str(
            config.pipeline_root / "partitions" / str(evidence_date) / spec["work_id"]
        ),
    }
