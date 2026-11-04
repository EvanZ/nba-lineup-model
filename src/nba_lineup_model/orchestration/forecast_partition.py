"""Dated forecast requests resolve the accepted model automatically, without training."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
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
    artifact_files,
    bind_execution,
    config_identity,
    prepare_partition,
    validate_execution,
)
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact


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


def prepare_forecast_date(template: DailyPipelineConfig, forecast_date: date) -> dict:
    """The Dagster boundary: date in, verified selected model and release out."""
    live = template.live_pointer.resolve(strict=True)
    base_ref = file_reference(live / "runtime-manifest.json")
    base = read_object(live / "runtime-manifest.json")["approved_manifest"]["identity"]
    bindings = read_object(live / "serving-bindings.json")
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
        }
    )
    output = template.pipeline_root / "forecasts" / str(forecast_date) / run_identity
    schedule = (live / "schedule").resolve()
    config = replace(
        template,
        pipeline_root=output / "pipeline",
        data_root=output / "data",
        model_root=output / "model",
        schedule_root=schedule,
        catalog_path=live / "catalog/games.parquet",
        frontend_root=live / "web",
        preseason_rankings_path=live / "preseason/rankings.parquet",
        preseason_profiles_path=live / "preseason/profiles.parquet",
        forecast_date=str(forecast_date),
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
    result = prepare_partition(config, spec)
    return {
        **result,
        "forecast_date": str(forecast_date),
        "model_date": str(evidence_date),
        "model_sha256": model["bundle_sha256"],
        "config_path": str(output / "config.json"),
        "work_root": str(
            config.pipeline_root / "partitions" / str(evidence_date) / spec["work_id"]
        ),
    }
