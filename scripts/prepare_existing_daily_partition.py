"""Pin an approved missing-day release using existing canonical artifacts.

This command performs no downloads, training, publication, or pointer changes.
The caller must supply the actual external approval reference.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from nba_lineup_model.orchestration.daily_pipeline import (
    DailyPipelineConfig, require_clean_code_revision, expected_slate_ids,
    causal_hca_replay_config,
)
from nba_lineup_model.orchestration.daily_work import atomic_json, file_reference, read_object, work_spec
from nba_lineup_model.orchestration.repair_plan import require_approval
from nba_lineup_model.orchestration.resumable_pipeline import (
    artifact_files, bind_execution, config_identity, validate_execution,
)
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.source_policy import require_clean_artifact_sources
from nba_lineup_model.season.serving_selection import verify_schedule_selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("live", "target-data", "target-model", "plan", "output", "code",
                 "coverage", "minutes", "hca-configuration", "observation-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--approval-reference", required=True)
    p.add_argument("--approved-by", required=True)
    p.add_argument("--model-provenance-matchup-url", required=True)
    args = p.parse_args()
    live = args.live.resolve(strict=True)
    runtime = read_object(live / "runtime-manifest.json")
    base = runtime["approved_manifest"]["identity"]
    bindings = read_object(live / "serving-bindings.json")
    data = verify_daily_data_artifact(args.target_data)
    model = verify_daily_model_artifact(args.target_model, data_bundle=args.target_data)
    target = date.fromisoformat(data["identity"]["game_date_pacific"])
    plan = read_object(args.plan)
    approval = {"plan_id":plan["plan_id"], "approved_by":args.approved_by,
                "approval_reference":args.approval_reference,
                "actions":["select_historical_data", "causal_replay"]}
    require_approval(plan, approval, "causal_replay")
    if model["identity"]["model_provenance"] != base["model"]["model_provenance"]:
        raise ValueError("Existing model is not the accepted model configuration")
    schedule = (live / "schedule").resolve()
    verify_schedule_selection(schedule)
    model_tip = Path(base["model"]["bundle_path"])
    data_tip = Path(base["data"]["bundle_path"])
    model_root = model_tip.parents[2]
    data_root = data_tip.parents[2]
    models_by_hash = {path.parent.name.removeprefix("model_sha256="):path
                      for path in model_root.glob("season=*/game_date_pacific=*/model_sha256=*/manifest.json")}
    model_manifests, data_manifests = [], []
    current = base["model"]["bundle_sha256"]
    while current:
        path = models_by_hash[current]
        manifest = verify_daily_model_artifact(path.parent)
        info = manifest["identity"]
        selected_data = data_root / f"season={info['season']}" / f"game_date_pacific={info['game_date_pacific']}" / f"data_sha256={info['input_data_sha256']}"
        verify_daily_model_artifact(path.parent, data_bundle=selected_data)
        model_manifests.append(path)
        data_manifests.append(selected_data / "manifest.json")
        current = info["parent_model_sha256"]
    model_manifests.reverse()
    data_manifests.reverse()
    data_manifests.append(args.target_data / "manifest.json")
    for path in [*data_manifests, *model_manifests, args.target_model / "manifest.json"]:
        require_clean_artifact_sources(path.parent)
    require_clean_artifact_sources(schedule)
    files = {"coverage.json":file_reference(args.coverage),
             "catalog.parquet":file_reference(live / "catalog/games.parquet"),
             "rankings.parquet":file_reference(live / "preseason/rankings.parquet"),
             "profiles.parquet":file_reference(live / "preseason/profiles.parquet"),
             "minutes.json":file_reference(args.minutes),
             "model-configuration.json":file_reference(args.hca_configuration)}
    schedule_files = {str(path.relative_to(schedule)):file_reference(path)
                      for path in sorted(schedule.rglob("*")) if path.is_file()}
    refs = list(files.values()) + list(schedule_files.values())
    for path in [*data_manifests, *model_manifests, args.target_model / "manifest.json"]:
        refs.extend(artifact_files(path.parent))
    base_ref = file_reference(live / "runtime-manifest.json")
    refs.append(base_ref)
    provenance = model["identity"]["model_provenance"]
    config = DailyPipelineConfig(
        environment_root=args.live.parent.resolve(), pipeline_root=args.output / "pipeline",
        data_root=args.output / "data", model_root=args.output / "model",
        schedule_root=schedule, coverage_path=args.coverage,
        catalog_path=live / "catalog/games.parquet",
        preseason_rankings_path=live / "preseason/rankings.parquet",
        preseason_minutes_path=args.minutes,
        preseason_profiles_path=live / "preseason/profiles.parquet",
        release_root=Path(bindings["historical_runtime"]), code_root=args.code,
        frontend_root=live / "web", live_pointer=args.live,
        matchup_url="isolated://frozen-context", model_track=provenance["model_version"],
        model_code_revision=provenance["code_revision"],
        model_configuration_path=args.hca_configuration,
        model_configuration_sha256=provenance["configuration_sha256"],
        model_provenance_matchup_url=args.model_provenance_matchup_url,
        historical_runtime_root=Path(bindings["historical_runtime"]),
        observation_root=args.observation_root, preserve_schedule_selections=True,
    )
    config.validate()
    causal_hca_replay_config(config)
    if expected_slate_ids(config, target) != set(data["identity"]["game_ids"]):
        raise ValueError("Selected daily data does not cover the complete official slate")
    spec = work_spec(
        season=base["season"], game_date=target,
        source_revisions=data["identity"]["game_sources"],
        model_configuration=provenance, parent_model=base["model"]["bundle_sha256"],
        edition_id=bindings["edition_id"], code_identity=require_clean_code_revision(args.code),
        inputs={ref["path"]:ref for ref in refs}, expected_game_ids=data["identity"]["game_ids"],
    )
    execution = {"files":files, "schedule_files":schedule_files,
                 "base_release_manifest":base_ref,
                 "data_manifests":[file_reference(path) for path in data_manifests],
                 "model_manifests":[file_reference(path) for path in model_manifests],
                 "target_model_manifest":file_reference(args.target_model / "manifest.json"),
                 "historical_data_selection":True, "causal_replay":True,
                 "forecast_replacement":False, "impact_plan":plan}
    spec = bind_execution(spec, config, execution)
    validate_execution(config, spec, approval)
    atomic_json(args.output / "config.json", config_identity(config), immutable=True)
    atomic_json(args.output / "spec.json", spec, immutable=True)
    atomic_json(args.output / "approval.json", approval, immutable=True)
    print({"work_id":spec["work_id"], "target":str(target),
           "existing_model_games":model["identity"]["cumulative_game_count"],
           "files_pinned":len(spec["identity"]["inputs"]), "executed":False})


if __name__ == "__main__":
    main()
