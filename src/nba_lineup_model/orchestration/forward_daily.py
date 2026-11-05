"""Pin a forward date from the accepted chain and run the canonical checkpoints."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from nba_lineup_model.orchestration.daily_work import atomic_json, digest, file_reference, read_object, work_spec
from nba_lineup_model.orchestration.daily_pipeline import require_clean_code_revision
from nba_lineup_model.orchestration.resumable_pipeline import (
    PartitionSteps, artifact_files, bind_execution, config_identity, prepare_partition,
)
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact


def accepted_chain(live):
    approved = read_object(live / "runtime-manifest.json")["approved_manifest"]["identity"]
    tip = Path(approved["model"]["bundle_path"])
    data_root = Path(approved["data"]["bundle_path"]).parents[2]
    model_root = tip.parents[2]
    expected = approved["model"]["bundle_sha256"]
    models, data = [], []
    while True:
        model = verify_daily_model_artifact(tip)
        if model["bundle_sha256"] != expected:
            raise ValueError("Accepted model chain checksum mismatch")
        info = model["identity"]
        observed = data_root / f"season={info['season']}" / f"game_date_pacific={info['game_date_pacific']}" / f"data_sha256={info['input_data_sha256']}"
        verify_daily_model_artifact(tip, data_bundle=observed)
        models.append(tip); data.append(observed)
        expected = info["parent_model_sha256"]
        if not expected:
            break
        tip = model_root / f"season={info['season']}" / f"game_date_pacific={info['parent_game_date_pacific']}" / f"model_sha256={expected}"
    return list(reversed(models)), list(reversed(data))


def capture_daily_inputs(config, day, output):
    """Use canonical final captures and archived injury PDF; never forecast rows as actuals."""
    from nba_lineup_model.season.cdn_lake import ingest_completed_cdn
    from nba_lineup_model.season.daily_data_replay import (
        replay_daily_data_chain, OfficialBoxscore, OfficialInjuryReport,
    )
    from nba_lineup_model.season.observed_revisions import archive_capture
    from nba_lineup_model.web_api.win_projections import _parse_official_injury_report
    coverage = output / "coverage.json"
    if not coverage.exists():
        shutil.copyfile(config.coverage_path, coverage)
    ingest_completed_cdn(state_dir=config.observation_root, season=config.season,
        start_date=day, end_date=day, coverage_path=coverage, ignore_grace=True)
    def box(gid):
        source = config.observation_root / "final" / f"{gid}.nba.json"
        ref = archive_capture(output / "captures", source)
        return OfficialBoxscore(payload=read_object(Path(ref["path"])), sha256=ref["sha256"],
            immutable_uri=Path(ref["path"]).as_uri(),
            source_url=f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json")
    def injury(report_day):
        board = read_object(config.observation_root / "latest" / f"{report_day}.json")
        ref = board["injury_report_artifact"]
        raw = subprocess.check_output(["aws", "s3", "cp", ref["s3_uri"], "-", "--only-show-errors"])
        if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise ValueError("Archived injury PDF checksum mismatch")
        report = _parse_official_injury_report(raw)
        report.update(source_url=board["injury_report"]["source_url"],
                      published_at=board["injury_report"]["published_at"])
        return OfficialInjuryReport(payload=report, sha256=ref["sha256"],
            immutable_uri=ref["s3_uri"], source_url=report["source_url"])
    result = replay_daily_data_chain(start_date=day, end_date=day,
        coverage_path=coverage, output_root=output / "data", game_catalog_path=config.catalog_path,
        season=config.season, boxscore_loader=box, injury_report_loader=injury)
    if len(result) != 1:
        raise ValueError("Forward daily work must produce exactly one complete date")
    return result[0].path, coverage


class ForwardPartitionSteps(PartitionSteps):
    def build_target_model(self, target):
        if self.spec["identity"]["expected_game_ids"]:
            return super().build_target_model(target)
        from nba_lineup_model.season.daily_model_artifact import publish_daily_model_artifact
        from nba_lineup_model.season.daily_model_replay import _write_latest_pointer
        parent = Path(self.model_inputs[-1]["path"]).parent
        manifest = verify_daily_model_artifact(parent)
        source = Path(self.data_inputs[-1]["path"]).parent
        observed = self.config.data_root / f"season={self.config.season}" / f"game_date_pacific={target}" / source.name
        if verify_daily_data_artifact(observed)["identity"]["game_ids"]:
            raise ValueError("Carry-forward requires a verified empty date")
        artifact = publish_daily_model_artifact(root=self.config.model_root,
            data_bundle=observed, parent_bundle=parent,
            components={k:parent / v["path"] for k,v in manifest["identity"]["components"].items()},
            model_provenance=manifest["identity"]["model_provenance"])
        _write_latest_pointer(self.config.model_root, artifact.path)

    def validate_and_stage(self, target):
        # The worker may be a newer revision than the site. Data publication
        # packages the already deployed application, never the worker checkout.
        base = Path(self.spec["identity"]["execution"]["base_release_manifest"]["path"]).parent
        old_config, old_revision = self.config, self.revision
        self.config = replace(self.config, code_root=base / "code", frontend_root=base / "web")
        self.revision = read_object(base / "runtime-manifest.json")["identity"]["code_revision"]
        try:
            return super().validate_and_stage(target)
        finally:
            self.config, self.revision = old_config, old_revision


def advance_daily_environment(template, day: date):
    from nba_lineup_model.orchestration.daily_activation import activation_report_from_work, activate_daily_release
    live = template.live_pointer.resolve(strict=True)
    base = read_object(live / "runtime-manifest.json")
    approved = base["approved_manifest"]["identity"]
    if day != date.fromisoformat(approved["evidence_through_date"]) + timedelta(days=1):
        raise ValueError("Forward processing must immediately follow the accepted evidence date")
    models, data = accepted_chain(live)
    identity = digest({"base": base["runtime_sha256"], "day": str(day),
                       "worker": require_clean_code_revision(template.code_root)})
    output = template.pipeline_root / "forward" / str(day) / identity
    output.mkdir(parents=True, exist_ok=True)
    bindings = read_object(live / "serving-bindings.json")
    config = replace(template, pipeline_root=output / "pipeline", data_root=output / "data",
        model_root=output / "model", schedule_root=live / "selected-schedule",
        catalog_path=live / "catalog/games.parquet", frontend_root=live / "web",
        preseason_rankings_path=live / "preseason/rankings.parquet",
        preseason_profiles_path=live / "preseason/profiles.parquet",
        historical_runtime_root=Path(bindings["historical_runtime"]),
        activate_after_validation=False, preserve_schedule_selections=True)
    config.validate()
    selected_path = output / "selected-data.json"
    if selected_path.exists():
        target = Path(read_object(selected_path)["path"])
        verify_daily_data_artifact(target)
        coverage = output / "inputs/coverage.json"
    else:
        inputs = output / "inputs"
        inputs.mkdir(exist_ok=True)
        target, coverage = capture_daily_inputs(config, day, inputs)
        atomic_json(selected_path, {"path": str(target)}, immutable=True)
    data.append(target)
    observed = verify_daily_data_artifact(target)
    config = replace(config, coverage_path=coverage)
    files = {"coverage.json":file_reference(coverage), "catalog.parquet":file_reference(config.catalog_path),
        "rankings.parquet":file_reference(config.preseason_rankings_path),
        "profiles.parquet":file_reference(config.preseason_profiles_path),
        "minutes.json":file_reference(config.preseason_minutes_path),
        "model-configuration.json":file_reference(config.model_configuration_path)}
    schedule_files = {str(p.relative_to(config.schedule_root)):file_reference(p)
                      for p in sorted(config.schedule_root.rglob("*")) if p.is_file()}
    refs = [file_reference(live / "runtime-manifest.json"), *files.values(), *schedule_files.values()]
    for bundle in [*models, *data]:
        refs.extend(artifact_files(bundle))
    spec = work_spec(season=config.season, game_date=day,
        source_revisions=observed["identity"]["game_sources"],
        model_configuration=approved["model"]["model_provenance"],
        parent_model=approved["model"]["bundle_sha256"], edition_id=bindings["edition_id"],
        code_identity=require_clean_code_revision(config.code_root),
        inputs={r["path"]:r for r in refs}, expected_game_ids=observed["identity"]["game_ids"])
    spec = bind_execution(spec, config, {"files":files, "schedule_files":schedule_files,
        "base_release_manifest":refs[0], "data_manifests":[file_reference(p / "manifest.json") for p in data],
        "model_manifests":[file_reference(p / "manifest.json") for p in models],
        "historical_data_selection":False, "causal_replay":False, "forecast_replacement":False,
        "preserve_application_code":True})
    atomic_json(output / "config.json", config_identity(config), immutable=True)
    atomic_json(output / "spec.json", spec, immutable=True)
    result = prepare_partition(config, spec, steps_class=ForwardPartitionSteps)
    report = activation_report_from_work(config, config.pipeline_root / "partitions" / str(day) / spec["work_id"])
    if template.activate_after_validation:
        activate_daily_release(config, report)
    return {**result, "publication": report}
