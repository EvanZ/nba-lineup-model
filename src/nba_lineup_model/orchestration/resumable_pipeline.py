"""Pinned, resumable daily execution; never activates an environment."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from uuid import uuid4

from nba_lineup_model.orchestration.daily_pipeline import (
    CanonicalPipelineSteps, DailyPipelineConfig, require_clean_code_revision,
)
from nba_lineup_model.orchestration.daily_work import (
    DailyWork, digest, file_reference, read_object, verify_reference, verify_spec,
)
from nba_lineup_model.orchestration.repair_plan import require_approval
from nba_lineup_model.season.daily_data_artifact import (
    promote_daily_data_artifact, verify_daily_data_artifact,
)
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.source_policy import require_clean_artifact_sources


def artifact_files(bundle: Path) -> list[dict]:
    if not bundle.is_dir():
        raise ValueError(f"Missing artifact directory: {bundle}")
    return [file_reference(path) for path in sorted(bundle.rglob("*")) if path.is_file()]


def _install_bundle(source: Path, destination: Path) -> None:
    """Only immutable files share disk blocks; never share discovery pointers."""
    if destination.exists():
        for path in source.rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(source)
                if file_reference(target)["sha256"] != file_reference(path)["sha256"]:
                    raise ValueError("Private checkpoint conflicts with selected input")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".seed-{uuid4().hex}"
    try:
        shutil.copytree(source, temporary, copy_function=os.link)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _input(spec: dict, reference: dict) -> Path:
    if reference not in spec["identity"]["inputs"].values():
        raise ValueError("Execution input is not part of the work identity")
    return verify_reference(reference)


class PartitionSteps(CanonicalPipelineSteps):
    def __init__(self, config, candidate, revision, spec):
        self.spec = spec
        execution = spec["identity"]["execution"]
        for relative, reference in {
            **{f"inputs/{key}": ref for key, ref in execution["files"].items()},
            **{f"schedule/{key}": ref for key, ref in execution["schedule_files"].items()},
        }.items():
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Input escapes the private workspace")
            source = _input(spec, reference)
            target = candidate / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if file_reference(target)["sha256"] != reference["sha256"]:
                raise ValueError("Pinned working input changed")
        require_clean_artifact_sources(candidate / "schedule")
        private = replace(
            config, data_root=candidate / "daily-data", model_root=candidate / "daily-model",
            schedule_root=candidate / "schedule", coverage_path=candidate / "inputs/coverage.json",
            catalog_path=candidate / "inputs/catalog.parquet",
            preseason_rankings_path=candidate / "inputs/rankings.parquet",
            preseason_minutes_path=candidate / "inputs/minutes.json",
            preseason_profiles_path=candidate / "inputs/profiles.parquet",
            preserve_schedule_selections=True, activate_after_validation=False,
        )
        if config.model_configuration_path is not None:
            private = replace(private, model_configuration_path=candidate / "inputs/model-configuration.json")
        super().__init__(private, candidate, revision)
        self.target = date.fromisoformat(spec["identity"]["game_date_pacific"])
        self.data_inputs = execution["data_manifests"]
        self.model_inputs = execution["model_manifests"]

    def data(self, target):
        from nba_lineup_model.season.daily_input_selection import build_selected_daily_data
        has_target = False
        for reference in self.data_inputs:
            source = _input(self.spec, reference).parent
            manifest = verify_daily_data_artifact(source)
            identity = manifest["identity"]
            if identity["game_date_pacific"] > target.isoformat():
                raise ValueError("Future data cannot enter this daily work unit")
            if identity["game_date_pacific"] == target.isoformat() and identity["game_sources"] != self.spec["identity"]["source_revisions"]:
                raise ValueError("Existing daily data differs from the pinned source selection")
            for ref in artifact_files(source):
                _input(self.spec, ref)
            destination = (self.config.data_root / f"season={identity['season']}" /
                           f"game_date_pacific={identity['game_date_pacific']}" / source.name)
            _install_bundle(source, destination)
            promote_daily_data_artifact(destination, root=self.config.data_root)
            has_target |= identity["game_date_pacific"] == target.isoformat()
        if not has_target:
            artifact = build_selected_daily_data(
                selection=self.spec["identity"]["source_revisions"],
                catalog_reference=file_reference(self.config.catalog_path),
                season=self.config.season, game_date=target, output_root=self.config.data_root,
            )
            promote_daily_data_artifact(artifact.path, root=self.config.data_root)

    def model(self, target):
        from nba_lineup_model.season.daily_model_replay import _write_latest_pointer
        previous = None
        for reference in self.model_inputs:
            source = _input(self.spec, reference).parent
            manifest = verify_daily_model_artifact(source)
            identity = manifest["identity"]
            if identity["game_date_pacific"] >= target.isoformat():
                raise ValueError("Cannot seed target/future model results")
            if identity["parent_model_sha256"] != previous:
                raise ValueError("Selected model prefix is discontinuous")
            for ref in artifact_files(source):
                _input(self.spec, ref)
            destination = (self.config.model_root / f"season={identity['season']}" /
                           f"game_date_pacific={identity['game_date_pacific']}" / source.name)
            _install_bundle(source, destination)
            previous = manifest["bundle_sha256"]
        if previous != self.spec["identity"]["parent_model"]:
            raise ValueError("Parent model differs from work identity")
        if self.model_inputs and not (self.config.model_root / "latest.json").exists():
            _write_latest_pointer(self.config.model_root, destination)
        super().model(target)

    def stage_receipt(self, stage, result):
        roots = {"data": self.config.data_root, "model": self.config.model_root,
                 "serving": self.candidate / "serving", "lineups": self.candidate / "lineups"}
        if stage == "validate_and_stage":
            from nba_lineup_model.season.production_site_release import verify_production_site_release
            verify_production_site_release(result)
            return {"artifacts": [file_reference(Path(result) / "runtime-manifest.json"),
                                  file_reference(Path(result) / "serving-bindings.json")],
                    "result": {"runtime_release": str(result)}}
        return {"artifacts": artifact_files(roots[stage]), "result": {}}

def validate_execution(config, spec, approval):
    """Derive approval needs from selected history, never trust caller flags."""
    identity = spec["identity"]
    execution = identity["execution"]
    base = read_object(_input(spec, execution["base_release_manifest"]))
    base_id = base.get("runtime_sha256") or base.get("release_sha256")
    active = read_object(config.live_pointer / "runtime-manifest.json")
    if (active.get("runtime_sha256") or active.get("release_sha256")) != base_id:
        raise ValueError("Active release changed; reassess the daily work plan")
    approved = base["approved_manifest"]["identity"]
    target = date.fromisoformat(identity["game_date_pacific"])
    through = date.fromisoformat(approved["evidence_through_date"])
    historical = target <= through
    plan = execution.get("impact_plan")
    if historical or execution.get("historical_data_selection") or execution.get("causal_replay"):
        if not plan or plan["identity"]["base_release_id"] != base_id:
            raise PermissionError("Repair must cite the exact selected base release")
        if target.isoformat() not in plan["identity"]["affected_games"]:
            raise PermissionError("Approved plan does not cover this Pacific date")
        if sorted(plan["identity"]["affected_games"][target.isoformat()]) != identity["expected_game_ids"]:
            raise PermissionError("Approved game scope differs from the work unit")
        if historical or execution.get("causal_replay"):
            require_approval(plan, approval, "causal_replay")
            if target.isoformat() not in plan["identity"]["model_suffix"]:
                raise PermissionError("Work date is outside the approved model suffix")
        if execution.get("historical_data_selection"):
            require_approval(plan, approval, "select_historical_data")
    parent = None
    accepted_prefix_seen = False
    for reference in execution["model_manifests"]:
        manifest = verify_daily_model_artifact(_input(spec, reference).parent)
        info = manifest["identity"]
        if info["parent_model_sha256"] != (parent["bundle_sha256"] if parent else None):
            raise ValueError("Selected model prefix is discontinuous")
        if info["game_date_pacific"] == through.isoformat():
            accepted_prefix_seen = manifest["bundle_sha256"] == approved["model"]["bundle_sha256"]
        if parent is not None and (date.fromisoformat(info["game_date_pacific"]) -
                                  date.fromisoformat(parent["identity"]["game_date_pacific"])).days != 1:
            raise ValueError("Selected model dates are not contiguous")
        parent = manifest
    if not historical and not accepted_prefix_seen:
        raise ValueError("Forward update must extend the accepted model lineage")
    if parent:
        parent_day = date.fromisoformat(parent["identity"]["game_date_pacific"])
        if (target - parent_day).days != 1:
            raise ValueError("Work must immediately follow its selected parent day")
        if parent["bundle_sha256"] != identity["parent_model"]:
            raise ValueError("Parent model differs from pinned identity")
    elif target != config.season_start or identity["parent_model"] is not None:
        raise ValueError("Missing parent model; cannot skip earlier dates")
    selected_data = {}
    for ref in execution["data_manifests"]:
        manifest = verify_daily_data_artifact(_input(spec, ref).parent)
        day = manifest["identity"]["game_date_pacific"]
        if day in selected_data:
            raise ValueError("More than one data revision selected for a date")
        selected_data[day] = manifest["bundle_sha256"]
    for ref in execution["model_manifests"]:
        model = verify_daily_model_artifact(verify_reference(ref).parent)["identity"]
        if selected_data.get(model["game_date_pacific"]) != model["input_data_sha256"]:
            raise ValueError("Repaired input invalidates this parent; request its causal suffix")


def config_identity(config):
    return {k: str(v) if isinstance(v, (Path, date)) else v for k, v in asdict(config).items()}


def bind_execution(spec, config, execution):
    identity = {**spec["identity"], "execution": execution,
                "configuration_digest": digest(config_identity(config))}
    return {"work_id": digest(identity), "identity": identity}


def prepare_partition(config, spec, *, approval=None, steps_class=PartitionSteps):
    config.validate()
    verify_spec(spec)
    identity = spec["identity"]
    execution = identity.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("An explicit execution-input selection is required")
    if identity["season"] != config.season:
        raise ValueError("Work season and pipeline configuration disagree")
    if identity.get("configuration_digest") != digest(config_identity(config)):
        raise ValueError("Configuration differs from the pinned work specification")
    if not config.historical_runtime_root or not config.observation_root:
        raise ValueError("Daily execution requires #55 release-bound serving inputs")
    validate_execution(config, spec, approval)
    plan = execution.get("impact_plan")
    for flag, action in (("historical_data_selection", "select_historical_data"),
                         ("causal_replay", "causal_replay"),
                         ("forecast_replacement", "replace_forecast_edition")):
        if execution.get(flag):
            if not plan:
                raise PermissionError("Historical work requires an impact plan")
            require_approval(plan, approval, action)
    if execution.get("forecast_replacement"):
        raise ValueError("Forecast-edition construction is separate from the daily updater")
    revision = require_clean_code_revision(config.code_root)
    if revision != identity["code_identity"]:
        raise ValueError("Running code differs from the pinned work specification")
    work = DailyWork(config.pipeline_root / "partitions", spec)
    config.pipeline_root.mkdir(parents=True, exist_ok=True)
    with (config.pipeline_root / ".partition-runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidate = work.root / "working"
        candidate.mkdir(parents=True, exist_ok=True)
        steps = steps_class(config, candidate, revision, spec)
        callbacks = {}
        for stage in ("data", "model", "serving", "lineups", "validate_and_stage"):
            def execute(name=stage):
                method = getattr(steps, name)
                result = method(date.fromisoformat(identity["game_date_pacific"])) if name in {
                    "data", "model", "validate_and_stage"} else method()
                return steps.stage_receipt(name, result)
            callbacks[stage] = execute
        result = work.run(callbacks)
    result["runtime_release"] = result["checkpoints"][-1]["result"]["runtime_release"]
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    result = prepare_partition(DailyPipelineConfig.read(args.config), read_object(args.spec),
                               approval=read_object(args.approval) if args.approval else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
