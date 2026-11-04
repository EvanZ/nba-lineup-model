"""Bind an existing reviewed dev release without replaying data or models.

This staging command does not activate an environment or fabricate approvals.
It verifies the current published model chain and preserves every historical
forecast selection. Activation still belongs to release_publisher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from nba_lineup_model.season.dev_site_release import (
    collect_api_validation,
    publish_dev_site_release,
)
from nba_lineup_model.season.production_site_release import stage_production_site_release
from nba_lineup_model.season.serving_selection import build_schedule_selection, read_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--historical-runtime", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    live = args.live.resolve()
    manifest = read_object(live / "runtime-manifest.json")
    approved = manifest["approved_manifest"]["identity"]
    revision = subprocess.check_output(
        ["git", "-C", str(args.code), "rev-parse", "HEAD"], text=True
    ).strip()
    output = args.output / revision
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "reviewed-schedule"
    if not snapshot.exists():
        source = (live / "schedule").resolve()
        edition = str(read_object(source / "manifest.json")["candidate_id"])
        build_schedule_selection(
            source, snapshot, edition_id=edition, observation_root=args.observations.resolve()
        )
    evidence = collect_api_validation(
        args.api,
        evidence_date=approved["evidence_through_date"],
        daily_model_root=args.model_root,
    )
    (output / "baseline-api-validation.json").write_text(json.dumps(evidence, indent=2))
    publish_dev_site_release(
        daily_data_root=args.data_root,
        daily_model_root=args.model_root,
        serving_root=live / "current-season-serving",
        lineups_root=live / "current-season-lineups",
        schedule_state_dir=snapshot,
        catalog_path=live / "catalog/games.parquet",
        preseason_rankings_path=live / "preseason/rankings.parquet",
        preseason_profiles_path=live / "preseason/profiles.parquet",
        code_revision=revision,
        api_validation=evidence,
        output_root=output / "site-releases",
    )
    candidate = stage_production_site_release(
        dev_site_release_root=output / "site-releases",
        serving_root=live / "current-season-serving",
        lineups_root=live / "current-season-lineups",
        schedule_snapshot_root=snapshot,
        schedule_overlay_path=snapshot,
        catalog_path=live / "catalog/games.parquet",
        preseason_rankings_path=live / "preseason/rankings.parquet",
        preseason_profiles_path=live / "preseason/profiles.parquet",
        code_root=args.code,
        frontend_root=args.frontend,
        output_root=output / "runtime-releases",
        live_pointer=args.live,
        historical_runtime_root=args.historical_runtime,
    )
    current = read_object(candidate / "runtime-manifest.json")["approved_manifest"]["identity"]
    for key in ("data", "model", "serving", "lineups"):
        if current[key]["bundle_sha256"] != approved[key]["bundle_sha256"]:
            raise ValueError(f"Unexpected {key} advance while binding an existing release")
    print(candidate, flush=True)


if __name__ == "__main__":
    main()
