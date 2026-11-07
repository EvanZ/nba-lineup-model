"""Resolve the immutable components of a loaded release exactly once."""

from __future__ import annotations

from nba_lineup_model.storage import website_data as data_io

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.serving_selection import (
    INDEX_FILENAME,
    ObservationSnapshot,
    build_schedule_selection,
    digest,
    read_object,
    verify_schedule_selection,
)

BINDINGS_FILENAME = "serving-bindings.json"


def build_serving_bindings(
    release: Path,
    *,
    runtime_identity: dict[str, Any],
    approved: dict[str, Any],
    historical_runtime: Path,
    observation_root: Path | None = None,
    forecast_registry_root: Path | None = None,
    version_references: dict[str, str] | None = None,
    environment: str = "dev",
) -> dict[str, Any]:
    """Bind already materialized artifacts; do not train or replay anything."""
    historical_runtime = data_io.resolve(historical_runtime)
    model_artifact = "forward_nail_rapm_v1212_residualized_lambda"
    model_root = historical_runtime / "artifacts/models" / model_artifact / "2025-26"
    run_id = str(read_object(model_root / "latest.json")["run_id"])
    historical_files = {}
    roots = [
        model_root / run_id,
        historical_runtime / "artifacts/web",
        historical_runtime / "data/analytical/player_season_panel/player_seasons.parquet",
        historical_runtime / "artifacts/models/analysis/forward_four_factor_offense_defense",
    ]
    for root in roots:
        for path in sorted(data_io.rglob(root, "*")) if data_io.is_dir(root) else [root]:
            if data_io.is_file(path):
                historical_files[str(path.relative_to(historical_runtime))] = file_sha256(path)
    if not historical_files or not data_io.is_file(model_root / run_id / "metadata.json"):
        raise ValueError("Historical runtime inputs are missing")
    schedule_manifest = read_object(release / "schedule-snapshot/manifest.json")
    # Preserve the already approved edition identity. This is not a new replay.
    edition_id = str(
        schedule_manifest.get("candidate_id")
        or schedule_manifest.get("identity")
        or runtime_identity["approved_release_sha256"]
    )
    selection = build_schedule_selection(
        release / "schedule-snapshot",
        release / "selected-schedule",
        edition_id=edition_id,
        observation_root=observation_root,
        forecast_registry_root=forecast_registry_root,
        version_references=version_references,
        environment=environment,
    )
    serving_pointer = read_object(release / "current-season-serving/latest.json")
    lineup_pointer = read_object(release / "current-season-lineups/latest.json")
    references = {
        "serving_bundle": f"current-season-serving/{serving_pointer['bundle_path']}",
        "lineup_bundle": f"current-season-lineups/{lineup_pointer['bundle_path']}",
        "catalog": "catalog/games.parquet",
        "preseason_rankings": "preseason/rankings.parquet",
        "preseason_profiles": "preseason/profiles.parquet",
        "schedule": "selected-schedule",
    }
    components = {
        "model_state_id": approved["model"]["bundle_sha256"],
        "observed_data_revision": approved["data"]["bundle_sha256"],
        "serving_bundle_id": runtime_identity["serving_bundle_sha256"],
        "lineup_bundle_id": runtime_identity["lineup_bundle_sha256"],
        "catalog_id": runtime_identity["catalog_sha256"],
        "forecast_selection_id": selection["selection_map_sha256"],
        "historical_inputs_id": digest(historical_files),
    }
    reconciliation = runtime_identity.get('forecast_initialization', {}).get('reference') or runtime_identity.get('reconciliation_reference')
    if reconciliation:
        components['reconciliation_id'] = reconciliation['content_id']
    bindings = {
        "player_pages": [
            "model_state_id",
            "observed_data_revision",
            "serving_bundle_id",
            "forecast_selection_id",
            "historical_inputs_id",
        ],
        "rankings": ["model_state_id", "serving_bundle_id", "catalog_id", "historical_inputs_id"],
        "teams": [
            "model_state_id",
            "serving_bundle_id",
            "lineup_bundle_id",
            "catalog_id",
            "historical_inputs_id",
        ],
        "lineups": [
            "model_state_id",
            "serving_bundle_id",
            "lineup_bundle_id",
            "historical_inputs_id",
        ],
        "wins": [
            "model_state_id",
            "serving_bundle_id",
            "forecast_selection_id",
            "catalog_id",
            "historical_inputs_id",
        ],
        "schedule": ["model_state_id", "forecast_selection_id", "catalog_id"],
    }
    if reconciliation:
        for keys in bindings.values():
            keys.append('reconciliation_id')
    result = {
        "schema_version": 1,
        "release_id": runtime_identity["approved_release_sha256"],
        "edition_id": edition_id,
        "edition_source": schedule_manifest.get("model_track"),
        "evidence_through_date": approved["evidence_through_date"],
        "references": references,
        "components": components,
        "surface_bindings": {
            name: {key: components[key] for key in keys} for name, keys in bindings.items()
        },
        "historical_runtime": str(historical_runtime),
        "historical_run_id": run_id,
        "historical_files": historical_files,
    }
    result["bindings_sha256"] = digest(result)
    (release / BINDINGS_FILENAME).write_bytes(canonical_json(result) + b"\n")
    return result


def verify_serving_bindings(release: Path) -> dict[str, Any]:
    result = read_object(release / BINDINGS_FILENAME)
    if result.get("schema_version") != 1 or digest(
        {key: value for key, value in result.items() if key != "bindings_sha256"}
    ) != result.get("bindings_sha256"):
        raise ValueError("Serving binding checksum/schema mismatch")
    required = {"player_pages", "rankings", "teams", "lineups", "wins", "schedule"}
    if set(result["surface_bindings"]) != required:
        raise ValueError("Release surface bindings are incomplete")
    for binding in result["surface_bindings"].values():
        for key, value in binding.items():
            if result["components"].get(key) != value:
                raise ValueError(f"Incompatible surface binding: {key}")
    for key, relative in result["references"].items():
        # Publisher generation entries are symlinks to the immutable candidate.
        # The paths inside each component must still be explicit and relative.
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Invalid component path: {key}")
        if not data_io.exists(release / relative):
            raise ValueError(f"Missing bound component: {key}")
    selection = verify_schedule_selection(release / result["references"]["schedule"])
    if (
        selection["edition_id"] != result["edition_id"]
        or selection["selection_map_sha256"] != result["components"]["forecast_selection_id"]
    ):
        raise ValueError("Schedule does not match the selected edition")
    from nba_lineup_model.season.serving_recovery import historical_path
    history = historical_path(release)
    for relative, checksum in result["historical_files"].items():
        path = data_io.resolve(history / relative)
        if not path.is_relative_to(history) or file_sha256(path) != checksum:
            raise ValueError(f"Historical runtime input changed: {relative}")
    return result


@dataclass(frozen=True)
class ReleaseServing:
    root: Path
    manifest: dict[str, Any]
    bindings: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path | None) -> ReleaseServing | None:
        if root is None:
            return None
        resolved = data_io.resolve(Path(root))
        # Old releases remain usable during migration. New release artifacts
        # declaring bindings are required to pass all checks; no downgrade.
        manifest = read_object(resolved / "runtime-manifest.json")
        if not manifest["identity"].get("serving_bindings_sha256"):
            return None
        if manifest["identity"].get("durable_serving_sha256"):
            from nba_lineup_model.storage.website_data import activate
            from nba_lineup_model.storage.s3_lake import _new_s3_client
            from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
            manifest, _ = activate(resolved, manifest, lake_uri=DEFAULT_LAKE_URI, client=_new_s3_client(1))
        bindings = verify_serving_bindings(resolved)
        if manifest["identity"]["serving_bindings_sha256"] != bindings["bindings_sha256"]:
            raise ValueError("Runtime and serving binding identities disagree")
        if bindings["release_id"] != manifest["identity"]["approved_release_sha256"]:
            raise ValueError("Loaded release and serving bindings disagree")
        return cls(resolved, manifest, bindings)

    def path(self, name: str) -> Path:
        return data_io.resolve(self.root / self.bindings["references"][name])

    @property
    def release_id(self) -> str:
        return self.manifest["runtime_sha256"]

    def metadata(self) -> dict[str, Any]:
        return {
            **{
                key: self.bindings[key]
                for key in (
                    "edition_id",
                    "edition_source",
                    "evidence_through_date",
                    "components",
                    "surface_bindings",
                )
            },
            "release_id": self.release_id,
        }

    def evaluator_kwargs(self) -> dict[str, Any]:
        # Historical loaders have relative cache paths. Their process working
        # directory is an immutable release directory and must match this pin.
        from nba_lineup_model.season.serving_recovery import historical_path, inventory
        from nba_lineup_model.storage.website_data import active
        if active() is not None:
            return {
                "run_id": self.bindings["historical_run_id"],
                "artifacts_dir": active().history / "artifacts/models",
                "preseason_rankings_override": self.path("preseason_rankings"),
                "preseason_profiles_override": self.path("preseason_profiles"),
                "current_serving_bundle": self.path("serving_bundle"),
                "current_lineups_bundle": self.path("lineup_bundle"),
            }
        history = historical_path(self.root)
        if inventory(self.root) is not None:
            import os
            os.chdir(history)
        if data_io.resolve(Path.cwd()) != history:
            raise ValueError("API working directory differs from bound historical runtime")
        return {
            "run_id": self.bindings["historical_run_id"],
            "preseason_rankings_override": self.path("preseason_rankings"),
            "preseason_profiles_override": self.path("preseason_profiles"),
            "current_serving_bundle": self.path("serving_bundle"),
            "current_lineups_bundle": self.path("lineup_bundle"),
        }

    def frozen_game_ratings(self, selected: dict) -> dict[int, float]:
        from nba_lineup_model.season.frozen_game_ratings import load_frozen_game_ratings
        model = self.manifest["approved_manifest"]["identity"]["model"]
        from nba_lineup_model.season.serving_recovery import daily_path
        return load_frozen_game_ratings(selected, model_bundle=daily_path(Path(model["bundle_path"]), release=self.root),
            model_sha256=model["bundle_sha256"], preseason_rankings=self.path("preseason_rankings"))

    def observations(self, root: Path | None) -> ObservationSnapshot:
        import os
        if self.manifest['identity'].get('durable_serving_sha256') and not os.environ.get('GESTALT_OBSERVATION_S3_URI'):
            raise ValueError('Durable website observations require an explicit S3 source')
        index = read_object(self.path("schedule") / INDEX_FILENAME)
        return ObservationSnapshot.capture(root, sorted(index["dates"]))
