"""Inventory and migrate only the artifacts reachable from the active dev release."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from nba_lineup_model.season.daily_data_artifact import (
    verify_daily_data_artifact,
)
from nba_lineup_model.season.daily_model_artifact import (
    verify_daily_model_artifact,
)
from nba_lineup_model.season.durable_artifact_storage import (
    daily_data_reference,
    daily_model_dependencies,
    daily_model_reference,
    publish_daily_data_bundle,
    publish_daily_model_bundle,
)
from nba_lineup_model.season.durable_release_storage import (
    production_release_reference,
    publish_production_release,
    required_release_dependencies,
)
from nba_lineup_model.season.production_site_release import (
    verify_production_site_release,
)
from nba_lineup_model.season.release_publisher import read_active_activation
from nba_lineup_model.storage.artifact_store import (
    ArtifactClient,
    ArtifactReference,
    resolve_artifact_file,
)
from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
from nba_lineup_model.storage.environment_selection import (
    activate_environment_selection,
    read_environment_selection,
)


@dataclass(frozen=True)
class ActiveMigrationPlan:
    environment: str
    activation_generation: int
    activation_operation_id: str
    release: Path
    release_reference: ArtifactReference
    data_bundles: tuple[Path, ...]
    model_bundles: tuple[Path, ...]

    @property
    def dependencies(self) -> tuple[ArtifactReference, ...]:
        references = [
            *(daily_data_reference(path) for path in self.data_bundles),
            *(daily_model_reference(path) for path in self.model_bundles),
        ]
        return tuple(sorted(references))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_kind": "active_artifact_migration_plan",
            "environment": self.environment,
            "activation_generation": self.activation_generation,
            "activation_operation_id": self.activation_operation_id,
            "release": {
                **self.release_reference.as_dict(),
                "source_path": self.release.as_posix(),
            },
            "daily_data": [
                {
                    **daily_data_reference(path).as_dict(),
                    "source_path": path.as_posix(),
                }
                for path in self.data_bundles
            ],
            "daily_model": [
                {
                    **daily_model_reference(path).as_dict(),
                    "source_path": path.as_posix(),
                }
                for path in self.model_bundles
            ],
            "selection_policy": (
                "Only the active recorded release and its exact transitive "
                "daily model/data lineage are included."
            ),
        }


def _bundle_path(root: Path, reference: ArtifactReference) -> Path:
    return (
        root
        / f"season={reference.season}"
        / f"game_date_pacific={reference.game_date_pacific.isoformat()}"
        / reference.content_id
    )


def build_active_migration_plan(
    live_pointer: Path | str,
) -> ActiveMigrationPlan:
    """Follow canonical identities from the recorded activation, never a scan."""
    active = read_active_activation(Path(live_pointer))
    if active is None:
        raise ValueError("Dev migration requires a recorded active release")
    if active.get("environment") != "dev":
        raise ValueError("This migration path is restricted to dev")

    release = Path(str(active["release_path"])).resolve(strict=True)
    runtime = verify_production_site_release(release)
    if runtime["runtime_sha256"] != active["runtime_sha256"]:
        raise ValueError(
            "Active release bytes differ from the recorded activation"
        )
    release_reference = production_release_reference(runtime)
    approved = runtime["approved_manifest"]["identity"]
    data_tip = Path(str(approved["data"]["bundle_path"])).resolve(strict=True)
    model_tip = Path(str(approved["model"]["bundle_path"])).resolve(strict=True)
    data_root = data_tip.parents[2]
    model_root = model_tip.parents[2]

    data_paths: dict[ArtifactReference, Path] = {}
    model_paths: dict[ArtifactReference, Path] = {}
    current = model_tip
    while True:
        verify_daily_model_artifact(current)
        model_reference = daily_model_reference(current)
        if model_reference in model_paths:
            raise ValueError("Selected daily model ancestry contains a cycle")
        model_paths[model_reference] = current
        dependencies = daily_model_dependencies(current)
        data_dependency = next(
            dependency
            for dependency in dependencies
            if dependency.artifact_kind == "daily_data"
        )
        data_path = _bundle_path(data_root, data_dependency).resolve(
            strict=True
        )
        verify_daily_data_artifact(data_path)
        if daily_data_reference(data_path) != data_dependency:
            raise ValueError(
                "Selected model resolves a different daily data artifact"
            )
        data_paths[data_dependency] = data_path
        parents = [
            dependency
            for dependency in dependencies
            if dependency.artifact_kind == "daily_model"
        ]
        if not parents:
            break
        if len(parents) != 1:
            raise ValueError(
                "Selected daily model has multiple parent identities"
            )
        current = _bundle_path(model_root, parents[0]).resolve(strict=True)

    verify_daily_data_artifact(data_tip)
    tip_data_reference = daily_data_reference(data_tip)
    if tip_data_reference not in data_paths:
        raise ValueError(
            "Approved data tip is absent from the selected model lineage"
        )
    required = set(required_release_dependencies(runtime))
    available = {*data_paths, *model_paths}
    if not required.issubset(available):
        raise ValueError(
            "Active release dependencies are absent from its selected lineage"
        )

    return ActiveMigrationPlan(
        environment="dev",
        activation_generation=int(active["generation"]),
        activation_operation_id=str(active["operation_id"]),
        release=release,
        release_reference=release_reference,
        data_bundles=tuple(
            data_paths[reference] for reference in sorted(data_paths)
        ),
        model_bundles=tuple(
            model_paths[reference] for reference in sorted(model_paths)
        ),
    )


def migrate_active_dev_artifacts(
    plan: ActiveMigrationPlan,
    *,
    cache_root: Path,
    client: ArtifactClient | None = None,
    lake_uri: str = DEFAULT_LAKE_URI,
) -> dict[str, object]:
    """Publish the planned lineage, release, immutable selection, and readback."""
    if plan.environment != "dev":
        raise ValueError("Only dev migration is authorized")
    data_publications = [
        publish_daily_data_bundle(
            path,
            cache_root=cache_root,
            client=client,
            lake_uri=lake_uri,
        )
        for path in plan.data_bundles
    ]
    model_publications = [
        publish_daily_model_bundle(
            path,
            cache_root=cache_root,
            client=client,
            lake_uri=lake_uri,
        )
        for path in plan.model_bundles
    ]
    release_publication = publish_production_release(
        plan.release,
        cache_root=cache_root,
        dependencies=plan.dependencies,
        client=client,
        lake_uri=lake_uri,
    )
    selected = read_environment_selection(
        "dev",
        client=client,
        lake_uri=lake_uri,
    )
    previous_generation = (
        int(selected["current"]["generation"]) if selected else 0
    )
    selection = activate_environment_selection(
        environment="dev",
        release=release_publication.reference,
        operation_id=(
            "ec2-active-migration-"
            f"{plan.activation_generation}-"
            f"{plan.release_reference.content_id}"
        ),
        reason=(
            "Migrate recorded EC2 dev activation "
            f"{plan.activation_generation} to immutable S3 storage"
        ),
        expected_previous_generation=previous_generation,
        client=client,
        lake_uri=lake_uri,
    )
    fresh_cache = (
        cache_root
        / "migration-readback"
        / plan.release_reference.content_id
        / uuid4().hex
    )
    restored_manifest = resolve_artifact_file(
        reference=release_publication.reference,
        relative_path="runtime-manifest.json",
        cache_root=fresh_cache,
        client=client,
        lake_uri=lake_uri,
    )
    if restored_manifest.read_bytes() != (
        plan.release / "runtime-manifest.json"
    ).read_bytes():
        raise RuntimeError(
            "Fresh-cache runtime manifest differs from the active release"
        )
    return {
        "status": "verified",
        "data_artifact_count": len(data_publications),
        "model_artifact_count": len(model_publications),
        "release": release_publication.reference.as_dict(),
        "selection_generation": selection.generation,
        "selection_activated": selection.activated,
        "fresh_cache_manifest": restored_manifest.as_posix(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-pointer", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = build_active_migration_plan(args.live_pointer)
    if not args.execute:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return
    if args.cache_root is None:
        raise SystemExit("--cache-root is required with --execute")
    result = migrate_active_dev_artifacts(
        plan,
        cache_root=args.cache_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
