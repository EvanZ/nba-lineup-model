"""Immutable model checkpoints derived from verified daily data checkpoints."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from nba_lineup_model.season.daily_artifact_common import (
    canonical_json,
    copy_components,
    read_manifest,
    require_logical_name,
    require_sha256,
    verify_components,
    write_manifest,
)
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.schema import validate_season

SCHEMA_VERSION = 1
ARTIFACT_KIND = "daily_model"
DEFAULT_REQUIRED_COMPONENTS = frozenset(
    {
        "player_ratings",
        "minutes_state",
        "rating_trajectory",
        "lineup_values",
        "game_projections",
        "evaluation_metrics",
    }
)


@dataclass(frozen=True)
class PublishedDailyModelArtifact:
    path: Path
    created: bool
    manifest: dict[str, Any]


def publish_daily_model_artifact(
    *,
    root: Path | str,
    data_bundle: Path | str,
    components: Mapping[str, Path | str],
    model_provenance: Mapping[str, object],
    input_data_uri: str | None = None,
    parent_bundle: Path | str | None = None,
    excluded_games: Sequence[Mapping[str, object] | str] = (),
    required_components: Iterable[str] = DEFAULT_REQUIRED_COMPONENTS,
    generated_at: datetime | None = None,
) -> PublishedDailyModelArtifact:
    """Publish one model checkpoint that cites one exact clean data checkpoint."""

    if excluded_games:
        raise ValueError("Daily model artifact cannot publish with excluded games")
    data_manifest = verify_daily_data_artifact(data_bundle)
    data_identity = data_manifest["identity"]
    season = validate_season(str(data_identity["season"]))
    game_date = date.fromisoformat(data_identity["game_date_pacific"])
    parent = _parent_identity(parent_bundle, data_manifest)
    provenance = _normalize_provenance(model_provenance)
    if parent is not None and parent["identity"]["model_provenance"] != provenance:
        raise ValueError("Daily model parent uses different model provenance")
    normalized_data_uri = str(
        input_data_uri
        or f"artifact://daily-data/{data_manifest['bundle_sha256']}"
    ).strip()
    if not normalized_data_uri:
        raise ValueError("Daily model input data URI cannot be empty")
    required = {require_logical_name(name, "Required component") for name in required_components}
    component_names = {
        require_logical_name(name, "Component name") for name in components
    }
    missing_components = sorted(required - component_names)
    if missing_components:
        raise ValueError(
            "Daily model artifact is missing components: " + ", ".join(missing_components)
        )
    previous_game_ids = parent["identity"]["cumulative_game_ids"] if parent else []
    daily_game_ids = list(data_identity["game_ids"])
    overlap = sorted(set(previous_game_ids) & set(daily_game_ids))
    if overlap:
        raise ValueError(
            "Daily model data repeats games from its parent: " + ", ".join(overlap)
        )
    cumulative_game_ids = sorted([*previous_game_ids, *daily_game_ids])

    root_path = Path(root).resolve()
    staging = root_path / f".daily-model.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        inventory = copy_components(components, staging=staging)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "season": season,
            "game_date_pacific": game_date.isoformat(),
            "input_data_sha256": data_manifest["bundle_sha256"],
            "input_data_uri": normalized_data_uri,
            "parent_model_sha256": parent["bundle_sha256"] if parent else None,
            "parent_game_date_pacific": (
                parent["identity"]["game_date_pacific"] if parent else None
            ),
            "daily_game_ids": daily_game_ids,
            "cumulative_game_ids": cumulative_game_ids,
            "cumulative_game_count": len(cumulative_game_ids),
            "model_provenance": provenance,
            "components": inventory,
            "validation": {
                "status": "clean",
                "excluded_game_count": 0,
                "excluded_games": [],
            },
        }
        bundle_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
        manifest = {
            "bundle_sha256": bundle_sha,
            "generated_at": _utc_timestamp(generated_at or datetime.now(UTC)),
            "identity": identity,
        }
        write_manifest(staging, manifest)
        target = (
            root_path
            / f"season={season}"
            / f"game_date_pacific={game_date.isoformat()}"
            / f"model_sha256={bundle_sha}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(staging)
            return PublishedDailyModelArtifact(
                target, False, verify_daily_model_artifact(target, data_bundle=data_bundle)
            )
        staging.replace(target)
        return PublishedDailyModelArtifact(
            target, True, verify_daily_model_artifact(target, data_bundle=data_bundle)
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_daily_model_artifact(
    path: Path | str,
    *,
    data_bundle: Path | str | None = None,
) -> dict[str, Any]:
    bundle = Path(path).resolve()
    manifest = read_manifest(bundle)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Daily model manifest identity is missing")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported daily model artifact schema")
    if identity.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Artifact is not a daily model checkpoint")
    validate_season(str(identity.get("season", "")))
    date.fromisoformat(str(identity.get("game_date_pacific", "")))
    bundle_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    if manifest.get("bundle_sha256") != bundle_sha:
        raise ValueError("Daily model artifact checksum does not match its identity")
    if bundle.name != f"model_sha256={bundle_sha}":
        raise ValueError("Daily model artifact directory is not content-addressed")
    validation = identity.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Daily model validation result is missing")
    if (
        validation.get("status") != "clean"
        or int(validation.get("excluded_game_count", -1)) != 0
        or validation.get("excluded_games") != []
    ):
        raise ValueError("Daily model artifact contains excluded games")
    cumulative = identity.get("cumulative_game_ids")
    if not isinstance(cumulative, list) or cumulative != sorted(set(map(str, cumulative))):
        raise ValueError("Daily model cumulative game coverage is invalid")
    if len(cumulative) != int(identity.get("cumulative_game_count", -1)):
        raise ValueError("Daily model cumulative game count is invalid")
    _normalize_provenance(identity.get("model_provenance", {}))
    verify_components(bundle, identity.get("components"))
    if not str(identity.get("input_data_uri", "")).strip():
        raise ValueError("Daily model input data URI is missing")
    require_sha256(identity.get("input_data_sha256"), "Input data checksum")
    if data_bundle is not None:
        data_manifest = verify_daily_data_artifact(data_bundle)
        if data_manifest["bundle_sha256"] != identity.get("input_data_sha256"):
            raise ValueError("Daily model input data checksum does not match")
        data_identity = data_manifest["identity"]
        if (
            data_identity["season"] != identity["season"]
            or data_identity["game_date_pacific"] != identity["game_date_pacific"]
            or data_identity["game_ids"] != identity.get("daily_game_ids")
        ):
            raise ValueError("Daily model input data identity does not match")
    return manifest


def _parent_identity(
    parent_bundle: Path | str | None,
    data_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    data_identity = data_manifest["identity"]
    data_date = date.fromisoformat(data_identity["game_date_pacific"])
    if parent_bundle is None:
        return None
    parent = verify_daily_model_artifact(parent_bundle)
    parent_identity = parent["identity"]
    if parent_identity["season"] != data_identity["season"]:
        raise ValueError("Daily model parent season does not match")
    parent_date = date.fromisoformat(parent_identity["game_date_pacific"])
    if parent_date + timedelta(days=1) != data_date:
        raise ValueError("Daily model parent must be the immediately preceding Pacific date")
    return parent


def _normalize_provenance(provenance: Mapping[str, object]) -> dict[str, str]:
    required = {"code_revision", "model_version", "configuration_sha256", "prior_sha256"}
    missing = sorted(required - set(provenance))
    if missing:
        raise ValueError("Daily model provenance is missing: " + ", ".join(missing))
    normalized = {str(key): str(value).strip() for key, value in sorted(provenance.items())}
    for key, value in normalized.items():
        if not value:
            raise ValueError(f"Daily model provenance is empty: {key}")
    require_sha256(normalized["configuration_sha256"], "Configuration checksum")
    require_sha256(normalized["prior_sha256"], "Prior checksum")
    return normalized


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Artifact generation time must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
