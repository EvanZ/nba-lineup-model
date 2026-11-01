"""Immutable daily data checkpoints for the active NBA season.

A data checkpoint contains only observed/source data.  It deliberately contains
no ratings, minutes estimates, lineup values, projections, or other model output.
Every checkpoint names the exact Pacific slate and the preceding data checkpoint.
"""

from __future__ import annotations

import hashlib
import html
import json
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
from nba_lineup_model.season.schema import validate_season

SCHEMA_VERSION = 1
ARTIFACT_KIND = "daily_data"
POINTER_SCHEMA_VERSION = 1
REPORT_FILENAME = "report.html"
DEFAULT_REQUIRED_COMPONENTS = frozenset(
    {
        "final_games",
        "nba_boxscores",
        "actual_minutes",
        "observed_stints",
        "observed_lineups",
        "player_identity",
        "rosters",
        "injuries",
    }
)
DEFAULT_REQUIRED_GAME_SOURCES = frozenset(
    {"nba_schedule", "nba_box_score", "nba_play_by_play"}
)


@dataclass(frozen=True)
class PublishedDailyDataArtifact:
    path: Path
    created: bool
    manifest: dict[str, Any]


def publish_daily_data_artifact(
    *,
    root: Path | str,
    season: str,
    game_date_pacific: date,
    game_ids: Sequence[str],
    game_sources: Mapping[str, Mapping[str, Mapping[str, object]]],
    game_quality: Mapping[str, Mapping[str, object]],
    components: Mapping[str, Path | str],
    parent_bundle: Path | str | None = None,
    validation_errors: Sequence[Mapping[str, object] | str] = (),
    repairs: Sequence[Mapping[str, object]] = (),
    required_components: Iterable[str] = DEFAULT_REQUIRED_COMPONENTS,
    required_game_sources: Iterable[str] = DEFAULT_REQUIRED_GAME_SOURCES,
    generated_at: datetime | None = None,
) -> PublishedDailyDataArtifact:
    """Validate and publish one fail-closed daily source-data checkpoint."""

    season = validate_season(season)
    normalized_game_ids = _normalize_game_ids(game_ids)
    errors = _normalize_errors(validation_errors)
    if errors:
        raise ValueError(
            "Daily data artifact has unresolved validation errors: "
            + "; ".join(issue["message"] for issue in errors)
        )
    parent = _parent_identity(parent_bundle, season, game_date_pacific)
    sources = _normalize_game_sources(
        game_sources,
        game_ids=normalized_game_ids,
        required_names=set(required_game_sources),
    )
    quality = _normalize_game_quality(game_quality, game_ids=normalized_game_ids)
    required = {require_logical_name(name, "Required component") for name in required_components}
    component_names = {
        require_logical_name(name, "Component name") for name in components
    }
    missing_components = sorted(required - component_names)
    if missing_components:
        raise ValueError(
            "Daily data artifact is missing components: " + ", ".join(missing_components)
        )

    root_path = Path(root).resolve()
    staging = root_path / f".daily-data.tmp-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        inventory = copy_components(components, staging=staging)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "season": season,
            "game_date_pacific": game_date_pacific.isoformat(),
            "parent_data_sha256": parent["bundle_sha256"] if parent else None,
            "parent_game_date_pacific": (
                parent["identity"]["game_date_pacific"] if parent else None
            ),
            "game_ids": normalized_game_ids,
            "game_count": len(normalized_game_ids),
            "game_sources": sources,
            "game_quality": quality,
            "components": inventory,
            "validation": {
                "status": "clean",
                "error_count": 0,
                "errors": [],
                "repairs": _normalize_repairs(repairs),
            },
        }
        report_path = staging / REPORT_FILENAME
        report_path.write_text(_render_report(identity), encoding="utf-8")
        identity["report_sha256"] = _file_sha256(report_path)
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
            / f"game_date_pacific={game_date_pacific.isoformat()}"
            / f"data_sha256={bundle_sha}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(staging)
            return PublishedDailyDataArtifact(target, False, verify_daily_data_artifact(target))
        staging.replace(target)
        return PublishedDailyDataArtifact(target, True, verify_daily_data_artifact(target))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_daily_data_artifact(path: Path | str) -> dict[str, Any]:
    bundle = Path(path).resolve()
    manifest = read_manifest(bundle)
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Daily data manifest identity is missing")
    if identity.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported daily data artifact schema")
    if identity.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Artifact is not a daily data checkpoint")
    validate_season(str(identity.get("season", "")))
    date.fromisoformat(str(identity.get("game_date_pacific", "")))
    bundle_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    if manifest.get("bundle_sha256") != bundle_sha:
        raise ValueError("Daily data artifact checksum does not match its identity")
    if bundle.name != f"data_sha256={bundle_sha}":
        raise ValueError("Daily data artifact directory is not content-addressed")
    game_ids = _normalize_game_ids(identity.get("game_ids", []))
    if len(game_ids) != int(identity.get("game_count", -1)):
        raise ValueError("Daily data game count does not match its manifest")
    sources = identity.get("game_sources")
    if not isinstance(sources, dict) or sorted(sources) != game_ids:
        raise ValueError("Daily data source inventory does not match its exact slate")
    _validate_normalized_sources(sources)
    quality = identity.get("game_quality")
    if not isinstance(quality, dict) or sorted(quality) != game_ids:
        raise ValueError("Daily data quality checks do not match its exact slate")
    _validate_normalized_game_quality(quality)
    validation = identity.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Daily data validation result is missing")
    if validation.get("status") != "clean" or int(validation.get("error_count", -1)) != 0:
        raise ValueError("Daily data artifact contains unresolved validation errors")
    if validation.get("errors") != []:
        raise ValueError("Daily data artifact carries non-empty validation errors")
    verify_components(bundle, identity.get("components"))
    report_sha = require_sha256(identity.get("report_sha256"), "Data report checksum")
    if _file_sha256(bundle / REPORT_FILENAME) != report_sha:
        raise ValueError("Daily data report checksum changed")
    return manifest


def promote_daily_data_artifact(
    bundle_path: Path | str,
    *,
    root: Path | str,
    promoted_at: datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    """Advance the data-chain pointer only when its parent is the current tip."""

    root_path = Path(root).resolve()
    bundle = Path(bundle_path).resolve()
    try:
        relative = bundle.relative_to(root_path)
    except ValueError as error:
        raise ValueError("Daily data artifact must live under its chain root") from error
    manifest = verify_daily_data_artifact(bundle)
    identity = manifest["identity"]
    latest_path = root_path / "latest.json"
    current: dict[str, Any] | None = None
    if latest_path.is_file():
        current = resolve_daily_data_pointer(root_path, _read_json(latest_path))
        if current["bundle_sha256"] == manifest["bundle_sha256"]:
            return _read_json(latest_path), False
        if identity["parent_data_sha256"] != current["bundle_sha256"]:
            raise ValueError("Daily data artifact does not extend the current chain tip")
    elif identity["parent_data_sha256"] is not None:
        raise ValueError("Cannot initialize a daily data pointer from a chained artifact")
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "season": identity["season"],
        "game_date_pacific": identity["game_date_pacific"],
        "bundle_sha256": manifest["bundle_sha256"],
        "bundle_path": relative.as_posix(),
        "promoted_at": _utc_timestamp(promoted_at or datetime.now(UTC)),
    }
    _atomic_write_json(latest_path, pointer)
    history = root_path / "pointer_history" / f"{manifest['bundle_sha256']}.json"
    if not history.exists():
        _atomic_write_json(history, pointer)
    return pointer, True


def resolve_daily_data_pointer(root: Path | str, pointer: Mapping[str, object]) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise ValueError("Unsupported daily data pointer schema")
    if pointer.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Pointer does not reference a daily data artifact")
    raw_path = pointer.get("bundle_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Daily data pointer path is missing")
    bundle = (root_path / raw_path).resolve()
    try:
        bundle.relative_to(root_path)
    except ValueError as error:
        raise ValueError("Daily data pointer escapes its root") from error
    manifest = verify_daily_data_artifact(bundle)
    if manifest["bundle_sha256"] != pointer.get("bundle_sha256"):
        raise ValueError("Daily data pointer checksum does not match its artifact")
    return manifest


def _parent_identity(
    parent_bundle: Path | str | None,
    season: str,
    game_date_pacific: date,
) -> dict[str, Any] | None:
    if parent_bundle is None:
        return None
    parent = verify_daily_data_artifact(parent_bundle)
    parent_identity = parent["identity"]
    if parent_identity["season"] != season:
        raise ValueError("Daily data parent season does not match")
    parent_date = date.fromisoformat(parent_identity["game_date_pacific"])
    if parent_date + timedelta(days=1) != game_date_pacific:
        raise ValueError("Daily data parent must be the immediately preceding Pacific date")
    return parent


def _normalize_game_ids(game_ids: Iterable[object]) -> list[str]:
    normalized = [str(game_id).strip() for game_id in game_ids]
    if any(not game_id for game_id in normalized):
        raise ValueError("Daily data game IDs cannot be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Daily data game IDs must be unique")
    return sorted(normalized)


def _normalize_game_sources(
    sources: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    game_ids: Sequence[str],
    required_names: set[str],
) -> dict[str, dict[str, dict[str, str]]]:
    if sorted(map(str, sources)) != list(game_ids):
        raise ValueError("Daily data source inventory must cover the exact Pacific slate")
    normalized: dict[str, dict[str, dict[str, str]]] = {}
    required = {require_logical_name(name, "Required game source") for name in required_names}
    for game_id in game_ids:
        raw_sources = sources[game_id]
        normalized_names = {
            require_logical_name(name, "Game source name") for name in raw_sources
        }
        missing = sorted(required - normalized_names)
        if missing:
            raise ValueError(f"Game {game_id} is missing sources: {', '.join(missing)}")
        normalized[game_id] = {}
        for raw_name, raw_reference in sorted(raw_sources.items()):
            name = require_logical_name(raw_name, "Game source name")
            if not isinstance(raw_reference, Mapping):
                raise ValueError(f"Game source reference is invalid: {game_id}/{name}")
            uri = str(raw_reference.get("uri", "")).strip()
            if not uri:
                raise ValueError(f"Game source URI is missing: {game_id}/{name}")
            normalized[game_id][name] = {
                "uri": uri,
                "sha256": require_sha256(
                    raw_reference.get("sha256"), f"Game source {game_id}/{name} checksum"
                ),
            }
    return normalized


def _validate_normalized_sources(sources: Mapping[str, object]) -> None:
    for game_id, raw_sources in sources.items():
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise ValueError(f"Daily data game has no source inventory: {game_id}")
        for name, raw_reference in raw_sources.items():
            require_logical_name(name, "Game source name")
            if not isinstance(raw_reference, dict) or not str(raw_reference.get("uri", "")):
                raise ValueError(f"Daily data source reference is invalid: {game_id}/{name}")
            require_sha256(raw_reference.get("sha256"), f"{game_id}/{name} checksum")


def _normalize_game_quality(
    quality: Mapping[str, Mapping[str, object]],
    *,
    game_ids: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if sorted(map(str, quality)) != list(game_ids):
        raise ValueError("Daily data quality checks must cover the exact Pacific slate")
    normalized: dict[str, dict[str, float | int]] = {}
    for game_id in game_ids:
        raw = quality[game_id]
        score_fields = (
            "final_home_score",
            "final_away_score",
            "box_home_points",
            "box_away_points",
        )
        possession_fields = ("play_by_play_possessions", "observed_stint_possessions")
        try:
            values: dict[str, float | int] = {
                field: int(raw[field]) for field in score_fields
            }
            values.update({field: float(raw[field]) for field in possession_fields})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Game quality checks are incomplete: {game_id}") from error
        _validate_game_quality(game_id, values)
        normalized[game_id] = values
    return normalized


def _validate_normalized_game_quality(quality: Mapping[str, object]) -> None:
    for game_id, raw in quality.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Game quality checks are invalid: {game_id}")
        _validate_game_quality(game_id, raw)


def _validate_game_quality(game_id: str, values: Mapping[str, object]) -> None:
    try:
        final_home = int(values["final_home_score"])
        final_away = int(values["final_away_score"])
        box_home = int(values["box_home_points"])
        box_away = int(values["box_away_points"])
        pbp_possessions = float(values["play_by_play_possessions"])
        stint_possessions = float(values["observed_stint_possessions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Game quality checks are incomplete: {game_id}") from error
    if min(final_home, final_away, box_home, box_away) < 0:
        raise ValueError(f"Game {game_id} has a negative score")
    if (final_home, final_away) != (box_home, box_away):
        raise ValueError(f"Game {game_id} fails score conservation")
    if pbp_possessions <= 0 or stint_possessions <= 0:
        raise ValueError(f"Game {game_id} must have positive possession totals")
    if abs(pbp_possessions - stint_possessions) > 1e-9:
        raise ValueError(f"Game {game_id} fails possession conservation")


def _normalize_errors(
    errors: Sequence[Mapping[str, object] | str],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for error in errors:
        if isinstance(error, str):
            message = error.strip()
            game_id = ""
        else:
            message = str(error.get("message", "")).strip()
            game_id = str(error.get("game_id", "")).strip()
        if not message:
            raise ValueError("Daily data validation errors require a message")
        normalized.append({"game_id": game_id, "message": message})
    return normalized


def _normalize_repairs(repairs: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for repair in repairs:
        action = str(repair.get("action", "")).strip()
        if not action:
            raise ValueError("Daily data repairs require an action")
        normalized.append(
            {
                "game_id": str(repair.get("game_id", "")).strip(),
                "action": action,
                "source": str(repair.get("source", "")).strip(),
            }
        )
    return sorted(normalized, key=lambda item: (item["game_id"], item["action"]))


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Artifact generation time must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _render_report(identity: Mapping[str, Any]) -> str:
    rows: list[str] = []
    quality = identity["game_quality"]
    for game_id in identity["game_ids"]:
        game = quality[game_id]
        rows.append(
            "<tr>"
            f"<td>{html.escape(game_id)}</td>"
            f"<td>{game['final_away_score']}-{game['final_home_score']}</td>"
            f"<td>{game['observed_stint_possessions']:.1f}</td>"
            "<td class='ok'>PASS</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4'>No games scheduled.</td></tr>")
    repairs = identity["validation"]["repairs"]
    repair_rows = "".join(
        "<li>"
        f"{html.escape(repair['game_id'] or 'slate')}: "
        f"{html.escape(repair['action'])}"
        + (f" ({html.escape(repair['source'])})" if repair["source"] else "")
        + "</li>"
        for repair in repairs
    ) or "<li>None</li>"
    parent_sha = html.escape(str(identity["parent_data_sha256"] or "GENESIS"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Daily data reconciliation — {html.escape(identity['game_date_pacific'])}</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:960px}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
.ok{{color:#176b3a;font-weight:700}}
code{{background:#f4f4f4;padding:.15rem .3rem}}</style></head><body>
<h1>Daily data reconciliation</h1>
<p><strong>{html.escape(identity['season'])}</strong> · Pacific slate
<strong>{html.escape(identity['game_date_pacific'])}</strong> · {identity['game_count']} games</p>
<p>Status: <span class="ok">CLEAN</span>. All expected games, scores, possessions, players,
and required observed-data components passed publication checks.</p>
<table><thead><tr><th>Game ID</th><th>Away–home score</th>
<th>Possessions</th><th>Checks</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><h2>Repairs</h2><ul>{repair_rows}</ul>
<h2>Lineage</h2><p>Parent data checksum: <code>{parent_sha}</code></p>
</body></html>"""


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
